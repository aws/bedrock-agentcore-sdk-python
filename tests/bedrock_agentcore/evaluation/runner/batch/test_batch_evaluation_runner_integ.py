"""Integration test for BatchEvaluationRunner against a real AgentCore runtime agent.

Requires AWS credentials with access to the agent runtime.
Run with:
    python -m pytest tests/bedrock_agentcore/evaluation/runner/batch/test_batch_evaluation_runner_integ.py \
        -v -s --log-cli-level=INFO
"""

import json
import logging
from datetime import datetime, timezone

import boto3
import pytest

from bedrock_agentcore.evaluation.runner.batch.batch_evaluation_models import (
    BatchEvaluationRunConfig,
    BatchEvaluatorConfig,
    CloudWatchSessionSourceConfig,
)
from bedrock_agentcore.evaluation.runner.batch.batch_evaluation_runner import (
    BatchEvaluationRunner,
)
from bedrock_agentcore.evaluation.runner.dataset_types import Dataset, PredefinedScenario, Turn
from bedrock_agentcore.evaluation.runner.invoker_types import AgentInvokerInput, AgentInvokerOutput

logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

TEST_AGENT_ARN = "arn:aws:bedrock-agentcore:us-west-2:053460373529:runtime/MyEvoProject_MyAgent-D6pczK2cUA"
TEST_REGION = "us-west-2"

_RUNTIME_ID = TEST_AGENT_ARN.split("/")[-1]
_LOG_GROUP = f"/aws/bedrock-agentcore/runtimes/{_RUNTIME_ID}-DEFAULT"

TEST_EXECUTION_ROLE_ARN = "arn:aws:iam::053460373529:role/BedrockAgentCoreEvaluationRole"


def _create_agent_invoker():
    client = boto3.client("bedrock-agentcore", region_name=TEST_REGION)

    def invoke(invoker_input: AgentInvokerInput) -> AgentInvokerOutput:
        payload = invoker_input.payload
        if isinstance(payload, str):
            payload = json.dumps({"prompt": payload}).encode()
        elif isinstance(payload, dict):
            payload = json.dumps(payload).encode()

        print(f"\n[INVOKER] session_id={invoker_input.session_id} payload={payload}")
        response = client.invoke_agent_runtime(
            agentRuntimeArn=TEST_AGENT_ARN,
            runtimeSessionId=invoker_input.session_id,
            payload=payload,
        )

        raw_output = response["response"].read().decode("utf-8")
        text_parts = []
        for line in raw_output.splitlines():
            if line.startswith("data: "):
                chunk = line[len("data: "):]
                if chunk.startswith('"') and chunk.endswith('"'):
                    chunk = json.loads(chunk)
                text_parts.append(chunk)
        agent_output = "".join(text_parts) if text_parts else raw_output
        print(f"[INVOKER] Agent output: {agent_output[:300]}")

        return AgentInvokerOutput(agent_output=agent_output)

    return invoke


def _create_dataset() -> Dataset:
    return Dataset(
        scenarios=[
            PredefinedScenario(
                scenario_id="healthcare-000",
                turns=[
                    Turn(
                        input="How does the flu spread?",
                        expected_response="Through respiratory droplets from coughs and sneezes, "
                        "and by touching contaminated surfaces.",
                    ),
                    Turn(
                        input="My coworker came to work sick yesterday. Am I at risk?",
                        expected_response="Yes, close contact with a sick person increases your risk "
                        "of catching the flu.",
                    ),
                    Turn(
                        input="How long after exposure would I show symptoms?",
                        expected_response="The flu incubation period is 1-4 days, typically about 2 days.",
                    ),
                ],
                assertions=["Provide accurate information about flu transmission and exposure"],
            ),
            PredefinedScenario(
                scenario_id="healthcare-001",
                turns=[
                    Turn(
                        input="How does the flu spread?",
                        expected_response="Through respiratory droplets from coughs and sneezes, "
                        "and by touching contaminated surfaces.",
                    ),
                    Turn(
                        input="My coworker came to work sick yesterday. Am I at risk?",
                        expected_response="Yes, close contact with a sick person increases your risk "
                        "of catching the flu.",
                    ),
                    Turn(
                        input="How long after exposure would I show symptoms?",
                        expected_response="The flu incubation period is 1-4 days, typically about 2 days.",
                    ),
                ],
                assertions=["Provide accurate information about flu transmission and exposure"],
            ),
        ]
    )


@pytest.mark.integ
def test_batch_evaluation_runner_with_healthcare_agent():
    """Run BatchEvaluationRunner end-to-end against the MyEvoProject agent."""
    print("\n[TEST] Creating agent invoker...")
    agent_invoker = _create_agent_invoker()

    print("[TEST] Creating dataset (2 scenarios, 3 turns each)...")
    dataset = _create_dataset()

    print("[TEST] Creating BatchEvaluationRunner...")
    runner = BatchEvaluationRunner(region=TEST_REGION)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config = BatchEvaluationRunConfig(
        name=f"HealthcareBatchIntegTest_{timestamp}",
        execution_role_arn=TEST_EXECUTION_ROLE_ARN,
        evaluator_config=BatchEvaluatorConfig(
            evaluator_ids=["Builtin.Helpfulness", "Builtin.Correctness"],
        ),
        session_source=CloudWatchSessionSourceConfig(
            service_names=[_RUNTIME_ID],
            log_group_names=[_LOG_GROUP],
            ingestion_delay_seconds=180,
        ),
        max_concurrent_scenarios=2,
        polling_timeout_seconds=1800,
        polling_interval_seconds=10,
    )

    print(f"[TEST] Running batch evaluation (name={config.name}, log_group={_LOG_GROUP})...")
    result = runner.run_dataset_evaluation(
        config=config,
        dataset=dataset,
        agent_invoker=agent_invoker,
    )

    print("\n[TEST] Result:")
    print(f"  batch_evaluate_id:  {result.batch_evaluate_id}")
    print(f"  batch_evaluate_arn: {result.batch_evaluate_arn}")
    print(f"  status:             {result.status}")
    print(f"  created_at:         {result.created_at}")

    if result.evaluation_results:
        er = result.evaluation_results
        print(f"  sessions_completed: {er.sessions_completed}")
        print(f"  sessions_failed:    {er.sessions_failed}")
        print(f"  total_sessions:     {er.total_sessions}")
        if er.evaluator_summaries:
            for summary in er.evaluator_summaries:
                print(f"  evaluator: {summary.evaluator_id or summary.evaluator_name}")
                if summary.statistics:
                    print(f"    average_score: {summary.statistics.average_score}")

    if result.agent_invocation_failures:
        print(f"  agent_invocation_failures: {[f.scenario_id for f in result.agent_invocation_failures]}")

    assert result.status == "COMPLETED"
    assert result.batch_evaluate_id
    assert not result.agent_invocation_failures, (
        f"Scenarios failed during agent invocation: {result.agent_invocation_failures}"
    )

    print("\n[TEST] Fetching evaluation results from CloudWatch...")
    if result.output_data_config:
        print(f"  log_group:  {result.output_data_config.log_group_name}")
        print(f"  log_stream: {result.output_data_config.log_stream_name}")
        eval_events = runner.fetch_evaluation_events(result)
        print(f"  events_fetched: {len(eval_events)}")
        for i, event in enumerate(eval_events):
            print(f"  event[{i}]: {json.dumps(event, default=str)}")
    else:
        print("  output_data_config not present in result — skipping CloudWatch fetch")
