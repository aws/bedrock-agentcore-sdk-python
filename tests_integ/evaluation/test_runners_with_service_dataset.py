"""Integration tests for OnDemandEvaluationDatasetRunner using DatasetManagementServiceProvider.

These tests verify the full pipeline:
  DatasetManagementServiceProvider → Dataset → Runner → Agent Invocation → Evaluation

Required env vars:
    INTEG_AGENT_RUNTIME_ARN: ARN of a deployed, invokable agent runtime
        (region is extracted from the ARN automatically)

Optional env vars:
    INTEG_AGENT_LOG_GROUP: CloudWatch log group for the agent's spans
        (defaults to /aws/bedrock-agentcore/runtimes/{runtime_id}-DEFAULT)

Run with:
    export INTEG_AGENT_RUNTIME_ARN=<agent-runtime-arn>
    uv run pytest tests_integ/evaluation/test_runners_with_service_dataset.py -xvs
"""

import json
import os
import time
import uuid

import boto3
import pytest

from bedrock_agentcore.evaluation.dataset_client import DatasetClient
from bedrock_agentcore.evaluation.runner.dataset_providers import DatasetManagementServiceProvider
from bedrock_agentcore.evaluation.runner.invoker_types import AgentInvokerInput, AgentInvokerOutput
from bedrock_agentcore.evaluation.runner.on_demand.config import EvaluationRunConfig, EvaluatorConfig
from bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner import OnDemandEvaluationDatasetRunner

RUNTIME_ARN = os.environ.get("INTEG_AGENT_RUNTIME_ARN")
REGION = RUNTIME_ARN.split(":")[3] if RUNTIME_ARN else os.environ.get("BEDROCK_TEST_REGION", "us-west-2")


def _make_invoker(runtime_arn: str, region: str):
    """Create an invoker compatible with the hosted echo agent."""
    dp = boto3.client("bedrock-agentcore", region_name=region)

    def invoker(input: AgentInvokerInput) -> AgentInvokerOutput:
        prompt = input.payload if isinstance(input.payload, str) else json.dumps(input.payload)
        resp = dp.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn,
            payload=json.dumps({"prompt": prompt}).encode(),
            runtimeSessionId=(
                input.session_id if input.session_id and len(input.session_id) >= 33 else str(uuid.uuid4())
            ),
        )
        body = resp["response"].read().decode()
        result = json.loads(body) if body else {}
        return AgentInvokerOutput(agent_output=result.get("result", result))

    return invoker


@pytest.mark.integration
@pytest.mark.skipif(not RUNTIME_ARN, reason="INTEG_AGENT_RUNTIME_ARN not set")
class TestOnDemandRunnerWithServiceDataset:
    """OnDemandEvaluationDatasetRunner + DatasetManagementServiceProvider end-to-end."""

    @classmethod
    def setup_class(cls):
        cls.region = REGION
        cls.runtime_arn = RUNTIME_ARN
        cls.runtime_id = cls.runtime_arn.split("/")[-1]  # type: ignore[union-attr]

        cls.log_group = os.environ.get(
            "INTEG_AGENT_LOG_GROUP",
            f"/aws/bedrock-agentcore/runtimes/{cls.runtime_id}-DEFAULT",
        )

        cls.client = DatasetClient(region_name=cls.region)
        cls.test_prefix = f"sdk_integ_runner_{int(time.time())}"
        cls.dataset_ids = []

        ds = cls.client.create_dataset_and_wait(
            datasetName=f"{cls.test_prefix}_ondemand",
            schemaType="AGENTCORE_EVALUATION_PREDEFINED_V1",
            source={
                "inlineExamples": {
                    "examples": [
                        {
                            "scenario_id": "greeting",
                            "turns": [{"input": "Hello", "expected_response": "Hi there!"}],
                            "assertions": ["Agent should respond politely"],
                        },
                        {
                            "scenario_id": "math",
                            "turns": [{"input": "What is 2+2?", "expected_response": "4"}],
                        },
                    ]
                }
            },
        )
        cls.dataset_id = ds["datasetId"]
        cls.dataset_ids.append(cls.dataset_id)

    @classmethod
    def teardown_class(cls):
        for did in cls.dataset_ids:
            try:
                cls.client.delete_dataset(datasetId=did)
            except Exception as e:
                print(f"Failed to delete dataset {did}: {e}")

    @pytest.mark.order(1)
    def test_on_demand_runner_executes_scenarios(self):
        """OnDemandRunner invokes agent for each scenario from DatasetManagementServiceProvider."""
        provider = DatasetManagementServiceProvider(
            dataset_id=self.dataset_id,
            client=self.client,
        )
        dataset = provider.get_dataset()
        assert len(dataset.scenarios) == 2

        runner = OnDemandEvaluationDatasetRunner(region=self.region)
        invoker = _make_invoker(self.runtime_arn, self.region)

        from bedrock_agentcore.evaluation.agent_span_collector import CloudWatchAgentSpanCollector

        collector = CloudWatchAgentSpanCollector(
            log_group_name=self.log_group,
            region=self.region,
        )

        config = EvaluationRunConfig(
            evaluator_config=EvaluatorConfig(evaluator_ids=["Builtin.Helpfulness"]),
            evaluation_delay_seconds=30,
            max_concurrent_scenarios=2,
        )

        result = runner.run(
            config=config,
            dataset=dataset,
            agent_invoker=invoker,
            span_collector=collector,
        )

        # Both scenarios should have been executed
        assert len(result.scenario_results) == 2
        scenario_ids = {r.scenario_id for r in result.scenario_results}
        assert "greeting" in scenario_ids
        assert "math" in scenario_ids
        # Both should complete (agent invocation succeeded)
        for sr in result.scenario_results:
            assert sr.status == "COMPLETED", f"Scenario {sr.scenario_id} failed: {sr.error}"
