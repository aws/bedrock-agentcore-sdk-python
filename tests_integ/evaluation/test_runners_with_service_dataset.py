"""Integration tests for OnDemandEvaluationDatasetRunner and BatchEvaluationRunner
using ServiceDatasetProvider.

These tests verify the full pipeline:
  ServiceDatasetProvider → Dataset → Runner → Agent Invocation → Evaluation

Requires a deployed agent runtime. Set INTEG_AGENT_NAME env var to the name of a
READY agent runtime, or the test will attempt to create one using defaults in helpers.py.

Run with:
    uv run pytest tests_integ/evaluation/test_runners_with_service_dataset.py -xvs --log-cli-level=INFO
"""

import os
import time

import pytest

from bedrock_agentcore.evaluation.dataset_client import DatasetClient
from bedrock_agentcore.evaluation.runner.dataset_providers import ServiceDatasetProvider
from bedrock_agentcore.evaluation.runner.on_demand.config import EvaluationRunConfig, EvaluatorConfig
from bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner import OnDemandEvaluationDatasetRunner

from .helpers import get_or_create_agent_runtime, make_agent_invoker


@pytest.mark.integration
@pytest.mark.skip(reason="Requires a warm deployed agent runtime. See helpers.py for setup.")
class TestOnDemandRunnerWithServiceDataset:
    """OnDemandEvaluationDatasetRunner + ServiceDatasetProvider end-to-end."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        agent_name = os.environ.get("INTEG_AGENT_NAME", None)

        kwargs = {"region": cls.region}
        if agent_name:
            kwargs["agent_name"] = agent_name

        agent_info = get_or_create_agent_runtime(**kwargs)
        cls.runtime_arn = agent_info["runtime_arn"]
        cls.runtime_id = agent_info["runtime_id"]

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
                            "turns": [
                                {"input": "Hello", "expected_response": "Hi there!"}
                            ],
                            "assertions": ["Agent should respond politely"],
                        },
                        {
                            "scenario_id": "math",
                            "turns": [
                                {"input": "What is 2+2?", "expected_response": "4"}
                            ],
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

    def test_on_demand_runner_executes_scenarios(self):
        """OnDemandRunner invokes agent for each scenario from ServiceDatasetProvider."""
        provider = ServiceDatasetProvider(
            dataset_id=self.dataset_id,
            region_name=self.region,
        )
        dataset = provider.get_dataset()

        runner = OnDemandEvaluationDatasetRunner(region=self.region)
        invoker = make_agent_invoker(self.runtime_arn, self.region)

        from bedrock_agentcore.evaluation.agent_span_collector import CloudWatchAgentSpanCollector

        collector = CloudWatchAgentSpanCollector(
            log_group_name=f"/aws/bedrock-agentcore/runtimes/{self.runtime_id}-DEFAULT",
            region=self.region,
        )

        config = EvaluationRunConfig(
            evaluator_config=EvaluatorConfig(evaluator_ids=["Builtin.Helpfulness"]),
            evaluation_delay_seconds=60,
            max_concurrent_scenarios=2,
        )

        result = runner.run(
            config=config,
            dataset=dataset,
            agent_invoker=invoker,
            span_collector=collector,
        )

        assert len(result.scenario_results) == 2
        scenario_ids = {r.scenario_id for r in result.scenario_results}
        assert "greeting" in scenario_ids
        assert "math" in scenario_ids
