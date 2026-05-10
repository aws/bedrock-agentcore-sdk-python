"""Integration tests for ServiceDatasetProvider.

Tests that ServiceDatasetProvider can fetch a dataset from the service
and return it as an SDK Dataset object usable by OnDemandEvaluationDatasetRunner.

Run with:
    uv run pytest tests_integ/evaluation/test_service_dataset_provider.py -xvs --log-cli-level=INFO
"""

import os
import time

import pytest

from bedrock_agentcore.evaluation.dataset_client import DatasetClient
from bedrock_agentcore.evaluation.runner.dataset_providers import ServiceDatasetProvider
from bedrock_agentcore.evaluation.runner.dataset_types import (
    Dataset,
    PredefinedScenario,
    SimulatedScenario,
)


@pytest.mark.integration
class TestServiceDatasetProvider:
    """Tests ServiceDatasetProvider with both PREDEFINED and SYNTHETIC schema types."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.client = DatasetClient(region_name=cls.region)
        cls.test_prefix = f"sdk_integ_provider_{int(time.time())}"
        cls.dataset_ids = []

        # Create a PREDEFINED_TURNS dataset
        predefined = cls.client.create_dataset_and_wait(
            datasetName=f"{cls.test_prefix}_predefined",
            schemaType="AGENTCORE_EVALUATION_PREDEFINED_V1",
            source={
                "inlineExamples": {
                    "examples": [
                        {
                            "scenario_id": "greeting-scenario",
                            "turns": [
                                {"input": "Hello", "expected_response": "Hi!"},
                                {"input": "How are you?", "expected_response": "I'm good!"},
                            ],
                            "assertions": ["Agent should be polite"],
                            "expected_trajectory": ["greet_user"],
                        },
                        {
                            "scenario_id": "math-scenario",
                            "turns": [
                                {"input": "What is 2+2?", "expected_response": "4"},
                            ],
                        },
                    ]
                }
            },
        )
        cls.predefined_dataset_id = predefined["datasetId"]
        cls.dataset_ids.append(cls.predefined_dataset_id)

        # Create a version for version-specific fetch test
        cls.client.create_dataset_version_and_wait(datasetId=cls.predefined_dataset_id)

        # Create a SYNTHETIC dataset
        synthetic = cls.client.create_dataset_and_wait(
            datasetName=f"{cls.test_prefix}_synthetic",
            schemaType="AGENTCORE_EVALUATION_SIMULATED_V1",
            source={
                "inlineExamples": {
                    "examples": [
                        {
                            "scenario_id": "frustrated-customer",
                            "scenario_description": "Customer wants a refund",
                            "actor_profile": {
                                "traits": {"personality": "impatient"},
                                "context": "Has been waiting 3 days",
                                "goal": "Get a refund",
                            },
                            "input": "I want my money back!",
                            "max_turns": 5,
                            "assertions": ["Agent should empathize"],
                        },
                    ]
                }
            },
        )
        cls.synthetic_dataset_id = synthetic["datasetId"]
        cls.dataset_ids.append(cls.synthetic_dataset_id)

    @classmethod
    def teardown_class(cls):
        for did in cls.dataset_ids:
            try:
                cls.client.delete_dataset(datasetId=did)
            except Exception as e:
                print(f"Failed to delete dataset {did}: {e}")

    # --- Predefined turns tests ---

    @pytest.mark.order(1)
    def test_get_predefined_dataset(self):
        """ServiceDatasetProvider returns a valid SDK Dataset with PredefinedScenarios."""
        provider = ServiceDatasetProvider(
            dataset_id=self.predefined_dataset_id,
            region_name=self.region,
        )
        dataset = provider.get_dataset()

        assert isinstance(dataset, Dataset)
        assert len(dataset.scenarios) == 2
        for scenario in dataset.scenarios:
            assert isinstance(scenario, PredefinedScenario)

    @pytest.mark.order(2)
    def test_predefined_fields_preserved(self):
        """Scenario fields (turns, assertions, trajectory) are preserved."""
        provider = ServiceDatasetProvider(
            dataset_id=self.predefined_dataset_id,
            region_name=self.region,
        )
        dataset = provider.get_dataset()

        greeting = next(s for s in dataset.scenarios if s.scenario_id == "greeting-scenario")
        assert len(greeting.turns) == 2
        assert greeting.turns[0].input == "Hello"
        assert greeting.turns[0].expected_response == "Hi!"
        assert greeting.assertions == ["Agent should be polite"]
        assert greeting.expected_trajectory == ["greet_user"]

    @pytest.mark.order(3)
    def test_get_predefined_with_version(self):
        """ServiceDatasetProvider can fetch a specific version."""
        versions_resp = self.client.list_dataset_versions(datasetId=self.predefined_dataset_id)
        version_id = str(versions_resp["versions"][0]["datasetVersion"])

        provider = ServiceDatasetProvider(
            dataset_id=self.predefined_dataset_id,
            version_id=version_id,
            region_name=self.region,
        )
        dataset = provider.get_dataset()

        assert isinstance(dataset, Dataset)
        assert len(dataset.scenarios) == 2

    # --- Synthetic tests ---

    @pytest.mark.order(4)
    def test_get_synthetic_dataset(self):
        """ServiceDatasetProvider returns SimulatedScenarios for SYNTHETIC schema."""
        provider = ServiceDatasetProvider(
            dataset_id=self.synthetic_dataset_id,
            region_name=self.region,
        )
        dataset = provider.get_dataset()

        assert isinstance(dataset, Dataset)
        assert len(dataset.scenarios) == 1
        scenario = dataset.scenarios[0]
        assert isinstance(scenario, SimulatedScenario)
        assert scenario.scenario_id == "frustrated-customer"
        assert scenario.actor_profile.goal == "Get a refund"
        assert scenario.max_turns == 5
        assert scenario.assertions == ["Agent should empathize"]
