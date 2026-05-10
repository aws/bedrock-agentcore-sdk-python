"""Tests for ServiceDatasetProvider."""

from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.evaluation.runner.dataset_providers import ServiceDatasetProvider
from bedrock_agentcore.evaluation.runner.dataset_types import (
    Dataset,
    PredefinedScenario,
    SimulatedScenario,
)

PATCH_TARGET = "bedrock_agentcore.evaluation.dataset_client.DatasetClient"


class TestServiceDatasetProvider:
    def _make_provider(self, mock_client_instance, dataset_id="ds-123", version_id=None, region_name="us-west-2"):
        with patch(PATCH_TARGET, return_value=mock_client_instance):
            provider = ServiceDatasetProvider(dataset_id=dataset_id, version_id=version_id, region_name=region_name)
            return provider.get_dataset()

    def test_get_dataset_predefined(self):
        mock_client = MagicMock()
        mock_client.list_dataset_examples.return_value = {
            "examples": [
                {
                    "scenario_id": "s1",
                    "turns": [
                        {"input": "Hello", "expected_response": "Hi!"},
                    ],
                    "assertions": ["Be polite"],
                    "expected_trajectory": ["greet"],
                },
                {
                    "scenario_id": "s2",
                    "turns": [{"input": "What is 2+2?"}],
                },
            ],
        }

        dataset = self._make_provider(mock_client)

        assert isinstance(dataset, Dataset)
        assert len(dataset.scenarios) == 2

        s1 = dataset.scenarios[0]
        assert isinstance(s1, PredefinedScenario)
        assert s1.scenario_id == "s1"
        assert len(s1.turns) == 1
        assert s1.turns[0].input == "Hello"
        assert s1.turns[0].expected_response == "Hi!"
        assert s1.assertions == ["Be polite"]
        assert s1.expected_trajectory == ["greet"]

        s2 = dataset.scenarios[1]
        assert isinstance(s2, PredefinedScenario)
        assert s2.scenario_id == "s2"

    def test_get_dataset_simulated(self):
        mock_client = MagicMock()
        mock_client.list_dataset_examples.return_value = {
            "examples": [
                {
                    "scenario_id": "sim-1",
                    "scenario_description": "Frustrated customer",
                    "actor_profile": {
                        "traits": {"personality": "impatient"},
                        "context": "Waiting 3 days",
                        "goal": "Get a refund",
                    },
                    "input": "I want my money back!",
                    "max_turns": 5,
                    "assertions": ["Agent should empathize"],
                },
            ],
        }

        dataset = self._make_provider(mock_client, dataset_id="ds-456")

        assert isinstance(dataset, Dataset)
        assert len(dataset.scenarios) == 1
        scenario = dataset.scenarios[0]
        assert isinstance(scenario, SimulatedScenario)
        assert scenario.scenario_id == "sim-1"
        assert scenario.actor_profile.goal == "Get a refund"
        assert scenario.max_turns == 5
        assert scenario.assertions == ["Agent should empathize"]

    def test_get_dataset_with_pagination(self):
        mock_client = MagicMock()
        mock_client.list_dataset_examples.side_effect = [
            {
                "examples": [{"scenario_id": "s1", "turns": [{"input": "a"}]}],
                "nextToken": "token-1",
            },
            {
                "examples": [{"scenario_id": "s2", "turns": [{"input": "b"}]}],
            },
        ]

        dataset = self._make_provider(mock_client, dataset_id="ds-789")

        assert len(dataset.scenarios) == 2
        assert mock_client.list_dataset_examples.call_count == 2

    def test_get_dataset_with_version_id(self):
        mock_client = MagicMock()
        mock_client.list_dataset_examples.return_value = {
            "examples": [{"scenario_id": "s1", "turns": [{"input": "hi"}]}],
        }

        with patch(PATCH_TARGET, return_value=mock_client):
            provider = ServiceDatasetProvider(
                dataset_id="ds-123",
                version_id="v-1",
                region_name="us-west-2",
            )
            provider.get_dataset()

        call_kwargs = mock_client.list_dataset_examples.call_args[1]
        assert call_kwargs["datasetId"] == "ds-123"
        assert call_kwargs["datasetVersion"] == "v-1"

    def test_get_dataset_empty_raises(self):
        mock_client = MagicMock()
        mock_client.list_dataset_examples.return_value = {"examples": []}

        with pytest.raises(ValueError, match="scenarios must not be empty"):
            self._make_provider(mock_client, dataset_id="ds-empty")
