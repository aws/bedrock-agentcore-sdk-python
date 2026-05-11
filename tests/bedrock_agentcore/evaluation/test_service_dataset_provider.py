"""Tests for ServiceDatasetProvider."""

from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.evaluation.runner.dataset_providers import ServiceDatasetProvider
from bedrock_agentcore.evaluation.runner.dataset_types import (
    Dataset,
    PredefinedScenario,
    SimulatedScenario,
)

PATCH_CLIENT = "bedrock_agentcore.evaluation.dataset_client.DatasetClient"
PATCH_REQUESTS = "bedrock_agentcore.evaluation.runner.dataset_providers.requests"


def _jsonl(*examples):
    """Build a JSONL string from example dicts."""
    import json

    return "\n".join(json.dumps(e) for e in examples)


class TestServiceDatasetProvider:
    def _make_provider(self, jsonl_content, dataset_id="ds-123", version_id=None, get_response=None):
        mock_client = MagicMock()
        if get_response is None:
            get_response = {"datasetId": dataset_id, "status": "ACTIVE", "downloadUrl": "https://example.com/dataset.jsonl"}
        mock_client.get_dataset.return_value = get_response

        mock_response = MagicMock()
        mock_response.text = jsonl_content
        mock_response.raise_for_status = MagicMock()

        with patch(PATCH_CLIENT, return_value=mock_client), patch(PATCH_REQUESTS) as mock_requests:
            mock_requests.get.return_value = mock_response
            provider = ServiceDatasetProvider(dataset_id=dataset_id, version_id=version_id, region_name="us-west-2")
            return provider.get_dataset(), mock_client, mock_requests

    def test_get_dataset_predefined(self):
        content = _jsonl(
            {
                "exampleId": "e1",
                "scenario_id": "s1",
                "turns": [{"input": "Hello", "expected_response": "Hi!"}],
                "assertions": ["Be polite"],
                "expected_trajectory": ["greet"],
            },
            {
                "exampleId": "e2",
                "scenario_id": "s2",
                "turns": [{"input": "What is 2+2?"}],
            },
        )

        dataset, mock_client, mock_requests = self._make_provider(content)

        assert isinstance(dataset, Dataset)
        assert len(dataset.scenarios) == 2

        s1 = dataset.scenarios[0]
        assert isinstance(s1, PredefinedScenario)
        assert s1.scenario_id == "s1"
        assert s1.turns[0].input == "Hello"
        assert s1.turns[0].expected_response == "Hi!"
        assert s1.assertions == ["Be polite"]
        assert s1.expected_trajectory == ["greet"]

        s2 = dataset.scenarios[1]
        assert isinstance(s2, PredefinedScenario)
        assert s2.scenario_id == "s2"

    def test_get_dataset_simulated(self):
        content = _jsonl(
            {
                "exampleId": "e1",
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
        )

        dataset, _, _ = self._make_provider(content, dataset_id="ds-456")

        assert isinstance(dataset, Dataset)
        assert len(dataset.scenarios) == 1
        scenario = dataset.scenarios[0]
        assert isinstance(scenario, SimulatedScenario)
        assert scenario.scenario_id == "sim-1"
        assert scenario.actor_profile.goal == "Get a refund"
        assert scenario.max_turns == 5
        assert scenario.assertions == ["Agent should empathize"]

    def test_downloads_from_presigned_url(self):
        content = _jsonl({"scenario_id": "s1", "turns": [{"input": "hi"}]})

        _, mock_client, mock_requests = self._make_provider(content)

        mock_client.get_dataset.assert_called_once_with(datasetId="ds-123")
        mock_requests.get.assert_called_once_with("https://example.com/dataset.jsonl")

    def test_get_dataset_with_version_id(self):
        content = _jsonl({"scenario_id": "s1", "turns": [{"input": "hi"}]})

        _, mock_client, _ = self._make_provider(content, dataset_id="ds-123", version_id="1")

        mock_client.get_dataset.assert_called_once_with(datasetId="ds-123", datasetVersion="1")

    def test_get_dataset_no_download_url_raises(self):
        mock_client = MagicMock()
        mock_client.get_dataset.return_value = {"datasetId": "ds-123", "status": "CREATING"}

        with patch(PATCH_CLIENT, return_value=mock_client), patch(PATCH_REQUESTS):
            provider = ServiceDatasetProvider(dataset_id="ds-123")
            with pytest.raises(ValueError, match="no downloadUrl"):
                provider.get_dataset()

    def test_get_dataset_empty_raises(self):
        with pytest.raises(ValueError, match="scenarios must not be empty"):
            self._make_provider("", dataset_id="ds-empty")
