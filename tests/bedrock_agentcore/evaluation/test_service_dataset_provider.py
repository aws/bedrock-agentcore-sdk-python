"""Tests for ServiceDatasetProvider."""

from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.evaluation.runner.dataset_providers import ServiceDatasetProvider
from bedrock_agentcore.evaluation.runner.dataset_types import (
    Dataset,
    PredefinedScenario,
    SimulatedScenario,
)

PATCH_REQUESTS = "bedrock_agentcore.evaluation.runner.dataset_providers.requests"


def _jsonl(*examples):
    """Build a JSONL string from example dicts."""
    import json

    return "\n".join(json.dumps(e) for e in examples)


class TestServiceDatasetProvider:
    def _run_provider(
        self, jsonl_content, dataset_id="ds-123", version_id=None, schema_type="AGENTCORE_EVALUATION_PREDEFINED_V1"
    ):
        mock_client = MagicMock()
        mock_client.get_dataset.return_value = {
            "datasetId": dataset_id,
            "status": "ACTIVE",
            "schemaType": schema_type,
            "downloadUrl": "https://example.com/dataset.jsonl",
        }

        mock_response = MagicMock()
        mock_response.content = jsonl_content.encode("utf-8")
        mock_response.raise_for_status = MagicMock()

        with patch(PATCH_REQUESTS) as mock_requests:
            mock_requests.get.return_value = mock_response
            provider = ServiceDatasetProvider(dataset_id=dataset_id, version_id=version_id, client=mock_client)
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

        dataset, mock_client, mock_requests = self._run_provider(content)

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

        dataset, _, _ = self._run_provider(
            content, dataset_id="ds-456", schema_type="AGENTCORE_EVALUATION_SIMULATED_V1"
        )

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

        _, mock_client, mock_requests = self._run_provider(content)

        mock_client.get_dataset.assert_called_once_with(datasetId="ds-123")
        mock_requests.get.assert_called_once_with("https://example.com/dataset.jsonl", timeout=60)

    def test_get_dataset_with_version_id(self):
        content = _jsonl({"scenario_id": "s1", "turns": [{"input": "hi"}]})

        _, mock_client, _ = self._run_provider(content, dataset_id="ds-123", version_id="1")

        mock_client.get_dataset.assert_called_once_with(datasetId="ds-123", datasetVersion="1")

    def test_get_dataset_no_download_url_raises(self):
        mock_client = MagicMock()
        mock_client.get_dataset.return_value = {
            "datasetId": "ds-123",
            "status": "CREATING",
            "schemaType": "AGENTCORE_EVALUATION_PREDEFINED_V1",
        }

        provider = ServiceDatasetProvider(dataset_id="ds-123", client=mock_client)
        with pytest.raises(ValueError, match="no downloadUrl"):
            provider.get_dataset()

    def test_get_dataset_empty_raises(self):
        mock_client = MagicMock()
        mock_client.get_dataset.return_value = {
            "datasetId": "ds-empty",
            "status": "ACTIVE",
            "schemaType": "AGENTCORE_EVALUATION_PREDEFINED_V1",
            "downloadUrl": "https://example.com/dataset.jsonl",
        }

        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.raise_for_status = MagicMock()

        with patch(PATCH_REQUESTS) as mock_requests:
            mock_requests.get.return_value = mock_response
            provider = ServiceDatasetProvider(dataset_id="ds-empty", client=mock_client)
            with pytest.raises(ValueError, match="scenarios must not be empty"):
                provider.get_dataset()

    def test_unsupported_schema_type_raises(self):
        mock_client = MagicMock()
        mock_client.get_dataset.return_value = {
            "datasetId": "ds-123",
            "status": "ACTIVE",
            "schemaType": "RAGAS_V1",
            "downloadUrl": "https://example.com/dataset.jsonl",
        }

        provider = ServiceDatasetProvider(dataset_id="ds-123", client=mock_client)
        with pytest.raises(ValueError, match="not supported by the evaluation runners"):
            provider.get_dataset()

    def test_download_failure_raises_runtime_error(self):
        import requests as real_requests

        mock_client = MagicMock()
        mock_client.get_dataset.return_value = {
            "datasetId": "ds-123",
            "status": "ACTIVE",
            "schemaType": "AGENTCORE_EVALUATION_PREDEFINED_V1",
            "downloadUrl": "https://example.com/dataset.jsonl",
        }

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = real_requests.HTTPError("403 Forbidden")

        with patch(PATCH_REQUESTS) as mock_requests:
            mock_requests.get.return_value = mock_response
            mock_requests.RequestException = real_requests.RequestException
            provider = ServiceDatasetProvider(dataset_id="ds-123", client=mock_client)
            with pytest.raises(RuntimeError, match="Couldn't download dataset"):
                provider.get_dataset()
