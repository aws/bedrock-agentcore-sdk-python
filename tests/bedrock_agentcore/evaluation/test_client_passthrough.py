"""Tests for EvaluationClient passthrough and *_and_wait methods."""

from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.evaluation.client import EvaluationClient


class TestEvaluationClientPassthrough:
    """Tests for __getattr__ passthrough."""

    def _make_client(self):
        with patch("boto3.client"):
            client = EvaluationClient(region_name="us-west-2")
        client._cp_client = Mock()
        client._dp_client = Mock()
        return client

    def test_cp_method_forwarded(self):
        client = self._make_client()
        client._cp_client.get_evaluator.return_value = {"evaluatorId": "e-123"}
        result = client.get_evaluator(evaluatorId="e-123")
        client._cp_client.get_evaluator.assert_called_once_with(evaluatorId="e-123")
        assert result["evaluatorId"] == "e-123"

    def test_dp_method_forwarded(self):
        client = self._make_client()
        client._dp_client.evaluate.return_value = {"evaluationResults": []}
        result = client.evaluate(evaluatorId="e-123")
        client._dp_client.evaluate.assert_called_once_with(evaluatorId="e-123")
        assert result["evaluationResults"] == []

    def test_snake_case_kwargs_converted(self):
        client = self._make_client()
        client._cp_client.get_evaluator.return_value = {"evaluatorId": "e-123"}
        client.get_evaluator(evaluator_id="e-123")
        client._cp_client.get_evaluator.assert_called_once_with(evaluatorId="e-123")

    def test_non_allowlisted_method_raises(self):
        client = self._make_client()
        with pytest.raises(AttributeError, match="has no attribute"):
            client.not_a_real_method()

    def test_all_cp_methods_in_allowlist(self):
        expected = {
            "create_evaluator",
            "get_evaluator",
            "list_evaluators",
            "update_evaluator",
            "delete_evaluator",
            "create_online_evaluation_config",
            "get_online_evaluation_config",
            "list_online_evaluation_configs",
            "update_online_evaluation_config",
            "delete_online_evaluation_config",
        }
        assert expected == EvaluationClient._ALLOWED_CP_METHODS

    def test_all_dp_methods_in_allowlist(self):
        expected = {"evaluate"}
        assert expected == EvaluationClient._ALLOWED_DP_METHODS


class TestEvaluationAndWait:
    """Tests for *_and_wait methods."""

    def _make_client(self):
        with patch("boto3.client"):
            client = EvaluationClient(region_name="us-west-2")
        client._cp_client = Mock()
        client._dp_client = Mock()
        return client

    def test_create_evaluator_and_wait(self):
        client = self._make_client()
        client._cp_client.create_evaluator.return_value = {"evaluatorId": "e-123"}
        client._cp_client.get_evaluator.return_value = {
            "status": "ACTIVE",
            "evaluatorId": "e-123",
        }
        result = client.create_evaluator_and_wait(evaluatorName="test")
        assert result["status"] == "ACTIVE"

    def test_create_evaluator_and_wait_failed(self):
        client = self._make_client()
        client._cp_client.create_evaluator.return_value = {"evaluatorId": "e-123"}
        client._cp_client.get_evaluator.return_value = {"status": "CREATE_FAILED"}
        with pytest.raises(RuntimeError, match="CREATE_FAILED"):
            client.create_evaluator_and_wait(evaluatorName="test")

    def test_update_evaluator_and_wait(self):
        client = self._make_client()
        client._cp_client.update_evaluator.return_value = {"evaluatorId": "e-123"}
        client._cp_client.get_evaluator.return_value = {
            "status": "ACTIVE",
            "evaluatorId": "e-123",
        }
        result = client.update_evaluator_and_wait(evaluatorId="e-123")
        assert result["status"] == "ACTIVE"

    def test_create_online_eval_config_and_wait(self):
        client = self._make_client()
        client._cp_client.create_online_evaluation_config.return_value = {
            "onlineEvaluationConfigId": "c-123",
        }
        client._cp_client.get_online_evaluation_config.return_value = {
            "status": "ACTIVE",
            "onlineEvaluationConfigId": "c-123",
        }
        result = client.create_online_evaluation_config_and_wait(
            onlineEvaluationConfigName="test",
        )
        assert result["status"] == "ACTIVE"

    def test_update_online_eval_config_and_wait(self):
        client = self._make_client()
        client._cp_client.update_online_evaluation_config.return_value = {
            "onlineEvaluationConfigId": "c-123",
        }
        client._cp_client.get_online_evaluation_config.return_value = {
            "status": "ACTIVE",
            "onlineEvaluationConfigId": "c-123",
        }
        result = client.update_online_evaluation_config_and_wait(
            onlineEvaluationConfigId="c-123",
        )
        assert result["status"] == "ACTIVE"

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    def test_delete_evaluator_and_wait(self, _mock_sleep):
        client = self._make_client()
        client._cp_client.delete_evaluator.return_value = {"evaluatorId": "e-123"}
        client._cp_client.get_evaluator.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "gone"}},
            "GetEvaluator",
        )
        client.delete_evaluator_and_wait(evaluatorId="e-123")
        client._cp_client.delete_evaluator.assert_called_once()

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    def test_delete_online_eval_config_and_wait(self, _mock_sleep):
        client = self._make_client()
        client._cp_client.delete_online_evaluation_config.return_value = {
            "onlineEvaluationConfigId": "c-123",
        }
        client._cp_client.get_online_evaluation_config.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "gone"}},
            "GetOnlineEvaluationConfig",
        )
        client.delete_online_evaluation_config_and_wait(
            onlineEvaluationConfigId="c-123",
        )
        client._cp_client.delete_online_evaluation_config.assert_called_once()
