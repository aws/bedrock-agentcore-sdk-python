"""Tests for AgentCoreRuntimeClient passthrough, *_and_wait, and orchestration methods."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.runtime.agent_core_runtime_client import AgentCoreRuntimeClient


class TestRuntimeClientPassthrough:
    """Tests for __getattr__ passthrough."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = AgentCoreRuntimeClient(session=mock_session)
        client.cp_client = Mock()
        client.dp_client = Mock()
        return client

    def test_cp_method_forwarded(self):
        client = self._make_client()
        client.cp_client.get_agent_runtime.return_value = {"agentRuntimeId": "r-123"}
        result = client.get_agent_runtime(agentRuntimeId="r-123")
        client.cp_client.get_agent_runtime.assert_called_once_with(agentRuntimeId="r-123")
        assert result["agentRuntimeId"] == "r-123"

    def test_dp_method_forwarded(self):
        client = self._make_client()
        client.dp_client.invoke_agent_runtime.return_value = {"output": "hello"}
        result = client.invoke_agent_runtime(agentRuntimeId="r-123")
        client.dp_client.invoke_agent_runtime.assert_called_once_with(agentRuntimeId="r-123")
        assert result["output"] == "hello"

    def test_snake_case_kwargs_converted(self):
        client = self._make_client()
        client.cp_client.get_agent_runtime.return_value = {"agentRuntimeId": "r-123"}
        client.get_agent_runtime(agent_runtime_id="r-123")
        client.cp_client.get_agent_runtime.assert_called_once_with(agentRuntimeId="r-123")

    def test_non_allowlisted_method_raises(self):
        client = self._make_client()
        with pytest.raises(AttributeError, match="has no attribute"):
            client.not_a_real_method()

    def test_all_cp_methods_in_allowlist(self):
        expected = {
            "create_agent_runtime",
            "update_agent_runtime",
            "get_agent_runtime",
            "delete_agent_runtime",
            "list_agent_runtimes",
            "create_agent_runtime_endpoint",
            "get_agent_runtime_endpoint",
            "update_agent_runtime_endpoint",
            "delete_agent_runtime_endpoint",
            "list_agent_runtime_endpoints",
            "list_agent_runtime_versions",
            "delete_agent_runtime_version",
        }
        assert expected == AgentCoreRuntimeClient._ALLOWED_CP_METHODS

    def test_all_dp_methods_in_allowlist(self):
        expected = {"invoke_agent_runtime", "stop_runtime_session"}
        assert expected == AgentCoreRuntimeClient._ALLOWED_DP_METHODS


class TestRuntimeAndWait:
    """Tests for *_and_wait methods."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = AgentCoreRuntimeClient(session=mock_session)
        client.cp_client = Mock()
        client.dp_client = Mock()
        return client

    def test_create_agent_runtime_and_wait(self):
        client = self._make_client()
        client.cp_client.create_agent_runtime.return_value = {"agentRuntimeId": "r-123"}
        client.cp_client.get_agent_runtime.return_value = {
            "status": "READY",
            "agentRuntimeId": "r-123",
        }
        result = client.create_agent_runtime_and_wait(agentRuntimeName="test")
        assert result["status"] == "READY"

    def test_create_agent_runtime_and_wait_failed(self):
        client = self._make_client()
        client.cp_client.create_agent_runtime.return_value = {"agentRuntimeId": "r-123"}
        client.cp_client.get_agent_runtime.return_value = {
            "status": "CREATE_FAILED",
            "failureReason": "bad config",
        }
        with pytest.raises(RuntimeError, match="CREATE_FAILED"):
            client.create_agent_runtime_and_wait(agentRuntimeName="test")

    def test_update_agent_runtime_and_wait(self):
        client = self._make_client()
        client.cp_client.update_agent_runtime.return_value = {"agentRuntimeId": "r-123"}
        client.cp_client.get_agent_runtime.return_value = {
            "status": "READY",
            "agentRuntimeId": "r-123",
        }
        result = client.update_agent_runtime_and_wait(agentRuntimeId="r-123")
        assert result["status"] == "READY"

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    def test_delete_agent_runtime_and_wait(self, _mock_sleep):
        client = self._make_client()
        client.cp_client.delete_agent_runtime.return_value = {"agentRuntimeId": "r-123"}
        client.cp_client.get_agent_runtime.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "gone"}},
            "GetAgentRuntime",
        )
        client.delete_agent_runtime_and_wait(agentRuntimeId="r-123")
        client.cp_client.delete_agent_runtime.assert_called_once()

    def test_get_agent_runtime_endpoint_and_wait(self):
        client = self._make_client()
        client.cp_client.update_agent_runtime_endpoint.return_value = {
            "name": "DEFAULT",
        }
        client.cp_client.get_agent_runtime_endpoint.return_value = {
            "status": "READY",
            "endpointName": "DEFAULT",
        }
        result = client.update_agent_runtime_endpoint_and_wait(
            agentRuntimeId="r-123",
            endpointName="DEFAULT",
        )
        assert result["status"] == "READY"


class TestRuntimeOrchestration:
    """Tests for get_aggregated_status and teardown."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = AgentCoreRuntimeClient(session=mock_session)
        client.cp_client = Mock()
        client.dp_client = Mock()
        return client

    def test_get_aggregated_status(self):
        client = self._make_client()
        client.cp_client.get_agent_runtime.return_value = {"status": "READY"}
        client.cp_client.get_agent_runtime_endpoint.return_value = {"status": "READY"}
        result = client.get_aggregated_status("r-123")
        assert result["runtime"]["status"] == "READY"
        assert result["endpoint"]["status"] == "READY"

    def test_get_aggregated_status_not_found(self):
        client = self._make_client()
        client.cp_client.get_agent_runtime.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": ""}},
            "GetAgentRuntime",
        )
        client.cp_client.get_agent_runtime_endpoint.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": ""}},
            "GetAgentRuntimeEndpoint",
        )
        result = client.get_aggregated_status("r-123")
        assert "error" in result["runtime"]
        assert "error" in result["endpoint"]

    def test_get_aggregated_status_reraises_non_not_found(self):
        client = self._make_client()
        client.cp_client.get_agent_runtime.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
            "GetAgentRuntime",
        )
        with pytest.raises(ClientError):
            client.get_aggregated_status("r-123")

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    def test_teardown_deletes_endpoint_then_runtime(self, _mock_sleep):
        client = self._make_client()
        client.cp_client.get_agent_runtime_endpoint.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": ""}},
            "GetAgentRuntimeEndpoint",
        )
        client.cp_client.delete_agent_runtime.return_value = {"agentRuntimeId": "r-123"}
        client.cp_client.get_agent_runtime.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": ""}},
            "GetAgentRuntime",
        )
        client.teardown_endpoint_and_runtime("r-123")
        client.cp_client.delete_agent_runtime_endpoint.assert_called_once()
        client.cp_client.delete_agent_runtime.assert_called_once()

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    def test_teardown_skips_missing_endpoint(self, _mock_sleep):
        client = self._make_client()
        client.cp_client.delete_agent_runtime_endpoint.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": ""}},
            "DeleteAgentRuntimeEndpoint",
        )
        client.cp_client.delete_agent_runtime.return_value = {"agentRuntimeId": "r-123"}
        client.cp_client.get_agent_runtime.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": ""}},
            "GetAgentRuntime",
        )
        client.teardown_endpoint_and_runtime("r-123")
        client.cp_client.delete_agent_runtime.assert_called_once()
