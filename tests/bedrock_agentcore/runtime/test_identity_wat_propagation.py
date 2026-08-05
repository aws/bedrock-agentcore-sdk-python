"""Tests for X-Amz-Bedrock-AgentCore-Identity-WAT header propagation."""

import contextvars
from unittest.mock import MagicMock

from bedrock_agentcore._utils.identity_propagation import (
    _inject_identity_wat_header,
    register_identity_wat_propagation,
)
from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
from bedrock_agentcore.runtime.models import (
    IDENTITY_WAT_HEADER,
    is_forwardable_header,
)


class TestIdentityWatHeader:
    """Tests for the IDENTITY_WAT_HEADER constant and forwarding rules."""

    def test_identity_wat_header_value(self):
        """Test that the header constant has the correct value."""
        assert IDENTITY_WAT_HEADER == "X-Amz-Bedrock-AgentCore-Identity-WAT"

    def test_identity_wat_header_is_forwardable(self):
        """Test that the Identity WAT header passes the forwardable check."""
        assert is_forwardable_header(IDENTITY_WAT_HEADER) is True

    def test_identity_wat_header_is_forwardable_case_insensitive(self):
        """Test that the Identity WAT header passes regardless of casing."""
        assert is_forwardable_header("x-amz-bedrock-agentcore-identity-wat") is True
        assert is_forwardable_header("X-AMZ-BEDROCK-AGENTCORE-IDENTITY-WAT") is True

    def test_other_x_amz_headers_still_blocked(self):
        """Test that other x-amz- headers are still blocked."""
        assert is_forwardable_header("X-Amz-Date") is False
        assert is_forwardable_header("X-Amz-Security-Token") is False
        assert is_forwardable_header("x-amz-content-sha256") is False


class TestInjectIdentityWatHeader:
    """Tests for the _inject_identity_wat_header event handler."""

    def test_injects_header_when_wat_present(self):
        """Test that the WAT header is injected when present in context."""
        token = "my-wat-token"
        BedrockAgentCoreContext.set_workload_access_token(token)

        request = MagicMock()
        request.headers = {}
        _inject_identity_wat_header(request)

        assert request.headers[IDENTITY_WAT_HEADER] == token

    def test_does_not_inject_when_wat_absent(self):
        """Test that no header is injected when no WAT in context."""
        ctx = contextvars.Context()

        def test_in_new_context():
            request = MagicMock()
            request.headers = {}
            _inject_identity_wat_header(request)
            return request.headers

        headers = ctx.run(test_in_new_context)
        assert IDENTITY_WAT_HEADER not in headers


class TestRegisterIdentityWatPropagation:
    """Tests for the register_identity_wat_propagation utility."""

    def test_registers_event_handler_per_operation(self):
        """Test that event handlers are registered for each operation."""
        mock_client = MagicMock()
        register_identity_wat_propagation(mock_client)

        calls = mock_client.meta.events.register.call_args_list
        registered_events = [call[0][0] for call in calls]

        assert "before-sign.bedrock-agentcore.InvokeAgentRuntime" in registered_events
        assert "before-sign.bedrock-agentcore.InvokeAgentRuntimeCommand" in registered_events
        assert "before-sign.bedrock-agentcore.InvokeHarness" in registered_events
        assert "before-sign.bedrock-agentcore.InvokeGateway" in registered_events

    def test_does_not_register_for_identity_operations(self):
        """Test that Identity operations are not registered."""
        mock_client = MagicMock()
        register_identity_wat_propagation(mock_client)

        calls = mock_client.meta.events.register.call_args_list
        registered_events = [call[0][0] for call in calls]

        for event in registered_events:
            assert "GetWorkloadAccessToken" not in event
            assert "GetResourceOauth2Token" not in event


class TestAppExtractsWatHeader:
    """Tests for WAT header extraction in _build_request_context."""

    def test_workload_access_token_stored_in_context(self):
        """Test that app extracts WorkloadAccessToken and stores in context."""
        from bedrock_agentcore.runtime.app import BedrockAgentCoreApp

        app = BedrockAgentCoreApp()
        mock_request = MagicMock()
        mock_request.headers = {
            "WorkloadAccessToken": "wat-from-runtime",
        }

        app._build_request_context(mock_request)

        assert BedrockAgentCoreContext.get_workload_access_token() == "wat-from-runtime"
