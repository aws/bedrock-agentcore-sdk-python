"""Tests for Bedrock AgentCore context functionality."""

import contextvars

from bedrock_agentcore.runtime.context import BedrockAgentCoreContext


class TestBedrockAgentCoreContext:
    """Test BedrockAgentCoreContext functionality."""

    def test_set_and_get_workload_access_token(self):
        """Test setting and getting workload access token."""
        token = "test-token-123"

        BedrockAgentCoreContext.set_workload_access_token(token)
        result = BedrockAgentCoreContext.get_workload_access_token()

        assert result == token

    def test_get_workload_access_token_when_none_set(self):
        """Test getting workload access token when none is set."""
        # Run this test in a completely fresh context to avoid interference from other tests
        ctx = contextvars.Context()

        def test_in_new_context():
            result = BedrockAgentCoreContext.get_workload_access_token()
            return result

        result = ctx.run(test_in_new_context)
        assert result is None

    def test_set_and_get_request_context(self):
        """Test setting and getting request and session IDs."""
        request_id = "test-request-123"
        session_id = "test-session-456"

        BedrockAgentCoreContext.set_request_context(request_id, session_id)

        assert BedrockAgentCoreContext.get_request_id() == request_id
        assert BedrockAgentCoreContext.get_session_id() == session_id

    def test_set_request_context_without_session(self):
        """Test setting request context without session ID."""
        request_id = "test-request-789"

        BedrockAgentCoreContext.set_request_context(request_id, None)

        assert BedrockAgentCoreContext.get_request_id() == request_id
        assert BedrockAgentCoreContext.get_session_id() is None

    def test_get_request_id_when_none_set(self):
        """Test getting request ID when none is set."""
        ctx = contextvars.Context()

        def test_in_new_context():
            return BedrockAgentCoreContext.get_request_id()

        result = ctx.run(test_in_new_context)
        assert result is None

    def test_get_session_id_when_none_set(self):
        """Test getting session ID when none is set."""
        ctx = contextvars.Context()

        def test_in_new_context():
            return BedrockAgentCoreContext.get_session_id()

        result = ctx.run(test_in_new_context)
        assert result is None
