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
