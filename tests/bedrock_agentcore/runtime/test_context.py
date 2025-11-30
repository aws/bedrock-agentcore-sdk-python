"""Tests for Bedrock AgentCore context functionality."""

import contextvars
from unittest.mock import MagicMock

from bedrock_agentcore.runtime.context import BedrockAgentCoreContext, RequestContext


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

    def test_set_and_get_oauth2_callback_url(self):
        oauth2_callback_url = "http://unit-test"

        BedrockAgentCoreContext.set_oauth2_callback_url(oauth2_callback_url)
        result = BedrockAgentCoreContext.get_oauth2_callback_url()

        assert result == oauth2_callback_url

    def test_get_oauth2_callback_url_when_none_set(self):
        # Run this test in a completely fresh context to avoid interference from other tests
        ctx = contextvars.Context()

        def test_in_new_context():
            return BedrockAgentCoreContext.get_oauth2_callback_url()

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

    def test_set_and_get_request_headers(self):
        """Test setting and getting request headers."""
        headers = {"Authorization": "Bearer token-123", "X-Amzn-Bedrock-AgentCore-Runtime-Custom-Key": "custom-value"}

        BedrockAgentCoreContext.set_request_headers(headers)
        result = BedrockAgentCoreContext.get_request_headers()

        assert result == headers

    def test_get_request_headers_when_none_set(self):
        """Test getting request headers when none are set."""
        ctx = contextvars.Context()

        def test_in_new_context():
            return BedrockAgentCoreContext.get_request_headers()

        result = ctx.run(test_in_new_context)
        assert result is None

    def test_request_headers_isolation_between_contexts(self):
        """Test that request headers are isolated between different contexts."""
        headers1 = {"Authorization": "Bearer token-1"}
        headers2 = {"Authorization": "Bearer token-2"}

        # Set headers in current context
        BedrockAgentCoreContext.set_request_headers(headers1)

        # Run test in different context
        ctx = contextvars.Context()

        def test_in_new_context():
            BedrockAgentCoreContext.set_request_headers(headers2)
            return BedrockAgentCoreContext.get_request_headers()

        result_in_new_context = ctx.run(test_in_new_context)

        # Headers should be different in each context
        assert BedrockAgentCoreContext.get_request_headers() == headers1
        assert result_in_new_context == headers2

    def test_empty_request_headers(self):
        """Test setting empty request headers."""
        empty_headers = {}

        BedrockAgentCoreContext.set_request_headers(empty_headers)
        result = BedrockAgentCoreContext.get_request_headers()

        assert result == empty_headers

    def test_request_headers_with_various_custom_headers(self):
        """Test request headers with multiple custom headers."""
        headers = {
            "Authorization": "Bearer token-123",
            "X-Amzn-Bedrock-AgentCore-Runtime-Custom-Header1": "value1",
            "X-Amzn-Bedrock-AgentCore-Runtime-Custom-Header2": "value2",
            "X-Amzn-Bedrock-AgentCore-Runtime-Custom-Special": "special-chars-!@#$%",
        }

        BedrockAgentCoreContext.set_request_headers(headers)
        result = BedrockAgentCoreContext.get_request_headers()

        assert result == headers
        assert len(result) == 4


class TestRequestContext:
    """Test RequestContext functionality."""

    def test_request_context_initialization_with_headers(self):
        """Test RequestContext initialization with request headers."""
        headers = {"Authorization": "Bearer test-token", "X-Amzn-Bedrock-AgentCore-Runtime-Custom-Key": "custom-value"}

        context = RequestContext(session_id="test-session-123", request_headers=headers)

        assert context.session_id == "test-session-123"
        assert context.request_headers == headers

    def test_request_context_initialization_without_headers(self):
        """Test RequestContext initialization without request headers."""
        context = RequestContext(session_id="test-session-456")

        assert context.session_id == "test-session-456"
        assert context.request_headers is None

    def test_request_context_initialization_minimal(self):
        """Test RequestContext initialization with minimal data."""
        context = RequestContext()

        assert context.session_id is None
        assert context.request_headers is None
        assert context.request is None

    def test_request_context_with_empty_headers(self):
        """Test RequestContext with empty headers dictionary."""
        context = RequestContext(session_id="test-session-789", request_headers={})

        assert context.session_id == "test-session-789"
        assert context.request_headers == {}

    def test_request_context_initialization_with_request_object(self):
        """Test RequestContext initialization with request object."""
        mock_request = MagicMock()
        mock_request.state.user_id = "123"
        mock_request.state.tenant = "acme"

        context = RequestContext(session_id="test-session-123", request=mock_request)

        assert context.session_id == "test-session-123"
        assert context.request is mock_request
        assert context.request.state.user_id == "123"
        assert context.request.state.tenant == "acme"

    def test_request_context_request_default_none(self):
        """Test RequestContext request defaults to None."""
        context = RequestContext(session_id="test-session-456")

        assert context.session_id == "test-session-456"
        assert context.request is None

    def test_request_context_initialization_minimal_has_none_request(self):
        """Test RequestContext with minimal initialization has None request."""
        context = RequestContext()

        assert context.session_id is None
        assert context.request_headers is None
        assert context.request is None

    def test_request_context_with_all_fields(self):
        """Test RequestContext with all fields populated."""
        headers = {"Authorization": "Bearer test-token", "X-Amzn-Bedrock-AgentCore-Runtime-Custom-Key": "custom-value"}
        mock_request = MagicMock()
        mock_request.state.middleware_processed = True
        mock_request.state.auth_result = {"user": "test-user", "roles": ["admin"]}

        context = RequestContext(session_id="full-session-123", request_headers=headers, request=mock_request)

        assert context.session_id == "full-session-123"
        assert context.request_headers == headers
        assert context.request is mock_request
        assert context.request.state.middleware_processed is True
        assert context.request.state.auth_result["user"] == "test-user"

    def test_request_context_request_state_with_nested_structures(self):
        """Test RequestContext with complex nested request.state data."""
        mock_request = MagicMock()
        mock_request.state.level1 = {"level2": {"level3": {"deep_value": "found"}}}
        mock_request.state.list_data = [1, 2, {"nested_in_list": True}]

        context = RequestContext(request=mock_request)

        assert context.request.state.level1["level2"]["level3"]["deep_value"] == "found"
        assert context.request.state.list_data[2]["nested_in_list"] is True

    def test_request_context_allows_arbitrary_types(self):
        """Test RequestContext allows arbitrary types via Config."""
        # This tests that arbitrary_types_allowed = True works
        mock_request = MagicMock()

        # Should not raise ValidationError
        context = RequestContext(request=mock_request)

        assert context.request is mock_request
