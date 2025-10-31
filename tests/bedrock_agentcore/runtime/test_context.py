"""Tests for Bedrock AgentCore context functionality."""

import contextvars

from pydantic import ValidationError

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

    def test_request_context_with_empty_headers(self):
        """Test RequestContext with empty headers dictionary."""
        context = RequestContext(session_id="test-session-789", request_headers={})

        assert context.session_id == "test-session-789"
        assert context.request_headers == {}


class TestProcessingContext:
    """Test ProcessingContext with namespace isolation."""

    def test_processing_context_set_and_get(self):
        """Test setting and getting namespaced data."""
        from bedrock_agentcore.runtime.context import ProcessingContext, StandardNamespaces

        context = ProcessingContext()
        context.set(StandardNamespaces.AUTH, "user_id", "alice")

        result = context.get(StandardNamespaces.AUTH, "user_id")
        assert result == "alice"

    def test_processing_context_namespace_isolation(self):
        """Test that namespaces are isolated."""
        from bedrock_agentcore.runtime.context import ProcessingContext

        context = ProcessingContext()
        context.set("auth", "user_id", "alice")
        context.set("timing", "user_id", "bob")  # Different namespace, same key

        assert context.get("auth", "user_id") == "alice"
        assert context.get("timing", "user_id") == "bob"

    def test_processing_context_get_namespace(self):
        """Test getting entire namespace."""
        from bedrock_agentcore.runtime.context import ProcessingContext

        context = ProcessingContext()
        context.set("auth", "user_id", "alice")
        context.set("auth", "role", "admin")

        auth_data = context.get_namespace("auth")
        assert auth_data == {"user_id": "alice", "role": "admin"}

    def test_processing_context_has_namespace(self):
        """Test checking namespace existence."""
        from bedrock_agentcore.runtime.context import ProcessingContext

        context = ProcessingContext()
        assert not context.has_namespace("auth")

        context.set("auth", "user_id", "alice")
        assert context.has_namespace("auth")

    def test_processing_context_validation(self):
        """Test optional validation."""
        import pytest

        from bedrock_agentcore.runtime.context import ProcessingContext

        context = ProcessingContext()

        # Should work without validation
        context.set("custom", "data", "x" * 200, validate=False)

        # Should fail with validation (exceeds max_size)
        with pytest.raises(ValueError, match="Value too large"):
            context.set("custom", "large", "x" * 200, validate=True, max_size=100)

        # Should fail with malicious pattern
        with pytest.raises(ValueError, match="Potentially unsafe pattern"):
            context.set("custom", "bad", "test\x00null", validate=True)


class TestAgentContext:
    """Test AgentContext integration."""

    def test_agent_context_backward_compatibility(self):
        """Test backward compatibility properties."""
        from bedrock_agentcore.runtime.context import AgentContext, ProcessingContext, RequestContext

        request = RequestContext(session_id="test-123", request_headers={"Authorization": "Bearer token"})
        processing = ProcessingContext()

        context = AgentContext(request=request, processing=processing)

        # Backward compatibility
        assert context.session_id == "test-123"
        assert context.request_headers is not None
        assert "Authorization" in context.request_headers

    def test_agent_context_immutability(self):
        """Test that request context is immutable."""
        from types import MappingProxyType

        import pytest

        from bedrock_agentcore.runtime.context import AgentContext, ProcessingContext, RequestContext

        request = RequestContext(session_id="test-123", request_headers={"Authorization": "Bearer token"})
        processing = ProcessingContext()
        context = AgentContext(request=request, processing=processing)

        # Cannot modify frozen request
        with pytest.raises((ValidationError, AttributeError, TypeError)):
            context.request.session_id = "hacked"

        # Cannot modify headers (MappingProxyType)
        assert isinstance(context.request_headers, MappingProxyType)
        with pytest.raises(TypeError):
            context.request_headers["new"] = "value"

        # CAN modify processing context
        context.processing.set("custom", "key", "value")
        assert context.processing.get("custom", "key") == "value"


class TestStandardNamespaces:
    """Test StandardNamespaces constants."""

    def test_standard_namespaces_values(self):
        """Test that standard namespace constants have expected values."""
        from bedrock_agentcore.runtime.context import StandardNamespaces

        assert StandardNamespaces.AUTH == "auth"
        assert StandardNamespaces.TIMING == "timing"
        assert StandardNamespaces.AUDIT == "audit"
        assert StandardNamespaces.METRICS == "metrics"
        assert StandardNamespaces.CUSTOM == "custom"
