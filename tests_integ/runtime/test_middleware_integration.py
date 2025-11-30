"""Integration tests for middleware → handler data flow.

These tests verify the complete flow:
1. Middleware sets request.state attributes
2. SDK passes the request object through in _build_request_context
3. Handler accesses it via context.request.state
"""

import time

import pytest
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.testclient import TestClient

# =============================================================================
# Test Middleware Definitions
# =============================================================================


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware that adds timing data."""

    async def dispatch(self, request, call_next):
        start_time = time.time()
        request.state.start_time = start_time

        response = await call_next(request)

        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that adds auth data."""

    async def dispatch(self, request, call_next):
        # Check for auth header
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            request.state.user_id = "test_user_123"
            request.state.authenticated = True
        else:
            request.state.authenticated = False

        return await call_next(request)


class MetadataMiddleware(BaseHTTPMiddleware):
    """Middleware that adds various metadata."""

    async def dispatch(self, request, call_next):
        request.state.client_ip = request.client.host if request.client else "unknown"
        request.state.path = request.url.path

        return await call_next(request)


# =============================================================================
# Integration Tests
# =============================================================================


class TestMiddlewareIntegration:
    """Integration tests for middleware data flow."""

    def test_single_middleware_data_visible(self):
        """Data from a single middleware is visible in handler."""
        from bedrock_agentcore.runtime import BedrockAgentCoreApp

        app = BedrockAgentCoreApp(middleware=[Middleware(TimingMiddleware)])

        @app.entrypoint
        def handler(payload, context):
            start_time = getattr(context.request.state, "start_time", None)
            return {"has_start_time": start_time is not None, "start_time_type": type(start_time).__name__}

        client = TestClient(app)
        response = client.post("/invocations", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["has_start_time"] is True
        assert data["start_time_type"] == "float"

    def test_auth_middleware_authenticated(self):
        """Auth middleware data visible when authenticated."""
        from bedrock_agentcore.runtime import BedrockAgentCoreApp

        app = BedrockAgentCoreApp(middleware=[Middleware(AuthMiddleware)])

        @app.entrypoint
        def handler(payload, context):
            return {
                "authenticated": getattr(context.request.state, "authenticated", None),
                "user_id": getattr(context.request.state, "user_id", None),
            }

        client = TestClient(app)

        # With auth header
        response = client.post("/invocations", json={}, headers={"Authorization": "Bearer test-token"})

        assert response.status_code == 200
        data = response.json()
        assert data["authenticated"] is True
        assert data["user_id"] == "test_user_123"

    def test_auth_middleware_not_authenticated(self):
        """Auth middleware data visible when not authenticated."""
        from bedrock_agentcore.runtime import BedrockAgentCoreApp

        app = BedrockAgentCoreApp(middleware=[Middleware(AuthMiddleware)])

        @app.entrypoint
        def handler(payload, context):
            return {
                "authenticated": getattr(context.request.state, "authenticated", None),
                "user_id": getattr(context.request.state, "user_id", None),
            }

        client = TestClient(app)

        # Without auth header
        response = client.post("/invocations", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["authenticated"] is False
        assert data["user_id"] is None

    def test_multiple_middleware_data_merged(self):
        """Data from multiple middleware is merged and visible."""
        from bedrock_agentcore.runtime import BedrockAgentCoreApp

        app = BedrockAgentCoreApp(
            middleware=[
                Middleware(TimingMiddleware),
                Middleware(AuthMiddleware),
                Middleware(MetadataMiddleware),
            ]
        )

        @app.entrypoint
        def handler(payload, context):
            state = context.request.state
            # Access the internal _state dict to get keys
            state_keys = list(state._state.keys()) if hasattr(state, "_state") else []
            return {
                "has_start_time": hasattr(state, "start_time"),
                "has_authenticated": hasattr(state, "authenticated"),
                "has_path": hasattr(state, "path"),
                "path": getattr(state, "path", None),
                "keys": state_keys,
            }

        client = TestClient(app)
        response = client.post("/invocations", json={}, headers={"Authorization": "Bearer token"})

        assert response.status_code == 200
        data = response.json()
        assert data["has_start_time"] is True
        assert data["has_authenticated"] is True
        assert data["has_path"] is True
        assert data["path"] == "/invocations"

        # All keys present
        keys = data["keys"]
        assert "start_time" in keys
        assert "authenticated" in keys
        assert "user_id" in keys
        assert "path" in keys
        assert "client_ip" in keys

    def test_no_middleware_empty_processing_data(self):
        """Without middleware, request.state has no custom attributes."""
        from bedrock_agentcore.runtime import BedrockAgentCoreApp

        app = BedrockAgentCoreApp()  # No middleware

        @app.entrypoint
        def handler(payload, context):
            # Access the internal _state dict to get custom attributes
            state_attrs = list(context.request.state._state.keys()) if hasattr(context.request.state, "_state") else []
            return {"state_attrs": state_attrs, "is_empty": len(state_attrs) == 0}

        client = TestClient(app)
        response = client.post("/invocations", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["is_empty"] is True
        assert data["state_attrs"] == []

    def test_handler_can_modify_processing_data(self):
        """Handler can add to request.state (though it won't persist)."""
        from bedrock_agentcore.runtime import BedrockAgentCoreApp

        app = BedrockAgentCoreApp(middleware=[Middleware(TimingMiddleware)])

        @app.entrypoint
        def handler(payload, context):
            # Add data in handler
            context.request.state.handler_added = "yes"
            context.request.state.processed_at = time.time()

            return {
                "has_middleware_data": hasattr(context.request.state, "start_time"),
                "has_handler_data": hasattr(context.request.state, "handler_added"),
                "handler_added": getattr(context.request.state, "handler_added", None),
            }

        client = TestClient(app)
        response = client.post("/invocations", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["has_middleware_data"] is True
        assert data["has_handler_data"] is True
        assert data["handler_added"] == "yes"

    def test_processing_data_with_session_and_headers(self):
        """request.state works alongside session_id and request_headers."""
        from bedrock_agentcore.runtime import BedrockAgentCoreApp

        app = BedrockAgentCoreApp(middleware=[Middleware(AuthMiddleware)])

        @app.entrypoint
        def handler(payload, context):
            return {
                "session_id": context.session_id,
                "has_auth_header": context.request_headers is not None and "Authorization" in context.request_headers,
                "authenticated": getattr(context.request.state, "authenticated", None),
                "user_id": getattr(context.request.state, "user_id", None),
            }

        client = TestClient(app)
        response = client.post(
            "/invocations",
            json={},
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "session-abc", "Authorization": "Bearer token123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session-abc"
        assert data["has_auth_header"] is True
        assert data["authenticated"] is True
        assert data["user_id"] == "test_user_123"


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_middleware_sets_empty_dict(self):
        """Middleware that sets an empty dict on request.state."""
        from bedrock_agentcore.runtime import BedrockAgentCoreApp

        class EmptyMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.empty_dict = {}
                return await call_next(request)

        app = BedrockAgentCoreApp(middleware=[Middleware(EmptyMiddleware)])

        @app.entrypoint
        def handler(payload, context):
            empty_dict = getattr(context.request.state, "empty_dict", None)
            return {"is_dict": isinstance(empty_dict, dict)}

        client = TestClient(app)
        response = client.post("/invocations", json={})

        assert response.status_code == 200
        assert response.json()["is_dict"] is True

    def test_middleware_sets_nested_data(self):
        """Middleware can set nested data structures."""
        from bedrock_agentcore.runtime import BedrockAgentCoreApp

        class NestedMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.auth = {"user_id": "alice", "roles": ["admin", "user"]}
                request.state.metrics = {"request_count": 42}
                return await call_next(request)

        app = BedrockAgentCoreApp(middleware=[Middleware(NestedMiddleware)])

        @app.entrypoint
        def handler(payload, context):
            auth = getattr(context.request.state, "auth", {})
            metrics = getattr(context.request.state, "metrics", {})
            return {
                "user_id": auth.get("user_id"),
                "roles": auth.get("roles"),
                "request_count": metrics.get("request_count"),
            }

        client = TestClient(app)
        response = client.post("/invocations", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "alice"
        assert data["roles"] == ["admin", "user"]
        assert data["request_count"] == 42

    def test_large_processing_data(self):
        """Handler can receive large data via request.state."""
        from bedrock_agentcore.runtime import BedrockAgentCoreApp

        class LargeDataMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.large_data = {f"key_{i}": f"value_{i}" for i in range(100)}
                return await call_next(request)

        app = BedrockAgentCoreApp(middleware=[Middleware(LargeDataMiddleware)])

        @app.entrypoint
        def handler(payload, context):
            large_data = getattr(context.request.state, "large_data", {})
            return {"count": len(large_data), "has_key_50": "key_50" in large_data}

        client = TestClient(app)
        response = client.post("/invocations", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 100
        assert data["has_key_50"] is True


# =============================================================================
# Test Real-World Scenario
# =============================================================================


class TestRealWorldScenario:
    """Test realistic agent scenario."""

    def test_complete_agent_flow(self):
        """Test a complete agent with auth, timing, and business logic."""
        from bedrock_agentcore.runtime import BedrockAgentCoreApp

        class ProductionAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                auth = request.headers.get("Authorization", "")
                if auth.startswith("Bearer "):
                    # Simulate JWT validation
                    request.state.user_id = "user_12345"
                    request.state.user_email = "user@example.com"
                    request.state.user_role = "developer"
                    request.state.authenticated = True
                else:
                    request.state.authenticated = False

                return await call_next(request)

        class ProductionTimingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                start = time.time()
                request.state.request_start = start

                response = await call_next(request)

                # Note: This won't update request.state for the handler
                # but shows the pattern
                return response

        app = BedrockAgentCoreApp(
            middleware=[
                Middleware(ProductionTimingMiddleware),
                Middleware(ProductionAuthMiddleware),
            ]
        )

        @app.entrypoint
        def ai_agent(payload, context):
            state = context.request.state

            # Check auth
            if not getattr(state, "authenticated", False):
                return {"error": "Unauthorized"}, 401

            # Get user info
            user_id = getattr(state, "user_id", None)
            user_email = getattr(state, "user_email", None)
            user_role = getattr(state, "user_role", None)

            # Process request
            user_message = payload.get("message", "")

            # Generate response
            response = {
                "reply": f"Hello {user_email}! You asked: {user_message}",
                "user": {"id": user_id, "email": user_email, "role": user_role},
                "session": context.session_id,
            }

            return response

        client = TestClient(app)

        # Test authenticated request
        response = client.post(
            "/invocations",
            json={"message": "What is machine learning?"},
            headers={
                "Authorization": "Bearer valid-token",
                "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "session-xyz",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "Hello user@example.com" in data["reply"]
        assert data["user"]["id"] == "user_12345"
        assert data["user"]["role"] == "developer"
        assert data["session"] == "session-xyz"

        # Test unauthenticated request
        response = client.post("/invocations", json={"message": "Hello"}, headers={})

        assert response.status_code == 200
        # Returns tuple (data, status_code)
        result = response.json()
        assert result[0]["error"] == "Unauthorized"
        assert result[1] == 401


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
