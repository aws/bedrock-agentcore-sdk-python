"""Tests for BedrockAgentCoreA2AApp."""

import asyncio
import contextlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from bedrock_agentcore.runtime import (
    A2AArtifact,
    AgentCard,
    AgentSkill,
    BedrockAgentCoreA2AApp,
)
from bedrock_agentcore.runtime.a2a_app import A2ARequestContextFormatter
from bedrock_agentcore.runtime.models import PingStatus


@pytest.fixture
def agent_card():
    """Create a test AgentCard."""
    return AgentCard(
        name="Test Agent",
        description="A test agent for unit testing",
        skills=[
            AgentSkill(id="test", name="Test Skill", description="A test skill"),
        ],
    )


@pytest.fixture
def app(agent_card):
    """Create a test A2A app."""
    return BedrockAgentCoreA2AApp(agent_card=agent_card)


class TestBedrockAgentCoreA2AAppInitialization:
    def test_basic_initialization(self, agent_card):
        """Test basic app initialization."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        assert app.agent_card == agent_card
        assert app.handlers == {}
        assert app.debug is False

    def test_initialization_with_debug(self, agent_card):
        """Test app initialization with debug mode."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card, debug=True)
        assert app.debug is True

    def test_routes_registered(self, agent_card):
        """Test that required routes are registered."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        route_paths = [route.path for route in app.routes]
        assert "/" in route_paths
        assert "/.well-known/agent-card.json" in route_paths
        assert "/ping" in route_paths


class TestAgentCardEndpoint:
    def test_agent_card_endpoint(self, app, agent_card):
        """Test GET /.well-known/agent-card.json returns agent card."""
        client = TestClient(app)
        response = client.get("/.well-known/agent-card.json")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == agent_card.name
        assert data["description"] == agent_card.description
        assert data["protocolVersion"] == agent_card.protocol_version
        assert data["url"] == "http://testserver/"
        assert len(data["skills"]) == 1
        assert data["skills"][0]["id"] == "test"

    def test_agent_card_with_runtime_url(self, agent_card):
        """Test agent card includes URL when AGENTCORE_RUNTIME_URL is set."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)

        with patch.dict(os.environ, {"AGENTCORE_RUNTIME_URL": "https://example.com/agent"}):
            client = TestClient(app)
            response = client.get("/.well-known/agent-card.json")

            assert response.status_code == 200
            data = response.json()
            assert data["url"] == "https://example.com/agent"


class TestPingEndpoint:
    def test_ping_endpoint(self, app):
        """Test GET /ping returns healthy status."""
        client = TestClient(app)
        response = client.get("/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["Healthy", "HEALTHY"]
        assert "time_of_last_update" in data

    def test_custom_ping_handler(self, agent_card):
        """Test custom ping handler."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)

        @app.ping
        def custom_ping():
            return "HealthyBusy"

        client = TestClient(app)
        response = client.get("/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["HealthyBusy", "HEALTHY_BUSY"]


class TestEntrypointDecorator:
    def test_entrypoint_decorator(self, app):
        """Test @app.entrypoint registers handler."""

        @app.entrypoint
        def handler(request, context):
            return {"result": "success"}

        assert "main" in app.handlers
        assert app.handlers["main"] == handler
        assert hasattr(handler, "run")

    def test_entrypoint_without_context(self, app):
        """Test entrypoint handler without context parameter."""

        @app.entrypoint
        def handler(request):
            return {"result": request.method}

        client = TestClient(app)
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "req-001",
                "method": "message/send",
                "params": {},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "req-001"
        assert data["result"]["result"] == "message/send"


class TestJsonRpcHandling:
    def test_valid_jsonrpc_request(self, app):
        """Test valid JSON-RPC request."""

        @app.entrypoint
        def handler(request, context):
            return {"artifacts": [{"artifactId": "art-1", "name": "response", "parts": []}]}

        client = TestClient(app)
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "req-001",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Hello"}],
                    }
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "req-001"
        assert "result" in data
        assert "artifacts" in data["result"]

    def test_dataclass_response_is_serialized(self, app):
        """Test A2A helper models are serialized in JSON-RPC responses."""

        @app.entrypoint
        def handler(request, context):
            return {"artifacts": [A2AArtifact.from_text("art-1", "response", "Hello")]}

        client = TestClient(app)
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "req-001",
                "method": "message/send",
                "params": {},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["result"]["artifacts"][0]["artifactId"] == "art-1"
        assert data["result"]["artifacts"][0]["parts"][0]["text"] == "Hello"

    def test_invalid_jsonrpc_version(self, app):
        """Test invalid JSON-RPC version returns error."""

        @app.entrypoint
        def handler(request, context):
            return {}

        client = TestClient(app)
        response = client.post(
            "/",
            json={
                "jsonrpc": "1.0",  # Invalid version
                "id": "req-001",
                "method": "test",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32600  # Invalid request

    def test_missing_method(self, app):
        """Test missing method returns error."""

        @app.entrypoint
        def handler(request, context):
            return {}

        client = TestClient(app)
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "req-001",
                # Missing method
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32600  # Invalid request

    def test_non_object_request_returns_invalid_request(self, app):
        """Test non-object JSON-RPC payload returns invalid request."""

        @app.entrypoint
        def handler(request, context):
            return {}

        client = TestClient(app)
        response = client.post(
            "/",
            json=[],
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32600  # Invalid request
        assert data["id"] is None

    def test_no_entrypoint_defined(self, app):
        """Test error when no entrypoint is defined."""
        client = TestClient(app)
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "req-001",
                "method": "message/send",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32603  # Internal error

    def test_invalid_json(self, app):
        """Test invalid JSON returns parse error."""

        @app.entrypoint
        def handler(request, context):
            return {}

        client = TestClient(app)
        response = client.post(
            "/",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32700  # Parse error

    def test_handler_exception(self, app):
        """Test handler exception returns internal error."""

        @app.entrypoint
        def handler(request, context):
            raise ValueError("Test error")

        client = TestClient(app)
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "req-001",
                "method": "message/send",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32603  # Internal error
        assert data["error"]["message"] == "Internal error"


class TestAsyncHandler:
    def test_async_handler(self, app):
        """Test async handler."""

        @app.entrypoint
        async def handler(request, context):
            await asyncio.sleep(0.01)
            return {"result": "async success"}

        client = TestClient(app)
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "req-001",
                "method": "message/send",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["result"]["result"] == "async success"


class TestStreamingResponse:
    def test_async_generator_response(self, app):
        """Test async generator for streaming response."""

        @app.entrypoint
        async def handler(request, context):
            async def generate():
                yield {"chunk": 1}
                yield {"chunk": 2}
                yield {"chunk": 3}

            return generate()

        client = TestClient(app)
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "req-001",
                "method": "message/stream",
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE events
        events = response.text.split("\n\n")
        events = [e for e in events if e.strip()]

        assert len(events) == 3
        for i, event in enumerate(events, 1):
            assert event.startswith("data: ")
            data = json.loads(event[6:])
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == "req-001"
            assert data["result"]["chunk"] == i

    def test_sync_generator_response(self, app):
        """Test sync generator for streaming response."""

        @app.entrypoint
        def handler(request, context):
            def generate():
                yield {"part": "A"}
                yield {"part": "B"}

            return generate()

        client = TestClient(app)
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "req-001",
                "method": "message/send",
            },
        )

        assert response.status_code == 200
        events = response.text.split("\n\n")
        events = [e for e in events if e.strip()]
        assert len(events) == 2

    def test_streaming_dataclass_response_is_serialized(self, app):
        """Test streaming payloads serialize A2A helper models."""

        @app.entrypoint
        def handler(request, context):
            def generate():
                yield {"artifacts": [A2AArtifact.from_text("art-1", "response", "Hello")]}

            return generate()

        client = TestClient(app)
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "req-001",
                "method": "message/stream",
            },
        )

        assert response.status_code == 200
        events = response.text.split("\n\n")
        events = [e for e in events if e.strip()]
        assert len(events) == 1

        data = json.loads(events[0][6:])
        assert data["result"]["artifacts"][0]["artifactId"] == "art-1"
        assert data["result"]["artifacts"][0]["parts"][0]["text"] == "Hello"


class TestSessionHeader:
    def test_session_id_from_header(self, app):
        """Test session ID is extracted from header."""
        captured_session_id = None

        @app.entrypoint
        def handler(request, context):
            nonlocal captured_session_id
            captured_session_id = context.session_id
            return {"session": context.session_id}

        client = TestClient(app)
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "req-001",
                "method": "message/send",
            },
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "test-session-123"},
        )

        assert response.status_code == 200
        assert captured_session_id == "test-session-123"


class TestRunMethod:
    @patch("uvicorn.run")
    def test_run_default_port(self, mock_uvicorn, app):
        """Test run uses default A2A port 9000."""
        app.run()

        mock_uvicorn.assert_called_once()
        call_kwargs = mock_uvicorn.call_args[1]
        assert call_kwargs["port"] == 9000
        assert call_kwargs["host"] == "127.0.0.1"

    @patch("uvicorn.run")
    def test_run_custom_port(self, mock_uvicorn, app):
        """Test run with custom port."""
        app.run(port=8080)

        mock_uvicorn.assert_called_once()
        call_kwargs = mock_uvicorn.call_args[1]
        assert call_kwargs["port"] == 8080

    @patch.dict(os.environ, {"DOCKER_CONTAINER": "true"})
    @patch("uvicorn.run")
    def test_run_in_docker(self, mock_uvicorn, agent_card):
        """Test run in Docker environment."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        app.run()

        mock_uvicorn.assert_called_once()
        call_kwargs = mock_uvicorn.call_args[1]
        assert call_kwargs["host"] == "0.0.0.0"


class TestLifespan:
    def test_lifespan_startup_and_shutdown(self, agent_card):
        """Test lifespan startup and shutdown."""
        startup_called = False
        shutdown_called = False

        @contextlib.asynccontextmanager
        async def lifespan(app):
            nonlocal startup_called, shutdown_called
            startup_called = True
            yield
            shutdown_called = True

        app = BedrockAgentCoreA2AApp(agent_card=agent_card, lifespan=lifespan)

        with TestClient(app):
            assert startup_called is True
        assert shutdown_called is True


class TestIntegrationScenario:
    def test_full_message_flow(self, app):
        """Test complete message flow with A2A protocol."""

        @app.entrypoint
        def handler(request, context):
            # Extract message from params
            params = request.params or {}
            message = params.get("message", {})
            parts = message.get("parts", [])
            user_text = ""
            for part in parts:
                if part.get("kind") == "text":
                    user_text = part.get("text", "")
                    break

            # Return A2A formatted response
            return {
                "artifacts": [
                    {
                        "artifactId": str(uuid.uuid4()),
                        "name": "agent_response",
                        "parts": [{"kind": "text", "text": f"Received: {user_text}"}],
                    }
                ]
            }

        client = TestClient(app)

        # Send message
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "req-001",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "What is 2 + 2?"}],
                        "messageId": "msg-001",
                    }
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "req-001"
        assert "result" in data
        assert "artifacts" in data["result"]
        assert len(data["result"]["artifacts"]) == 1
        assert "Received: What is 2 + 2?" in data["result"]["artifacts"][0]["parts"][0]["text"]


class TestA2ARequestContextFormatter:
    def test_format_basic_record(self):
        """Test basic log record formatting."""
        formatter = A2ARequestContextFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="Test message", args=(), exc_info=None,
        )
        result = json.loads(formatter.format(record))
        assert result["level"] == "INFO"
        assert result["message"] == "Test message"
        assert result["protocol"] == "A2A"

    def test_format_with_exc_info(self):
        """Test log record formatting with exception info."""
        formatter = A2ARequestContextFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="test.py",
            lineno=1, msg="Error occurred", args=(), exc_info=exc_info,
        )
        result = json.loads(formatter.format(record))
        assert result["errorType"] == "ValueError"
        assert result["errorMessage"] == "test error"
        assert "stackTrace" in result
        assert "location" in result

    def test_format_with_request_and_session_ids(self, app):
        """Test log record includes request and session IDs from context."""
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        BedrockAgentCoreContext.set_request_context("req-123", "sess-456")
        formatter = A2ARequestContextFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="Test", args=(), exc_info=None,
        )
        result = json.loads(formatter.format(record))
        assert result["requestId"] == "req-123"
        assert result["sessionId"] == "sess-456"


class TestPingStatusAdvanced:
    def test_ping_with_forced_status(self, agent_card):
        """Test forced ping status overrides everything."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        app._forced_ping_status = PingStatus.HEALTHY_BUSY

        status = app.get_current_ping_status()
        assert status == PingStatus.HEALTHY_BUSY

    def test_ping_with_active_tasks(self, agent_card):
        """Test automatic HEALTHY_BUSY when tasks are active."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        app._active_tasks = {1: {"name": "task1"}}

        status = app.get_current_ping_status()
        assert status == PingStatus.HEALTHY_BUSY

    def test_ping_custom_handler_returns_ping_status(self, agent_card):
        """Test custom ping handler returning PingStatus enum directly."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)

        @app.ping
        def custom_ping():
            return PingStatus.HEALTHY_BUSY

        status = app.get_current_ping_status()
        assert status == PingStatus.HEALTHY_BUSY

    def test_ping_custom_handler_returns_string(self, agent_card):
        """Test custom ping handler returning string value."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)

        @app.ping
        def custom_ping():
            return "HealthyBusy"

        status = app.get_current_ping_status()
        assert status == PingStatus.HEALTHY_BUSY

    def test_ping_custom_handler_exception_falls_back(self, agent_card):
        """Test custom ping handler exception falls back to automatic."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)

        @app.ping
        def broken_ping():
            raise RuntimeError("ping failed")

        status = app.get_current_ping_status()
        assert status == PingStatus.HEALTHY

    def test_ping_status_unchanged_does_not_update_timestamp(self, agent_card):
        """Test timestamp is not updated when status doesn't change."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        app.get_current_ping_status()
        first_time = app._last_status_update_time

        app.get_current_ping_status()
        assert app._last_status_update_time == first_time

    def test_ping_endpoint_exception(self, agent_card):
        """Test ping endpoint handles exception gracefully."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)

        # Force get_current_ping_status to throw
        original = app.get_current_ping_status
        def broken():
            raise RuntimeError("broken")
        app.get_current_ping_status = broken

        client = TestClient(app)
        response = client.get("/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["Healthy", "HEALTHY"]


class TestBuildRequestContext:
    def test_context_with_all_headers(self, app):
        """Test context building with access token, oauth, auth, and custom headers."""
        captured_context = None

        @app.entrypoint
        def handler(request, context):
            nonlocal captured_context
            captured_context = context
            return {"ok": True}

        client = TestClient(app)
        client.post(
            "/",
            json={"jsonrpc": "2.0", "id": "1", "method": "test"},
            headers={
                "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "sess-1",
                "X-Amzn-Bedrock-AgentCore-Runtime-Request-Id": "req-1",
                "WorkloadAccessToken": "token-abc",
                "OAuth2CallbackUrl": "https://callback.example.com",
                "Authorization": "Bearer xyz",
                "X-Amzn-Bedrock-AgentCore-Runtime-Custom-MyHeader": "custom-val",
            },
        )

        assert captured_context is not None
        assert captured_context.session_id == "sess-1"

    def test_context_build_exception_fallback(self, agent_card):
        """Test context building falls back gracefully on exception."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)

        @app.entrypoint
        def handler(request, context):
            return {"session": context.session_id}

        # Patch set_request_context: first call raises, second (in except) succeeds
        call_count = 0
        original_set = __import__(
            "bedrock_agentcore.runtime.context", fromlist=["BedrockAgentCoreContext"]
        ).BedrockAgentCoreContext.set_request_context

        def failing_then_ok(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("context failure")
            return original_set(*args, **kwargs)

        with patch(
            "bedrock_agentcore.runtime.a2a_app.BedrockAgentCoreContext.set_request_context",
            side_effect=failing_then_ok,
        ):
            client = TestClient(app)
            response = client.post(
                "/",
                json={"jsonrpc": "2.0", "id": "1", "method": "test"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["result"]["session"] is None


class TestTakesContext:
    def test_takes_context_with_exception(self, app):
        """Test _takes_context returns False on exception."""
        # Use a mock that raises when inspecting signature
        mock_handler = MagicMock(spec=[])
        mock_handler.__name__ = "mock"
        with patch("inspect.signature", side_effect=ValueError("bad")):
            assert app._takes_context(mock_handler) is False


class TestStreamingErrorHandling:
    def test_async_streaming_error(self, app):
        """Test error during async streaming yields error SSE event."""

        @app.entrypoint
        async def handler(request, context):
            async def generate():
                yield {"chunk": 1}
                raise ValueError("stream error")

            return generate()

        client = TestClient(app)
        response = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": "req-001", "method": "message/stream"},
        )

        assert response.status_code == 200
        events = [e for e in response.text.split("\n\n") if e.strip()]
        assert len(events) >= 2
        # Last event should be an error
        last_data = json.loads(events[-1].replace("data: ", ""))
        assert "error" in last_data

    def test_sync_streaming_error(self, app):
        """Test error during sync streaming yields error SSE event."""

        @app.entrypoint
        def handler(request, context):
            def generate():
                yield {"chunk": 1}
                raise ValueError("sync stream error")

            return generate()

        client = TestClient(app)
        response = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": "req-001", "method": "message/stream"},
        )

        assert response.status_code == 200
        events = [e for e in response.text.split("\n\n") if e.strip()]
        assert len(events) >= 2
        last_data = json.loads(events[-1].replace("data: ", ""))
        assert "error" in last_data


class TestConvertToSerializable:
    def test_set_converted_to_list(self, app):
        """Test set objects are converted to lists."""
        result = app._convert_to_serializable({"tags", "a2a"})
        assert isinstance(result, list)
        assert set(result) == {"tags", "a2a"}

    def test_tuple_converted(self, app):
        """Test tuple objects are converted to lists."""
        result = app._convert_to_serializable((1, 2, 3))
        assert result == [1, 2, 3]

    def test_object_with_to_dict(self, app):
        """Test objects with to_dict method are serialized."""

        @dataclass
        class FakeModel:
            name: str
            def to_dict(self):
                return {"name": self.name}

        result = app._convert_to_serializable(FakeModel(name="test"))
        assert result == {"name": "test"}

    def test_nested_dict_serialization(self, app):
        """Test nested dict with mixed types."""
        result = app._convert_to_serializable({
            "items": [A2AArtifact.from_text("1", "resp", "hello")],
            "tags": {"a", "b"},
        })
        assert isinstance(result["items"], list)
        assert result["items"][0]["artifactId"] == "1"
        assert isinstance(result["tags"], list)


class TestSafeSerialize:
    def test_normal_json(self, app):
        """Test normal JSON serialization."""
        result = app._safe_serialize_to_json_string({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_non_serializable_fallback(self, app):
        """Test non-serializable object uses convert then str fallback."""
        result = app._safe_serialize_to_json_string({"data": {1, 2, 3}})
        parsed = json.loads(result)
        assert isinstance(parsed["data"], list)

    def test_totally_unserializable_uses_str(self, app):
        """Test totally unserializable object falls back to str()."""

        class Unserializable:
            def __repr__(self):
                return "<unserializable>"

        # Patch _convert_to_serializable to also fail
        original = app._convert_to_serializable
        def broken(obj):
            raise TypeError("cannot convert")
        app._convert_to_serializable = broken

        result = app._safe_serialize_to_json_string(Unserializable())
        assert "<unserializable>" in result

        app._convert_to_serializable = original


class TestAgentCardEndpointError:
    def test_agent_card_exception(self, agent_card):
        """Test agent card endpoint handles exception."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)

        # Make to_dict raise
        original = agent_card.to_dict
        agent_card.to_dict = MagicMock(side_effect=RuntimeError("card error"))

        client = TestClient(app)
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 500
        assert "error" in response.json()

        agent_card.to_dict = original


class TestInvokeHandlerException:
    def test_sync_handler_exception_propagates(self, app):
        """Test exception from sync handler propagates through _invoke_handler."""

        @app.entrypoint
        def handler(request, context):
            raise RuntimeError("handler failed")

        client = TestClient(app)
        response = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": "1", "method": "test"},
        )
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32603

    def test_async_handler_exception_propagates(self, app):
        """Test exception from async handler propagates through _invoke_handler."""

        @app.entrypoint
        async def handler(request, context):
            raise RuntimeError("async handler failed")

        client = TestClient(app)
        response = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": "1", "method": "test"},
        )
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32603


class TestRunDockerDetection:
    @patch("os.path.exists", return_value=True)
    @patch("uvicorn.run")
    def test_run_detects_dockerenv_file(self, mock_uvicorn, mock_exists, agent_card):
        """Test run detects /.dockerenv file for Docker environment."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        app.run()

        mock_uvicorn.assert_called_once()
        call_kwargs = mock_uvicorn.call_args[1]
        assert call_kwargs["host"] == "0.0.0.0"

    @patch("uvicorn.run")
    def test_run_with_custom_host(self, mock_uvicorn, agent_card):
        """Test run with explicitly specified host."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        app.run(host="192.168.1.1")

        call_kwargs = mock_uvicorn.call_args[1]
        assert call_kwargs["host"] == "192.168.1.1"

    @patch("uvicorn.run")
    def test_run_passes_extra_kwargs(self, mock_uvicorn, agent_card):
        """Test run passes extra kwargs to uvicorn."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        app.run(workers=4)

        call_kwargs = mock_uvicorn.call_args[1]
        assert call_kwargs["workers"] == 4


class TestGetRuntimeUrl:
    def test_runtime_url_from_env(self, app):
        """Test runtime URL from environment variable."""
        with patch.dict(os.environ, {"AGENTCORE_RUNTIME_URL": "https://runtime.example.com"}):
            url = app._get_runtime_url()
            assert url == "https://runtime.example.com"

    def test_runtime_url_from_request_base_url(self, app):
        """Test runtime URL fallback to request.base_url."""
        mock_request = MagicMock()
        mock_request.base_url = "http://localhost:9000/"
        with patch.dict(os.environ, {}, clear=True):
            # Remove AGENTCORE_RUNTIME_URL if set
            os.environ.pop("AGENTCORE_RUNTIME_URL", None)
            url = app._get_runtime_url(request=mock_request)
            assert url == "http://localhost:9000/"

    def test_runtime_url_none_when_nothing_available(self, app):
        """Test runtime URL returns None when nothing is available."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AGENTCORE_RUNTIME_URL", None)
            url = app._get_runtime_url()
            assert url is None


class TestAsyncTaskDecorator:
    def test_async_task_decorator(self, agent_card):
        """Test @async_task decorator tracks task and returns result."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)

        @app.async_task
        async def my_task():
            return "done"

        assert my_task.__name__ == "my_task"
        result = asyncio.get_event_loop().run_until_complete(my_task())
        assert result == "done"
        assert len(app._active_tasks) == 0  # task completed

    def test_async_task_rejects_sync(self, agent_card):
        """Test @async_task raises ValueError for sync functions."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)

        with pytest.raises(ValueError, match="async"):
            @app.async_task
            def sync_func():
                pass

    def test_async_task_exception_cleanup(self, agent_card):
        """Test @async_task cleans up task on exception."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)

        @app.async_task
        async def failing_task():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            asyncio.get_event_loop().run_until_complete(failing_task())
        assert len(app._active_tasks) == 0


class TestAsyncTaskManagement:
    def test_add_and_complete_task(self, agent_card):
        """Test add_async_task and complete_async_task."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        task_id = app.add_async_task("test-task", metadata={"key": "val"})
        assert len(app._active_tasks) == 1
        assert app._active_tasks[task_id]["name"] == "test-task"
        assert app._active_tasks[task_id]["metadata"] == {"key": "val"}

        result = app.complete_async_task(task_id)
        assert result is True
        assert len(app._active_tasks) == 0

    def test_complete_unknown_task(self, agent_card):
        """Test completing a non-existent task returns False."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        result = app.complete_async_task(99999)
        assert result is False

    def test_get_async_task_info(self, agent_card):
        """Test get_async_task_info returns correct data."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        app.add_async_task("job-1")
        app.add_async_task("job-2")

        info = app.get_async_task_info()
        assert info["active_count"] == 2
        assert len(info["running_jobs"]) == 2
        names = {j["name"] for j in info["running_jobs"]}
        assert names == {"job-1", "job-2"}

    def test_force_and_clear_ping_status(self, agent_card):
        """Test force_ping_status and clear_forced_ping_status."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        app.force_ping_status(PingStatus.HEALTHY_BUSY)
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

        app.clear_forced_ping_status()
        assert app.get_current_ping_status() == PingStatus.HEALTHY


class TestContextVarsPropagation:
    def test_sync_handler_preserves_context_vars(self, agent_card):
        """Test sync handler receives context variables via contextvars.copy_context()."""
        app = BedrockAgentCoreA2AApp(agent_card=agent_card)
        captured_request_id = None

        @app.entrypoint
        def handler(request, context):
            nonlocal captured_request_id
            from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
            captured_request_id = BedrockAgentCoreContext.get_request_id()
            return {"ok": True}

        client = TestClient(app)
        client.post(
            "/",
            json={"jsonrpc": "2.0", "id": "1", "method": "test"},
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Request-Id": "ctx-req-123"},
        )

        assert captured_request_id == "ctx-req-123"
