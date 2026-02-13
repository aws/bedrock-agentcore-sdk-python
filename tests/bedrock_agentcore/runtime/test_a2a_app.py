"""Tests for BedrockAgentCoreA2AApp."""

import asyncio
import contextlib
import json
import os
import uuid
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from bedrock_agentcore.runtime import (
    AgentCard,
    AgentSkill,
    BedrockAgentCoreA2AApp,
    JsonRpcRequest,
)


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
        assert "Test error" in data["error"]["message"]


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
