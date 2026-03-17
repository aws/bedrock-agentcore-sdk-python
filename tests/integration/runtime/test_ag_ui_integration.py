"""Integration tests for AG-UI protocol support.

Uses a real AGUIApp with ag-ui-protocol types and encoder --
no mocks for the ag-ui layer. Every test sends real HTTP requests through the
full stack.
"""

import contextvars
import json

import pytest
from ag_ui.core import (
    RunAgentInput,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
)
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from bedrock_agentcore.runtime.ag_ui import build_ag_ui_app
from bedrock_agentcore.runtime.context import RequestContext


def _make_run_input(**overrides):
    defaults = {
        "thread_id": "t-integ",
        "run_id": "r-integ",
        "state": [],
        "messages": [{"role": "user", "content": "integration test", "id": "msg-1"}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    defaults.update(overrides)
    return defaults


async def _echo_agent(input_data: RunAgentInput):
    """Simple echo agent that yields start, content, and finished events."""
    yield RunStartedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)
    user_text = ""
    if input_data.messages:
        msg = input_data.messages[0]
        user_text = getattr(msg, "content", str(msg))
    yield TextMessageContentEvent(message_id="m-1", delta=f"echo: {user_text}")
    yield RunFinishedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)


async def _echo_agent_with_context(input_data: RunAgentInput, context: RequestContext):
    """Echo agent that also captures context for inspection."""
    _echo_agent_with_context.last_context = context
    async for event in _echo_agent(input_data):
        yield event


_echo_agent_with_context.last_context = None


@pytest.fixture()
def ag_ui_client():
    app = build_ag_ui_app(_echo_agent)
    return TestClient(app, raise_server_exceptions=False)


def _parse_sse_event_types(text: str) -> list[str]:
    """Extract event types from an SSE response body."""
    event_types = []
    for line in text.split("\n"):
        if line.startswith("data:"):
            data = line[len("data:") :].strip()
            if data:
                event_types.append(json.loads(data).get("type"))
    return event_types


def _collect_ws_event_types(ws) -> list[str]:
    """Read all text frames from a WebSocket and return event types."""
    event_types = []
    while True:
        try:
            msg = ws.receive_text()
        except WebSocketDisconnect:
            break
        data_str = msg.removeprefix("data: ").strip()
        if data_str:
            event_types.append(json.loads(data_str).get("type"))
    return event_types


@pytest.mark.integration
class TestAGUIServerIntegration:
    # -- SSE transport (POST /invocations) ------------------------------------

    def test_invocation_streams_events(self, ag_ui_client):
        resp = ag_ui_client.post("/invocations", json=_make_run_input())
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        event_types = _parse_sse_event_types(resp.text)
        assert len(event_types) >= 3
        assert "RUN_STARTED" in event_types
        assert "TEXT_MESSAGE_CONTENT" in event_types
        assert "RUN_FINISHED" in event_types

    def test_ping_endpoint(self, ag_ui_client):
        resp = ag_ui_client.get("/ping")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "Healthy"
        assert isinstance(body["time_of_last_update"], int)

    def test_bedrock_headers_propagated_via_sse(self):
        def _test():
            app = build_ag_ui_app(_echo_agent_with_context)
            client = TestClient(app, raise_server_exceptions=False)

            _echo_agent_with_context.last_context = None

            resp = client.post(
                "/invocations",
                json=_make_run_input(),
                headers={
                    "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "integ-sess",
                    "X-Amzn-Bedrock-AgentCore-Runtime-Request-Id": "integ-req",
                    "WorkloadAccessToken": "integ-token",
                    "OAuth2CallbackUrl": "https://callback.example.com",
                },
            )
            assert resp.status_code == 200

            ctx = _echo_agent_with_context.last_context
            assert ctx is not None
            assert ctx.session_id == "integ-sess"

        ctx = contextvars.copy_context()
        ctx.run(_test)

    def test_invalid_json_returns_error(self, ag_ui_client):
        resp = ag_ui_client.post(
            "/invocations",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_invalid_run_agent_input_returns_400(self, ag_ui_client):
        resp = ag_ui_client.post(
            "/invocations",
            json={"thread_id": "t-1"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"] == "Invalid RunAgentInput"

    def test_framework_agent_with_run_method(self):
        class FrameworkAgent:
            async def run(self, input_data: RunAgentInput):
                yield RunStartedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)
                yield RunFinishedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)

        app = build_ag_ui_app(FrameworkAgent())
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/invocations", json=_make_run_input())
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_error_emits_run_error_event_via_sse(self):
        async def failing_agent(input_data: RunAgentInput):
            yield RunStartedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)
            raise RuntimeError("agent exploded")

        app = build_ag_ui_app(failing_agent)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/invocations", json=_make_run_input())
        assert resp.status_code == 200
        assert "RUN_ERROR" in _parse_sse_event_types(resp.text)

    # -- WebSocket transport (/ws) --------------------------------------------

    def test_websocket_streams_same_events_as_sse(self, ag_ui_client):
        """Same entrypoint handler produces same events over /ws."""
        with ag_ui_client.websocket_connect("/ws") as ws:
            ws.send_json(_make_run_input())
            event_types = _collect_ws_event_types(ws)

        assert "RUN_STARTED" in event_types
        assert "TEXT_MESSAGE_CONTENT" in event_types
        assert "RUN_FINISHED" in event_types

    def test_websocket_bedrock_headers_propagated(self):
        def _test():
            app = build_ag_ui_app(_echo_agent_with_context)
            client = TestClient(app, raise_server_exceptions=False)

            _echo_agent_with_context.last_context = None

            with client.websocket_connect(
                "/ws",
                headers={
                    "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "ws-integ-sess",
                    "WorkloadAccessToken": "ws-integ-token",
                },
            ) as ws:
                ws.send_json(_make_run_input())
                _collect_ws_event_types(ws)

            ctx = _echo_agent_with_context.last_context
            assert ctx is not None
            assert ctx.session_id == "ws-integ-sess"

        ctx = contextvars.copy_context()
        ctx.run(_test)

    def test_websocket_error_emits_run_error_event(self):
        async def failing_agent(input_data: RunAgentInput):
            yield RunStartedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)
            raise RuntimeError("ws exploded")

        app = build_ag_ui_app(failing_agent)
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            ws.send_json(_make_run_input())
            event_types = _collect_ws_event_types(ws)

        assert "RUN_STARTED" in event_types
        assert "RUN_ERROR" in event_types

    def test_websocket_invalid_input_closes_1003(self):
        app = build_ag_ui_app(_echo_agent)
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"thread_id": "t-1"})
                ws.receive_text()
        assert exc_info.value.code == 1003
