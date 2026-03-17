import contextvars
import json
import uuid
from unittest.mock import patch

import pytest
from ag_ui.core import RunAgentInput, RunFinishedEvent, RunStartedEvent, TextMessageContentEvent
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from bedrock_agentcore.runtime.ag_ui import (
    AGUIApp,
    build_ag_ui_app,
    serve_ag_ui,
)
from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
from bedrock_agentcore.runtime.models import PingStatus


def _make_run_input(**overrides):
    defaults = {
        "thread_id": "t-1",
        "run_id": "r-1",
        "state": [],
        "messages": [{"role": "user", "content": "hello", "id": "msg-1"}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    defaults.update(overrides)
    return defaults


async def _echo_agent(input_data: RunAgentInput):
    yield RunStartedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)
    user_text = ""
    if input_data.messages:
        msg = input_data.messages[0]
        user_text = getattr(msg, "content", str(msg))
    yield TextMessageContentEvent(message_id="m-1", delta=f"echo: {user_text}")
    yield RunFinishedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)


class TestBuildAGUIApp:
    def test_routes_present(self):
        app = build_ag_ui_app(_echo_agent)
        paths = [r.path for r in app.routes]
        assert "/invocations" in paths
        assert "/ping" in paths
        assert "/ws" in paths

    def test_ping_returns_healthy_by_default(self):
        app = build_ag_ui_app(_echo_agent)
        client = TestClient(app)
        resp = client.get("/ping")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "Healthy"
        assert isinstance(body["time_of_last_update"], int)
        assert body["time_of_last_update"] > 0

    def test_custom_ping_handler(self):
        app = AGUIApp()
        app.entrypoint(_echo_agent)

        @app.ping
        def busy_ping():
            return PingStatus.HEALTHY_BUSY

        client = TestClient(app)
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json()["status"] == "HealthyBusy"

    def test_ping_fallback_on_error(self):
        def bad_ping():
            raise RuntimeError("boom")

        app = build_ag_ui_app(_echo_agent, ping_handler=bad_ping)
        client = TestClient(app)
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json()["status"] == "Healthy"

    def test_invocations_endpoint_exists(self):
        app = build_ag_ui_app(_echo_agent)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/invocations", json=_make_run_input())
        assert resp.status_code != 404


class TestEntrypoint:
    def test_decorator_form_async_generator(self):
        app = AGUIApp()

        @app.entrypoint
        async def my_agent(input_data: RunAgentInput):
            yield RunStartedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)
            yield RunFinishedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)

        assert app._handler is my_agent

    def test_object_form_with_run_method(self):
        class FrameworkAgent:
            async def run(self, input_data: RunAgentInput):
                yield RunStartedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)

        agent = FrameworkAgent()
        app = AGUIApp()
        app.entrypoint(agent)
        # Bound methods create new objects each access, so compare underlying function
        assert app._handler.__func__ is FrameworkAgent.run

    def test_context_arg_detection(self):
        async def handler_with_ctx(input_data, context):
            yield RunStartedEvent(thread_id="t", run_id="r")

        async def handler_no_ctx(input_data):
            yield RunStartedEvent(thread_id="t", run_id="r")

        app = AGUIApp()
        assert app._takes_context(handler_with_ctx) is True
        assert app._takes_context(handler_no_ctx) is False


class TestWebSocket:
    """Test that the entrypoint handler is served over WebSocket at /ws."""

    def test_websocket_streams_same_events_as_sse(self):
        app = build_ag_ui_app(_echo_agent)
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            ws.send_json(_make_run_input())
            messages = []
            while True:
                try:
                    msg = ws.receive_text()
                    messages.append(msg)
                except WebSocketDisconnect:
                    break

        event_types = []
        for msg in messages:
            data_str = msg.removeprefix("data: ").strip()
            if data_str:
                event_types.append(json.loads(data_str).get("type"))

        assert "RUN_STARTED" in event_types
        assert "TEXT_MESSAGE_CONTENT" in event_types
        assert "RUN_FINISHED" in event_types

    def test_websocket_no_entrypoint_closes_1011(self):
        app = AGUIApp()
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws") as ws:
                ws.send_json(_make_run_input())
                ws.receive_text()
        assert exc_info.value.code == 1011

    def test_websocket_invalid_run_input_closes_1003(self):
        app = build_ag_ui_app(_echo_agent)
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"thread_id": "t-1"})  # Missing required fields
                ws.receive_text()
        assert exc_info.value.code == 1003

    def test_websocket_error_emits_run_error_event(self):
        async def failing_agent(input_data: RunAgentInput):
            yield RunStartedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)
            raise RuntimeError("ws boom")

        app = build_ag_ui_app(failing_agent)
        client = TestClient(app)

        with client.websocket_connect("/ws") as ws:
            ws.send_json(_make_run_input())
            messages = []
            while True:
                try:
                    msg = ws.receive_text()
                    messages.append(msg)
                except WebSocketDisconnect:
                    break

        event_types = []
        for msg in messages:
            data_str = msg.removeprefix("data: ").strip()
            if data_str:
                event_types.append(json.loads(data_str).get("type"))

        assert "RUN_STARTED" in event_types
        assert "RUN_ERROR" in event_types

    def test_websocket_receives_bedrock_headers(self):
        captured = {}

        async def agent_with_ctx(input_data: RunAgentInput, context):
            captured["session_id"] = context.session_id
            yield RunStartedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)
            yield RunFinishedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)

        app = build_ag_ui_app(agent_with_ctx)
        client = TestClient(app)

        with client.websocket_connect(
            "/ws",
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "ws-sess-1"},
        ) as ws:
            ws.send_json(_make_run_input())
            while True:
                try:
                    ws.receive_text()
                except WebSocketDisconnect:
                    break

        assert captured["session_id"] == "ws-sess-1"


class TestBedrockHeaderExtraction:
    def _run_in_isolated_context(self, fn):
        ctx = contextvars.copy_context()
        return ctx.run(fn)

    def test_extracts_all_bedrock_headers(self):
        def _test():
            from starlette.requests import Request

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/invocations",
                "headers": [
                    (b"x-amzn-bedrock-agentcore-runtime-session-id", b"sess-abc"),
                    (b"x-amzn-bedrock-agentcore-runtime-request-id", b"req-456"),
                    (b"workloadaccesstoken", b"tok-xyz"),
                    (b"oauth2callbackurl", b"https://callback.example.com"),
                    (b"x-amzn-bedrock-agentcore-runtime-custom-foo", b"bar"),
                    (b"authorization", b"Bearer test-token"),
                ],
                "query_string": b"",
            }
            request = Request(scope)
            app = AGUIApp()
            rc = app._build_request_context(request)

            assert rc.session_id == "sess-abc"
            assert BedrockAgentCoreContext.get_session_id() == "sess-abc"
            assert BedrockAgentCoreContext.get_request_id() == "req-456"
            assert BedrockAgentCoreContext.get_workload_access_token() == "tok-xyz"
            assert BedrockAgentCoreContext.get_oauth2_callback_url() == "https://callback.example.com"

            headers = BedrockAgentCoreContext.get_request_headers()
            assert headers is not None
            assert "Authorization" in headers or "authorization" in headers
            custom_found = any(k.lower().startswith("x-amzn-bedrock-agentcore-runtime-custom-") for k in headers)
            assert custom_found

        self._run_in_isolated_context(_test)

    def test_auto_generates_request_id_when_missing(self):
        def _test():
            from starlette.requests import Request

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/invocations",
                "headers": [],
                "query_string": b"",
            }
            request = Request(scope)
            app = AGUIApp()
            app._build_request_context(request)

            generated_id = BedrockAgentCoreContext.get_request_id()
            assert generated_id is not None
            uuid.UUID(generated_id)

        self._run_in_isolated_context(_test)


class TestServeAGUI:
    @patch("uvicorn.run")
    def test_default_localhost(self, mock_uvicorn_run):
        with patch.dict("os.environ", {}, clear=False):
            with patch("os.path.exists", return_value=False):
                serve_ag_ui(_echo_agent)
        kw = mock_uvicorn_run.call_args[1]
        assert kw["host"] == "127.0.0.1"
        assert kw["port"] == 8080

    @patch("uvicorn.run")
    def test_docker_detection_dockerenv(self, mock_uvicorn_run):
        with patch("os.path.exists", return_value=True):
            serve_ag_ui(_echo_agent)
        assert mock_uvicorn_run.call_args[1]["host"] == "0.0.0.0"

    @patch("uvicorn.run")
    def test_docker_detection_env_var(self, mock_uvicorn_run):
        with patch.dict("os.environ", {"DOCKER_CONTAINER": "true"}):
            with patch("os.path.exists", return_value=False):
                serve_ag_ui(_echo_agent)
        assert mock_uvicorn_run.call_args[1]["host"] == "0.0.0.0"

    @patch("uvicorn.run")
    def test_custom_host_port(self, mock_uvicorn_run):
        serve_ag_ui(_echo_agent, host="10.0.0.1", port=9090)
        kw = mock_uvicorn_run.call_args[1]
        assert kw["host"] == "10.0.0.1"
        assert kw["port"] == 9090

    @patch("uvicorn.run")
    def test_kwargs_passthrough(self, mock_uvicorn_run):
        serve_ag_ui(_echo_agent, host="127.0.0.1", workers=4, log_level="debug")
        kw = mock_uvicorn_run.call_args[1]
        assert kw["workers"] == 4
        assert kw["log_level"] == "debug"
