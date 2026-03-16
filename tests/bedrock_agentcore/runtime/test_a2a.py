import contextvars
import uuid
from unittest.mock import patch

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Part, TextPart
from a2a.utils import new_task
from starlette.testclient import TestClient

from bedrock_agentcore.runtime.a2a import (
    BedrockCallContextBuilder,
    build_a2a_app,
    serve_a2a,
)
from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
from bedrock_agentcore.runtime.models import PingStatus


class _EchoExecutor(AgentExecutor):
    """Echoes user input back as an artifact. Records call_context for inspection."""

    def __init__(self):
        self.last_call_context = None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        self.last_call_context = context.call_context
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        user_text = context.get_user_input()
        await updater.add_artifact([Part(root=TextPart(text=f"echo: {user_text}"))])
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def _make_agent_card() -> AgentCard:
    return AgentCard(
        name="test-agent",
        description="A test agent",
        url="http://localhost:9000",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        skills=[AgentSkill(id="echo", name="echo", description="Echoes input", tags=["echo"])],
        default_input_modes=["text"],
        default_output_modes=["text"],
    )


def _jsonrpc_request(method: str, params: dict | None = None) -> dict:
    body: dict = {"jsonrpc": "2.0", "method": method, "id": 1}
    if params is not None:
        body["params"] = params
    return body


def _send_message_params(text: str = "hello") -> dict:
    return {
        "message": {
            "message_id": str(uuid.uuid4()),
            "role": "user",
            "parts": [{"kind": "text", "text": text}],
        }
    }


class TestBuildA2AApp:
    def test_ping_returns_healthy_with_timestamp(self):
        app = build_a2a_app(_EchoExecutor(), _make_agent_card())
        client = TestClient(app)
        resp = client.get("/ping")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "Healthy"
        assert isinstance(body["time_of_last_update"], int)
        assert body["time_of_last_update"] > 0

    def test_custom_ping_handler(self):
        app = build_a2a_app(
            _EchoExecutor(),
            _make_agent_card(),
            ping_handler=lambda: PingStatus.HEALTHY_BUSY,
        )
        client = TestClient(app)
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json()["status"] == "HealthyBusy"

    def test_ping_fallback_on_error(self):
        def bad_ping():
            raise RuntimeError("boom")

        app = build_a2a_app(
            _EchoExecutor(),
            _make_agent_card(),
            ping_handler=bad_ping,
        )
        client = TestClient(app)
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json()["status"] == "Healthy"

    def test_agent_card_returns_card_data(self):
        app = build_a2a_app(_EchoExecutor(), _make_agent_card())
        client = TestClient(app)
        resp = client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "test-agent"
        assert body["version"] == "1.0.0"
        assert body["skills"][0]["id"] == "echo"

    def test_message_send_executes_and_returns_completed_task(self):
        """Verify the full RPC path: JSON-RPC request -> executor -> completed task."""
        app = build_a2a_app(_EchoExecutor(), _make_agent_card())
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/", json=_jsonrpc_request("message/send", _send_message_params("unit-test")))
        assert resp.status_code == 200
        body = resp.json()
        assert "result" in body
        task = body["result"]
        assert task["status"]["state"] == "completed"
        assert task["artifacts"][0]["parts"][0]["text"] == "echo: unit-test"

    def test_custom_task_store_is_used_for_persistence(self):
        """Verify that a custom task_store actually stores the task."""
        store = InMemoryTaskStore()
        app = build_a2a_app(_EchoExecutor(), _make_agent_card(), task_store=store)
        client = TestClient(app, raise_server_exceptions=False)

        # Send a message so a task gets created
        resp = client.post("/", json=_jsonrpc_request("message/send", _send_message_params("store-test")))
        task_id = resp.json()["result"]["id"]

        # Retrieve from the same store via tasks/get
        resp2 = client.post("/", json=_jsonrpc_request("tasks/get", {"id": task_id}))
        assert resp2.json()["result"]["id"] == task_id

    def test_agent_card_url_auto_populated_from_env(self):
        """When AGENTCORE_RUNTIME_URL is set, agent_card.url is overridden."""
        card = _make_agent_card()
        assert card.url == "http://localhost:9000"
        with patch.dict("os.environ", {"AGENTCORE_RUNTIME_URL": "https://deployed.example.com/"}):
            build_a2a_app(_EchoExecutor(), card)
        assert card.url == "https://deployed.example.com/"

    def test_agent_card_url_unchanged_without_env(self):
        """When AGENTCORE_RUNTIME_URL is not set, agent_card.url stays as-is."""
        card = _make_agent_card()
        original_url = card.url
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("AGENTCORE_RUNTIME_URL", None)
            build_a2a_app(_EchoExecutor(), card)
        assert card.url == original_url

    def test_auto_builds_card_when_none_provided(self):
        """When agent_card is omitted, a default card is built automatically."""
        app = build_a2a_app(_EchoExecutor())
        client = TestClient(app)
        resp = client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "agent"
        assert body["url"] == "http://localhost:9000/"

    def test_auto_builds_card_from_strands_executor(self):
        """When executor has .agent with name/description, card is built from it."""

        class _FakeAgent:
            name = "My Strands Agent"
            description = "Does cool stuff"

        executor = _EchoExecutor()
        executor.agent = _FakeAgent()  # type: ignore[attr-defined]

        app = build_a2a_app(executor)
        client = TestClient(app)
        resp = client.get("/.well-known/agent-card.json")
        body = resp.json()
        assert body["name"] == "My Strands Agent"
        assert body["description"] == "Does cool stuff"

    def test_auto_card_uses_runtime_url_env(self):
        """Auto-built card picks up AGENTCORE_RUNTIME_URL."""
        with patch.dict("os.environ", {"AGENTCORE_RUNTIME_URL": "https://prod.example.com/"}):
            app = build_a2a_app(_EchoExecutor())
        client = TestClient(app)
        resp = client.get("/.well-known/agent-card.json")
        assert resp.json()["url"] == "https://prod.example.com/"


class TestBedrockCallContextBuilder:
    def _run_in_isolated_context(self, fn):
        """Run fn in a fresh contextvars.Context so tests don't leak state."""
        ctx = contextvars.copy_context()
        return ctx.run(fn)

    def test_extracts_all_bedrock_headers(self):
        """Verify every Bedrock header is extracted into both ServerCallContext.state and contextvars."""

        def _test():
            from starlette.requests import Request

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/",
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
            builder = BedrockCallContextBuilder()
            result = builder.build(request)

            # Check ServerCallContext state dict
            assert result.state["session_id"] == "sess-abc"
            assert result.state["request_id"] == "req-456"
            assert result.state["workload_access_token"] == "tok-xyz"
            assert result.state["oauth2_callback_url"] == "https://callback.example.com"

            # Check contextvars were actually set
            assert BedrockAgentCoreContext.get_session_id() == "sess-abc"
            assert BedrockAgentCoreContext.get_request_id() == "req-456"
            assert BedrockAgentCoreContext.get_workload_access_token() == "tok-xyz"
            assert BedrockAgentCoreContext.get_oauth2_callback_url() == "https://callback.example.com"

            # Check custom + authorization headers collected
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
                "path": "/",
                "headers": [],
                "query_string": b"",
            }
            request = Request(scope)
            builder = BedrockCallContextBuilder()
            result = builder.build(request)

            generated_id = result.state["request_id"]
            assert generated_id is not None
            # Verify it's a valid UUID
            uuid.UUID(generated_id)
            assert BedrockAgentCoreContext.get_request_id() == generated_id

        self._run_in_isolated_context(_test)

    def test_optional_headers_omitted_from_state_when_absent(self):
        """When optional headers (token, oauth2) are missing, they must not appear in state."""

        def _test():
            from starlette.requests import Request

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/",
                "headers": [
                    (b"x-amzn-bedrock-agentcore-runtime-session-id", b"sess-only"),
                ],
                "query_string": b"",
            }
            request = Request(scope)
            builder = BedrockCallContextBuilder()
            result = builder.build(request)

            assert result.state["session_id"] == "sess-only"
            assert "workload_access_token" not in result.state
            assert "oauth2_callback_url" not in result.state

        self._run_in_isolated_context(_test)

    def test_headers_reach_executor_through_full_app(self):
        """End-to-end: HTTP headers -> CallContextBuilder -> a2a-sdk -> executor.call_context."""

        def _test():
            executor = _EchoExecutor()
            builder = BedrockCallContextBuilder()
            app = build_a2a_app(executor, _make_agent_card(), context_builder=builder)
            client = TestClient(app, raise_server_exceptions=False)

            client.post(
                "/",
                json=_jsonrpc_request("message/send", _send_message_params("ctx-test")),
                headers={
                    "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "unit-sess",
                    "X-Amzn-Bedrock-AgentCore-Runtime-Request-Id": "unit-req",
                    "WorkloadAccessToken": "unit-token",
                },
            )

            ctx = executor.last_call_context
            assert ctx is not None
            assert ctx.state["session_id"] == "unit-sess"
            assert ctx.state["request_id"] == "unit-req"
            assert ctx.state["workload_access_token"] == "unit-token"

        self._run_in_isolated_context(_test)


class TestServeA2A:
    @patch("uvicorn.run")
    def test_default_localhost(self, mock_uvicorn_run):
        with patch.dict("os.environ", {}, clear=False):
            with patch("os.path.exists", return_value=False):
                serve_a2a(_EchoExecutor(), _make_agent_card())
        kw = mock_uvicorn_run.call_args[1]
        assert kw["host"] == "127.0.0.1"
        assert kw["port"] == 9000

    @patch("uvicorn.run")
    def test_docker_detection_dockerenv(self, mock_uvicorn_run):
        with patch("os.path.exists", return_value=True):
            serve_a2a(_EchoExecutor(), _make_agent_card())
        assert mock_uvicorn_run.call_args[1]["host"] == "0.0.0.0"

    @patch("uvicorn.run")
    def test_docker_detection_env_var(self, mock_uvicorn_run):
        with patch.dict("os.environ", {"DOCKER_CONTAINER": "true"}):
            with patch("os.path.exists", return_value=False):
                serve_a2a(_EchoExecutor(), _make_agent_card())
        assert mock_uvicorn_run.call_args[1]["host"] == "0.0.0.0"

    @patch("uvicorn.run")
    def test_custom_host_port(self, mock_uvicorn_run):
        serve_a2a(_EchoExecutor(), _make_agent_card(), host="10.0.0.1", port=8888)
        kw = mock_uvicorn_run.call_args[1]
        assert kw["host"] == "10.0.0.1"
        assert kw["port"] == 8888

    @patch("uvicorn.run")
    def test_kwargs_passthrough(self, mock_uvicorn_run):
        serve_a2a(
            _EchoExecutor(),
            _make_agent_card(),
            host="127.0.0.1",
            workers=4,
            log_level="debug",
        )
        kw = mock_uvicorn_run.call_args[1]
        assert kw["workers"] == 4
        assert kw["log_level"] == "debug"

    @patch("uvicorn.run")
    def test_serve_without_agent_card(self, mock_uvicorn_run):
        """serve_a2a works with just an executor, no card needed."""
        with patch("os.path.exists", return_value=False):
            serve_a2a(_EchoExecutor())
        mock_uvicorn_run.assert_called_once()
