"""Integration tests for A2A protocol support.

Uses real a2a-sdk v1 route factories + DefaultRequestHandler with a concrete
executor. Every test sends real HTTP requests through the full stack.
"""

import json
import uuid

import pytest
from a2a.helpers import new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Part,
)
from a2a.utils.errors import UnsupportedOperationError
from starlette.testclient import TestClient

from bedrock_agentcore.runtime.a2a import BedrockCallContextBuilder, build_a2a_app


class EchoExecutor(AgentExecutor):
    """Echoes user input back as an artifact. Records context for inspection."""

    def __init__(self):
        self.last_call_context = None
        self.last_user_text = None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        self.last_call_context = context.call_context
        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        user_text = context.get_user_input()
        self.last_user_text = user_text

        await updater.add_artifact([Part(text=f"echo: {user_text}")])
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError()


def _make_card() -> AgentCard:
    return AgentCard(
        name="echo-agent",
        description="Integration test echo agent",
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=True),
        skills=[AgentSkill(id="echo", name="echo", description="Echoes input", tags=["echo"])],
        default_input_modes=["text"],
        default_output_modes=["text"],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url="http://localhost:9000",
            )
        ],
    )


def _jsonrpc_request(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    body: dict = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params is not None:
        body["params"] = params
    return body


def _send_message_params(text: str = "hello") -> dict:
    return {
        "message": {
            "messageId": str(uuid.uuid4()),
            "role": "ROLE_USER",
            "parts": [{"text": text}],
        }
    }


@pytest.fixture()
def echo_executor():
    return EchoExecutor()


@pytest.fixture()
def a2a_client(echo_executor):
    """Full app with BedrockCallContextBuilder wired in -- exercises our glue."""
    app = build_a2a_app(echo_executor, _make_card(), context_builder=BedrockCallContextBuilder())
    return TestClient(app, raise_server_exceptions=False, headers={"A2A-Version": "1.0"})


@pytest.mark.integration
class TestA2AServerIntegration:
    def test_message_send_returns_completed_task_with_echo_artifact(self, a2a_client):
        resp = a2a_client.post(
            "/",
            json=_jsonrpc_request("SendMessage", _send_message_params("hi")),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "result" in body
        task = body["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        artifacts = task["artifacts"]
        assert len(artifacts) == 1
        assert artifacts[0]["parts"][0]["text"] == "echo: hi"

    def test_message_send_stream_produces_sse_with_artifact_and_status(self, a2a_client):
        resp = a2a_client.post(
            "/",
            json=_jsonrpc_request("SendStreamingMessage", _send_message_params("stream-test")),
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE data lines. Each envelope contains one StreamResponse field.
        results = []
        for line in resp.text.split("\n"):
            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
                if data_str:
                    envelope = json.loads(data_str)
                    result = envelope.get("result", {})
                    results.append(result)

        assert len(results) >= 2, f"Expected at least 2 SSE events, got {len(results)}"

        assert any("artifactUpdate" in result for result in results)
        assert any("statusUpdate" in result for result in results)

        # Verify the artifact content in the artifact update event
        artifact_event = next(r["artifactUpdate"] for r in results if "artifactUpdate" in r)
        assert artifact_event["artifact"]["parts"][0]["text"] == "echo: stream-test"

        # Verify the final status is completed
        status_event = next(r["statusUpdate"] for r in results if "statusUpdate" in r)
        assert status_event["status"]["state"] == "TASK_STATE_COMPLETED"

    def test_get_task_returns_previously_created_task(self, a2a_client):
        send_resp = a2a_client.post(
            "/",
            json=_jsonrpc_request("SendMessage", _send_message_params("for-get")),
        )
        task_id = send_resp.json()["result"]["task"]["id"]

        resp = a2a_client.post(
            "/",
            json=_jsonrpc_request("GetTask", {"id": task_id}),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["id"] == task_id
        assert body["result"]["status"]["state"] == "TASK_STATE_COMPLETED"
        assert body["result"]["artifacts"][0]["parts"][0]["text"] == "echo: for-get"

    def test_cancel_task_returns_unsupported_error(self, a2a_client):
        send_resp = a2a_client.post(
            "/",
            json=_jsonrpc_request("SendMessage", _send_message_params("for-cancel")),
        )
        task_id = send_resp.json()["result"]["task"]["id"]

        resp = a2a_client.post(
            "/",
            json=_jsonrpc_request("CancelTask", {"id": task_id}),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "error" in body

    def test_unknown_method_returns_method_not_found(self, a2a_client):
        resp = a2a_client.post(
            "/",
            json=_jsonrpc_request("nonexistent/method"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["error"]["code"] == -32601

    def test_invalid_params_returns_error(self, a2a_client):
        resp = a2a_client.post(
            "/",
            json=_jsonrpc_request("SendMessage", {"bad_key": "bad_value"}),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "error" in body

    def test_agent_card_endpoint_returns_full_card(self, a2a_client):
        resp = a2a_client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "echo-agent"
        assert body["version"] == "0.1.0"
        assert len(body["skills"]) == 1
        assert body["skills"][0]["id"] == "echo"
        assert body["capabilities"]["streaming"] is True

    def test_ping_endpoint_returns_bedrock_health_format(self, a2a_client):
        resp = a2a_client.get("/ping")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "Healthy"
        assert "time_of_last_update" not in body

    def test_bedrock_headers_propagated_to_executor(self, echo_executor):
        builder = BedrockCallContextBuilder()
        app = build_a2a_app(echo_executor, _make_card(), context_builder=builder)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/",
            json=_jsonrpc_request("SendMessage", _send_message_params("headers-test")),
            headers={
                "A2A-Version": "1.0",
                "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "integ-sess-1",
                "X-Amzn-Bedrock-AgentCore-Runtime-Request-Id": "integ-req-1",
                "WorkloadAccessToken": "integ-token",
                "OAuth2CallbackUrl": "https://callback.example.com",
            },
        )
        assert resp.status_code == 200
        # Verify the task completed (executor actually ran)
        assert resp.json()["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"

        # Verify Bedrock headers reached the executor via ServerCallContext
        ctx = echo_executor.last_call_context
        assert ctx is not None
        assert ctx.state["session_id"] == "integ-sess-1"
        assert ctx.state["bedrock_request_id"] == "integ-req-1"
        assert ctx.state["workload_access_token"] == "integ-token"
        assert ctx.state["oauth2_callback_url"] == "https://callback.example.com"

    def test_user_input_reaches_executor(self, a2a_client, echo_executor):
        """Verify the user message text flows all the way to the executor."""
        a2a_client.post(
            "/",
            json=_jsonrpc_request("SendMessage", _send_message_params("verify-input")),
        )
        assert echo_executor.last_user_text == "verify-input"
