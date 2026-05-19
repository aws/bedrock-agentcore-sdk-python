"""Integration tests: assert user content is NOT present in OTEL spans.

COE Finding: User prompts and agent responses must never leak into exported
OTEL span attributes, events, or resource attributes.
"""

import json
import uuid

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from starlette.testclient import TestClient

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.runtime import tracing as tracing_mod

SENTINEL_PROMPT = "SENSITIVE_USER_PROMPT_a1b2c3d4e5"
SENTINEL_RESPONSE = "SENSITIVE_AGENT_RESPONSE_f6g7h8i9j0"


def _assert_no_sentinel_in_spans(spans, sentinels):
    """Assert none of the sentinel strings appear anywhere in exported span data."""
    for span in spans:
        for sentinel in sentinels:
            assert sentinel not in span.name, f"Sentinel found in span name: {span.name}"

            for key, value in (span.attributes or {}).items():
                serialized = json.dumps(value) if not isinstance(value, str) else value
                assert sentinel not in str(key), f"Sentinel in attribute key: {key}"
                assert sentinel not in serialized, f"Sentinel in attribute value for key={key}: {serialized}"

            for event in span.events:
                assert sentinel not in event.name, f"Sentinel in event name: {event.name}"
                for key, value in (event.attributes or {}).items():
                    serialized = json.dumps(value) if not isinstance(value, str) else value
                    assert sentinel not in str(key), f"Sentinel in event attribute key: {key}"
                    assert sentinel not in serialized, f"Sentinel in event attribute value for key={key}: {serialized}"

            if hasattr(span, "resource") and span.resource:
                for key, value in (span.resource.attributes or {}).items():
                    serialized = json.dumps(value) if not isinstance(value, str) else value
                    assert sentinel not in str(key), f"Sentinel in resource attribute key: {key}"
                    assert sentinel not in serialized, (
                        f"Sentinel in resource attribute value for key={key}: {serialized}"
                    )


@pytest.fixture()
def otel_exporter():
    """Set up TracerProvider with InMemorySpanExporter for span capture."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    import opentelemetry.trace

    original_provider = opentelemetry.trace.get_tracer_provider()
    opentelemetry.trace.set_tracer_provider(provider)
    tracing_mod._registered_on = None

    yield exporter

    provider.shutdown()
    opentelemetry.trace.set_tracer_provider(original_provider)


class TestBedrockAgentCoreAppNoUserContent:
    """Verify BedrockAgentCoreApp does not leak user content into OTEL spans."""

    def test_sync_invocation_no_user_content_in_spans(self, otel_exporter):
        app = BedrockAgentCoreApp()

        @app.entrypoint
        def handler(payload):
            return {"response": SENTINEL_RESPONSE, "echo": payload.get("prompt")}

        client = TestClient(app)
        response = client.post(
            "/invocations",
            json={"prompt": SENTINEL_PROMPT},
            headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "test-session"},
        )

        assert response.status_code == 200
        spans = otel_exporter.get_finished_spans()
        _assert_no_sentinel_in_spans(spans, [SENTINEL_PROMPT, SENTINEL_RESPONSE])

    def test_streaming_invocation_no_user_content_in_spans(self, otel_exporter):
        app = BedrockAgentCoreApp()

        @app.entrypoint
        async def handler(payload):
            yield f"chunk1: {SENTINEL_RESPONSE}"
            yield f"chunk2: {payload.get('prompt')}"

        client = TestClient(app)
        response = client.post(
            "/invocations",
            json={"prompt": SENTINEL_PROMPT},
            headers={
                "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "test-session",
                "Accept": "text/event-stream",
            },
        )

        assert response.status_code == 200
        spans = otel_exporter.get_finished_spans()
        _assert_no_sentinel_in_spans(spans, [SENTINEL_PROMPT, SENTINEL_RESPONSE])


class TestA2ANoUserContent:
    """Verify A2A app does not leak user content into OTEL spans."""

    def test_a2a_message_send_no_user_content_in_spans(self, otel_exporter):
        a2a_sdk = pytest.importorskip("a2a")

        from a2a.server.agent_execution import AgentExecutor, RequestContext
        from a2a.server.events import EventQueue
        from a2a.server.tasks import TaskUpdater
        from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Part, TextPart
        from a2a.utils import new_task
        from a2a.utils.errors import ServerError

        from bedrock_agentcore.runtime.a2a import build_a2a_app

        class SentinelExecutor(AgentExecutor):
            async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
                task = context.current_task or new_task(context.message)
                if not context.current_task:
                    await event_queue.enqueue_event(task)
                updater = TaskUpdater(event_queue, task.id, task.context_id)
                await updater.add_artifact([Part(root=TextPart(text=SENTINEL_RESPONSE))])
                await updater.complete()

            async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
                raise ServerError(error=a2a.types.UnsupportedOperationError())

        card = AgentCard(
            name="sentinel-agent",
            description="Test agent",
            url="http://localhost:9000",
            version="0.1.0",
            capabilities=AgentCapabilities(streaming=True),
            skills=[AgentSkill(id="test", name="test", description="test", tags=["test"])],
            default_input_modes=["text"],
            default_output_modes=["text"],
        )

        app = build_a2a_app(executor=SentinelExecutor(), agent_card=card)
        client = TestClient(app, raise_server_exceptions=False)

        jsonrpc_body = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "id": 1,
            "params": {
                "message": {
                    "message_id": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": SENTINEL_PROMPT}],
                }
            },
        }

        response = client.post("/", json=jsonrpc_body)
        assert response.status_code == 200

        spans = otel_exporter.get_finished_spans()
        _assert_no_sentinel_in_spans(spans, [SENTINEL_PROMPT, SENTINEL_RESPONSE])


class TestAGUINoUserContent:
    """Verify AG-UI app does not leak user content into OTEL spans."""

    def test_ag_ui_no_user_content_in_spans(self, otel_exporter):
        pytest.importorskip("ag_ui")

        from ag_ui.core import RunAgentInput, RunFinishedEvent, RunStartedEvent, TextMessageContentEvent

        from bedrock_agentcore.runtime.ag_ui import build_ag_ui_app

        async def sentinel_agent(input_data: RunAgentInput):
            yield RunStartedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)
            yield TextMessageContentEvent(message_id="m-1", delta=SENTINEL_RESPONSE)
            yield RunFinishedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)

        app = build_ag_ui_app(sentinel_agent)
        client = TestClient(app, raise_server_exceptions=False)

        run_input = {
            "thread_id": "t-otel-test",
            "run_id": "r-otel-test",
            "state": [],
            "messages": [{"role": "user", "content": SENTINEL_PROMPT, "id": "msg-1"}],
            "tools": [],
            "context": [],
            "forwardedProps": {},
        }

        response = client.post("/invocations", json=run_input)
        assert response.status_code == 200

        spans = otel_exporter.get_finished_spans()
        _assert_no_sentinel_in_spans(spans, [SENTINEL_PROMPT, SENTINEL_RESPONSE])
