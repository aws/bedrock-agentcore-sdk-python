"""Integration tests: assert user content is NOT present in OTEL spans.

COE Finding: User prompts and agent responses must never leak into exported
OTEL span attributes, events, or resource attributes.
"""

import json

import opentelemetry.trace
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
