"""Tests for BaggageSpanProcessor and auto-registration helper."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.runtime.tracing import BaggageSpanProcessor, _ensure_baggage_processor_registered

# ---------------------------------------------------------------------------
# BaggageSpanProcessor.on_start
# ---------------------------------------------------------------------------


class TestBaggageSpanProcessorOnStart:
    def setup_method(self):
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        BedrockAgentCoreContext.set_routing_experiment(None, None)

    def _make_span(self):
        span = MagicMock()
        return span

    def test_sets_both_attributes_when_both_present(self):
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        BedrockAgentCoreContext.set_routing_experiment("arn:aws:bedrock:us-east-1:123:exp/e1", "blue")
        span = self._make_span()
        BaggageSpanProcessor().on_start(span)

        calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert calls["aws.agentcore.gateway.routing_experiment_arn"] == "arn:aws:bedrock:us-east-1:123:exp/e1"
        assert calls["aws.agentcore.gateway.routing_experiment_variant_name"] == "blue"

    def test_sets_only_arn_when_variant_absent(self):
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        BedrockAgentCoreContext.set_routing_experiment("arn:aws:bedrock:us-east-1:123:exp/e1", None)
        span = self._make_span()
        BaggageSpanProcessor().on_start(span)

        calls = {c[0][0] for c in span.set_attribute.call_args_list}
        assert "aws.agentcore.gateway.routing_experiment_arn" in calls
        assert "aws.agentcore.gateway.routing_experiment_variant_name" not in calls

    def test_sets_only_variant_when_arn_absent(self):
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        BedrockAgentCoreContext.set_routing_experiment(None, "green")
        span = self._make_span()
        BaggageSpanProcessor().on_start(span)

        calls = {c[0][0] for c in span.set_attribute.call_args_list}
        assert "aws.agentcore.gateway.routing_experiment_variant_name" in calls
        assert "aws.agentcore.gateway.routing_experiment_arn" not in calls

    def test_sets_no_attributes_when_both_absent(self):
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        BedrockAgentCoreContext.set_routing_experiment(None, None)
        span = self._make_span()
        BaggageSpanProcessor().on_start(span)

        span.set_attribute.assert_not_called()

    def test_falls_back_to_parent_context_baggage_when_contextvars_empty(self):
        """ASGI entry span: ContextVars not yet set, baggage in parent_context."""
        from opentelemetry import baggage as otel_baggage
        from opentelemetry import context as otel_context

        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        BedrockAgentCoreContext.set_routing_experiment(None, None)

        parent_ctx = otel_baggage.set_baggage(
            "aws.agentcore.gateway.routing_experiment_arn",
            "arn:exp-fallback",
            otel_baggage.set_baggage(
                "aws.agentcore.gateway.routing_experiment_variant_name",
                "green",
                otel_context.get_current(),
            ),
        )

        span = self._make_span()
        BaggageSpanProcessor().on_start(span, parent_context=parent_ctx)

        calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert calls["aws.agentcore.gateway.routing_experiment_arn"] == "arn:exp-fallback"
        assert calls["aws.agentcore.gateway.routing_experiment_variant_name"] == "green"

    def test_contextvar_takes_priority_over_parent_context_baggage(self):
        """ContextVar value wins for both fields when both sources are set."""
        from opentelemetry import baggage as otel_baggage
        from opentelemetry import context as otel_context

        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        BedrockAgentCoreContext.set_routing_experiment("arn:exp-ctx", "blue")

        parent_ctx = otel_baggage.set_baggage(
            "aws.agentcore.gateway.routing_experiment_arn",
            "arn:exp-baggage",
            otel_baggage.set_baggage(
                "aws.agentcore.gateway.routing_experiment_variant_name",
                "green",
                otel_context.get_current(),
            ),
        )

        span = self._make_span()
        BaggageSpanProcessor().on_start(span, parent_context=parent_ctx)

        calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert calls["aws.agentcore.gateway.routing_experiment_arn"] == "arn:exp-ctx"
        assert calls["aws.agentcore.gateway.routing_experiment_variant_name"] == "blue"

    def test_parent_context_fallback_skipped_when_opentelemetry_not_installed(self):
        """When opentelemetry-api is absent, parent_context baggage fallback is silently skipped."""
        import sys

        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        BedrockAgentCoreContext.set_routing_experiment(None, None)
        span = self._make_span()

        saved = {k: v for k, v in sys.modules.items() if k == "opentelemetry" or k.startswith("opentelemetry.")}
        for key in saved:
            del sys.modules[key]
        try:
            # Must not raise; span gets no attributes (no ContextVar, no baggage fallback).
            BaggageSpanProcessor().on_start(span, parent_context=MagicMock())
        finally:
            sys.modules.update(saved)

        span.set_attribute.assert_not_called()

    def test_different_contexts_get_different_values(self):
        """Concurrent requests must not bleed experiment values into each other."""
        import contextvars

        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        processor = BaggageSpanProcessor()
        results = {}

        def run_in_context(name, arn, variant):
            ctx = contextvars.copy_context()

            def _inner():
                BedrockAgentCoreContext.set_routing_experiment(arn, variant)
                span = MagicMock()
                processor.on_start(span)
                results[name] = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}

            ctx.run(_inner)

        t1 = threading.Thread(target=run_in_context, args=("req1", "arn:exp-A", "blue"))
        t2 = threading.Thread(target=run_in_context, args=("req2", "arn:exp-B", "green"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["req1"]["aws.agentcore.gateway.routing_experiment_arn"] == "arn:exp-A"
        assert results["req1"]["aws.agentcore.gateway.routing_experiment_variant_name"] == "blue"
        assert results["req2"]["aws.agentcore.gateway.routing_experiment_arn"] == "arn:exp-B"
        assert results["req2"]["aws.agentcore.gateway.routing_experiment_variant_name"] == "green"


class TestBaggageSpanProcessorNoOpMethods:
    def test_on_end_does_not_raise(self):
        BaggageSpanProcessor().on_end(MagicMock())

    def test_shutdown_does_not_raise(self):
        BaggageSpanProcessor().shutdown()

    def test_force_flush_returns_true(self):
        assert BaggageSpanProcessor().force_flush() is True


# ---------------------------------------------------------------------------
# _ensure_baggage_processor_registered
# ---------------------------------------------------------------------------


class TestEnsureBaggageProcessorRegistered:
    def setup_method(self):
        # Reset module-level state before each test.
        import bedrock_agentcore.runtime.tracing as tracing_mod

        tracing_mod._registered_on = None

    def test_registers_on_first_call(self):
        mock_provider = MagicMock()

        with patch("opentelemetry.trace.get_tracer_provider", return_value=mock_provider):
            _ensure_baggage_processor_registered()

        mock_provider.add_span_processor.assert_called_once()
        args = mock_provider.add_span_processor.call_args[0]
        assert isinstance(args[0], BaggageSpanProcessor)

    def test_skips_registration_on_same_provider(self):
        mock_provider = MagicMock()

        with patch("opentelemetry.trace.get_tracer_provider", return_value=mock_provider):
            _ensure_baggage_processor_registered()
            _ensure_baggage_processor_registered()

        assert mock_provider.add_span_processor.call_count == 1

    def test_re_registers_when_provider_replaced(self):
        provider_a = MagicMock()
        provider_b = MagicMock()

        with patch("opentelemetry.trace.get_tracer_provider", return_value=provider_a):
            _ensure_baggage_processor_registered()

        with patch("opentelemetry.trace.get_tracer_provider", return_value=provider_b):
            _ensure_baggage_processor_registered()

        provider_a.add_span_processor.assert_called_once()
        provider_b.add_span_processor.assert_called_once()

    def test_noop_when_opentelemetry_not_installed(self):
        import sys

        saved = {k: v for k, v in sys.modules.items() if k == "opentelemetry" or k.startswith("opentelemetry.")}
        for key in saved:
            del sys.modules[key]
        try:
            _ensure_baggage_processor_registered()  # must not raise
        finally:
            sys.modules.update(saved)

    def test_thread_safe_registration(self):
        """Only one add_span_processor call even under concurrent first requests."""
        mock_provider = MagicMock()
        barrier = threading.Barrier(10)

        def _call():
            barrier.wait()
            with patch("opentelemetry.trace.get_tracer_provider", return_value=mock_provider):
                _ensure_baggage_processor_registered()

        threads = [threading.Thread(target=_call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert mock_provider.add_span_processor.call_count == 1


# ---------------------------------------------------------------------------
# End-to-end: baggage header → ContextVar → span attribute (via app.py)
# ---------------------------------------------------------------------------


class TestBaggageEndToEnd:
    def test_baggage_processor_registered_on_provider(self):
        """BaggageSpanProcessor is registered on the TracerProvider when app is created."""
        import bedrock_agentcore.runtime.tracing as tracing_mod
        from bedrock_agentcore.runtime import BedrockAgentCoreApp

        tracing_mod._registered_on = None
        mock_provider = MagicMock()

        with patch("opentelemetry.trace.get_tracer_provider", return_value=mock_provider):
            BedrockAgentCoreApp()

        mock_provider.add_span_processor.assert_called_once()
        processor = mock_provider.add_span_processor.call_args[0][0]
        assert isinstance(processor, BaggageSpanProcessor)

    def test_contextvars_populated_during_handler(self):
        """Baggage header → ContextVars are set by the time the handler runs."""
        from starlette.testclient import TestClient

        from bedrock_agentcore.runtime import BedrockAgentCoreApp
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        app = BedrockAgentCoreApp()
        captured = {}

        @app.entrypoint
        def handler(payload):
            captured["arn"] = BedrockAgentCoreContext.get_routing_experiment_arn()
            captured["variant"] = BedrockAgentCoreContext.get_routing_experiment_variant()
            return {"ok": True}

        client = TestClient(app)
        baggage = (
            "aws.agentcore.gateway.routing_experiment_arn=arn:aws:bedrock:us-east-1:123:exp/e1,"
            "aws.agentcore.gateway.routing_experiment_variant_name=canary"
        )
        response = client.post("/invocations", json={}, headers={"baggage": baggage})

        assert response.status_code == 200
        assert captured["arn"] == "arn:aws:bedrock:us-east-1:123:exp/e1"
        assert captured["variant"] == "canary"

    def test_no_baggage_clears_experiment_context(self):
        """Request without baggage sets both experiment ContextVars to None."""
        from starlette.testclient import TestClient

        from bedrock_agentcore.runtime import BedrockAgentCoreApp
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        app = BedrockAgentCoreApp()
        captured = {}

        @app.entrypoint
        def handler(payload):
            captured["arn"] = BedrockAgentCoreContext.get_routing_experiment_arn()
            captured["variant"] = BedrockAgentCoreContext.get_routing_experiment_variant()
            return {"ok": True}

        client = TestClient(app)
        response = client.post("/invocations", json={})

        assert response.status_code == 200
        assert captured["arn"] is None
        assert captured["variant"] is None

    def test_extract_baggage_error_clears_experiment_context(self):
        """When _extract_baggage raises, all_baggage defaults to {} and ContextVars are set to None."""
        from starlette.testclient import TestClient

        from bedrock_agentcore.runtime import BedrockAgentCoreApp
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        app = BedrockAgentCoreApp()
        captured = {}

        @app.entrypoint
        def handler(payload):
            captured["arn"] = BedrockAgentCoreContext.get_routing_experiment_arn()
            captured["variant"] = BedrockAgentCoreContext.get_routing_experiment_variant()
            return {"ok": True}

        with patch(
            "bedrock_agentcore.runtime.app._extract_baggage",
            side_effect=ValueError("malformed"),
        ):
            client = TestClient(app)
            response = client.post("/invocations", json={}, headers={"baggage": "bad=data"})

        assert response.status_code == 200
        assert captured["arn"] is None
        assert captured["variant"] is None


# ---------------------------------------------------------------------------
# A2A: BedrockCallContextBuilder experiment baggage extraction
# ---------------------------------------------------------------------------


class TestA2ACallContextBuilderBaggage:
    """BedrockCallContextBuilder.build() sets experiment ContextVars from W3C baggage."""

    def setup_method(self):
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        BedrockAgentCoreContext.set_routing_experiment(None, None)

    def _make_request(self, headers: dict):
        mock_request = MagicMock()
        mock_request.headers = headers
        return mock_request

    def test_baggage_processor_registered_on_provider(self):
        """BaggageSpanProcessor is registered on the TracerProvider when BedrockCallContextBuilder is created."""
        import bedrock_agentcore.runtime.tracing as tracing_mod

        tracing_mod._registered_on = None
        mock_provider = MagicMock()

        with patch("opentelemetry.trace.get_tracer_provider", return_value=mock_provider):
            from bedrock_agentcore.runtime.a2a import BedrockCallContextBuilder

            BedrockCallContextBuilder()

        mock_provider.add_span_processor.assert_called_once()
        assert isinstance(mock_provider.add_span_processor.call_args[0][0], BaggageSpanProcessor)

    def test_baggage_sets_experiment_context(self):
        """Baggage header → experiment ContextVars are set by the time build() returns."""
        from bedrock_agentcore.runtime.a2a import BedrockCallContextBuilder
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
        from bedrock_agentcore.runtime.models import REQUEST_ID_HEADER, SESSION_HEADER

        headers = {
            "baggage": (
                "aws.agentcore.gateway.routing_experiment_arn=arn:aws:bedrock:us-east-1:123:exp/e1,"
                "aws.agentcore.gateway.routing_experiment_variant_name=blue"
            ),
            REQUEST_ID_HEADER: "req-123",
            SESSION_HEADER: "sess-456",
        }
        builder = BedrockCallContextBuilder()
        ctx = builder.build(self._make_request(headers))

        assert BedrockAgentCoreContext.get_routing_experiment_arn() == "arn:aws:bedrock:us-east-1:123:exp/e1"
        assert BedrockAgentCoreContext.get_routing_experiment_variant() == "blue"
        # A2A protocol contract: request_id and session_id in ServerCallContext.state
        assert ctx.state["request_id"] == "req-123"
        assert ctx.state["session_id"] == "sess-456"

    def test_no_baggage_clears_experiment_context(self):
        """Request without baggage sets both experiment ContextVars to None."""
        from bedrock_agentcore.runtime.a2a import BedrockCallContextBuilder
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        builder = BedrockCallContextBuilder()
        builder.build(self._make_request({}))

        assert BedrockAgentCoreContext.get_routing_experiment_arn() is None
        assert BedrockAgentCoreContext.get_routing_experiment_variant() is None

    def test_extract_baggage_error_clears_experiment_context(self):
        """When _extract_baggage raises, all_baggage defaults to {} and ContextVars are set to None."""
        from bedrock_agentcore.runtime.a2a import BedrockCallContextBuilder
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        builder = BedrockCallContextBuilder()
        with patch(
            "bedrock_agentcore.runtime.a2a._extract_baggage",
            side_effect=ValueError("malformed"),
        ):
            builder.build(self._make_request({"baggage": "bad=data"}))

        assert BedrockAgentCoreContext.get_routing_experiment_arn() is None
        assert BedrockAgentCoreContext.get_routing_experiment_variant() is None


# ---------------------------------------------------------------------------
# AG-UI: AGUIApp._build_request_context experiment baggage extraction
# ---------------------------------------------------------------------------


class TestAGUIBaggageExtraction:
    """AGUIApp._build_request_context() sets experiment ContextVars from W3C baggage."""

    def setup_method(self):
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        BedrockAgentCoreContext.set_routing_experiment(None, None)

    def _make_request(self, headers: dict):
        mock_request = MagicMock()
        mock_request.headers = headers
        return mock_request

    def test_baggage_processor_registered_on_provider(self):
        """BaggageSpanProcessor is registered on the TracerProvider when AGUIApp is created."""
        pytest.importorskip("ag_ui")

        import bedrock_agentcore.runtime.tracing as tracing_mod

        tracing_mod._registered_on = None
        mock_provider = MagicMock()

        with patch("opentelemetry.trace.get_tracer_provider", return_value=mock_provider):
            from bedrock_agentcore.runtime.ag_ui import AGUIApp

            AGUIApp()

        mock_provider.add_span_processor.assert_called_once()
        assert isinstance(mock_provider.add_span_processor.call_args[0][0], BaggageSpanProcessor)

    def test_baggage_sets_experiment_context(self):
        """Baggage header → experiment ContextVars are set by the time _build_request_context() returns."""
        pytest.importorskip("ag_ui")

        from bedrock_agentcore.runtime.ag_ui import AGUIApp
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        app = AGUIApp()
        headers = {
            "baggage": (
                "aws.agentcore.gateway.routing_experiment_arn=arn:aws:bedrock:us-east-1:123:exp/e2,"
                "aws.agentcore.gateway.routing_experiment_variant_name=green"
            )
        }
        app._build_request_context(self._make_request(headers))

        assert BedrockAgentCoreContext.get_routing_experiment_arn() == "arn:aws:bedrock:us-east-1:123:exp/e2"
        assert BedrockAgentCoreContext.get_routing_experiment_variant() == "green"

    def test_no_baggage_clears_experiment_context(self):
        """Request without baggage sets both experiment ContextVars to None."""
        pytest.importorskip("ag_ui")

        from bedrock_agentcore.runtime.ag_ui import AGUIApp
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        app = AGUIApp()
        app._build_request_context(self._make_request({}))

        assert BedrockAgentCoreContext.get_routing_experiment_arn() is None
        assert BedrockAgentCoreContext.get_routing_experiment_variant() is None

    def test_extract_baggage_error_clears_experiment_context(self):
        """When _extract_baggage raises, all_baggage defaults to {} and ContextVars are set to None."""
        pytest.importorskip("ag_ui")

        from bedrock_agentcore.runtime.ag_ui import AGUIApp
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        app = AGUIApp()
        with patch(
            "bedrock_agentcore.runtime.ag_ui._extract_baggage",
            side_effect=ValueError("malformed"),
        ):
            app._build_request_context(self._make_request({"baggage": "bad=data"}))

        assert BedrockAgentCoreContext.get_routing_experiment_arn() is None
        assert BedrockAgentCoreContext.get_routing_experiment_variant() is None
