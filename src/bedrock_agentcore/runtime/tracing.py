"""OpenTelemetry span processor for Bedrock AgentCore routing experiment attributes.

Provides ``BaggageSpanProcessor``, which stamps every span with the routing
experiment ARN and variant name extracted from the request's W3C baggage header.
Values are read from two sources in priority order:

1. ``BedrockAgentCoreContext`` ContextVars — set by ``_build_request_context``
   after the baggage header is parsed.  Covers agent/tool spans created during
   handler execution.
2. OTel baggage in the span's ``parent_context`` — covers spans started by ASGI
   instrumentation *before* ``_build_request_context`` runs (e.g. the root
   ``POST /invocations`` server span), where the propagator has already
   extracted the W3C baggage into the OTel context.

Auto-registration
-----------------
``_ensure_baggage_processor_registered()`` is called by the SDK on every
request.  It registers ``BaggageSpanProcessor`` on the active
``TracerProvider`` the first time it is called, and re-registers whenever
``get_tracer_provider()`` returns a different provider instance than the one
last seen — which handles the case where a framework replaces the global
provider after the app is constructed.
"""

import logging
import threading
from typing import Optional

from .context import BedrockAgentCoreContext as _context

logger = logging.getLogger(__name__)

# Module-level state for provider-tracking auto-registration.
_registration_lock = threading.Lock()
_registered_on: Optional[object] = None  # the TracerProvider instance we last registered on


def _ensure_baggage_processor_registered() -> None:
    """Register ``BaggageSpanProcessor`` on the current ``TracerProvider`` if needed.

    No-ops when ``opentelemetry-api`` is not installed.
    Re-registers automatically when the global provider has been replaced since
    the last call (e.g. by a framework that calls set_tracer_provider at startup).
    """
    global _registered_on
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if provider is _registered_on:
            return
        with _registration_lock:
            # Re-check inside the lock — another thread may have registered first.
            provider = trace.get_tracer_provider()
            if provider is _registered_on:
                return
            provider.add_span_processor(BaggageSpanProcessor())
            _registered_on = provider
            logger.debug("BaggageSpanProcessor registered on %s", type(provider).__name__)
    except ImportError:
        logger.debug("opentelemetry-api not installed; BaggageSpanProcessor registration skipped")
    except Exception:
        logger.debug("Could not register BaggageSpanProcessor", exc_info=True)


def _get_base_class() -> type:
    """Return the OTel SDK SpanProcessor base if available, otherwise object.

    ``opentelemetry-sdk`` is not a required dependency of this package, so the
    import may fail.  Deferring it here (rather than at module level) means the
    module loads cleanly regardless — a module-level ImportError would crash
    ``BedrockAgentCoreApp.__init__`` even for users who don't use OTel at all.

    When the SDK *is* present, inheriting from ``SpanProcessor`` is required:
    ``SynchronousMultiSpanProcessor`` calls internal hooks like ``_on_ending``
    that only exist on the SDK base class.

    When the SDK is absent, ``ProxyTracerProvider`` has no ``add_span_processor``,
    so ``_ensure_baggage_processor_registered`` no-ops before the processor is
    ever used — the ``object`` fallback is effectively dead code at runtime.
    """
    try:
        from opentelemetry.sdk.trace import SpanProcessor  # type: ignore[import]

        return SpanProcessor
    except ImportError:
        return object


class BaggageSpanProcessor(_get_base_class()):  # type: ignore[misc]
    """SpanProcessor that stamps every span with routing experiment attributes.

    .. warning::
        This feature is in preview and may change in future releases.

    Reads ``BedrockAgentCoreContext.get_routing_experiment_arn()`` and
    ``get_routing_experiment_variant()`` from ContextVars on ``on_start``,
    so each concurrent request gets its own values with no cross-talk.

    Span attributes set (when the corresponding baggage key is present):
      - ``aws.agentcore.gateway.routing_experiment_arn``
      - ``aws.agentcore.gateway.routing_experiment_variant_name``
    """

    def on_start(self, span: object, parent_context: Optional[object] = None) -> None:
        """Set routing experiment attributes on every new span.

        Primary source: ContextVars set by ``_build_request_context`` — covers
        all spans created after request parsing (agent spans, tool spans, etc.).

        Fallback: OTel baggage in ``parent_context`` — covers spans created by
        ASGI instrumentation before ``_build_request_context`` runs (e.g.
        ``POST /invocations``), where the propagator has already extracted the
        W3C baggage header into the context.
        """
        arn = _context.get_routing_experiment_arn()
        variant = _context.get_routing_experiment_variant()

        if (arn is None or variant is None) and parent_context is not None:
            try:
                from opentelemetry import baggage as otel_baggage

                if arn is None:
                    arn = otel_baggage.get_baggage("aws.agentcore.gateway.routing_experiment_arn", parent_context)
                if variant is None:
                    variant = otel_baggage.get_baggage(
                        "aws.agentcore.gateway.routing_experiment_variant_name", parent_context
                    )
            except ImportError:
                logger.debug("opentelemetry-api not installed; parent_context baggage fallback skipped")

        if arn is not None:
            span.set_attribute("aws.agentcore.gateway.routing_experiment_arn", arn)  # type: ignore[union-attr]
        if variant is not None:
            span.set_attribute("aws.agentcore.gateway.routing_experiment_variant_name", variant)  # type: ignore[union-attr]

    def on_end(self, span: object) -> None:
        """No-op."""

    def shutdown(self) -> None:
        """No-op."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op — returns True to indicate success."""
        return True
