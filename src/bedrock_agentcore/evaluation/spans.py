"""Helpers for identifying tool spans and extracting evaluation target IDs."""

from typing import Any, Dict, List, Optional


def is_tool_span(span: Dict[str, Any]) -> bool:
    """Return whether a span represents a tool execution.

    Recognizes OTel GenAI (``gen_ai.operation.name=execute_tool``),
    OpenInference (``openinference.span.kind=TOOL``), and Traceloop
    (``traceloop.span.kind=tool``) attributes. Missing or non-dict
    attributes are treated as non-tool spans.

    Args:
        span: A span dict containing an optional attributes dict.
    """
    attrs = span.get("attributes", {})
    if not isinstance(attrs, dict):
        return False
    return (
        attrs.get("gen_ai.operation.name") == "execute_tool"
        or attrs.get("openinference.span.kind") == "TOOL"
        or attrs.get("traceloop.span.kind") == "tool"
    )


def tool_span_ids(spans: List[Dict[str, Any]], trace_id: Optional[str] = None) -> List[str]:
    """Return tool span IDs in input order, preserving duplicates.

    Args:
        spans: Span dicts with attributes, spanId, and optionally traceId.
            Spans with missing or empty spanId values are skipped.
        trace_id: Restrict results to this trace. None or an empty string
            includes tool spans from all traces.
    """
    return [
        span["spanId"]
        for span in spans
        if is_tool_span(span) and span.get("spanId") and (not trace_id or span.get("traceId") == trace_id)
    ]


def trace_ids(spans: List[Dict[str, Any]]) -> List[str]:
    """Return unique trace IDs ordered by first appearance.

    Args:
        spans: Span dicts containing traceId values. Missing or empty
            traceId values are skipped.
    """
    return list(dict.fromkeys(span.get("traceId") for span in spans if span.get("traceId")))
