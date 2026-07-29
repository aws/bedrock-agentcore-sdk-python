"""Span mapping orchestration — uses strands-evals mappers with auto-detection."""

import logging
import warnings
from typing import Any, Dict, List, Optional

from strands_evals.mappers import detect_otel_mapper
from strands_evals.types.trace import AgentInvocationSpan, Session, ToolExecutionSpan

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
    FieldExtractionError,
    SpanMapResult,
)

logger = logging.getLogger(__name__)


def _extract_session_id(session_spans: List[Dict[str, Any]]) -> str:
    """Extract session ID from span attributes."""
    for span in session_spans:
        attrs = span.get("attributes", {})
        if isinstance(attrs, dict):
            session_id = attrs.get("session.id")
            if session_id:
                return session_id
    return "default"


def _detect_mapper(session_spans: List[Dict[str, Any]]):
    """Detect the appropriate mapper using strands-evals auto-detection.

    When the service sends normalized dict spans (InMemory format without body),
    StrandsInMemorySessionMapper expects ReadableSpan objects and fails on dicts.
    In this case, fall back to CloudWatchSessionMapper which handles dicts.
    """
    from strands_evals.mappers import CloudWatchSessionMapper, StrandsInMemorySessionMapper

    mapper = detect_otel_mapper(session_spans)
    if isinstance(mapper, StrandsInMemorySessionMapper) and session_spans and isinstance(session_spans[0], dict):
        return CloudWatchSessionMapper()
    return mapper


def map_spans(
    session_spans: List[Dict[str, Any]],
    reference_inputs: Optional[List[Any]] = None,
    target_trace_id: Optional[str] = None,
) -> SpanMapResult:
    """Map session spans to evaluation fields using strands-evals mappers.

    Auto-detects the span format (Strands, OpenInference, OpenTelemetry LangChain)
    and delegates to the appropriate strands-evals mapper, then bridges the result
    to SpanMapResult for adapter consumption.

    Args:
        session_spans: Raw ADOT span dicts from the evaluation service.
        reference_inputs: Optional ReferenceInput list for expected_output/tools/assertions.
        target_trace_id: Optional trace ID to match reference_inputs by spanContext.

    Returns:
        SpanMapResult with extracted fields.

    Raises:
        ValueError: If no mapper can extract data from the spans.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="strands_evals")
        mapper = _detect_mapper(session_spans)

    session_id = _extract_session_id(session_spans)

    try:
        session = mapper.map_to_session(session_spans, session_id=session_id)
    except Exception as e:
        raise FieldExtractionError(
            f"Could not extract evaluation fields from spans using {type(mapper).__name__}: {e}. "
            f"Provide a custom_mapper for custom or unsupported span formats."
        ) from e

    try:
        result = _session_to_span_map_result(session)
    except FieldExtractionError:
        # Mapper couldn't find AgentInvocationSpan — try service format fallback
        result = None

    # Fallback: extract from service-normalized format (gen_ai events)
    if result is None or not result.input or not result.actual_output:
        service_result = _extract_from_service_format(session_spans)
        if service_result:
            result = service_result
        elif result is None:
            raise FieldExtractionError(
                "No AgentInvocationSpan found in session and service format extraction failed. "
                "Provide a custom_mapper for custom or unsupported span formats."
            )

    if reference_inputs:
        # Combine all relevant reference_inputs (session-level + trace-level).
        # A reference is relevant if it has no traceId (session-scoped) or its
        # traceId matches the target trace.
        for ref in reference_inputs:
            ctx = getattr(ref, "context", None)
            if isinstance(ctx, dict):
                span_ctx = ctx.get("spanContext", {})
                ref_trace = span_ctx.get("traceId") if isinstance(span_ctx, dict) else None
            else:
                ref_trace = None
            if ref_trace and target_trace_id and ref_trace != target_trace_id:
                continue

            expected = getattr(ref, "expected_response_text", None)
            if expected and not result.expected_output:
                result.expected_output = expected
            trajectory = getattr(ref, "expected_trajectory", None)
            if isinstance(trajectory, dict) and not result.expected_tools:
                tool_names = trajectory.get("toolNames")
                if isinstance(tool_names, list) and tool_names:
                    result.expected_tools = [{"name": name} for name in tool_names if isinstance(name, str)]
            assertions = getattr(ref, "assertions", None)
            if isinstance(assertions, list) and assertions and not result.assertions:
                assertion_texts = [a.get("text") for a in assertions if isinstance(a, dict) and a.get("text")]
                if assertion_texts:
                    result.assertions = assertion_texts

    return result


def _extract_from_service_format(session_spans: List[Dict[str, Any]]) -> Optional[SpanMapResult]:
    """Extract fields from service-normalized span format.

    The AgentCore evaluation service sends spans with gen_ai semantic convention
    events (gen_ai.user.message, gen_ai.choice) instead of body with input/output.
    This handles that format as a fallback when strands-evals mappers can't parse it.
    """
    import json as _json

    for span in session_spans:
        scope = span.get("scope", {}).get("name", "")
        events = span.get("events", [])
        if "strands" not in scope or not events:
            continue

        user_input = None
        assistant_output = None
        system_prompt = None

        for event in events:
            event_name = event.get("name", "")
            attrs = event.get("attributes", {})
            content = attrs.get("content", "")

            if event_name == "gen_ai.user.message" and content:
                try:
                    parts = _json.loads(content)
                    user_input = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
                except (ValueError, TypeError):
                    user_input = content
            elif event_name == "gen_ai.choice" and attrs.get("message"):
                try:
                    parts = _json.loads(attrs["message"])
                    assistant_output = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
                except (ValueError, TypeError):
                    assistant_output = attrs["message"]
            elif event_name == "gen_ai.system.message" and content:
                try:
                    parts = _json.loads(content)
                    system_prompt = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
                except (ValueError, TypeError):
                    system_prompt = content

        if user_input and assistant_output:
            return SpanMapResult(
                input=user_input,
                actual_output=assistant_output,
                system_prompt=system_prompt,
            )

    return None


def _session_to_span_map_result(session: Session) -> SpanMapResult:
    """Bridge strands-evals Session to SpanMapResult.

    Extracts the last AgentInvocationSpan for input/output (single-turn),
    all ToolExecutionSpans for retrieval_context and tools_called, and
    all AgentInvocationSpans as turns (multi-turn / session-level).
    """
    agent_span = None
    agent_spans: List[AgentInvocationSpan] = []
    tool_spans: List[ToolExecutionSpan] = []

    for trace in session.traces:
        for span in trace.spans:
            if isinstance(span, AgentInvocationSpan):
                agent_span = span
                agent_spans.append(span)
            elif isinstance(span, ToolExecutionSpan):
                tool_spans.append(span)

    if agent_span is None:
        raise FieldExtractionError(
            "No AgentInvocationSpan found in session. Provide a custom_mapper for custom or unsupported span formats."
        )

    retrieval_context = [ts.tool_result.content for ts in tool_spans if ts.tool_result and ts.tool_result.content]
    tools_called = [
        {
            "name": ts.tool_call.name,
            "input_parameters": ts.tool_call.arguments if ts.tool_call.arguments else None,
            "output": ts.tool_result.content if ts.tool_result else None,
        }
        for ts in tool_spans
        if ts.tool_call and ts.tool_call.name
    ]

    # Build turns for multi-turn / session-level evaluation
    turns = []
    for span in agent_spans:
        turns.append({"role": "user", "content": span.user_prompt})
        turns.append({"role": "assistant", "content": span.agent_response})

    return SpanMapResult(
        input=agent_span.user_prompt,
        actual_output=agent_span.agent_response,
        retrieval_context=retrieval_context if retrieval_context else None,
        context=None,
        system_prompt=agent_span.system_prompt,
        tools_called=tools_called if tools_called else None,
        turns=turns if len(turns) > 2 else None,
    )
