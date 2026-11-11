"""Span mapping orchestration — uses strands-evals mappers with auto-detection."""

import json
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

    # Multi-turn reachability: when the service collapses a session into a single
    # span with multiple span_events (one per turn), the CloudWatch mapper still
    # produces a valid single-turn input/actual_output from the first/last event,
    # so the `not result.input` guard below would never fire and the multi-turn
    # turns would be lost. Detect the collapsed multi-event shape up front and run
    # the service-format extractor so `turns` is populated even when the mapper
    # already found a valid single-turn pair.
    if _has_multi_event_span(session_spans):
        service_result = _extract_from_service_format(session_spans)
        if service_result is not None and service_result.turns:
            if result is None:
                result = service_result
            else:
                # Keep the mapper's richer fields (tools, retrieval_context, etc.)
                # but adopt the multi-turn conversation extracted from span_events,
                # including its input/actual_output (the latest turn) so single-turn
                # fields stay consistent with the extracted conversation.
                result.turns = service_result.turns
                result.input = service_result.input
                result.actual_output = service_result.actual_output

    # Fallback: extract from service-normalized format (span_events / gen_ai events)
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


def _has_multi_event_span(session_spans: List[Dict[str, Any]]) -> bool:
    """Return True if any span carries more than one span_event.

    This is the signature of the service-normalized SESSION format, where the
    evaluation service collapses all ADOT spans sharing a session.id into a
    single span with one span_event per conversation turn.
    """
    for span in session_spans:
        if isinstance(span, dict) and len(span.get("span_events", []) or []) > 1:
            return True
    return False


def _extract_message_text(messages: List[Dict[str, Any]], role: Optional[str] = None) -> Optional[str]:
    """Extract plain text from a service-format message list.

    Decodes the double-encoded ``content`` that real Strands ``invoke_agent``
    bodies emit (a JSON string like ``'[{"text": ...}]'``) into plain text,
    mirroring the single-turn decoding behavior (PR #454) rather than returning
    the serialized JSON literally. Plain strings and already-decoded
    ``[{"text": ...}]`` lists are also handled.

    Args:
        messages: The ``body.input.messages`` or ``body.output.messages`` list.
        role: If given, prefer the last message whose ``role`` matches (latest
            user / latest assistant, in chronological order). Falls back to any
            message with text.

    Returns:
        The extracted plain text, or None if no text could be parsed.
    """

    def _text_from_list(items: List[Any]) -> Optional[str]:
        text = " ".join(c.get("text", "") for c in items if isinstance(c, dict)).strip()
        return text or None

    def _text_from_raw(raw: Any) -> Optional[str]:
        # Decode the double-encoded JSON-string content: a string that parses to a
        # list of {"text": ...} blocks. Falls back to the plain string on failure.
        if isinstance(raw, str):
            stripped = raw.strip()
            if not stripped:
                return None
            try:
                parsed = json.loads(stripped)
            except (ValueError, TypeError):
                return stripped
            if isinstance(parsed, list):
                return _text_from_list(parsed)
            return stripped
        # Already-decoded list variant: [{"text": ...}, ...]
        if isinstance(raw, list):
            return _text_from_list(raw)
        return None

    def _text(msg: Dict[str, Any]) -> Optional[str]:
        content = msg.get("content", msg.get("message"))
        # Service shape: content/message is a dict wrapping the raw value under a
        # nested "content"/"message" key (double-encoded JSON string, plain string,
        # or already-decoded list).
        if isinstance(content, dict):
            inner = content.get("content", content.get("message"))
            return _text_from_raw(inner)
        # Top-level raw value: JSON string, plain string, or list.
        return _text_from_raw(content)

    if role is not None:
        # Prefer the latest message matching the requested role (chronological).
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == role:
                text = _text(msg)
                if text:
                    return text

    # Fallback: first message that yields any text.
    for msg in messages:
        if isinstance(msg, dict):
            text = _text(msg)
            if text:
                return text
    return None


def _extract_from_service_format(session_spans: List[Dict[str, Any]]) -> Optional[SpanMapResult]:
    """Extract fields from service-normalized span format.

    Handles two service formats:
    1. SESSION format with span_events[*].body (multi-turn conversations where the
       service collapses all ADOT spans into one span with multiple span_events)
    2. gen_ai semantic convention events (single-turn Strands spans)
    """
    # --- Multi-turn: extract from span_events[*].body ---
    # Only treat a span as a collapsed multi-turn conversation when it carries
    # more than one span_event. A single span_event is a single-turn span the
    # CloudWatch mapper already handles, and hijacking it here would mislabel
    # single-turn sessions.
    for span in session_spans:
        span_events = span.get("span_events", [])
        if len(span_events) > 1:
            turns: List[Dict[str, Any]] = []
            last_input = None
            last_output = None
            for se in span_events:
                body = se.get("body", {})
                inp_msgs = (body.get("input") or {}).get("messages", [])
                out_msgs = (body.get("output") or {}).get("messages", [])
                user_text = _extract_message_text(inp_msgs, role="user") if inp_msgs else None
                asst_text = _extract_message_text(out_msgs, role="assistant") if out_msgs else None
                if user_text:
                    turns.append({"role": "user", "content": user_text})
                    last_input = user_text
                if asst_text:
                    turns.append({"role": "assistant", "content": asst_text})
                    last_output = asst_text
            if turns and last_input and last_output:
                return SpanMapResult(
                    input=last_input,
                    actual_output=last_output,
                    turns=turns if len(turns) > 2 else None,
                )

    # --- Single-turn: extract from gen_ai semantic convention events ---
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
                    parts = json.loads(content)
                    user_input = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
                except (ValueError, TypeError):
                    user_input = content
            elif event_name == "gen_ai.choice" and attrs.get("message"):
                try:
                    parts = json.loads(attrs["message"])
                    assistant_output = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
                except (ValueError, TypeError):
                    assistant_output = attrs["message"]
            elif event_name == "gen_ai.system.message" and content:
                try:
                    parts = json.loads(content)
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
