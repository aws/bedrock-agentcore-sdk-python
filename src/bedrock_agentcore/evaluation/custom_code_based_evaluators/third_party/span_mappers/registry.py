"""Span mapping orchestration — uses strands-evals mappers with auto-detection."""

import logging
import warnings
from typing import Any, Dict, List, Optional

from strands_evals.mappers import (
    LangChainOtelSessionMapper,
    OpenInferenceSessionMapper,
    detect_otel_mapper,
)
from strands_evals.mappers.utils import get_scope_name
from strands_evals.types.trace import AgentInvocationSpan, Session, ToolExecutionSpan

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
    FieldExtractionError,
    SpanMapResult,
)

logger = logging.getLogger(__name__)

# Amazon ADOT distro scope not yet recognized by strands-evals detect_otel_mapper.
# Uses the same span format as opentelemetry.instrumentation.langchain — LangChainOtelSessionMapper
# handles both. Workaround until strands-evals adds native support.
SCOPE_AMAZON_OTEL_LANGCHAIN = "amazon.opentelemetry.distro.instrumentation.langchain"


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
    """Detect the appropriate mapper, extending strands-evals for edge cases.

    We check scope names before falling back to detect_otel_mapper because:
    - detect_otel_mapper doesn't recognize amazon.opentelemetry.distro scope
    - detect_otel_mapper misidentifies CloudWatch split format (body on separate
      entries) as StrandsInMemorySessionMapper (which expects ReadableSpan objects)
    - Our pre-check handles both cases correctly, then falls back for other formats
    """
    from strands_evals.mappers import CloudWatchSessionMapper
    from strands_evals.mappers.utils import get_body

    has_strands_scope = False
    has_body_entry = False

    for span in session_spans:
        scope_name = get_scope_name(span)
        if scope_name in (SCOPE_AMAZON_OTEL_LANGCHAIN, "opentelemetry.instrumentation.langchain"):
            return LangChainOtelSessionMapper()
        if scope_name == "openinference.instrumentation.langchain":
            return OpenInferenceSessionMapper()
        if scope_name == "strands.telemetry.tracer":
            has_strands_scope = True
        if get_body(span) is not None:
            has_body_entry = True

    # CloudWatch split format: Strands scope on metadata entries, body on log entries
    if has_strands_scope and has_body_entry:
        return CloudWatchSessionMapper()

    # Fallback to strands-evals auto-detection
    return detect_otel_mapper(session_spans)


def map_spans(
    session_spans: List[Dict[str, Any]],
    reference_inputs: Optional[List[Any]] = None,
) -> SpanMapResult:
    """Map session spans to evaluation fields using strands-evals mappers.

    Auto-detects the span format (Strands, OpenInference, OpenTelemetry LangChain)
    and delegates to the appropriate strands-evals mapper, then bridges the result
    to SpanMapResult for adapter consumption.

    Args:
        session_spans: Raw ADOT span dicts from the evaluation service.
        reference_inputs: Optional ReferenceInput list for expected_output/tools/assertions.

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

    result = _session_to_span_map_result(session)

    if reference_inputs:
        ref = reference_inputs[0]
        expected = getattr(ref, "expected_response_text", None)
        if expected:
            result.expected_output = expected
        trajectory = getattr(ref, "expected_trajectory", None)
        if isinstance(trajectory, dict):
            tool_names = trajectory.get("toolNames")
            if isinstance(tool_names, list) and tool_names:
                result.expected_tools = [
                    {"name": name} for name in tool_names if isinstance(name, str)
                ]
        assertions = getattr(ref, "assertions", None)
        if isinstance(assertions, list) and assertions:
            assertion_texts = [
                a.get("text") for a in assertions
                if isinstance(a, dict) and a.get("text")
            ]
            if assertion_texts:
                result.assertions = assertion_texts

    return result


def _session_to_span_map_result(session: Session) -> SpanMapResult:
    """Bridge strands-evals Session to SpanMapResult.

    Extracts the last AgentInvocationSpan for input/output and all
    ToolExecutionSpans for retrieval_context and tools_called.

    Note: Currently scoped to trace-level evaluation only. Only the last
    AgentInvocationSpan is used — session-level multi-turn evaluators
    (e.g. ConversationCompleteness) are not supported. Use custom_mapper
    for multi-turn evaluation needs.
    """
    agent_span = None
    tool_spans: List[ToolExecutionSpan] = []

    for trace in session.traces:
        for span in trace.spans:
            if isinstance(span, AgentInvocationSpan):
                agent_span = span
            elif isinstance(span, ToolExecutionSpan):
                tool_spans.append(span)

    if agent_span is None:
        raise FieldExtractionError(
            "No AgentInvocationSpan found in session. "
            "Provide a custom_mapper for custom or unsupported span formats."
        )

    retrieval_context = [
        ts.tool_result.content for ts in tool_spans
        if ts.tool_result and ts.tool_result.content
    ]
    tools_called = [
        {
            "name": ts.tool_call.name,
            "input_parameters": ts.tool_call.arguments if ts.tool_call.arguments else None,
            "output": ts.tool_result.content if ts.tool_result else None,
        }
        for ts in tool_spans
        if ts.tool_call and ts.tool_call.name
    ]

    return SpanMapResult(
        input=agent_span.user_prompt,
        actual_output=agent_span.agent_response,
        retrieval_context=retrieval_context if retrieval_context else None,
        context=retrieval_context if retrieval_context else None,
        system_prompt=agent_span.system_prompt,
        tools_called=tools_called if tools_called else None,
    )
