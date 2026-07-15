"""Span mapping orchestration — dispatches by OTel scope name via registry."""

import logging
from typing import Any, Dict, List, Optional, Set

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.base import (
    BaseSpanMapper,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
    SpanMapResult,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.langgraph.openinference_instrumentation_langchain_mapper import (
    OpenInferenceInstrumentationLangchainMapper,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.langgraph.opentelemetry_instrumentation_langchain_mapper import (
    OpenTelemetryInstrumentationLangchainMapper,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.strands import (
    StrandsTelemetryTracerMapper,
)

logger = logging.getLogger(__name__)

_REGISTRY: List[BaseSpanMapper] = [
    StrandsTelemetryTracerMapper(),
    OpenInferenceInstrumentationLangchainMapper(),
    OpenTelemetryInstrumentationLangchainMapper(),
]


def _detect_scope_names(session_spans: List[Dict[str, Any]]) -> Set[str]:
    """Collect unique scope names from all spans."""
    names: Set[str] = set()
    for span in session_spans:
        scope = span.get("scope", {})
        if isinstance(scope, dict) and scope.get("name"):
            names.add(scope["name"])
    return names


def map_spans(
    session_spans: List[Dict[str, Any]],
    reference_inputs: Optional[List[Any]] = None,
) -> SpanMapResult:
    """Map session spans to evaluation fields.

    Dispatches to the first registered mapper whose scope_name is found in the spans.

    Args:
        session_spans: Raw ADOT span dicts from the evaluation service.
        reference_inputs: Optional ReferenceInput list for expected_output.

    Returns:
        SpanMapResult with extracted fields.

    Raises:
        ValueError: If no mapper can extract data from the spans.
    """
    scope_names = _detect_scope_names(session_spans)

    for mapper in _REGISTRY:
        if scope_names & set(mapper.scope_names):
            result = mapper.map(session_spans)
            if result is not None:
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

    detected = ", ".join(sorted(scope_names)) if scope_names else "none"
    supported = ", ".join(f"'{n}'" for m in _REGISTRY for n in m.scope_names)
    raise ValueError(
        f"Could not extract evaluation fields from spans. "
        f"Detected scope names: [{detected}]. "
        f"Supported: {supported}. "
        f"Provide a custom_mapper for custom or unsupported span formats."
    )
