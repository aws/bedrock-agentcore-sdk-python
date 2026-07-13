"""Span mapping orchestration."""

import logging
from typing import Any, Dict, List, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
    SpanMapResult,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.strands import (
    StrandsSpanMapper,
)

logger = logging.getLogger(__name__)

_strands_mapper = StrandsSpanMapper()


def map_spans(
    session_spans: List[Dict[str, Any]],
    reference_inputs: Optional[List[Any]] = None,
) -> SpanMapResult:
    """Map session spans to evaluation fields.

    Currently supports Strands Agent SDK spans (scope.name == "strands.telemetry.tracer").
    Additional framework support can be added when real span data is available.

    Args:
        session_spans: Raw ADOT span dicts from the evaluation service.
        reference_inputs: Optional ReferenceInput list for expected_output.

    Returns:
        SpanMapResult with extracted fields.

    Raises:
        ValueError: If no mapper can extract data from the spans.
    """
    result = _strands_mapper.map(session_spans)
    if result is not None:
        if reference_inputs:
            ref = reference_inputs[0]
            expected = getattr(ref, "expected_response_text", None)
            if expected:
                result.expected_output = expected
        return result

    raise ValueError(
        "Could not extract evaluation fields from spans. "
        "No Strands agent span (scope.name=='strands.telemetry.tracer' with "
        "gen_ai.operation.name=='invoke_agent') found. "
        "Provide a customer_mapper for custom or unsupported span formats."
    )
