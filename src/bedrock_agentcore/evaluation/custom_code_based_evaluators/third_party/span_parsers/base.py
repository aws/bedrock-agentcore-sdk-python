"""Base span parsing logic and orchestration across format-specific parsers."""

import logging
from typing import Any, Dict, List, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_parsers.common import (
    SpanParseResult,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_parsers.strands import (
    parse_strands_spans,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_parsers.otel_langchain import (
    parse_otel_langchain_spans,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_parsers.openinference import (
    parse_openinference_spans,
)

logger = logging.getLogger(__name__)


_PARSERS = [
    parse_strands_spans,
    parse_otel_langchain_spans,
    parse_openinference_spans,
]


def parse_spans(
    session_spans: List[Dict[str, Any]],
    reference_inputs: Optional[List[Any]] = None,
) -> SpanParseResult:
    """Parse session spans using the first matching agent-level parser.

    Iterates through format-specific parsers (Strands, OTel LangChain,
    OpenInference) and returns the result from the first one that
    successfully extracts data.

    Args:
        session_spans: Raw ADOT span dicts from the evaluation service.
        reference_inputs: Optional ReferenceInput list for expected_output.

    Returns:
        SpanParseResult with extracted fields.

    Raises:
        ValueError: If no parser can extract data from the spans.
    """
    for parser in _PARSERS:
        result = parser(session_spans)
        if result is not None:
            if reference_inputs:
                ref = reference_inputs[0]
                expected = getattr(ref, "expected_response_text", None)
                if expected:
                    result.expected_output = expected
            return result

    raise ValueError(
        "Could not extract evaluation fields from spans. "
        "No agent-level span with gen_ai.operation.name=='invoke_agent' and "
        "valid span_events found. Provide a field_mapper for custom formats."
    )
