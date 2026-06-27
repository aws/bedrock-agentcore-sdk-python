"""Strands Agent SDK span parser."""

import logging
from typing import Any, Dict, List, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_parsers.common import (
    SpanParseResult,
    extract_from_agent_span_events,
)

logger = logging.getLogger(__name__)


def parse_strands_spans(session_spans: List[Dict[str, Any]]) -> Optional[SpanParseResult]:
    """Parse spans from Strands Agent SDK format.

    Looks for spans with gen_ai.operation.name == "invoke_agent" and
    extracts input/output from span_events.

    Args:
        session_spans: Raw ADOT span dicts.

    Returns:
        SpanParseResult if agent spans found, None otherwise.
    """
    return extract_from_agent_span_events(session_spans)
