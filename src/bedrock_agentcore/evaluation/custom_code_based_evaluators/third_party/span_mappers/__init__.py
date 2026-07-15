"""Span mappers for extracting evaluation fields from Agent SDK trace formats.

Uses strands-evals mappers for span format auto-detection and extraction,
bridged to SpanMapResult for adapter consumption.
"""

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
    SpanMapResult,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.registry import (
    map_spans,
)

__all__ = [
    "SpanMapResult",
    "map_spans",
]
