"""Span parsers for extracting evaluation fields from Agent SDK trace formats."""

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_parsers.base import (
    SpanParseResult,
    parse_spans,
)

__all__ = ["SpanParseResult", "parse_spans"]
