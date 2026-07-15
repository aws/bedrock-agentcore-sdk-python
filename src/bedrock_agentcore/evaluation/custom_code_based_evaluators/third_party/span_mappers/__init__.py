"""Span mappers for extracting evaluation fields from Agent SDK trace formats."""

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.base import (
    BaseSpanMapper,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.registry import (
    map_spans,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
    SpanMapResult,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.langgraph import (
    OpenInferenceInstrumentationLangchainMapper,
    OpenTelemetryInstrumentationLangchainMapper,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.strands import (
    StrandsTelemetryTracerMapper,
)

__all__ = [
    "BaseSpanMapper",
    "OpenInferenceInstrumentationLangchainMapper",
    "OpenTelemetryInstrumentationLangchainMapper",
    "SpanMapResult",
    "StrandsTelemetryTracerMapper",
    "map_spans",
]
