"""Agent span collector: collects OpenTelemetry spans for evaluation."""

from .agent_span_collector import AgentSpanCollector, CloudWatchAgentSpanCollector

__all__ = [
    "AgentSpanCollector",
    "CloudWatchAgentSpanCollector",
]
