"""Strands Agent SDK span mapper."""

import logging
from typing import Any, Dict, List, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
    SpanMapResult,
    _get_message_content,
    _parse_span_event_body,
    _try_parse_text_blocks,
)

logger = logging.getLogger(__name__)

SCOPE_STRANDS = "strands.telemetry.tracer"


class StrandsSpanMapper:
    """Maps Strands agent spans to evaluation fields.

    Extracts only what metrics consume: input, actual_output, retrieval_context,
    system_prompt. Supports two trace formats:
    - Inline events (unified ADOT): data in span.events[]
    - Span body (CloudWatch ADOT): data in span.span_events[].body
    """

    def map(self, session_spans: List[Dict[str, Any]]) -> Optional[SpanMapResult]:
        """Map session spans to evaluation fields.

        Args:
            session_spans: Raw ADOT span dicts (already filtered by evaluationTarget).

        Returns:
            SpanMapResult with extracted fields, or None if no Strands agent span found.
        """
        agent_span = self._find_invoke_agent_span(session_spans)
        if agent_span is None:
            return None

        if self._has_inline_events(agent_span):
            return self._extract_from_inline_events(agent_span)

        return self._extract_from_span_body(agent_span)

    def _find_invoke_agent_span(self, session_spans: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the invoke_agent span with strands.telemetry.tracer scope."""
        for span in session_spans:
            scope = span.get("scope", {})
            if not isinstance(scope, dict):
                continue
            if scope.get("name") != SCOPE_STRANDS:
                continue
            attributes = span.get("attributes", {})
            if attributes.get("gen_ai.operation.name") == "invoke_agent":
                return span
        return None

    def _has_inline_events(self, span: Dict[str, Any]) -> bool:
        """Check if span uses inline events format (unified ADOT)."""
        events = span.get("events", [])
        return any(
            e.get("name") in ("gen_ai.user.message", "gen_ai.choice")
            for e in events
        )

    def _extract_from_inline_events(self, agent_span: Dict[str, Any]) -> Optional[SpanMapResult]:
        """Extract fields from inline events[] (unified ADOT format).

        - input: first gen_ai.user.message → attributes.content (JSON text blocks)
        - actual_output: last gen_ai.choice → attributes.message (plain string)
        - system_prompt: from span attributes
        """
        events = agent_span.get("events", [])

        input_text = None
        actual_output = None

        for event in events:
            name = event.get("name")
            attrs = event.get("attributes", {})

            if name == "gen_ai.user.message" and input_text is None:
                content_str = attrs.get("content", "")
                input_text = _try_parse_text_blocks(content_str) or content_str

            elif name == "gen_ai.choice":
                actual_output = attrs.get("message", "")

        if not input_text and not actual_output:
            return None

        system_prompt = agent_span.get("attributes", {}).get("system_prompt")

        return SpanMapResult(
            input=input_text,
            actual_output=actual_output,
            system_prompt=system_prompt,
        )

    def _extract_from_span_body(self, agent_span: Dict[str, Any]) -> Optional[SpanMapResult]:
        """Extract fields from span_events[].body (CloudWatch ADOT format).

        Multi-turn: one span_event per turn.
        - input: first user message across all turns
        - actual_output: last assistant message across all turns
        - retrieval_context: all tool outputs
        - system_prompt: from system-role message or span attributes
        """
        input_text = None
        actual_output = None
        tool_outputs: List[str] = []
        system_prompt: Optional[str] = None

        for event in agent_span.get("span_events", []):
            body = _parse_span_event_body(event.get("body"))
            if not body:
                continue

            input_data = body.get("input", {})
            if isinstance(input_data, dict):
                for msg in input_data.get("messages", []):
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "")
                    if role == "system" and system_prompt is None:
                        content = _get_message_content(msg)
                        if content:
                            system_prompt = content
                    elif role == "user" and input_text is None:
                        content = _get_message_content(msg)
                        if content:
                            input_text = content

            output_data = body.get("output", {})
            if isinstance(output_data, dict):
                for msg in output_data.get("messages", []):
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "")
                    content = _get_message_content(msg)
                    if role == "assistant" and content:
                        actual_output = content
                    elif role == "tool" and content:
                        tool_outputs.append(content)

        if not input_text and not actual_output:
            return None

        if system_prompt is None:
            system_prompt = agent_span.get("attributes", {}).get("system_prompt")

        return SpanMapResult(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=tool_outputs if tool_outputs else None,
            context=tool_outputs if tool_outputs else None,
            system_prompt=system_prompt,
        )
