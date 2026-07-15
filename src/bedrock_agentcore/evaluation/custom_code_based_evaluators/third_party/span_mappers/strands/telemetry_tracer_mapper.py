"""Strands telemetry tracer span mapper.

Handles spans produced by the ``strands.telemetry.tracer`` instrumentation scope.
"""

import logging
from typing import Any, Dict, List, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.base import (
    BaseSpanMapper,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
    SpanMapResult,
    _get_message_content,
    _parse_span_event_body,
    _try_parse_json,
    _try_parse_text_blocks,
)

logger = logging.getLogger(__name__)

SCOPE_STRANDS = "strands.telemetry.tracer"


class StrandsTelemetryTracerMapper(BaseSpanMapper):
    """Maps Strands agent spans to evaluation fields.

    Extracts only what metrics consume: input, actual_output, retrieval_context,
    system_prompt. Supports two trace formats:
    - Inline events (unified ADOT): data in span.events[]
    - Span body (CloudWatch ADOT): data in span.span_events[].body
    """

    @property
    def scope_name(self) -> str:
        return SCOPE_STRANDS

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
            return self._extract_from_inline_events(agent_span, session_spans)

        return self._extract_from_span_body(agent_span, session_spans)

    def _find_invoke_agent_span(self, session_spans: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the last invoke_agent span with strands.telemetry.tracer scope.

        In multi-turn traces there may be multiple invoke_agent spans; the last one
        (latest end_time) is the most complete — matching server-side OtelSpanMapper behavior.
        """
        found = None
        for span in session_spans:
            scope = span.get("scope", {})
            if not isinstance(scope, dict):
                continue
            if scope.get("name") != SCOPE_STRANDS:
                continue
            attributes = span.get("attributes", {})
            if attributes.get("gen_ai.operation.name") == "invoke_agent":
                found = span
        return found

    def _has_inline_events(self, span: Dict[str, Any]) -> bool:
        """Check if span uses inline events format (unified ADOT)."""
        events = span.get("events", [])
        return any(
            e.get("name") in ("gen_ai.user.message", "gen_ai.choice")
            for e in events
        )

    def _extract_from_inline_events(
        self, agent_span: Dict[str, Any], session_spans: List[Dict[str, Any]]
    ) -> Optional[SpanMapResult]:
        """Extract fields from inline events[] (unified ADOT format).

        - input: first gen_ai.user.message → attributes.content (JSON text blocks)
        - actual_output: last gen_ai.choice → attributes.message (parsed text blocks)
        - retrieval_context: tool outputs from child execute_tool spans
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
                message_str = attrs.get("message", "")
                actual_output = _try_parse_text_blocks(message_str) or message_str

        if not input_text and not actual_output:
            return None

        system_prompt = agent_span.get("attributes", {}).get("system_prompt")
        tool_outputs, tools_called = self._extract_tool_data_from_sibling_spans(agent_span, session_spans)

        return SpanMapResult(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=tool_outputs if tool_outputs else None,
            context=tool_outputs if tool_outputs else None,
            system_prompt=system_prompt,
            tools_called=tools_called if tools_called else None,
        )

    def _extract_tool_data_from_sibling_spans(
        self, agent_span: Dict[str, Any], session_spans: List[Dict[str, Any]]
    ) -> tuple:
        """Extract tool outputs and tool calls from execute_tool spans in the same trace.

        Checks two relationships:
        1. Direct children: parentSpanId == agent span's spanId (in-memory format)
        2. Same trace: any execute_tool span with matching traceId (CloudWatch ADOT format,
           where execute_tool spans may be grandchildren via intermediate chat spans)

        Returns:
            Tuple of (tool_outputs: List[str], tools_called: List[Dict[str, Any]])
        """
        agent_span_id = agent_span.get("spanId")
        agent_trace_id = agent_span.get("traceId")
        if not agent_span_id:
            return [], []

        tool_outputs: List[str] = []
        tools_called: List[Dict[str, Any]] = []
        for span in session_spans:
            if span is agent_span:
                continue
            attrs = span.get("attributes", {})
            if attrs.get("gen_ai.operation.name") != "execute_tool":
                continue
            # Match by direct parent OR same trace
            is_child = span.get("parentSpanId") == agent_span_id
            is_same_trace = agent_trace_id and span.get("traceId") == agent_trace_id
            if not is_child and not is_same_trace:
                continue

            output = self._extract_tool_output_from_span(span)
            if output:
                tool_outputs.append(output)

            tool_call = self._extract_tool_call_from_span(span)
            if tool_call:
                tools_called.append(tool_call)

        return tool_outputs, tools_called

    def _extract_tool_call_from_span(self, tool_span: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract tool name, input parameters, and output from an execute_tool span.

        Returns a dict with keys: name, input_parameters, output.
        """
        attrs = tool_span.get("attributes", {})
        tool_name = attrs.get("gen_ai.tool.name")
        if not tool_name:
            return None

        input_parameters = None
        output = None

        # Try inline events for input parameters
        for event in tool_span.get("events", []):
            if event.get("name") == "gen_ai.tool.message":
                content_str = event.get("attributes", {}).get("content", "")
                if content_str:
                    parsed = _try_parse_json(content_str)
                    if isinstance(parsed, dict):
                        input_parameters = parsed

            if event.get("name") == "gen_ai.choice":
                message_str = event.get("attributes", {}).get("message", "")
                if message_str:
                    output = _try_parse_text_blocks(message_str) or message_str

        # Span body fallback for input parameters
        if input_parameters is None:
            for event in tool_span.get("span_events", []):
                body = _parse_span_event_body(event.get("body"))
                if not body:
                    continue
                input_data = body.get("input", {})
                if isinstance(input_data, dict):
                    for msg in input_data.get("messages", []):
                        if not isinstance(msg, dict):
                            continue
                        content = msg.get("content", {})
                        if isinstance(content, dict) and "content" in content:
                            parsed = _try_parse_json(content["content"]) if isinstance(content["content"], str) else content["content"]
                            if isinstance(parsed, dict):
                                input_parameters = parsed
                                break

        # Span body fallback for output
        if output is None:
            output = self._extract_tool_output_from_span(tool_span)

        result: Dict[str, Any] = {"name": tool_name}
        if input_parameters is not None:
            result["input_parameters"] = input_parameters
        if output is not None:
            result["output"] = output
        return result

    def _extract_tool_output_from_span(self, tool_span: Dict[str, Any]) -> Optional[str]:
        """Extract text output from a single execute_tool span.

        Tries inline gen_ai.choice event first, then falls back to span_events body.
        """
        # Inline events path
        for event in tool_span.get("events", []):
            if event.get("name") == "gen_ai.choice":
                message_str = event.get("attributes", {}).get("message", "")
                if message_str:
                    return _try_parse_text_blocks(message_str) or message_str

        # Span body fallback
        for event in tool_span.get("span_events", []):
            body = _parse_span_event_body(event.get("body"))
            if not body:
                continue
            output_data = body.get("output", {})
            if isinstance(output_data, dict):
                for msg in output_data.get("messages", []):
                    if not isinstance(msg, dict):
                        continue
                    content = _get_message_content(msg)
                    if content:
                        return content

        return None

    def _extract_from_span_body(self, agent_span: Dict[str, Any], session_spans: List[Dict[str, Any]]) -> Optional[SpanMapResult]:
        """Extract fields from span_events[].body (CloudWatch ADOT format).

        Multi-turn: one span_event per turn.
        - input: first user message across all turns
        - actual_output: last assistant message across all turns
        - retrieval_context: all tool outputs (from agent body + sibling execute_tool spans)
        - tools_called: tool name + parameters from tool-role input messages or sibling spans
        - system_prompt: from system-role message or span attributes
        """
        input_text = None
        actual_output = None
        tool_outputs: List[str] = []
        tools_called: List[Dict[str, Any]] = []
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
                    elif role == "tool":
                        tool_call = self._extract_tool_call_from_body_message(msg)
                        if tool_call:
                            tools_called.append(tool_call)

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

        # If no tool data from agent span body, extract from sibling execute_tool spans
        if not tool_outputs and not tools_called:
            tool_outputs, tools_called = self._extract_tool_data_from_sibling_spans(agent_span, session_spans)

        if system_prompt is None:
            system_prompt = agent_span.get("attributes", {}).get("system_prompt")

        return SpanMapResult(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=tool_outputs if tool_outputs else None,
            context=tool_outputs if tool_outputs else None,
            system_prompt=system_prompt,
            tools_called=tools_called if tools_called else None,
        )

    def _extract_tool_call_from_body_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract tool call info from a tool-role message in span body.

        Message format: {"role": "tool", "content": {"content": "{\"key\": \"val\"}", "role": "tool", "id": "tooluse_xxx"}}
        """
        content = msg.get("content", {})
        if not isinstance(content, dict):
            return None
        tool_call_id = content.get("id", "")
        input_str = content.get("content", "")
        input_parameters = None
        if isinstance(input_str, str) and input_str:
            parsed = _try_parse_json(input_str)
            if isinstance(parsed, dict):
                input_parameters = parsed
        result: Dict[str, Any] = {"name": tool_call_id}
        if input_parameters is not None:
            result["input_parameters"] = input_parameters
        return result
