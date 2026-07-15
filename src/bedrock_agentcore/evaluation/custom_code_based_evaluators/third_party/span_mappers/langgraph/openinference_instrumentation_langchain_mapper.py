"""OpenInference instrumentation LangChain span mapper.

Handles spans produced by ``openinference.instrumentation.langchain`` (Phoenix/Arize SDK).
"""

import json
import logging
from typing import Any, Dict, List, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.base import (
    BaseSpanMapper,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
    SpanMapResult,
    _parse_span_event_body,
    extract_agent_response_from_messages,
    extract_user_prompt_from_messages,
)

logger = logging.getLogger(__name__)

SCOPE_OPENINFERENCE_INSTRUMENTATION_LANGCHAIN = "openinference.instrumentation.langchain"


class OpenInferenceInstrumentationLangchainMapper(BaseSpanMapper):
    """Maps spans from openinference.instrumentation.langchain to evaluation fields.

    Extracts input, actual_output, retrieval_context, and system_prompt from
    Phoenix/Arize-instrumented LangChain/LangGraph traces.

    Supports two data formats:
    - Attributes path: data in span attributes (input.value / output.value)
    - Log-event path: data in span_events[].body (CloudWatch ADOT format)
    """

    @property
    def scope_name(self) -> str:
        return SCOPE_OPENINFERENCE_INSTRUMENTATION_LANGCHAIN

    def map(self, session_spans: List[Dict[str, Any]]) -> Optional[SpanMapResult]:
        """Map session spans to evaluation fields.

        Args:
            session_spans: Raw ADOT span dicts (already filtered by evaluationTarget).

        Returns:
            SpanMapResult with extracted fields, or None if no agent span found.
        """
        scope_spans = self._filter_scope_spans(session_spans)
        if not scope_spans:
            return None

        agent_span = self._find_agent_span(scope_spans)
        if agent_span is None:
            return None

        input_text = self._extract_input(agent_span, scope_spans)
        actual_output = self._extract_output(agent_span, scope_spans)
        retrieval_context, tools_called = self._extract_tool_data(scope_spans)
        system_prompt = self._extract_system_prompt(scope_spans)

        if not input_text and not actual_output:
            return None

        return SpanMapResult(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=retrieval_context if retrieval_context else None,
            context=retrieval_context if retrieval_context else None,
            system_prompt=system_prompt,
            tools_called=tools_called if tools_called else None,
        )

    def _find_agent_span(self, scope_spans: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the last top-level agent span.

        In multi-turn traces there may be multiple agent spans; the last one
        (latest end_time) is the most complete — matching server-side OtelSpanMapper behavior.

        Looks for:
        - openinference.span.kind == "CHAIN" AND name == "LangGraph"
        - OR openinference.span.kind == "AGENT" AND name != "agent" and not starting with "route_"
        """
        found = None
        for span in scope_spans:
            attrs = span.get("attributes", {})
            span_kind = attrs.get("openinference.span.kind", "")
            span_name = span.get("name", "")

            if span_kind == "CHAIN" and span_name == "LangGraph":
                found = span
            elif span_kind == "AGENT":
                if span_name != "agent" and not span_name.startswith("route_"):
                    found = span

        if found is not None:
            return found

        # Second pass: accept the last CHAIN span as fallback if no LangGraph found
        for span in scope_spans:
            attrs = span.get("attributes", {})
            span_kind = attrs.get("openinference.span.kind", "")
            if span_kind == "CHAIN":
                found = span

        return found

    def _extract_input(
        self, agent_span: Dict[str, Any], scope_spans: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Extract user input from the agent span.

        Fallback chain:
        1. Parse input.value as JSON → get ["messages"] → extract_user_prompt
        2. Use raw input.value string
        3. Log-event body: span_events[].body.input.messages → extract user message
        """
        attrs = agent_span.get("attributes", {})
        input_raw = attrs.get("input.value")

        if input_raw and isinstance(input_raw, str):
            parsed = self._try_parse_json(input_raw)
            if parsed is not None and isinstance(parsed, dict):
                messages = parsed.get("messages")
                if isinstance(messages, list) and messages:
                    result = extract_user_prompt_from_messages(messages)
                    if result:
                        return result

            # Fallback: use raw string
            if input_raw.strip():
                return input_raw.strip()

        # Log-event fallback: extract from span_events body
        return self._extract_input_from_span_body(agent_span)

    def _extract_output(
        self, agent_span: Dict[str, Any], scope_spans: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Extract agent output from the agent span.

        Fallback chain:
        1. Parse output.value as JSON → get ["messages"] → extract_agent_response
        2. Use raw output.value string
        3. Log-event body: span_events[].body.output.messages → extract assistant message
        """
        attrs = agent_span.get("attributes", {})
        output_raw = attrs.get("output.value")

        if output_raw and isinstance(output_raw, str):
            parsed = self._try_parse_json(output_raw)
            if parsed is not None and isinstance(parsed, dict):
                messages = parsed.get("messages")
                if isinstance(messages, list) and messages:
                    result = extract_agent_response_from_messages(messages)
                    if result:
                        return result

            # Fallback: use raw string
            if output_raw.strip():
                return output_raw.strip()

        # Log-event fallback: extract from span_events body
        return self._extract_output_from_span_body(agent_span)

    def _extract_input_from_span_body(self, agent_span: Dict[str, Any]) -> Optional[str]:
        """Extract user input from span_events[].body (CloudWatch ADOT format).

        Body format:
        {
            "input": {"messages": [{"content": "{\"messages\": [...]}", "role": "user"}]},
            "output": {...}
        }
        """
        for event in agent_span.get("span_events", []):
            body = _parse_span_event_body(event.get("body"))
            if not body:
                continue
            input_data = body.get("input", {})
            if not isinstance(input_data, dict):
                continue
            for msg in input_data.get("messages", []):
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if not content:
                    continue
                # Content may be JSON-encoded messages list
                if isinstance(content, str):
                    parsed = self._try_parse_json(content)
                    if isinstance(parsed, dict) and "messages" in parsed:
                        result = extract_user_prompt_from_messages(parsed["messages"])
                        if result:
                            return result
                    # Plain string content
                    if content.strip():
                        return content.strip()
                elif isinstance(content, dict) and "messages" in content:
                    result = extract_user_prompt_from_messages(content["messages"])
                    if result:
                        return result
        return None

    def _extract_output_from_span_body(self, agent_span: Dict[str, Any]) -> Optional[str]:
        """Extract agent output from span_events[].body (CloudWatch ADOT format).

        Body format:
        {
            "input": {...},
            "output": {"messages": [{"content": "{\"messages\": [...]}", "role": "assistant"}]}
        }

        Also handles "generations" format:
        {"content": "{\"generations\": [[{\"text\": \"...\"}]]}", "role": "assistant"}
        """
        for event in agent_span.get("span_events", []):
            body = _parse_span_event_body(event.get("body"))
            if not body:
                continue
            output_data = body.get("output", {})
            if not isinstance(output_data, dict):
                continue
            for msg in output_data.get("messages", []):
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content")
                if not content:
                    continue
                if isinstance(content, str):
                    parsed = self._try_parse_json(content)
                    if isinstance(parsed, dict):
                        # Try messages path
                        if "messages" in parsed:
                            result = extract_agent_response_from_messages(parsed["messages"])
                            if result:
                                return result
                        # Try generations path
                        if "generations" in parsed:
                            try:
                                return parsed["generations"][0][0]["text"]
                            except (IndexError, KeyError, TypeError):
                                pass
                    # Plain string fallback
                    if content.strip():
                        return content.strip()
                elif isinstance(content, dict):
                    if "messages" in content:
                        result = extract_agent_response_from_messages(content["messages"])
                        if result:
                            return result
        return None

    def _extract_tool_data(self, scope_spans: List[Dict[str, Any]]) -> tuple:
        """Extract retrieval context and tool calls from TOOL spans.

        Tool spans are identified by openinference.span.kind == "TOOL".
        Extracts from attributes["output.value"], parsing nested JSON structures.

        Returns:
            Tuple of (tool_outputs: List[str], tools_called: List[Dict[str, Any]])
        """
        tool_outputs: List[str] = []
        tools_called: List[Dict[str, Any]] = []
        for span in scope_spans:
            attrs = span.get("attributes", {})
            if attrs.get("openinference.span.kind") != "TOOL":
                continue

            # Extract tool call metadata
            tool_name = span.get("name") or attrs.get("tool.name")
            if tool_name:
                tool_call: Dict[str, Any] = {"name": tool_name}
                # Try to get input parameters from input.value
                input_val = attrs.get("input.value")
                if isinstance(input_val, str) and input_val.strip():
                    parsed = self._try_parse_json(input_val)
                    if isinstance(parsed, dict):
                        tool_call["input_parameters"] = parsed

                # Extract output
                output_val = attrs.get("output.value")
                if isinstance(output_val, str) and output_val.strip():
                    extracted = self._extract_tool_content_text(output_val.strip())
                    tool_call["output"] = extracted
                    tool_outputs.append(extracted)

                tools_called.append(tool_call)
            else:
                # No tool name — only extract output for retrieval_context
                output_val = attrs.get("output.value")
                if isinstance(output_val, str) and output_val.strip():
                    extracted = self._extract_tool_content_text(output_val.strip())
                    tool_outputs.append(extracted)

        return tool_outputs, tools_called

    def _extract_tool_content_text(self, raw_output: str) -> str:
        """Extract clean text from a tool output value.

        Handles formats:
        - Plain string → return as-is
        - '{"content": "actual text", ...}' → extract "content"
        - '{"data": {"content": "actual text", ...}}' → extract nested "content"
        - '[{"text": "block1"}, {"text": "block2"}]' → join text blocks
        """
        parsed = self._try_parse_json(raw_output)
        if parsed is None:
            return raw_output

        # Dict with nested "data" wrapper (openinference 0.1.62+)
        if isinstance(parsed, dict) and "data" in parsed:
            inner = parsed["data"]
            if isinstance(inner, dict) and "content" in inner:
                content = inner["content"]
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return self._join_text_blocks(content)

        # Dict with direct "content" field
        if isinstance(parsed, dict) and "content" in parsed:
            content = parsed["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return self._join_text_blocks(content)

        # List of text blocks
        if isinstance(parsed, list):
            joined = self._join_text_blocks(parsed)
            if joined:
                return joined

        return raw_output

    def _join_text_blocks(self, blocks: list) -> str:
        """Join text blocks like [{"text": "a"}, {"text": "b"}] into a single string."""
        parts = []
        for block in blocks:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else ""

    def _extract_system_prompt(self, scope_spans: List[Dict[str, Any]]) -> Optional[str]:
        """Extract system prompt from LLM inference spans.

        Looks in two locations:
        1. Span attributes: llm.input_messages.0.message.role == "system"
        2. Log-event body: input.messages where role == "system"
        """
        for span in scope_spans:
            attrs = span.get("attributes", {})
            if attrs.get("openinference.span.kind") != "LLM":
                continue

            # Try from span_events body
            for event in span.get("span_events", []):
                body = _parse_span_event_body(event.get("body"))
                if not body:
                    continue
                input_data = body.get("input", {})
                if not isinstance(input_data, dict):
                    continue
                for msg in input_data.get("messages", []):
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("role") != "user":
                        continue
                    content = msg.get("content")
                    if not isinstance(content, str):
                        continue
                    parsed = self._try_parse_json(content)
                    if not isinstance(parsed, dict) or "messages" not in parsed:
                        continue
                    for inner_msg in parsed["messages"]:
                        if not isinstance(inner_msg, dict):
                            continue
                        msg_type = self._resolve_message_type(inner_msg)
                        if msg_type == "system":
                            kwargs = inner_msg.get("kwargs") or inner_msg.get("data", {})
                            if isinstance(kwargs, dict):
                                sys_content = kwargs.get("content")
                                if isinstance(sys_content, str) and sys_content.strip():
                                    return sys_content.strip()
                                if isinstance(sys_content, list):
                                    parts = []
                                    for item in sys_content:
                                        if isinstance(item, dict) and "text" in item:
                                            parts.append(item["text"])
                                        elif isinstance(item, str):
                                            parts.append(item)
                                    if parts:
                                        return "\n\n".join(parts)

        return None

    @staticmethod
    def _resolve_message_type(msg: dict) -> Optional[str]:
        """Resolve the logical message type from a LangGraph message dict.

        Handles:
        - {"type": "human"} / {"type": "ai"} / {"type": "system"}
        - {"type": "constructor", "kwargs": {"type": "human"}}
        - {"id": ["langchain", "schema", "messages", "SystemMessage", ...]}
        """
        msg_type = msg.get("type", "")
        if isinstance(msg_type, str):
            type_lower = msg_type.lower()
            if type_lower in ("human", "humanmessage"):
                return "human"
            if type_lower in ("ai", "aimessage"):
                return "ai"
            if type_lower in ("system", "systemmessage"):
                return "system"
            if type_lower in ("tool", "toolmessage"):
                return "tool"
            if type_lower == "constructor":
                kwargs = msg.get("kwargs", {})
                if isinstance(kwargs, dict):
                    inner_type = kwargs.get("type", "")
                    if isinstance(inner_type, str):
                        return OpenInferenceInstrumentationLangchainMapper._resolve_message_type(
                            {"type": inner_type}
                        )

        # Check ID-based classification (e.g., ["langchain", ..., "SystemMessage", ...])
        msg_id = msg.get("id")
        if isinstance(msg_id, list):
            for part in msg_id:
                if isinstance(part, str):
                    part_lower = part.lower()
                    if "human" in part_lower:
                        return "human"
                    if "system" in part_lower:
                        return "system"
                    if part_lower in ("ai", "aimessage"):
                        return "ai"

        return None

    @staticmethod
    def _try_parse_json(val: str) -> Any:
        """Try to parse a JSON string, return None on failure."""
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return None
