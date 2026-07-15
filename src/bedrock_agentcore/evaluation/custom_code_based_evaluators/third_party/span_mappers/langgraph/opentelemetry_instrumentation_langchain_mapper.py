"""OpenTelemetry instrumentation LangChain span mapper.

Handles spans produced by ``opentelemetry.instrumentation.langchain`` (Traceloop SDK).
"""

import json
import logging
from typing import Any, Dict, List, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.base import (
    BaseSpanMapper,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
    SpanMapResult,
    extract_agent_response_from_messages,
    extract_user_prompt_from_messages,
)

logger = logging.getLogger(__name__)

SCOPE_OPENTELEMETRY_INSTRUMENTATION_LANGCHAIN = "opentelemetry.instrumentation.langchain"
SCOPE_AMAZON_OPENTELEMETRY_DISTRO_INSTRUMENTATION_LANGCHAIN = (
    "amazon.opentelemetry.distro.instrumentation.langchain"
)

# Well-known input field names in LangGraph state dicts, ordered by specificity
_INPUT_FIELD_NAMES = (
    "user_prompt", "user_message_text", "user_input", "user_query",
    "query", "prompt", "input_text", "question", "human_input",
)
# Well-known output field names in LangGraph state dicts, ordered by specificity
_OUTPUT_FIELD_NAMES = (
    "agent_response", "final_response", "primary_message", "response",
    "output_text", "answer", "result", "final_result", "assistant_response",
)


class OpenTelemetryInstrumentationLangchainMapper(BaseSpanMapper):
    """Maps spans from opentelemetry.instrumentation.langchain to evaluation fields.

    Extracts input, actual_output, retrieval_context, and system_prompt from
    Traceloop-instrumented LangChain/LangGraph traces.
    """

    @property
    def scope_name(self) -> str:
        return SCOPE_OPENTELEMETRY_INSTRUMENTATION_LANGCHAIN

    @property
    def scope_names(self) -> List[str]:
        return [
            SCOPE_OPENTELEMETRY_INSTRUMENTATION_LANGCHAIN,
            SCOPE_AMAZON_OPENTELEMETRY_DISTRO_INSTRUMENTATION_LANGCHAIN,
        ]

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
        """Find the last top-level agent/workflow span.

        In multi-turn traces there may be multiple agent spans; the last one
        (latest end_time) is the most complete — matching server-side OtelSpanMapper behavior.

        Looks for:
        - attributes["traceloop.span.kind"] == "workflow"
        - OR attributes["gen_ai.operation.name"] == "invoke_agent"
        """
        found = None
        for span in scope_spans:
            attrs = span.get("attributes", {})
            if attrs.get("traceloop.span.kind") == "workflow":
                found = span
            elif attrs.get("gen_ai.operation.name") == "invoke_agent":
                found = span
        return found

    def _extract_input(
        self, agent_span: Dict[str, Any], scope_spans: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Extract user input using the fallback chain.

        1. Parse gen_ai.task.input → ["inputs"]["messages"] → extract_user_prompt
        2. Parse gen_ai.task.input → try well-known field names
        3. Fallback: raw gen_ai.task.input string
        4. Final fallback: child LLM spans gen_ai.prompt.0.content (first user)
        """
        attrs = agent_span.get("attributes", {})
        task_input_raw = attrs.get("gen_ai.task.input")

        if task_input_raw and isinstance(task_input_raw, str):
            parsed = self._try_parse_json(task_input_raw)
            if parsed is not None:
                # Try inputs.messages path
                inputs = parsed.get("inputs") if isinstance(parsed, dict) else None
                if isinstance(inputs, dict):
                    messages = inputs.get("messages")
                    if isinstance(messages, list) and messages:
                        result = extract_user_prompt_from_messages(messages)
                        if result:
                            return result
                    # Try well-known field names in inputs
                    for field_name in _INPUT_FIELD_NAMES:
                        val = inputs.get(field_name)
                        if isinstance(val, str) and val.strip():
                            return val.strip()

                # Try top-level messages (some formats put them at root)
                if isinstance(parsed, dict):
                    messages = parsed.get("messages")
                    if isinstance(messages, list) and messages:
                        result = extract_user_prompt_from_messages(messages)
                        if result:
                            return result
                    # Try well-known field names at top level
                    for field_name in _INPUT_FIELD_NAMES:
                        val = parsed.get(field_name)
                        if isinstance(val, str) and val.strip():
                            return val.strip()

            # Fallback: use raw string if it looks like actual content
            if task_input_raw.strip() and not task_input_raw.strip().startswith("{"):
                return task_input_raw.strip()

        # Fallback: span_events body on chat/LLM spans (CloudWatch ADOT split format)
        input_from_body = self._extract_input_from_span_bodies(scope_spans)
        if input_from_body:
            return input_from_body

        # Final fallback: child LLM spans
        return self._extract_input_from_llm_spans(scope_spans)

    def _extract_output(
        self, agent_span: Dict[str, Any], scope_spans: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Extract agent output using the fallback chain.

        1. Parse gen_ai.task.output → ["outputs"]["messages"] → extract_agent_response
        2. Parse gen_ai.task.output → try well-known field names
        3. Fallback: raw gen_ai.task.output string
        4. Final fallback: child LLM spans gen_ai.completion.0.content (last assistant)
        """
        attrs = agent_span.get("attributes", {})
        task_output_raw = attrs.get("gen_ai.task.output")

        if task_output_raw and isinstance(task_output_raw, str):
            parsed = self._try_parse_json(task_output_raw)
            if parsed is not None:
                # Try outputs.messages path
                outputs = parsed.get("outputs") if isinstance(parsed, dict) else None
                if isinstance(outputs, dict):
                    messages = outputs.get("messages")
                    if isinstance(messages, list) and messages:
                        result = extract_agent_response_from_messages(messages)
                        if result:
                            return result
                    # Try well-known field names in outputs
                    for field_name in _OUTPUT_FIELD_NAMES:
                        val = outputs.get(field_name)
                        if isinstance(val, str) and val.strip():
                            return val.strip()

                # Try top-level messages
                if isinstance(parsed, dict):
                    messages = parsed.get("messages")
                    if isinstance(messages, list) and messages:
                        result = extract_agent_response_from_messages(messages)
                        if result:
                            return result
                    # Try well-known field names at top level
                    for field_name in _OUTPUT_FIELD_NAMES:
                        val = parsed.get(field_name)
                        if isinstance(val, str) and val.strip():
                            return val.strip()

            # Fallback: use raw string if it looks like actual content
            if task_output_raw.strip() and not task_output_raw.strip().startswith("{"):
                return task_output_raw.strip()

        # Fallback: span_events body on chat/LLM spans (CloudWatch ADOT split format)
        output_from_body = self._extract_output_from_span_bodies(scope_spans)
        if output_from_body:
            return output_from_body

        # Final fallback: child LLM spans
        return self._extract_output_from_llm_spans(scope_spans)

    def _extract_input_from_span_bodies(self, scope_spans: List[Dict[str, Any]]) -> Optional[str]:
        """Extract user input from span_events body (CloudWatch ADOT split format).

        In adot_v18+, chat span bodies contain input.messages with LangGraph message format.
        Looks for the first user message in the first chat span's body.
        """
        for span in scope_spans:
            for event in span.get("span_events", []):
                body = event.get("body")
                if not isinstance(body, dict):
                    continue
                input_data = body.get("input", {})
                if not isinstance(input_data, dict):
                    continue
                for msg in input_data.get("messages", []):
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("role") != "user":
                        continue
                    content = msg.get("content", "")
                    if not isinstance(content, str) or not content.strip():
                        continue
                    # Content may be JSON-encoded LangGraph messages
                    parsed = self._try_parse_json(content)
                    if isinstance(parsed, list):
                        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
                            extract_user_prompt_from_messages,
                        )
                        result = extract_user_prompt_from_messages(parsed)
                        if result:
                            return result
                        # Handle "parts" format: [{"role": "user", "parts": [{"type": "text", "content": "..."}]}]
                        for m in parsed:
                            if isinstance(m, dict) and m.get("role") in ("user", "human"):
                                parts = m.get("parts", [])
                                for p in parts:
                                    if isinstance(p, dict) and p.get("type") == "text" and p.get("content"):
                                        return p["content"]
                    # Plain string content
                    if content.strip():
                        return content.strip()
        return None

    def _extract_output_from_span_bodies(self, scope_spans: List[Dict[str, Any]]) -> Optional[str]:
        """Extract agent output from span_events body (CloudWatch ADOT split format).

        In adot_v18+, chat span bodies contain output.messages with LangGraph message format.
        Looks for the last assistant message across all chat span bodies.
        """
        last_output = None
        for span in scope_spans:
            for event in span.get("span_events", []):
                body = event.get("body")
                if not isinstance(body, dict):
                    continue
                output_data = body.get("output", {})
                if not isinstance(output_data, dict):
                    continue
                for msg in output_data.get("messages", []):
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("role") != "assistant":
                        continue
                    content = msg.get("content", "")
                    if not isinstance(content, str) or not content.strip():
                        continue
                    # Content may be JSON-encoded LangGraph messages
                    parsed = self._try_parse_json(content)
                    if isinstance(parsed, list):
                        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
                            extract_agent_response_from_messages,
                        )
                        result = extract_agent_response_from_messages(parsed)
                        if result:
                            last_output = result
                            continue
                        # Handle "parts" format: [{"role": "assistant", "parts": [{"type": "text", "content": "..."}]}]
                        for m in parsed:
                            if isinstance(m, dict) and m.get("role") in ("assistant", "ai"):
                                parts = m.get("parts", [])
                                for p in parts:
                                    if isinstance(p, dict) and p.get("type") == "text" and p.get("content"):
                                        last_output = p["content"]
                                        break
                                if last_output:
                                    break
                        if last_output:
                            continue
                    # Plain string content
                    if content.strip():
                        last_output = content.strip()
        return last_output

    def _extract_input_from_llm_spans(self, scope_spans: List[Dict[str, Any]]) -> Optional[str]:
        """Extract user input from the first LLM child span with gen_ai.prompt.0.content."""
        for span in scope_spans:
            attrs = span.get("attributes", {})
            if not self._is_llm_span(attrs):
                continue
            role = attrs.get("gen_ai.prompt.0.role", "")
            if role == "user":
                content = attrs.get("gen_ai.prompt.0.content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
            # Check higher indices for first user prompt
            for i in range(10):
                role_key = f"gen_ai.prompt.{i}.role"
                content_key = f"gen_ai.prompt.{i}.content"
                if role_key not in attrs:
                    break
                if attrs.get(role_key) == "user":
                    content = attrs.get(content_key)
                    if isinstance(content, str) and content.strip():
                        return content.strip()
        return None

    def _extract_output_from_llm_spans(self, scope_spans: List[Dict[str, Any]]) -> Optional[str]:
        """Extract assistant output from the last LLM child span with gen_ai.completion.0.content."""
        last_output = None
        for span in scope_spans:
            attrs = span.get("attributes", {})
            if not self._is_llm_span(attrs):
                continue
            content = attrs.get("gen_ai.completion.0.content")
            if isinstance(content, str) and content.strip():
                last_output = content.strip()
        return last_output

    def _is_tool_span(self, attrs: Dict[str, Any]) -> bool:
        """Check if span attributes indicate a tool execution span."""
        if attrs.get("traceloop.span.kind") == "tool":
            return True
        if attrs.get("gen_ai.operation.name") == "execute_tool":
            return True
        return False

    def _extract_tool_data(self, scope_spans: List[Dict[str, Any]]) -> tuple:
        """Extract retrieval context and tool calls from tool spans.

        Tool spans are identified by traceloop.span.kind == "tool"
        OR gen_ai.operation.name == "execute_tool" (Amazon OTEL distro).

        Returns:
            Tuple of (tool_outputs: List[str], tools_called: List[Dict[str, Any]])
        """
        tool_outputs: List[str] = []
        tools_called: List[Dict[str, Any]] = []
        for span in scope_spans:
            attrs = span.get("attributes", {})
            if not self._is_tool_span(attrs):
                continue

            # Extract tool call metadata
            tool_name = attrs.get("gen_ai.tool.name") or span.get("name")
            if tool_name:
                tool_call: Dict[str, Any] = {"name": tool_name}
                # Try to get input parameters
                tool_args_raw = attrs.get("gen_ai.tool.call.arguments")
                if isinstance(tool_args_raw, str) and tool_args_raw.strip():
                    parsed = self._try_parse_json(tool_args_raw)
                    if isinstance(parsed, dict):
                        tool_call["input_parameters"] = parsed

                # Tool output will be added below
                output_text = None

                # Priority 1: gen_ai.tool.call.result attribute
                tool_result_raw = attrs.get("gen_ai.tool.call.result")
                if isinstance(tool_result_raw, str) and tool_result_raw.strip():
                    extracted = self._extract_tool_result_content(tool_result_raw)
                    if extracted:
                        output_text = extracted

                # Priority 2: span_events body
                if output_text is None:
                    for event in span.get("span_events", []):
                        body = event.get("body")
                        if body is None:
                            continue
                        text = self._extract_text_from_body(body)
                        if text:
                            output_text = text
                            break

                # Priority 3: gen_ai.task.output
                if output_text is None:
                    task_output = attrs.get("gen_ai.task.output")
                    if isinstance(task_output, str) and task_output.strip():
                        output_text = task_output.strip()

                if output_text:
                    tool_call["output"] = output_text
                    tool_outputs.append(output_text)

                tools_called.append(tool_call)
            else:
                # No tool name — only extract output for retrieval_context
                tool_result_raw = attrs.get("gen_ai.tool.call.result")
                if isinstance(tool_result_raw, str) and tool_result_raw.strip():
                    extracted = self._extract_tool_result_content(tool_result_raw)
                    if extracted:
                        tool_outputs.append(extracted)
                        continue

                body_extracted = False
                for event in span.get("span_events", []):
                    body = event.get("body")
                    if body is None:
                        continue
                    text = self._extract_text_from_body(body)
                    if text:
                        tool_outputs.append(text)
                        body_extracted = True

                if body_extracted:
                    continue

                task_output = attrs.get("gen_ai.task.output")
                if isinstance(task_output, str) and task_output.strip():
                    tool_outputs.append(task_output.strip())

        return tool_outputs, tools_called

    def _extract_tool_result_content(self, raw_result: str) -> Optional[str]:
        """Extract clean text from a gen_ai.tool.call.result attribute value.

        Handles formats:
        - '{"output": {"kwargs": {"content": "actual text", ...}}}' (LangChain ToolMessage wrapper)
        - '{"output": {"kwargs": {"content": [...list of blocks...]}}}' (list content)
        - '{"output": "plain string"}' (direct string result)
        - '{"output": {...}}' (dict without kwargs → JSON dump)
        - Plain string (non-JSON)
        """
        parsed = self._try_parse_json(raw_result)
        if parsed is None:
            return raw_result.strip() if raw_result.strip() else None

        if not isinstance(parsed, dict):
            return raw_result.strip()

        output = parsed.get("output")
        if output is None:
            # No "output" wrapper — try direct content fields
            for key in ("content", "result", "text"):
                val = parsed.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return raw_result.strip()

        if isinstance(output, str):
            return output.strip() if output.strip() else None

        if isinstance(output, dict):
            # LangChain ToolMessage wrapper: output.kwargs.content
            kwargs = output.get("kwargs")
            if isinstance(kwargs, dict):
                content = kwargs.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    return self._join_content_blocks(content)
            # Plain dict result
            return json.dumps(output)

        if isinstance(output, list):
            return self._join_content_blocks(output)

        return raw_result.strip()

    def _join_content_blocks(self, blocks: list) -> str:
        """Join content blocks into a single string.

        Handles: [{"text": "a"}, {"text": "b"}] or ["a", "b"] or mixed.
        """
        parts = []
        for block in blocks:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else json.dumps(blocks)

    def _extract_system_prompt(self, scope_spans: List[Dict[str, Any]]) -> Optional[str]:
        """Extract system prompt from LLM spans.

        Looks for gen_ai.prompt.0.role == "system" with gen_ai.prompt.0.content.
        """
        for span in scope_spans:
            attrs = span.get("attributes", {})
            if not self._is_llm_span(attrs):
                continue
            if attrs.get("gen_ai.prompt.0.role") == "system":
                content = attrs.get("gen_ai.prompt.0.content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return None

    def _is_llm_span(self, attrs: Dict[str, Any]) -> bool:
        """Check if span attributes indicate an LLM call span."""
        if attrs.get("llm.request.type") == "chat":
            return True
        if attrs.get("gen_ai.operation.name") == "chat":
            return True
        return False

    def _extract_text_from_body(self, body: Any) -> Optional[str]:
        """Extract text content from a span event body."""
        if isinstance(body, str):
            parsed = self._try_parse_json(body)
            if parsed is not None and isinstance(parsed, dict):
                # Look for output/text fields
                for key in ("output", "text", "result", "content"):
                    val = parsed.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
            # Use raw string if non-empty
            if body.strip():
                return body.strip()
        elif isinstance(body, dict):
            for key in ("output", "text", "result", "content"):
                val = body.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return None

    @staticmethod
    def _try_parse_json(val: str) -> Any:
        """Try to parse a JSON string, return None on failure."""
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return None
