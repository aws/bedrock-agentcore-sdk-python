"""Common span parsing utilities shared across format-specific parsers."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpanParseResult:
    """Result of parsing spans into evaluation fields."""

    input: Optional[str] = None
    actual_output: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    context: Optional[List[str]] = None
    expected_output: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, omitting None values."""
        result: Dict[str, Any] = {}
        if self.input is not None:
            result["input"] = self.input
        if self.actual_output is not None:
            result["actual_output"] = self.actual_output
        if self.retrieval_context is not None:
            result["retrieval_context"] = self.retrieval_context
        if self.context is not None:
            result["context"] = self.context
        if self.expected_output is not None:
            result["expected_output"] = self.expected_output
        return result


def _get_message_content(message: Any) -> str:
    """Extract text content from a message object."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        for key in ("content", "message"):
            if key in message:
                val = message[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    return _get_message_content(val)
                if isinstance(val, list):
                    parts = []
                    for item in val:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
                    if parts:
                        return "\n".join(parts)
                return str(val)
    return ""


def _parse_span_event_body(body: Any) -> Dict[str, Any]:
    """Parse the body of a span event, handling both dict and JSON string."""
    if isinstance(body, str):
        try:
            return json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(body, dict):
        return body
    return {}


def extract_from_agent_span_events(
    session_spans: List[Dict[str, Any]],
) -> Optional[SpanParseResult]:
    """Extract evaluation fields from agent-level span events.

    Looks for spans where attributes.gen_ai.operation.name == "invoke_agent",
    then inspects span_events for input/output messages.

    Args:
        session_spans: Raw ADOT span dicts.

    Returns:
        SpanParseResult if agent span with valid events found, None otherwise.
    """
    user_messages: List[str] = []
    assistant_messages: List[str] = []
    tool_messages: List[str] = []

    found_agent_span = False

    for span in session_spans:
        attributes = span.get("attributes", {})
        operation_name = attributes.get("gen_ai.operation.name")
        if operation_name != "invoke_agent":
            continue

        found_agent_span = True
        span_events = span.get("span_events", [])

        for event in span_events:
            body = _parse_span_event_body(event.get("body"))
            if not body:
                continue

            input_data = body.get("input", {})
            if isinstance(input_data, dict):
                for msg in input_data.get("messages", []):
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "")
                    content = _get_message_content(msg)
                    if role == "user" and content:
                        user_messages.append(content)

            output_data = body.get("output", {})
            if isinstance(output_data, dict):
                for msg in output_data.get("messages", []):
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "")
                    content = _get_message_content(msg)
                    if role == "assistant" and content:
                        assistant_messages.append(content)
                    elif role == "tool" and content:
                        tool_messages.append(content)

    if not found_agent_span:
        return None

    if not user_messages and not assistant_messages:
        return None

    result = SpanParseResult()
    if user_messages:
        result.input = user_messages[0]
    if assistant_messages:
        result.actual_output = assistant_messages[-1]
    if tool_messages:
        result.retrieval_context = tool_messages
        result.context = tool_messages

    return result
