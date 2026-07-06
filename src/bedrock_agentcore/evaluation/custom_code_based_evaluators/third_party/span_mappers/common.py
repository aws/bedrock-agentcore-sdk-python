"""Common span mapping utilities shared across format-specific mappers."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpanMapResult:
    """Extraction result from spans — only what metrics consume."""

    input: Optional[str] = None
    actual_output: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    context: Optional[List[str]] = None
    system_prompt: Optional[str] = None
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
        if self.system_prompt is not None:
            result["system_prompt"] = self.system_prompt
        if self.expected_output is not None:
            result["expected_output"] = self.expected_output
        return result


def _try_parse_text_blocks(val: str) -> Optional[str]:
    """Try to parse a JSON-encoded list of text blocks.

    Strands encodes content as: '[{"text": "Hello"}, {"text": "world"}]'
    Returns joined text if parseable, None otherwise.
    """
    try:
        parsed = json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    parts = []
    for item in parsed:
        if isinstance(item, dict) and "text" in item:
            parts.append(item["text"])
    return "\n".join(parts) if parts else None


def _get_message_content(message: Any) -> str:
    """Extract text content from a message object.

    Handles Strands format variations:
    - {"content": "plain string"}
    - {"content": {"content": '[{"text": "..."}]'}}  (user messages)
    - {"content": {"message": "...", "finish_reason": "..."}}  (assistant messages)
    - {"content": '[{"text": "..."}, {"toolUse": {...}}]'}  (assistant with tool calls)
    """
    if isinstance(message, str):
        return _try_parse_text_blocks(message) or message
    if isinstance(message, dict):
        for key in ("content", "message"):
            if key in message:
                val = message[key]
                if isinstance(val, str):
                    return _try_parse_text_blocks(val) or val
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
