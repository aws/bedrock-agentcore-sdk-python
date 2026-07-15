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
    tools_called: Optional[List[Dict[str, Any]]] = None
    expected_tools: Optional[List[Dict[str, Any]]] = None
    assertions: Optional[List[str]] = None

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
        if self.tools_called is not None:
            result["tools_called"] = self.tools_called
        if self.expected_tools is not None:
            result["expected_tools"] = self.expected_tools
        if self.assertions is not None:
            result["assertions"] = self.assertions
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


def _try_parse_json(val: str) -> Any:
    """Try to parse a JSON string, return None on failure."""
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None


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


# --- LangChain/LangGraph message extraction utilities ---

# Role identifiers for human/user messages
_HUMAN_ROLES = {"human", "user", "humanmessage"}
# Role identifiers for AI/assistant messages
_AI_ROLES = {"ai", "assistant", "aimessage"}


def _get_message_role(msg: Any) -> Optional[str]:
    """Determine the normalized role of a LangGraph message.

    Handles multiple message formats:
    - Tuple: ("user", "text") or ("assistant", "text")
    - Dict with "type": {"type": "human", "content": "..."}
    - Dict with "role": {"role": "user", "content": "..."}
    - Dict with nested data: {"type": "human", "data": {"content": "..."}}
    - Dict with constructor kwargs: {"type": "constructor", "kwargs": {"type": "human", ...}}
    - Class name type: {"type": "HumanMessage", ...} or {"type": "AIMessage", ...}
    """
    if isinstance(msg, (list, tuple)) and len(msg) >= 2:
        role = str(msg[0]).lower()
        if role in _HUMAN_ROLES:
            return "human"
        if role in _AI_ROLES:
            return "ai"
        return None

    if not isinstance(msg, dict):
        return None

    # Check "role" field first (standard format)
    role = msg.get("role", "")
    if isinstance(role, str) and role:
        role_lower = role.lower()
        if role_lower in _HUMAN_ROLES:
            return "human"
        if role_lower in _AI_ROLES:
            return "ai"

    # Check "type" field (LangGraph format)
    msg_type = msg.get("type", "")
    if isinstance(msg_type, str) and msg_type:
        type_lower = msg_type.lower()
        if type_lower in _HUMAN_ROLES:
            return "human"
        if type_lower in _AI_ROLES:
            return "ai"
        # Constructor pattern: {"type": "constructor", "kwargs": {"type": "human", ...}}
        if type_lower == "constructor":
            kwargs = msg.get("kwargs", {})
            if isinstance(kwargs, dict):
                inner_type = kwargs.get("type", "")
                if isinstance(inner_type, str):
                    inner_lower = inner_type.lower()
                    if inner_lower in _HUMAN_ROLES:
                        return "human"
                    if inner_lower in _AI_ROLES:
                        return "ai"

    return None


def _get_langchain_message_content(msg: Any) -> Optional[str]:
    """Extract text content from a LangGraph/LangChain message.

    Handles:
    - Tuple: ("role", "text")
    - Dict with "content" (string or list of blocks)
    - Dict with "data": {"content": ...}
    - Dict with "kwargs": {"content": ...}
    """
    if isinstance(msg, (list, tuple)) and len(msg) >= 2:
        content = msg[1]
        if isinstance(content, str) and content.strip():
            return content.strip()
        return None

    if not isinstance(msg, dict):
        return None

    # Direct content field
    content = msg.get("content")
    if content is None:
        # Nested in "data"
        data = msg.get("data")
        if isinstance(data, dict):
            content = data.get("content")
        # Nested in "kwargs" (constructor pattern)
        if content is None:
            kwargs = msg.get("kwargs")
            if isinstance(kwargs, dict):
                content = kwargs.get("content")

    if content is None:
        return None

    if isinstance(content, str):
        return content.strip() if content.strip() else None

    # Content can be a list of blocks (e.g., [{"type": "text", "text": "..."}])
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Text block
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                # Skip tool_use blocks
                elif block.get("type") == "tool_use":
                    continue
        return "\n".join(parts).strip() if parts else None

    return None


def _is_tool_use_only(msg: Any) -> bool:
    """Check if a message contains only tool_use blocks (no text content)."""
    if not isinstance(msg, dict):
        return False
    content = msg.get("content")
    if content is None:
        data = msg.get("data")
        if isinstance(data, dict):
            content = data.get("content")
        if content is None:
            kwargs = msg.get("kwargs")
            if isinstance(kwargs, dict):
                content = kwargs.get("content")

    if not isinstance(content, list):
        return False

    # All blocks are tool_use with no text blocks
    has_tool_use = False
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "tool_use":
                has_tool_use = True
            elif block.get("type") == "text" and block.get("text", "").strip():
                return False
        elif isinstance(block, str) and block.strip():
            return False
    return has_tool_use


def extract_user_prompt_from_messages(messages: List[Any]) -> Optional[str]:
    """Extract the last human/user message content from a LangGraph message list.

    Iterates in reverse to find the most recent user message.

    Args:
        messages: List of messages in any supported LangGraph format.

    Returns:
        The text content of the last human message, or None if not found.
    """
    for msg in reversed(messages):
        role = _get_message_role(msg)
        if role == "human":
            content = _get_langchain_message_content(msg)
            if content:
                return content
    return None


def extract_agent_response_from_messages(messages: List[Any]) -> Optional[str]:
    """Extract the last AI/assistant message with text content from a LangGraph message list.

    Iterates in reverse. Skips AI messages that only contain tool_use blocks.

    Args:
        messages: List of messages in any supported LangGraph format.

    Returns:
        The text content of the last AI message with real text, or None if not found.
    """
    for msg in reversed(messages):
        role = _get_message_role(msg)
        if role == "ai":
            if _is_tool_use_only(msg):
                continue
            content = _get_langchain_message_content(msg)
            if content:
                return content
    return None
