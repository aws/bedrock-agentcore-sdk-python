"""OpenAI-format converter for AgentCore Memory.

Converts between Strands SessionMessages (Bedrock Converse format) and OpenAI message format
stored in AgentCore Memory STM events.
"""

import json
import logging
from typing import Any, Tuple

from strands.types.session import SessionMessage

from .protocol import exceeds_conversational_limit

logger = logging.getLogger(__name__)


def _bedrock_to_openai(message: dict) -> dict:
    """Convert a Bedrock Converse message dict to OpenAI message format.

    Args:
        message: Bedrock Converse format message with role and content list.

    Returns:
        OpenAI format message dict.
    """
    role = message.get("role", "user")
    content = message.get("content", [])

    # Check for toolResult — maps to OpenAI "tool" role
    if content and "toolResult" in content[0]:
        tool_result = content[0]["toolResult"]
        text_parts = [c.get("text", "") for c in tool_result.get("content", []) if "text" in c]
        result = {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": "\n".join(text_parts),
        }
        if "status" in tool_result:
            result["status"] = tool_result["status"]
        return result

    # Separate text and toolUse items
    text_parts = []
    tool_calls = []
    for item in content:
        if "text" in item:
            text = item["text"].strip()
            if text:
                text_parts.append(text)
        elif "toolUse" in item:
            tu = item["toolUse"]
            tool_calls.append({
                "id": tu["toolUseId"],
                "type": "function",
                "function": {
                    "name": tu["name"],
                    "arguments": json.dumps(tu.get("input", {})),
                },
            })

    result: dict[str, Any] = {"role": role}

    if tool_calls:
        # OpenAI convention: content is null when only tool_calls, string when both
        result["content"] = "\n".join(text_parts) if text_parts else None
        result["tool_calls"] = tool_calls
    else:
        result["content"] = "\n".join(text_parts) if text_parts else ""

    return result


def _openai_to_bedrock(openai_msg: dict) -> dict:
    """Convert an OpenAI message dict to Bedrock Converse format.

    Args:
        openai_msg: OpenAI format message dict.

    Returns:
        Bedrock Converse format message dict with role and content list.
    """
    role = openai_msg.get("role", "user")
    content_items: list[dict[str, Any]] = []

    # Handle tool response
    if role == "tool":
        tool_result: dict[str, Any] = {
            "toolUseId": openai_msg["tool_call_id"],
            "content": [{"text": openai_msg.get("content", "")}],
        }
        if "status" in openai_msg:
            tool_result["status"] = openai_msg["status"]
        return {
            "role": "user",
            "content": [{"toolResult": tool_result}],
        }

    # Handle system messages — Bedrock Converse has no system role
    if role == "system":
        return {
            "role": "user",
            "content": [{"text": openai_msg.get("content", "")}],
        }

    # Handle text content
    text_content = openai_msg.get("content")
    if text_content and isinstance(text_content, str):
        content_items.append({"text": text_content})

    # Handle tool_calls
    for tc in openai_msg.get("tool_calls", []):
        fn = tc.get("function", {})
        args_str = fn.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, ValueError):
            args = {}
        content_items.append({
            "toolUse": {
                "toolUseId": tc["id"],
                "name": fn["name"],
                "input": args,
            }
        })

    # Map role
    bedrock_role = "assistant" if role == "assistant" else "user"

    return {"role": bedrock_role, "content": content_items}


class OpenAIConverseConverter:
    """Converts between Strands SessionMessages (Bedrock Converse) and OpenAI message format in STM."""

    @staticmethod
    def message_to_payload(session_message: SessionMessage) -> list[Tuple[str, str]]:
        """Convert a SessionMessage (Bedrock Converse) to OpenAI-format STM payload.

        Args:
            session_message: The Strands session message to convert.

        Returns:
            List of (json_str, role) tuples for STM storage. Empty list if no content.
        """
        message = session_message.message
        content = message.get("content", [])
        if not content:
            return []

        # Filter empty/whitespace-only text items
        has_non_empty = any(
            ("text" in item and item["text"].strip())
            or "toolUse" in item
            or "toolResult" in item
            for item in content
        )
        if not has_non_empty:
            return []

        openai_msg = _bedrock_to_openai(message)
        role = openai_msg.get("role", "user")

        return [(json.dumps(openai_msg), role)]

    @staticmethod
    def events_to_messages(events: list[dict[str, Any]]) -> list[SessionMessage]:
        """Convert STM events containing OpenAI-format messages to SessionMessages (Bedrock Converse).

        Args:
            events: List of STM events. Each may contain conversational or blob payloads
                with OpenAI-format JSON.

        Returns:
            List of SessionMessage objects in Bedrock Converse format.
        """
        messages: list[SessionMessage] = []

        for event in events:
            for payload_item in event.get("payload", []):
                openai_msg = None

                if "conversational" in payload_item:
                    conv = payload_item["conversational"]
                    try:
                        openai_msg = json.loads(conv["content"]["text"])
                    except (json.JSONDecodeError, KeyError, ValueError):
                        logger.error("Failed to parse conversational payload as OpenAI message")
                        continue

                elif "blob" in payload_item:
                    try:
                        blob_data = json.loads(payload_item["blob"])
                        if isinstance(blob_data, (tuple, list)) and len(blob_data) == 2:
                            openai_msg = json.loads(blob_data[0])
                    except (json.JSONDecodeError, ValueError):
                        logger.error("Failed to parse blob payload: %s", payload_item)
                        continue

                if openai_msg and isinstance(openai_msg, dict):
                    bedrock_msg = _openai_to_bedrock(openai_msg)
                    if bedrock_msg.get("content"):
                        session_msg = SessionMessage(
                            message=bedrock_msg,
                            message_id=0,
                        )
                        messages.append(session_msg)

        return list(reversed(messages))

    @staticmethod
    def exceeds_conversational_limit(message: tuple[str, str]) -> bool:
        """Check if message exceeds conversational payload size limit."""
        return exceeds_conversational_limit(message)
