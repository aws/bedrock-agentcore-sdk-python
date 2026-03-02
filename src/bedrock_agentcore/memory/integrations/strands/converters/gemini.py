"""Gemini-format converter for AgentCore Memory.

Converts Strands-native messages to/from a Gemini-compatible shape:
- text -> {"text": ...}
- toolUse -> {"functionCall": {...}}
- toolResult -> {"functionResponse": {...}}
- reasoningContent -> {"thought": {"text": ..., "signature": ...}}
"""

from __future__ import annotations

import json
import logging
from typing import Any, Tuple

from strands.types.session import SessionMessage

from .protocol import exceeds_conversational_limit

logger = logging.getLogger(__name__)


class GeminiConverseConverter:
    """Converts between Strands SessionMessages and Gemini-like STM payloads."""

    @staticmethod
    def message_to_payload(session_message: SessionMessage) -> list[Tuple[str, str]]:
        """Convert a Strands SessionMessage into a Gemini-style payload tuple."""
        message = session_message.message
        content = message.get("content", [])
        if not content:
            return []

        has_non_empty = any(
            (isinstance(item.get("text"), str) and item["text"].strip())
            or "toolUse" in item
            or "toolResult" in item
            or "reasoningContent" in item
            for item in content
            if isinstance(item, dict)
        )
        if not has_non_empty:
            return []

        formatted_parts = [GeminiConverseConverter._to_gemini_part(item) for item in content if isinstance(item, dict)]
        formatted_parts = [item for item in formatted_parts if item is not None]
        if not formatted_parts:
            return []

        role = message.get("role", "user")
        if role not in {"user", "assistant"}:
            role = "user"

        gemini_msg = {"role": "model" if role == "assistant" else "user", "parts": formatted_parts}
        return [(json.dumps(gemini_msg), role)]

    @staticmethod
    def events_to_messages(events: list[dict[str, Any]]) -> list[SessionMessage]:
        """Restore Strands SessionMessages from Gemini-style payload events."""
        messages: list[SessionMessage] = []

        for event in reversed(events):
            for payload_item in event.get("payload", []):
                msg_dict: dict[str, Any] | None = None

                if "conversational" in payload_item:
                    conv = payload_item["conversational"]
                    try:
                        msg_dict = json.loads(conv["content"]["text"])
                    except (json.JSONDecodeError, KeyError, ValueError):
                        logger.error("Failed to parse conversational payload as Gemini message")
                        continue
                elif "blob" in payload_item:
                    try:
                        blob_data = json.loads(payload_item["blob"])
                        if isinstance(blob_data, (tuple, list)) and len(blob_data) == 2:
                            msg_dict = json.loads(blob_data[0])
                    except (json.JSONDecodeError, ValueError):
                        logger.error("Failed to parse blob payload: %s", payload_item)
                        continue

                if not (msg_dict and isinstance(msg_dict, dict)):
                    continue
                parts = msg_dict.get("parts")
                if not parts:
                    continue

                role = msg_dict.get("role", "user")
                role = "assistant" if role == "model" else "user"
                if role not in {"user", "assistant"}:
                    role = "user"
                strands_content = [
                    GeminiConverseConverter._from_gemini_part(part) for part in parts if isinstance(part, dict)
                ]
                strands_content = [item for item in strands_content if item is not None]
                if not strands_content:
                    continue
                messages.append(SessionMessage(message={"role": role, "content": strands_content}, message_id=0))

        return messages

    @staticmethod
    def exceeds_conversational_limit(message: tuple[str, str]) -> bool:
        """Check if serialized message exceeds conversational payload size limit."""
        return exceeds_conversational_limit(message)

    @staticmethod
    def _to_gemini_part(item: dict[str, Any]) -> dict[str, Any] | None:
        if "text" in item:
            return {"text": item.get("text", "")}

        if "toolUse" in item:
            tool_use = item["toolUse"]
            part: dict[str, Any] = {
                "functionCall": {
                    "id": tool_use.get("toolUseId", ""),
                    "name": tool_use.get("name", ""),
                    "args": tool_use.get("input", {}),
                }
            }
            if tool_use.get("reasoningSignature"):
                part["thoughtSignature"] = tool_use["reasoningSignature"]
            return part

        if "toolResult" in item:
            tool_result = item["toolResult"]
            response_output: list[Any] = []
            for result_block in tool_result.get("content", []):
                if "json" in result_block:
                    response_output.append(result_block["json"])
                elif "text" in result_block:
                    response_output.append({"text": result_block["text"]})
            return {
                "functionResponse": {
                    "id": tool_result.get("toolUseId", ""),
                    "name": tool_result.get("toolUseId", ""),
                    "response": {"output": response_output},
                }
            }

        if "reasoningContent" in item:
            reasoning_text = item["reasoningContent"].get("reasoningText", {})
            thought: dict[str, Any] = {"text": reasoning_text.get("text", "")}
            if reasoning_text.get("signature"):
                thought["signature"] = reasoning_text["signature"]
            return {"thought": thought}

        return None

    @staticmethod
    def _from_gemini_part(part: dict[str, Any]) -> dict[str, Any] | None:
        if "text" in part:
            return {"text": part.get("text", "")}

        if "functionCall" in part:
            fc = part["functionCall"]
            tool_use = {
                "toolUseId": fc.get("id", ""),
                "name": fc.get("name", ""),
                "input": fc.get("args", {}),
            }
            if part.get("thoughtSignature"):
                tool_use["reasoningSignature"] = part["thoughtSignature"]
            return {"toolUse": tool_use}

        if "functionResponse" in part:
            fr = part["functionResponse"]
            response = fr.get("response", {})
            output = response.get("output", [])
            content: list[dict[str, Any]] = []
            for item in output:
                if isinstance(item, dict) and "text" in item:
                    content.append({"text": item["text"]})
                else:
                    content.append({"json": item})
            return {
                "toolResult": {
                    "toolUseId": fr.get("id", ""),
                    "status": "success",
                    "content": content,
                }
            }

        if "thought" in part:
            thought = part["thought"] if isinstance(part["thought"], dict) else {}
            reasoning_text = {"text": thought.get("text", "")}
            if thought.get("signature"):
                reasoning_text["signature"] = thought["signature"]
            return {"reasoningContent": {"reasoningText": reasoning_text}}

        return None
