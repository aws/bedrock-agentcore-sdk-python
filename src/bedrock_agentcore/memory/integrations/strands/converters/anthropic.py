"""Anthropic-format converter for AgentCore Memory.

Converts Strands-native messages to/from an Anthropic-compatible message shape:
- text -> {"type":"text","text":...}
- toolUse -> {"type":"tool_use", ...}
- toolResult -> {"type":"tool_result", ...}
- reasoningContent -> {"type":"thinking", ...}
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Tuple

from strands.types.session import SessionMessage

from .protocol import exceeds_conversational_limit

logger = logging.getLogger(__name__)


class AnthropicConverseConverter:
    """Converts between Strands SessionMessages and Anthropic-like STM payloads."""

    @staticmethod
    def message_to_payload(session_message: SessionMessage) -> list[Tuple[str, str]]:
        """Convert a Strands SessionMessage into an Anthropic-style payload tuple."""
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

        formatted_content = [
            AnthropicConverseConverter._to_anthropic_block(item) for item in content if isinstance(item, dict)
        ]
        formatted_content = [item for item in formatted_content if item is not None]
        if not formatted_content:
            return []

        # Anthropic APIs accept user/assistant roles.
        role = message.get("role", "user")
        if role not in {"user", "assistant"}:
            role = "user"

        anthropic_msg = {"role": role, "content": formatted_content}
        return [(json.dumps(anthropic_msg), role)]

    @staticmethod
    def events_to_messages(events: list[dict[str, Any]]) -> list[SessionMessage]:
        """Restore Strands SessionMessages from Anthropic-style payload events."""
        messages: list[SessionMessage] = []

        for event in reversed(events):
            for payload_item in event.get("payload", []):
                msg_dict: dict[str, Any] | None = None

                if "conversational" in payload_item:
                    conv = payload_item["conversational"]
                    try:
                        msg_dict = json.loads(conv["content"]["text"])
                    except (json.JSONDecodeError, KeyError, ValueError):
                        logger.error("Failed to parse conversational payload as Anthropic message")
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
                if "content" not in msg_dict or not msg_dict["content"]:
                    continue

                role = msg_dict.get("role", "user")
                if role not in {"user", "assistant"}:
                    role = "user"
                strands_content = [
                    AnthropicConverseConverter._from_anthropic_block(block)
                    for block in msg_dict["content"]
                    if isinstance(block, dict)
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
    def _to_anthropic_block(item: dict[str, Any]) -> dict[str, Any] | None:
        if "text" in item:
            return {"type": "text", "text": item.get("text", "")}

        if "toolUse" in item:
            tool_use = item["toolUse"]
            return {
                "type": "tool_use",
                "id": tool_use.get("toolUseId"),
                "name": tool_use.get("name"),
                "input": tool_use.get("input", {}),
                **(
                    {"reasoning_signature": tool_use.get("reasoningSignature")}
                    if tool_use.get("reasoningSignature")
                    else {}
                ),
            }

        if "toolResult" in item:
            tool_result = item["toolResult"]
            content_blocks: list[dict[str, Any]] = []
            for result_block in tool_result.get("content", []):
                if "text" in result_block:
                    content_blocks.append({"type": "text", "text": result_block["text"]})
                elif "json" in result_block:
                    content_blocks.append({"type": "text", "text": json.dumps(result_block["json"])})
            return {
                "type": "tool_result",
                "tool_use_id": tool_result.get("toolUseId"),
                "is_error": tool_result.get("status") == "error",
                "content": content_blocks,
            }

        if "reasoningContent" in item:
            reasoning_text = item["reasoningContent"].get("reasoningText", {})
            return {
                "type": "thinking",
                "thinking": reasoning_text.get("text", ""),
                **({"signature": reasoning_text.get("signature")} if reasoning_text.get("signature") else {}),
            }

        return None

    @staticmethod
    def _from_anthropic_block(block: dict[str, Any]) -> dict[str, Any] | None:
        block_type = block.get("type")
        if block_type == "text":
            return {"text": block.get("text", "")}

        if block_type == "tool_use":
            tool_use = {
                "toolUseId": block.get("id", ""),
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            }
            if block.get("reasoning_signature"):
                tool_use["reasoningSignature"] = block["reasoning_signature"]
            return {"toolUse": tool_use}

        if block_type == "tool_result":
            content: list[dict[str, Any]] = []
            for content_block in block.get("content", []):
                if isinstance(content_block, dict) and content_block.get("type") == "text":
                    content.append({"text": content_block.get("text", "")})
            return {
                "toolResult": {
                    "toolUseId": block.get("tool_use_id", ""),
                    "status": "error" if block.get("is_error") else "success",
                    "content": content,
                }
            }

        if block_type == "thinking":
            reasoning_text = {"text": block.get("thinking", "")}
            if block.get("signature"):
                reasoning_text["signature"] = block["signature"]
            return {"reasoningContent": {"reasoningText": reasoning_text}}

        # Support a small compatibility path for binary thought signature if present.
        if block_type == "thinking_b64":
            reasoning_text = {"text": block.get("thinking", "")}
            if block.get("signature_b64"):
                try:
                    reasoning_text["signature"] = base64.b64decode(block["signature_b64"]).decode("utf-8")
                except Exception:
                    pass
            return {"reasoningContent": {"reasoningText": reasoning_text}}

        return None
