"""Bedrock AgentCore Memory conversion utilities."""

import base64
import json
import logging
from datetime import datetime
from typing import Any, Tuple

from strands.types.content import ContentBlock, Message
from strands.types.session import SessionMessage

logger = logging.getLogger(__name__)

CONVERSATIONAL_MAX_SIZE = 9000


class BedrockConverter:
    """Handles conversion between Strands and Bedrock AgentCore Memory formats."""

    @staticmethod
    def message_to_bedrock_payload(session_message: SessionMessage) -> list[Tuple[str, str]]:
        """Convert a SessionMessage to Bedrock AgentCore Memory message format.

        Args:
            session_message (SessionMessage): The session message to convert.

        Returns:
            list[Tuple[str, str]]: list of (text, role) tuples for Bedrock AgentCore Memory.
        """
        message = session_message.to_message()
        messages = []

        for content_block in message.get("content", []):
            if "text" in content_block:
                messages.append((content_block["text"], message["role"]))
            elif "toolUse" in content_block:
                tool_use = content_block["toolUse"]
                tool_text = (
                    f"Tool use: {tool_use.get('name', 'unknown')} with input: {json.dumps(tool_use.get('input', {}))}"
                )
                messages.append((tool_text, message["role"]))
            elif "toolResult" in content_block:
                tool_result = content_block["toolResult"]
                result_content = BedrockConverter._serialize_tool_result_content(tool_result.get("content", []))
                messages.append((result_content, message["role"]))

        return messages

    @staticmethod
    def _serialize_tool_result_content(content_list: list[dict]) -> str:
        """
        Serialize tool result content, handling different content types as blobs when needed.
        
        Args:
            content_list: List of ToolResultContent items

        Returns:
            str: Serialized content for storage
        """
        if not content_list:
            return "Tool result: (empty)"

        serialized_parts = []
        for content in content_list:
            if "text" in content:
                serialized_parts.append(content["text"])
            elif "json" in content:
                serialized_parts.append(f"JSON: {json.dumps(content['json'])}")
            elif "document" in content:
                doc = content["document"]
                doc_b64 = base64.b64encode(doc["source"]["bytes"]).decode()
                serialized_parts.append(f"Document({doc['format']}): {doc['name']} [base64:{doc_b64}]")
            elif "image" in content:
                img = content["image"]
                img_b64 = base64.b64encode(img["source"]["bytes"]).decode()
                serialized_parts.append(f"Image({img['format']}): [base64:{img_b64}]")

        return f"Tool result: {' | '.join(serialized_parts)}"

    @staticmethod
    def bedrock_events_to_messages(events: list[dict[str, Any]]) -> list[SessionMessage]:
        """Convert Bedrock AgentCore Memory events to SessionMessages.

        Args:
            events (list[dict[str, Any]]): list of events from Bedrock AgentCore Memory. Each individual event looks as follows:
                ```
                {
                    "memoryId": "unique_mem_id",
                    "actorId": "actor_id",
                    "sessionId": "session_id",
                    "eventId": "0000001756147154000#ffa53e54",
                    "eventTimestamp": datetime.datetime(2025, 8, 25, 15, 12, 34, tzinfo=tzlocal()),
                    "payload": [
                        {
                            "conversational": {
                                "content": {"text": "What is the weather?"},
                                "role": "USER",
                            }
                        }
                    ],
                    "branch": {"name": "main"},
                }
                ```

        Returns:
            list[SessionMessage]: list of SessionMessage objects.
        """
        messages = []
        for event in events:
            content_blocks: list[ContentBlock] = []
            role = None

            for payload_item in event.get("payload", []):
                if "conversational" in payload_item:
                    conv = payload_item["conversational"]
                    role = conv.get("role", "").lower()
                    if role is None:
                        raise ValueError("Role not set for conversational item")

                    content = conv.get("content", {})
                    if "text" in content:
                        content_blocks.append({"text": content["text"]})
                elif "blob" in payload_item:
                    try:
                        blob_data = json.loads(payload_item["blob"])
                        if isinstance(blob_data, (tuple, list)) and len(blob_data) == 2:
                            text, blob_role = blob_data
                            role = blob_role.lower()
                            content_blocks.append({"text": text})
                    except (json.JSONDecodeError, ValueError):
                        logger.error(f'Failed to parse blob content: {payload_item["blob"]}')

            if content_blocks and role:
                message: Message = {
                    "content": content_blocks,
                    "role": role,
                }
                session_message = SessionMessage(
                    message=message,
                    message_id=event.get("eventId"),
                    created_at=event.get("eventTimestamp", datetime.utcnow().isoformat()),
                    updated_at=event.get("eventTimestamp", datetime.utcnow().isoformat()),
                )
                messages.append(session_message)

        return messages

    @staticmethod
    def total_length(message: tuple[str, str]) -> int:
        """Calculate total length of a message tuple."""
        return sum(len(text) for text in message)

    @staticmethod
    def exceeds_conversational_limit(message: tuple[str, str]) -> bool:
        """Check if message exceeds conversational size limit."""
        return BedrockConverter.total_length(message) >= CONVERSATIONAL_MAX_SIZE
