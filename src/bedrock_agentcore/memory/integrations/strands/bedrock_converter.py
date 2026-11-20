"""Bedrock AgentCore Memory conversion utilities."""

import json
import logging
from typing import Any, Tuple

from strands.types.session import SessionMessage

logger = logging.getLogger(__name__)

# Bedrock AgentCore Data Plane conversational payload text max is 100000 chars.
# Ref: https://docs.aws.amazon.com/cli/latest/reference/bedrock-agentcore/create-event.html
CONVERSATIONAL_MAX_SIZE = 100000


def _is_message(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("role") not in ("user", "assistant"):
        return False
    content = value.get("content")
    return isinstance(content, list) and all(
        isinstance(block, dict) and isinstance(block.get("text", ""), str) for block in content
    )


class AgentCoreMemoryConverter:
    """Handles conversion between Strands and Bedrock AgentCore Memory formats."""

    @staticmethod
    def _filter_empty_text(message: dict) -> dict:
        """The Bedrock Converse API can't take empty text as input. So we need to filter out empty text."""
        content = message.get("content", [])
        filtered_content = [item for item in content if "text" not in item or item.get("text", "").strip() != ""]
        return {**message, "content": filtered_content}

    @staticmethod
    def message_to_payload(session_message: SessionMessage) -> list[Tuple[str, str]]:
        """Convert a SessionMessage to Bedrock AgentCore Memory message format.

        Args:
            session_message (SessionMessage): The session message to convert.

        Returns:
            list[Tuple[str, str]]: list of (text, role) tuples for Bedrock AgentCore Memory.
                Returns empty list if message has no content after filtering.
        """
        # First convert to dict (which encodes bytes to base64),
        # then filter empty text on the encoded version
        session_dict = session_message.to_dict()
        filtered_message = AgentCoreMemoryConverter._filter_empty_text(session_dict["message"])
        if not filtered_message.get("content"):
            logger.debug("Skipping message with no content after filtering empty text")
            return []
        session_dict["message"] = filtered_message
        return [(json.dumps(session_dict), filtered_message["role"])]

    @staticmethod
    def _parse_session_message(text: Any) -> SessionMessage | None:
        try:
            data = json.loads(text)
            session_msg = SessionMessage.from_dict(data) if isinstance(data, dict) else None
        except (TypeError, ValueError, RecursionError):
            return None
        if session_msg is None or not _is_message(session_msg.message):
            return None
        if session_msg.redact_message is not None and not _is_message(session_msg.redact_message):
            return None
        return session_msg

    @staticmethod
    def events_to_messages(events: list[dict[str, Any]]) -> list[SessionMessage]:
        """Convert Bedrock AgentCore Memory events to SessionMessages.

        Args:
            events (list[dict[str, Any]]): list of events from Bedrock AgentCore Memory.
                Each individual event looks as follows:
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
        for event in reversed(events):
            for payload_item in event.get("payload", []):
                if "conversational" in payload_item:
                    text = payload_item["conversational"].get("content", {}).get("text")
                elif "blob" in payload_item:
                    try:
                        blob_data = json.loads(payload_item["blob"])
                    except (TypeError, ValueError, RecursionError):
                        logger.error("Failed to parse blob content: %s", payload_item)
                        continue
                    if not (isinstance(blob_data, list) and len(blob_data) == 2):
                        continue
                    text = blob_data[0]
                else:
                    continue

                session_msg = AgentCoreMemoryConverter._parse_session_message(text)
                if session_msg is None:
                    logger.error("Skipping a payload that is not a SessionMessage in event %s", event.get("eventId"))
                    continue
                session_msg.message = AgentCoreMemoryConverter._filter_empty_text(session_msg.message)
                if session_msg.message.get("content"):
                    messages.append(session_msg)
        return messages

    @staticmethod
    def total_length(message: tuple[str, str]) -> int:
        """Calculate total length of a message tuple."""
        return sum(len(text) for text in message)

    @staticmethod
    def exceeds_conversational_limit(message: tuple[str, str]) -> bool:
        """Check if message exceeds conversational size limit."""
        return AgentCoreMemoryConverter.total_length(message) >= CONVERSATIONAL_MAX_SIZE
