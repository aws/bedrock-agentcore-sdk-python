"""Bedrock AgentCore Memory conversion utilities."""

import json
import logging
from typing import Any, Tuple

from strands.types.session import SessionMessage

logger = logging.getLogger(__name__)

CONVERSATIONAL_MAX_SIZE = 9000


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
        for event in events:
            for payload_item in event.get("payload", []):
                if "conversational" in payload_item:
                    conv = payload_item["conversational"]
                    session_msg = SessionMessage.from_dict(json.loads(conv["content"]["text"]))
                    session_msg.message = AgentCoreMemoryConverter._filter_empty_text(session_msg.message)
                    if session_msg.message.get("content"):
                        messages.append(session_msg)
                elif "blob" in payload_item:
                    try:
                        blob_data = json.loads(payload_item["blob"])
                        if isinstance(blob_data, (tuple, list)) and len(blob_data) == 2:
                            try:
                                session_msg = SessionMessage.from_dict(json.loads(blob_data[0]))
                                session_msg.message = AgentCoreMemoryConverter._filter_empty_text(session_msg.message)
                                if session_msg.message.get("content"):
                                    messages.append(session_msg)
                            except (json.JSONDecodeError, ValueError):
                                logger.error("This is not a SessionMessage but just a blob message. Ignoring")
                    except (json.JSONDecodeError, ValueError):
                        logger.error("Failed to parse blob content: %s", payload_item)
        return list(reversed(messages))

    @staticmethod
    def total_length(message: tuple[str, str]) -> int:
        """Calculate total length of a message tuple."""
        return sum(len(text) for text in message)

    @staticmethod
    def exceeds_conversational_limit(message: tuple[str, str]) -> bool:
        """Check if message exceeds conversational size limit."""
        return AgentCoreMemoryConverter.total_length(message) >= CONVERSATIONAL_MAX_SIZE
