"""Shared protocol and utilities for memory converters."""

from typing import Any, Protocol, Tuple

from strands.types.session import SessionMessage

CONVERSATIONAL_MAX_SIZE = 9000


class MemoryConverter(Protocol):
    """Protocol for converting between Strands messages and STM event payloads."""

    @staticmethod
    def message_to_payload(session_message: SessionMessage) -> list[Tuple[str, str]]:
        """Convert SessionMessage to STM event payload format."""
        ...

    @staticmethod
    def events_to_messages(events: list[dict[str, Any]]) -> list[SessionMessage]:
        """Convert STM events to SessionMessages."""
        ...

    @staticmethod
    def exceeds_conversational_limit(message: tuple[str, str]) -> bool:
        """Check if message exceeds conversational payload size limit."""
        ...


def exceeds_conversational_limit(message: tuple[str, str]) -> bool:
    """Check if message exceeds the conversational payload size limit.

    Args:
        message: A (text, role) tuple.

    Returns:
        True if the total length meets or exceeds CONVERSATIONAL_MAX_SIZE.
    """
    return sum(len(text) for text in message) >= CONVERSATIONAL_MAX_SIZE
