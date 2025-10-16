"""Module containing all the model classes."""

from typing import Any, Dict

from .DictWrapper import DictWrapper
from .filters import (
    EventMetadataFilter,
    LeftExpression,
    MetadataKey,
    MetadataValue,
    OperatorType,
    RightExpression,
    StringValue,
)


class ActorSummary(DictWrapper):
    """A class representing an actor summary."""

    def __init__(self, actor_summary: Dict[str, Any]):
        """Initialize an ActorSummary instance.

        Args:
            actor_summary: Dictionary containing actor summary data.
        """
        super().__init__(actor_summary)


class Branch(DictWrapper):
    """A class representing a branch."""

    def __init__(self, data: Dict[str, Any]):
        """Initialize a Branch instance.

        Args:
            data: Dictionary containing branch data.
        """
        super().__init__(data)


class Event(DictWrapper):
    """A class representing an event."""

    def __init__(self, data: Dict[str, Any]):
        """Initialize an Event instance.

        Args:
            data: Dictionary containing event data.
        """
        super().__init__(data)


class EventMessage(DictWrapper):
    """A class representing an event message."""

    def __init__(self, event_message: Dict[str, Any]):
        """Initialize an EventMessage instance.

        Args:
            event_message: Dictionary containing event message data.
        """
        super().__init__(event_message)


class MemoryRecord(DictWrapper):
    """A class representing a memory record."""

    def __init__(self, memory_record: Dict[str, Any]):
        """Initialize a MemoryRecord instance.

        Args:
            memory_record: Dictionary containing memory record data.
        """
        super().__init__(memory_record)


class SessionSummary(DictWrapper):
    """A class representing a session summary."""

    def __init__(self, session_summary: Dict[str, Any]):
        """Initialize a SessionSummary instance.

        Args:
            session_summary: Dictionary containing session summary data.
        """
        super().__init__(session_summary)


__all__ = [
    "DictWrapper",
    "ActorSummary",
    "Branch",
    "Event",
    "EventMessage",
    "MemoryRecord",
    "SessionSummary",
    "StringValue",
    "MetadataValue",
    "MetadataKey",
    "LeftExpression",
    "OperatorType",
    "RightExpression",
    "EventMetadataFilter",
]
