"""Framework-agnostic domain models and ADOT document builders.

This module contains the reusable components for converting telemetry data to ADOT format:
- Domain Models (Layer 1): Clean data structures representing telemetry concepts
- Base Extraction (Layer 2): Standard OTel span field extraction
- ADOT Transformation (Layer 3): Convert domain models to ADOT format

These components are framework-agnostic and can be reused across different
telemetry frameworks (Strands, LangGraph, etc.).
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ==============================================================================
# Domain Models - Framework-agnostic intermediate representation
# ==============================================================================


@dataclass
class SpanMetadata:
    """Core span identification and timing."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: int
    end_time: int
    duration: int
    kind: str
    flags: int
    status_code: str


@dataclass
class ResourceInfo:
    """Span resource and scope information."""

    resource_attributes: Dict[str, Any]
    scope_name: str
    scope_version: str


class ConversationTurn:
    """A single user-assistant conversation turn, with prior history preserved.

    ``input_messages`` holds every input event (``gen_ai.user.message`` and
    ``gen_ai.assistant.message``) in the order observed on the span, so
    downstream consumers can reconstruct the actual conversation flow.
    ``assistant_messages`` holds only the current turn's output, derived from
    ``gen_ai.choice`` events. ``tool_results`` holds any tool outputs observed
    on the span.

    The ``user_message`` attribute is retained as a backwards-compatible alias
    returning the most recent user turn's content; new code should iterate
    ``input_messages`` filtered by role.
    """

    def __init__(
        self,
        user_message: Optional[str] = None,
        assistant_messages: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[str]] = None,
        input_messages: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize a conversation turn.

        ``input_messages`` is the chronological list of input events
        (``{"role": "user"|"assistant", "content": {...}}``). For backwards
        compatibility, callers may instead pass a legacy ``user_message``
        scalar, which is converted into a single-entry ``input_messages``
        list. Supplying both is an error.
        """
        if input_messages is not None and user_message is not None:
            raise ValueError("Provide either input_messages or user_message, not both")
        if input_messages is not None:
            self.input_messages = list(input_messages)
        elif user_message is not None:
            self.input_messages = [{"content": {"content": user_message}, "role": "user"}]
        else:
            self.input_messages = []
        self.assistant_messages = list(assistant_messages or [])
        self.tool_results = list(tool_results or [])

    def __repr__(self) -> str:
        """Return a debug representation listing every stored field."""
        return (
            f"ConversationTurn(input_messages={self.input_messages!r}, "
            f"assistant_messages={self.assistant_messages!r}, "
            f"tool_results={self.tool_results!r})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare turns by the full set of instance fields."""
        if not isinstance(other, ConversationTurn):
            return NotImplemented
        return (
            self.input_messages == other.input_messages
            and self.assistant_messages == other.assistant_messages
            and self.tool_results == other.tool_results
        )

    @property
    def user_message(self) -> str:
        """Return the most recent user turn's content (backwards-compatible alias).

        Deprecated: iterate ``input_messages`` to access the full conversation
        history. This alias returns only the last user entry.
        """
        user_entries = [m for m in self.input_messages if m.get("role") == "user"]
        if not user_entries:
            return ""
        if len(user_entries) > 1 or any(m.get("role") == "assistant" for m in self.input_messages):
            warnings.warn(
                "ConversationTurn.user_message drops prior user turns and history "
                "when the span carries multiple input events. Iterate "
                "ConversationTurn.input_messages for the full conversation.",
                DeprecationWarning,
                stacklevel=2,
            )
        last = user_entries[-1]
        content = last.get("content")
        if isinstance(content, dict):
            inner = content.get("content", "")
            return inner if isinstance(inner, str) else ""
        return content if isinstance(content, str) else ""


@dataclass
class ToolExecution:
    """A single tool execution event."""

    tool_input: str
    tool_output: str
    tool_id: str


# ==============================================================================
# Base Extraction - Parse standard OTel span fields
# ==============================================================================


class SpanParser:
    """Extract structured data from raw OTel spans.

    This parser extracts standard OpenTelemetry span fields that are
    common across all frameworks.
    """

    @staticmethod
    def extract_metadata(span) -> SpanMetadata:
        """Extract core span metadata."""
        if not hasattr(span, "context") or not span.context:
            raise ValueError(f"Span '{getattr(span, 'name', 'unknown')}' missing required context")

        return SpanMetadata(
            trace_id=format(span.context.trace_id, "032x"),
            span_id=format(span.context.span_id, "016x"),
            parent_span_id=format(span.parent.span_id, "016x") if span.parent else None,
            name=span.name or "",
            start_time=span.start_time,
            end_time=span.end_time,
            duration=span.end_time - span.start_time,
            kind=str(span.kind).split(".")[-1],
            flags=span.context.trace_flags,
            status_code=str(span.status.status_code).split(".")[-1],
        )

    @staticmethod
    def extract_resource_info(span) -> ResourceInfo:
        """Extract resource and scope information."""
        resource_attrs = {}
        if hasattr(span, "resource") and span.resource and hasattr(span.resource, "attributes"):
            resource_attrs = dict(span.resource.attributes)

        scope_name = ""
        scope_version = ""
        if hasattr(span, "instrumentation_scope") and span.instrumentation_scope:
            scope_name = span.instrumentation_scope.name or ""
            scope_version = span.instrumentation_scope.version or ""

        return ResourceInfo(
            resource_attributes=resource_attrs,
            scope_name=scope_name,
            scope_version=scope_version,
        )

    @staticmethod
    def get_span_attributes(span) -> Dict[str, Any]:
        """Safely extract span attributes."""
        return dict(span.attributes) if hasattr(span, "attributes") and span.attributes else {}


# ==============================================================================
# ADOT Document Builders - Transform to ADOT format
# ==============================================================================


class ADOTDocumentBuilder:
    """Build ADOT-formatted documents from structured domain models.

    This builder is framework-agnostic and only works with the domain models,
    not with raw telemetry data.
    """

    LOG_SEVERITY_INFO = 9
    LOG_FLAGS_SAMPLED = 1
    OBSERVED_TIME_OFFSET_NS = 100_000

    @staticmethod
    def build_span_document(
        metadata: SpanMetadata,
        resource_info: ResourceInfo,
        attributes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build ADOT span document."""
        return {
            "resource": {"attributes": resource_info.resource_attributes},
            "scope": {
                "name": resource_info.scope_name,
                "version": resource_info.scope_version,
            },
            "traceId": metadata.trace_id,
            "spanId": metadata.span_id,
            "parentSpanId": metadata.parent_span_id,
            "flags": metadata.flags,
            "name": metadata.name,
            "kind": metadata.kind,
            "startTimeUnixNano": metadata.start_time,
            "endTimeUnixNano": metadata.end_time,
            "durationNano": metadata.duration,
            "attributes": attributes,
            "status": {"code": metadata.status_code},
        }

    @classmethod
    def _build_log_record_base(
        cls,
        metadata: SpanMetadata,
        resource_info: ResourceInfo,
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build base ADOT log record structure shared by all log types."""
        return {
            "resource": {"attributes": resource_info.resource_attributes},
            "scope": {"name": resource_info.scope_name},
            "timeUnixNano": metadata.end_time,
            "observedTimeUnixNano": metadata.end_time + cls.OBSERVED_TIME_OFFSET_NS,
            "severityNumber": cls.LOG_SEVERITY_INFO,
            "severityText": "",
            "body": body,
            "attributes": {"event.name": resource_info.scope_name},
            "flags": cls.LOG_FLAGS_SAMPLED,
            "traceId": metadata.trace_id,
            "spanId": metadata.span_id,
        }

    @classmethod
    def build_conversation_log_record(
        cls,
        conversation: ConversationTurn,
        metadata: SpanMetadata,
        resource_info: ResourceInfo,
    ) -> Dict[str, Any]:
        """Build ADOT log record for conversation turn.

        ``body.input.messages`` carries ``ConversationTurn.input_messages`` in
        event arrival order, so downstream consumers reconstruct the exact
        conversation the model received (user and prior assistant turns
        interleaved). ``body.output.messages`` carries only the current turn's
        output.
        """
        output_messages = []
        for i, msg in enumerate(conversation.assistant_messages):
            output_msg = msg.copy()
            if i == 0 and conversation.tool_results:
                if "content" not in output_msg:
                    output_msg["content"] = {}
                output_msg["content"]["tool.result"] = conversation.tool_results[0]
            output_messages.append(output_msg)

        for tool_result in conversation.tool_results:
            output_messages.append({"content": tool_result, "role": "assistant"})

        body = {
            "output": {"messages": output_messages},
            "input": {"messages": list(conversation.input_messages)},
        }

        return cls._build_log_record_base(metadata, resource_info, body)

    @classmethod
    def build_tool_log_record(
        cls,
        tool_exec: ToolExecution,
        metadata: SpanMetadata,
        resource_info: ResourceInfo,
    ) -> Dict[str, Any]:
        """Build ADOT log record for tool execution."""
        body = {
            "output": {
                "messages": [
                    {"content": {"message": tool_exec.tool_output, "id": tool_exec.tool_id}, "role": "assistant"}
                ]
            },
            "input": {
                "messages": [
                    {
                        "content": {"content": tool_exec.tool_input, "role": "tool", "id": tool_exec.tool_id},
                        "role": "tool",
                    }
                ]
            },
        }

        return cls._build_log_record_base(metadata, resource_info, body)
