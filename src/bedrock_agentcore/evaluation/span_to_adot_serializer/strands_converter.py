"""Strands-specific OTel span to ADOT converter.

This module contains the Strands-specific implementation for converting
OpenTelemetry spans to ADOT format:
- Event Extraction (Layer 2): Parse Strands-specific span events
- Orchestration (Layer 4): Coordinate the conversion pipeline for Strands

To add support for other frameworks (e.g., LangGraph), create a similar
converter module that implements framework-specific extractors and orchestration.
"""

import logging
from typing import Any, Dict, List, Optional

from .adot_models import (
    ADOTDocumentBuilder,
    ConversationTurn,
    SpanParser,
    ToolExecution,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Strands Event Extraction - Parse Strands-specific span events
# ==============================================================================


class StrandsEventParser:
    """Extract structured data from Strands-specific span events."""

    EVENT_USER_MESSAGE = "gen_ai.user.message"
    EVENT_CHOICE = "gen_ai.choice"
    EVENT_ASSISTANT_MESSAGE = "gen_ai.assistant.message"
    EVENT_TOOL_MESSAGE = "gen_ai.tool.message"

    @classmethod
    def extract_conversation_turn(cls, events: List[Any]) -> Optional[ConversationTurn]:
        """Extract conversation turn from Strands span events."""
        user_message = None
        assistant_messages = []
        tool_results = []

        for event in events:
            event_attrs = dict(event.attributes) if hasattr(event, "attributes") and event.attributes else {}

            match event.name:
                case cls.EVENT_USER_MESSAGE:
                    user_message = event_attrs.get("content", "")

                case cls.EVENT_CHOICE:
                    message = event_attrs.get("message", "")
                    finish_reason = event_attrs.get("finish_reason", "")
                    tool_result = event_attrs.get("tool.result", "")

                    if message:
                        msg_content = {"message": message}
                        if finish_reason:
                            msg_content["finish_reason"] = finish_reason
                        assistant_messages.append({"content": msg_content, "role": "assistant"})

                    if tool_result:
                        tool_results.append(tool_result)

                case cls.EVENT_ASSISTANT_MESSAGE:
                    content = event_attrs.get("content", "")
                    if content:
                        assistant_messages.append({"content": {"content": content}, "role": "assistant"})

                case cls.EVENT_TOOL_MESSAGE:
                    content = event_attrs.get("content", "")
                    if content:
                        tool_results.append(content)

        if user_message and assistant_messages:
            return ConversationTurn(
                user_message=user_message,
                assistant_messages=assistant_messages,
                tool_results=tool_results,
            )

        return None

    @classmethod
    def extract_tool_execution(cls, events: List[Any]) -> Optional[ToolExecution]:
        """Extract tool execution from Strands span events."""
        tool_input = ""
        tool_output = ""
        tool_id = ""

        for event in events:
            event_attrs = dict(event.attributes) if hasattr(event, "attributes") and event.attributes else {}

            match event.name:
                case cls.EVENT_TOOL_MESSAGE:
                    tool_input = event_attrs.get("content", "{}")
                    tool_id = event_attrs.get("id", "")

                case cls.EVENT_CHOICE:
                    tool_output = event_attrs.get("message", "")
                    if not tool_id:
                        tool_id = event_attrs.get("id", "")

        if tool_input and tool_output:
            return ToolExecution(
                tool_input=tool_input,
                tool_output=tool_output,
                tool_id=tool_id,
            )

        return None


# ==============================================================================
# Strands Converter - Orchestrates the conversion pipeline
# ==============================================================================


class StrandsToADOTConverter:
    """Convert Strands OTel spans to ADOT format."""

    def __init__(self):
        """Initialize converter with parsers and builder."""
        self.span_parser = SpanParser()
        self.event_parser = StrandsEventParser()
        self.doc_builder = ADOTDocumentBuilder()

    def convert_span(self, span) -> List[Dict[str, Any]]:
        """Convert a single span to ADOT documents."""
        documents = []

        try:
            metadata = self.span_parser.extract_metadata(span)
            resource_info = self.span_parser.extract_resource_info(span)
            attributes = self.span_parser.get_span_attributes(span)

            span_doc = self.doc_builder.build_span_document(metadata, resource_info, attributes)
            documents.append(span_doc)

            if hasattr(span, "events") and span.events:
                conversation = self.event_parser.extract_conversation_turn(span.events)
                if conversation:
                    conv_log = self.doc_builder.build_conversation_log_record(conversation, metadata, resource_info)
                    documents.append(conv_log)

                if attributes.get("gen_ai.operation.name") == "execute_tool":
                    tool_exec = self.event_parser.extract_tool_execution(span.events)
                    if tool_exec:
                        tool_log = self.doc_builder.build_tool_log_record(tool_exec, metadata, resource_info)
                        documents.append(tool_log)

        except Exception as e:
            logger.warning(
                "Failed to convert span '%s': %s",
                getattr(span, "name", "unknown"),
                e,
                exc_info=True,
            )

        return documents

    def convert(self, raw_spans: List[Any]) -> List[Dict[str, Any]]:
        """Convert list of Strands OTel spans to ADOT documents."""
        documents = []
        for span in raw_spans:
            span_documents = self.convert_span(span)
            documents.extend(span_documents)
        return documents


# ==============================================================================
# Public API
# ==============================================================================


def convert_strands_to_adot(raw_spans: List[Any]) -> List[Dict[str, Any]]:
    """Convert Strands OTel spans to ADOT format for AgentCore evaluation.

    Args:
        raw_spans: List of OpenTelemetry Span objects from Strands agent

    Returns:
        List of ADOT documents (spans and log records)

    Example:
        >>> from strands_evals.telemetry import StrandsEvalsTelemetry
        >>> telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
        >>> # ... run agent ...
        >>> raw_spans = telemetry.in_memory_exporter.get_finished_spans()
        >>> adot_docs = convert_strands_to_adot(raw_spans)
    """
    converter = StrandsToADOTConverter()
    return converter.convert(raw_spans)
