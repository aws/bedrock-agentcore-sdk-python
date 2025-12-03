"""Tests for Strands-specific converter."""

from unittest.mock import Mock

import pytest

from bedrock_agentcore.evaluation.span_to_adot_serializer import convert_strands_to_adot
from bedrock_agentcore.evaluation.span_to_adot_serializer.strands_converter import (
    StrandsEventParser,
    StrandsToADOTConverter,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_span_context():
    """Create a mock span context."""
    context = Mock()
    context.trace_id = 0x1234567890ABCDEF1234567890ABCDEF
    context.span_id = 0x1234567890ABCDEF
    context.trace_flags = 1
    return context


@pytest.fixture
def mock_resource():
    """Create a mock resource."""
    resource = Mock()
    resource.attributes = {"service.name": "test-service"}
    return resource


@pytest.fixture
def mock_instrumentation_scope():
    """Create a mock instrumentation scope."""
    scope = Mock()
    scope.name = "strands.agent"
    scope.version = "1.0.0"
    return scope


@pytest.fixture
def mock_status():
    """Create a mock status."""
    status = Mock()
    status.status_code = Mock()
    status.status_code.__str__ = Mock(return_value="StatusCode.OK")
    return status


@pytest.fixture
def mock_span(mock_span_context, mock_resource, mock_instrumentation_scope, mock_status):
    """Create a mock OTel span."""
    span = Mock()
    span.context = mock_span_context
    span.resource = mock_resource
    span.instrumentation_scope = mock_instrumentation_scope
    span.status = mock_status
    span.parent = None
    span.name = "test-span"
    span.start_time = 1000000000
    span.end_time = 2000000000
    span.kind = Mock()
    span.kind.__str__ = Mock(return_value="SpanKind.INTERNAL")
    span.attributes = {"gen_ai.operation.name": "chat"}
    span.events = []
    return span


@pytest.fixture
def mock_event():
    """Create a mock span event."""

    def _create_event(name, attributes):
        event = Mock()
        event.name = name
        event.attributes = attributes
        return event

    return _create_event


# ==============================================================================
# Strands Event Parser Tests
# ==============================================================================


class TestStrandsEventParser:
    """Test StrandsEventParser class."""

    def test_extract_conversation_turn(self, mock_event):
        """Test extracting conversation turn from events."""
        events = [
            mock_event("gen_ai.user.message", {"content": "Hello"}),
            mock_event("gen_ai.choice", {"message": "Hi there", "finish_reason": "stop"}),
        ]

        turn = StrandsEventParser.extract_conversation_turn(events)

        assert turn is not None
        assert turn.user_message == "Hello"
        assert len(turn.assistant_messages) == 1
        assert turn.assistant_messages[0]["content"]["message"] == "Hi there"
        assert turn.assistant_messages[0]["content"]["finish_reason"] == "stop"

    def test_extract_conversation_turn_with_tool_result(self, mock_event):
        """Test extracting conversation with tool results."""
        events = [
            mock_event("gen_ai.user.message", {"content": "Calculate 2+2"}),
            mock_event("gen_ai.choice", {"message": "4", "tool.result": "4"}),
        ]

        turn = StrandsEventParser.extract_conversation_turn(events)

        assert turn is not None
        assert len(turn.tool_results) == 1
        assert turn.tool_results[0] == "4"

    def test_extract_conversation_turn_assistant_message(self, mock_event):
        """Test extracting assistant message event."""
        events = [
            mock_event("gen_ai.user.message", {"content": "Hello"}),
            mock_event("gen_ai.assistant.message", {"content": "Hi there"}),
        ]

        turn = StrandsEventParser.extract_conversation_turn(events)

        assert turn is not None
        assert turn.assistant_messages[0]["content"]["content"] == "Hi there"

    def test_extract_conversation_turn_tool_message(self, mock_event):
        """Test extracting tool message as tool result."""
        events = [
            mock_event("gen_ai.user.message", {"content": "Hello"}),
            mock_event("gen_ai.choice", {"message": "Using tool"}),
            mock_event("gen_ai.tool.message", {"content": "tool output"}),
        ]

        turn = StrandsEventParser.extract_conversation_turn(events)

        assert turn is not None
        assert "tool output" in turn.tool_results

    def test_extract_conversation_turn_no_user_message(self, mock_event):
        """Test returns None when no user message."""
        events = [
            mock_event("gen_ai.choice", {"message": "Hi"}),
        ]

        turn = StrandsEventParser.extract_conversation_turn(events)

        assert turn is None

    def test_extract_conversation_turn_no_assistant_message(self, mock_event):
        """Test returns None when no assistant message."""
        events = [
            mock_event("gen_ai.user.message", {"content": "Hello"}),
        ]

        turn = StrandsEventParser.extract_conversation_turn(events)

        assert turn is None

    def test_extract_tool_execution(self, mock_event):
        """Test extracting tool execution from events."""
        events = [
            mock_event("gen_ai.tool.message", {"content": '{"x": 1}', "id": "tool-1"}),
            mock_event("gen_ai.choice", {"message": "result"}),
        ]

        tool = StrandsEventParser.extract_tool_execution(events)

        assert tool is not None
        assert tool.tool_input == '{"x": 1}'
        assert tool.tool_output == "result"
        assert tool.tool_id == "tool-1"

    def test_extract_tool_execution_id_from_choice(self, mock_event):
        """Test tool ID extracted from choice event if not in tool message."""
        events = [
            mock_event("gen_ai.tool.message", {"content": '{"x": 1}'}),
            mock_event("gen_ai.choice", {"message": "result", "id": "tool-2"}),
        ]

        tool = StrandsEventParser.extract_tool_execution(events)

        assert tool.tool_id == "tool-2"

    def test_extract_tool_execution_no_input(self, mock_event):
        """Test returns None when no tool input."""
        events = [
            mock_event("gen_ai.choice", {"message": "result"}),
        ]

        tool = StrandsEventParser.extract_tool_execution(events)

        assert tool is None

    def test_extract_tool_execution_no_output(self, mock_event):
        """Test returns None when no tool output."""
        events = [
            mock_event("gen_ai.tool.message", {"content": '{"x": 1}'}),
        ]

        tool = StrandsEventParser.extract_tool_execution(events)

        assert tool is None


# ==============================================================================
# Strands Converter Tests
# ==============================================================================


class TestStrandsToADOTConverter:
    """Test StrandsToADOTConverter class."""

    def test_convert_span_basic(self, mock_span):
        """Test converting a basic span."""
        converter = StrandsToADOTConverter()

        docs = converter.convert_span(mock_span)

        assert len(docs) == 1  # Just span document, no events
        assert docs[0]["name"] == "test-span"

    def test_convert_span_with_conversation(self, mock_span, mock_event):
        """Test converting span with conversation events."""
        mock_span.events = [
            mock_event("gen_ai.user.message", {"content": "Hello"}),
            mock_event("gen_ai.choice", {"message": "Hi"}),
        ]
        converter = StrandsToADOTConverter()

        docs = converter.convert_span(mock_span)

        assert len(docs) == 2  # Span + conversation log
        assert docs[1]["body"]["input"]["messages"][0]["content"]["content"] == "Hello"

    def test_convert_span_with_tool_execution(self, mock_span, mock_event):
        """Test converting span with tool execution."""
        mock_span.attributes = {"gen_ai.operation.name": "execute_tool"}
        mock_span.events = [
            mock_event("gen_ai.tool.message", {"content": '{"x": 1}', "id": "t1"}),
            mock_event("gen_ai.choice", {"message": "result"}),
        ]
        converter = StrandsToADOTConverter()

        docs = converter.convert_span(mock_span)

        assert len(docs) == 2  # Span + tool log
        assert docs[1]["body"]["input"]["messages"][0]["content"]["content"] == '{"x": 1}'

    def test_convert_span_error_handling(self):
        """Test converter handles errors gracefully."""
        bad_span = Mock()
        bad_span.context = None
        bad_span.name = "bad-span"

        converter = StrandsToADOTConverter()
        docs = converter.convert_span(bad_span)

        assert docs == []  # Returns empty list on error

    def test_convert_multiple_spans(self, mock_span):
        """Test converting multiple spans."""
        converter = StrandsToADOTConverter()

        docs = converter.convert([mock_span, mock_span])

        assert len(docs) == 2


# ==============================================================================
# Public API Tests
# ==============================================================================


class TestConvertStrandsToAdot:
    """Test convert_strands_to_adot function."""

    def test_empty_spans(self):
        """Test with empty span list."""
        result = convert_strands_to_adot([])

        assert result == []

    def test_basic_conversion(self, mock_span):
        """Test basic span conversion."""
        result = convert_strands_to_adot([mock_span])

        assert len(result) == 1
        assert result[0]["name"] == "test-span"

    def test_full_conversion(self, mock_span, mock_event):
        """Test full conversion with events."""
        mock_span.events = [
            mock_event("gen_ai.user.message", {"content": "Hello"}),
            mock_event("gen_ai.choice", {"message": "Hi"}),
        ]

        result = convert_strands_to_adot([mock_span])

        assert len(result) == 2
