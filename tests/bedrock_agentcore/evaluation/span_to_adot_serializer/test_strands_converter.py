"""Tests for Strands-specific converter."""

from unittest.mock import Mock

from bedrock_agentcore.evaluation.span_to_adot_serializer import convert_strands_to_adot
from bedrock_agentcore.evaluation.span_to_adot_serializer.strands_converter import (
    StrandsEventParser,
    StrandsToADOTConverter,
)

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
        assert turn.input_messages == [{"content": {"content": "Hello"}, "role": "user"}]
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

    def test_extract_conversation_turn_assistant_message_only_returns_none(self, mock_event):
        """No gen_ai.choice means no current-turn output, so no ConversationTurn is emitted."""
        events = [
            mock_event("gen_ai.user.message", {"content": "Hello"}),
            mock_event("gen_ai.assistant.message", {"content": "Hi there"}),
        ]

        turn = StrandsEventParser.extract_conversation_turn(events)

        assert turn is None

    def test_extract_conversation_turn_tool_results_only_returns_none(self, mock_event):
        """User + tool_result with no choice and no history must not emit a record."""
        events = [
            mock_event("gen_ai.user.message", {"content": "Hello"}),
            mock_event("gen_ai.tool.message", {"content": "tool output"}),
        ]

        turn = StrandsEventParser.extract_conversation_turn(events)

        assert turn is None

    def test_extract_conversation_turn_preserves_all_user_messages(self, mock_event):
        """Multiple gen_ai.user.message events are all preserved, not deduped."""
        events = [
            mock_event("gen_ai.user.message", {"content": "first"}),
            mock_event("gen_ai.user.message", {"content": "second"}),
            mock_event("gen_ai.user.message", {"content": "third"}),
            mock_event("gen_ai.choice", {"message": "ok"}),
        ]

        turn = StrandsEventParser.extract_conversation_turn(events)

        assert turn is not None
        user_contents = [m["content"]["content"] for m in turn.input_messages if m["role"] == "user"]
        assert user_contents == ["first", "second", "third"]

    def test_extract_conversation_turn_preserves_chronological_order(self, mock_event):
        """input_messages must interleave user and prior assistant turns in event arrival order."""
        events = [
            mock_event("gen_ai.user.message", {"content": "u1"}),
            mock_event("gen_ai.assistant.message", {"content": "prior-assistant-1"}),
            mock_event("gen_ai.user.message", {"content": "u2"}),
            mock_event("gen_ai.assistant.message", {"content": "prior-assistant-2"}),
            mock_event("gen_ai.user.message", {"content": "u3"}),
            mock_event("gen_ai.choice", {"message": "new-model-output", "finish_reason": "end_turn"}),
        ]

        turn = StrandsEventParser.extract_conversation_turn(events)

        assert turn is not None
        assert [m["role"] for m in turn.input_messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert [m["content"]["content"] for m in turn.input_messages] == [
            "u1",
            "prior-assistant-1",
            "u2",
            "prior-assistant-2",
            "u3",
        ]
        assert len(turn.assistant_messages) == 1
        assert turn.assistant_messages[0]["content"]["message"] == "new-model-output"

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
