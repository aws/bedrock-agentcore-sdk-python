"""Tests for framework-agnostic ADOT models and builders."""

import warnings
from unittest.mock import Mock

import pytest

from bedrock_agentcore.evaluation.span_to_adot_serializer.adot_models import (
    ADOTDocumentBuilder,
    ConversationTurn,
    ResourceInfo,
    SpanMetadata,
    SpanParser,
    ToolExecution,
)

# ==============================================================================
# Domain Model Tests
# ==============================================================================


class TestSpanMetadata:
    """Test SpanMetadata dataclass."""

    def test_creation(self):
        """Test SpanMetadata creation."""
        metadata = SpanMetadata(
            trace_id="abc123",
            span_id="def456",
            parent_span_id="parent123",
            name="test",
            start_time=1000,
            end_time=2000,
            duration=1000,
            kind="INTERNAL",
            flags=1,
            status_code="OK",
        )
        assert metadata.trace_id == "abc123"
        assert metadata.span_id == "def456"
        assert metadata.parent_span_id == "parent123"
        assert metadata.status_code == "OK"

    def test_optional_parent(self):
        """Test SpanMetadata with no parent."""
        metadata = SpanMetadata(
            trace_id="abc",
            span_id="def",
            parent_span_id=None,
            name="test",
            start_time=0,
            end_time=0,
            duration=0,
            kind="INTERNAL",
            flags=0,
            status_code="UNSET",
        )
        assert metadata.parent_span_id is None


class TestResourceInfo:
    """Test ResourceInfo dataclass."""

    def test_creation(self):
        """Test ResourceInfo creation."""
        info = ResourceInfo(
            resource_attributes={"service.name": "test"},
            scope_name="test.scope",
            scope_version="1.0.0",
        )
        assert info.resource_attributes == {"service.name": "test"}
        assert info.scope_name == "test.scope"
        assert info.scope_version == "1.0.0"


class TestConversationTurn:
    """Test ConversationTurn class."""

    def test_creation_legacy_scalar(self):
        """Legacy ``user_message`` scalar becomes a single-entry input_messages list."""
        turn = ConversationTurn(
            user_message="Hello",
            assistant_messages=[{"content": {"message": "Hi"}, "role": "assistant"}],
            tool_results=["result1"],
        )
        assert turn.user_message == "Hello"
        assert turn.input_messages == [{"content": {"content": "Hello"}, "role": "user"}]
        assert len(turn.assistant_messages) == 1
        assert len(turn.tool_results) == 1

    def test_creation_with_input_messages(self):
        """ConversationTurn accepts a chronological input_messages list."""
        input_msgs = [
            {"content": {"content": "first"}, "role": "user"},
            {"content": {"content": "prior"}, "role": "assistant"},
            {"content": {"content": "second"}, "role": "user"},
        ]
        turn = ConversationTurn(
            input_messages=input_msgs,
            assistant_messages=[{"content": {"message": "ok"}, "role": "assistant"}],
        )
        assert turn.input_messages == input_msgs

    def test_user_message_alias_returns_latest_user_entry(self):
        """user_message returns the last user entry and warns when history is present."""
        turn = ConversationTurn(
            input_messages=[
                {"content": {"content": "a"}, "role": "user"},
                {"content": {"content": "prior"}, "role": "assistant"},
                {"content": {"content": "b"}, "role": "user"},
            ],
            assistant_messages=[{}],
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert turn.user_message == "b"
            assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_user_message_and_input_messages_rejected(self):
        """Supplying both user_message and input_messages is an error."""
        with pytest.raises(ValueError):
            ConversationTurn(
                user_message="x",
                input_messages=[{"content": {"content": "y"}, "role": "user"}],
            )

    def test_equality_compares_full_instance_state(self):
        """__eq__ compares input_messages, assistant_messages, and tool_results."""
        a = ConversationTurn(
            input_messages=[{"content": {"content": "u"}, "role": "user"}],
            assistant_messages=[{"content": {"message": "out"}, "role": "assistant"}],
            tool_results=["t"],
        )
        b = ConversationTurn(
            input_messages=[{"content": {"content": "u"}, "role": "user"}],
            assistant_messages=[{"content": {"message": "out"}, "role": "assistant"}],
            tool_results=["t"],
        )
        c = ConversationTurn(
            input_messages=[{"content": {"content": "different"}, "role": "user"}],
            assistant_messages=[{"content": {"message": "out"}, "role": "assistant"}],
            tool_results=["t"],
        )
        assert a == b
        assert a != c


class TestToolExecution:
    """Test ToolExecution dataclass."""

    def test_creation(self):
        """Test ToolExecution creation."""
        tool = ToolExecution(
            tool_input='{"arg": "value"}',
            tool_output="result",
            tool_id="tool-123",
        )
        assert tool.tool_input == '{"arg": "value"}'
        assert tool.tool_output == "result"
        assert tool.tool_id == "tool-123"


# ==============================================================================
# Base Extraction Tests
# ==============================================================================


class TestSpanParser:
    """Test SpanParser class."""

    def test_extract_metadata(self, mock_span):
        """Test extracting metadata from span."""
        metadata = SpanParser.extract_metadata(mock_span)

        assert metadata.trace_id == "1234567890abcdef1234567890abcdef"
        assert metadata.span_id == "1234567890abcdef"
        assert metadata.parent_span_id is None
        assert metadata.name == "test-span"
        assert metadata.start_time == 1000000000
        assert metadata.end_time == 2000000000
        assert metadata.duration == 1000000000
        assert metadata.kind == "INTERNAL"
        assert metadata.flags == 1

    def test_extract_metadata_with_parent(self, mock_span):
        """Test extracting metadata from span with parent."""
        parent = Mock()
        parent.span_id = 0xFEDCBA0987654321
        mock_span.parent = parent

        metadata = SpanParser.extract_metadata(mock_span)

        assert metadata.parent_span_id == "fedcba0987654321"

    def test_extract_metadata_missing_context(self):
        """Test extracting metadata from span without context."""
        span = Mock()
        span.context = None
        span.name = "bad-span"

        with pytest.raises(ValueError, match="missing required context"):
            SpanParser.extract_metadata(span)

    def test_extract_resource_info(self, mock_span):
        """Test extracting resource info from span."""
        info = SpanParser.extract_resource_info(mock_span)

        assert info.resource_attributes == {"service.name": "test-service"}
        assert info.scope_name == "strands.agent"
        assert info.scope_version == "1.0.0"

    def test_extract_resource_info_missing_resource(self):
        """Test extracting resource info when resource is missing."""
        span = Mock()
        span.resource = None
        span.instrumentation_scope = None

        info = SpanParser.extract_resource_info(span)

        assert info.resource_attributes == {}
        assert info.scope_name == ""
        assert info.scope_version == ""

    def test_get_span_attributes(self, mock_span):
        """Test getting span attributes."""
        attrs = SpanParser.get_span_attributes(mock_span)

        assert attrs == {"gen_ai.operation.name": "chat"}

    def test_get_span_attributes_empty(self):
        """Test getting span attributes when empty."""
        span = Mock()
        span.attributes = None

        attrs = SpanParser.get_span_attributes(span)

        assert attrs == {}


# ==============================================================================
# ADOT Builder Tests
# ==============================================================================


class TestADOTDocumentBuilder:
    """Test ADOTDocumentBuilder class."""

    def test_build_span_document(self, span_metadata, resource_info):
        """Test building span document."""
        attributes = {"test.attr": "value"}

        doc = ADOTDocumentBuilder.build_span_document(span_metadata, resource_info, attributes)

        assert doc["traceId"] == "1234567890abcdef1234567890abcdef"
        assert doc["spanId"] == "1234567890abcdef"
        assert doc["name"] == "test-span"
        assert doc["kind"] == "INTERNAL"
        assert doc["startTimeUnixNano"] == 1000000000
        assert doc["endTimeUnixNano"] == 2000000000
        assert doc["durationNano"] == 1000000000
        assert doc["attributes"] == {"test.attr": "value"}
        assert doc["status"]["code"] == "OK"
        assert doc["resource"]["attributes"] == {"service.name": "test-service"}
        assert doc["scope"]["name"] == "strands.agent"

    def test_build_conversation_log_record(self, span_metadata, resource_info):
        """Test building conversation log record."""
        conversation = ConversationTurn(
            user_message="Hello",
            assistant_messages=[{"content": {"message": "Hi"}, "role": "assistant"}],
            tool_results=[],
        )

        doc = ADOTDocumentBuilder.build_conversation_log_record(conversation, span_metadata, resource_info)

        assert doc["traceId"] == "1234567890abcdef1234567890abcdef"
        assert doc["spanId"] == "1234567890abcdef"
        assert doc["severityNumber"] == 9
        assert doc["body"]["input"]["messages"][0]["content"]["content"] == "Hello"
        assert doc["body"]["output"]["messages"][0]["content"]["message"] == "Hi"

    def test_build_conversation_log_record_preserves_chronological_order(self, span_metadata, resource_info):
        """Builder emits input_messages in event arrival order (user/assistant interleaved)."""
        conversation = ConversationTurn(
            input_messages=[
                {"content": {"content": "u1"}, "role": "user"},
                {"content": {"content": "prior-1"}, "role": "assistant"},
                {"content": {"content": "u2"}, "role": "user"},
                {"content": {"content": "prior-2"}, "role": "assistant"},
                {"content": {"content": "u3"}, "role": "user"},
            ],
            assistant_messages=[{"content": {"message": "new-output"}, "role": "assistant"}],
        )

        doc = ADOTDocumentBuilder.build_conversation_log_record(conversation, span_metadata, resource_info)

        input_msgs = doc["body"]["input"]["messages"]
        assert [m.get("role") for m in input_msgs] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert [m["content"].get("content") for m in input_msgs] == [
            "u1",
            "prior-1",
            "u2",
            "prior-2",
            "u3",
        ]

        output_msgs = doc["body"]["output"]["messages"]
        assert len(output_msgs) == 1
        assert output_msgs[0]["content"]["message"] == "new-output"

    def test_build_conversation_log_record_with_tool_results(self, span_metadata, resource_info):
        """Test building conversation log record with tool results."""
        conversation = ConversationTurn(
            user_message="Calculate",
            assistant_messages=[{"content": {"message": "4"}, "role": "assistant"}],
            tool_results=["4"],
        )

        doc = ADOTDocumentBuilder.build_conversation_log_record(conversation, span_metadata, resource_info)

        # Tool result attached to first assistant message
        assert doc["body"]["output"]["messages"][0]["content"]["tool.result"] == "4"

    def test_build_tool_log_record(self, span_metadata, resource_info):
        """Test building tool log record."""
        tool_exec = ToolExecution(
            tool_input='{"x": 1}',
            tool_output="result",
            tool_id="tool-123",
        )

        doc = ADOTDocumentBuilder.build_tool_log_record(tool_exec, span_metadata, resource_info)

        assert doc["traceId"] == "1234567890abcdef1234567890abcdef"
        assert doc["body"]["input"]["messages"][0]["content"]["content"] == '{"x": 1}'
        assert doc["body"]["input"]["messages"][0]["content"]["id"] == "tool-123"
        assert doc["body"]["output"]["messages"][0]["content"]["message"] == "result"
