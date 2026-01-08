"""Tests for framework-agnostic ADOT models and builders."""

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
def span_metadata():
    """Create test SpanMetadata."""
    return SpanMetadata(
        trace_id="1234567890abcdef1234567890abcdef",
        span_id="1234567890abcdef",
        parent_span_id=None,
        name="test-span",
        start_time=1000000000,
        end_time=2000000000,
        duration=1000000000,
        kind="INTERNAL",
        flags=1,
        status_code="OK",
    )


@pytest.fixture
def resource_info():
    """Create test ResourceInfo."""
    return ResourceInfo(
        resource_attributes={"service.name": "test-service"},
        scope_name="strands.agent",
        scope_version="1.0.0",
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
    """Test ConversationTurn dataclass."""

    def test_creation(self):
        """Test ConversationTurn creation."""
        turn = ConversationTurn(
            user_message="Hello",
            assistant_messages=[{"content": {"message": "Hi"}, "role": "assistant"}],
            tool_results=["result1"],
        )
        assert turn.user_message == "Hello"
        assert len(turn.assistant_messages) == 1
        assert len(turn.tool_results) == 1


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
