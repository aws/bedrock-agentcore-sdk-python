"""Shared fixtures for span_to_adot_serializer tests."""

from unittest.mock import Mock

import pytest

from bedrock_agentcore.evaluation.span_to_adot_serializer.adot_models import (
    ResourceInfo,
    SpanMetadata,
)


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
    """Create a mock span event factory."""

    def _create_event(name, attributes):
        event = Mock()
        event.name = name
        event.attributes = attributes
        return event

    return _create_event


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
