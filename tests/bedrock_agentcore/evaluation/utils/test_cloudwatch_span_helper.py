"""Tests for CloudWatch span fetcher."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

from bedrock_agentcore.evaluation.utils.cloudwatch_span_helper import (
    CloudWatchSpanHelper,
    _is_valid_adot_document,
    fetch_spans_from_cloudwatch,
)


class TestIsValidAdotDocument:
    """Test _is_valid_adot_document helper."""

    def test_valid_adot_document(self):
        """Test valid ADOT document is recognized."""
        doc = {"scope": {"name": "test"}, "traceId": "123", "spanId": "456"}
        assert _is_valid_adot_document(doc) is True

    def test_missing_scope(self):
        """Test document missing scope is invalid."""
        doc = {"traceId": "123", "spanId": "456"}
        assert _is_valid_adot_document(doc) is False

    def test_missing_trace_id(self):
        """Test document missing traceId is invalid."""
        doc = {"scope": {"name": "test"}, "spanId": "456"}
        assert _is_valid_adot_document(doc) is False

    def test_missing_span_id(self):
        """Test document missing spanId is invalid."""
        doc = {"scope": {"name": "test"}, "traceId": "123"}
        assert _is_valid_adot_document(doc) is False

    def test_not_a_dict(self):
        """Test non-dict is invalid."""
        assert _is_valid_adot_document("not a dict") is False
        assert _is_valid_adot_document(None) is False


class TestCloudWatchSpanHelper:
    """Test CloudWatchSpanHelper class."""

    def test_query_log_group_successful(self):
        """Test successful CloudWatch query."""
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "query-123"}
        mock_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [
                [
                    {"field": "@timestamp", "value": "2024-01-01"},
                    {"field": "@message", "value": '{"scope": {"name": "test"}, "traceId": "123", "spanId": "456"}'},
                ]
            ],
        }

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

        results = helper.query_log_group("test-log-group", "session-123", start_time, end_time)

        assert len(results) == 1
        assert results[0]["scope"]["name"] == "test"

    def test_query_log_group_failure(self):
        """Test query failure handling."""
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "query-123"}
        mock_client.get_query_results.return_value = {"status": "Failed"}

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

        results = helper.query_log_group("test-log-group", "session-123", start_time, end_time)

        assert results == []

    def test_query_log_group_invalid_json(self):
        """Test handling of invalid JSON in messages."""
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "query-123"}
        mock_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [
                [
                    {"field": "@message", "value": "not valid json"},
                    {"field": "@message", "value": '{"valid": "json"}'},
                ]
            ],
        }

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

        results = helper.query_log_group("test-log-group", "session-123", start_time, end_time)

        assert len(results) == 1
        assert results[0]["valid"] == "json"


class TestFetchSpansFromCloudWatch:
    """Test fetch_spans_from_cloudwatch function."""

    def test_fetch_spans_from_cloudwatch(self):
        """Test fetching spans from CloudWatch."""
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "query-123"}
        mock_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [
                [
                    {"field": "@message", "value": '{"scope": {"name": "test"}, "traceId": "123", "spanId": "456"}'},
                ]
            ],
        }

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

        with patch("boto3.client", return_value=mock_client):
            spans = fetch_spans_from_cloudwatch(
                session_id="session-123",
                event_log_group="/aws/bedrock-agentcore/runtimes/my-agent-ABC-DEFAULT",
                start_time=start_time,
            )

        assert len(spans) == 2  # Called twice (aws/spans + event logs)
        assert all(_is_valid_adot_document(span) for span in spans)

    def test_fetch_spans_from_cloudwatch_filters_invalid(self):
        """Test that invalid documents are filtered out."""
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "query-123"}

        # First call returns valid, second returns invalid
        mock_client.get_query_results.side_effect = [
            {
                "status": "Complete",
                "results": [
                    [{"field": "@message", "value": '{"scope": {"name": "test"}, "traceId": "123", "spanId": "456"}'}]
                ],
            },
            {
                "status": "Complete",
                "results": [
                    [{"field": "@message", "value": '{"invalid": "document"}'}]  # Missing required fields
                ],
            },
        ]

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

        with patch("boto3.client", return_value=mock_client):
            spans = fetch_spans_from_cloudwatch(
                session_id="session-123",
                event_log_group="/aws/bedrock-agentcore/runtimes/my-agent-ABC-DEFAULT",
                start_time=start_time,
            )

        assert len(spans) == 1  # Only valid document
        assert spans[0]["scope"]["name"] == "test"
