"""Tests for CloudWatch span fetcher."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

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

    def test_query_log_group_default_query_includes_ispresent_filters(self):
        """Test default query includes ispresent filters for scope.name, traceId, spanId."""
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "query-123"}
        mock_client.get_query_results.return_value = {"status": "Complete", "results": []}

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

        helper.query_log_group("test-log-group", "session-123", start_time, end_time)

        query_string = mock_client.start_query.call_args[1]["queryString"]
        assert 'filter @message like "session-123"' in query_string
        assert "ispresent(scope.name)" in query_string
        assert "ispresent(traceId)" in query_string
        assert "ispresent(spanId)" in query_string

    def test_query_log_group_custom_query_string(self):
        """Test custom query_string overrides the default."""
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "query-123"}
        mock_client.get_query_results.return_value = {"status": "Complete", "results": []}

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
        custom_query = 'fields @message | filter attributes.session.id = "sess-1"'

        helper.query_log_group("test-log-group", "session-123", start_time, end_time, query_string=custom_query)

        query_string = mock_client.start_query.call_args[1]["queryString"]
        assert query_string == custom_query

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
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

        with patch("boto3.client", return_value=mock_client):
            spans = fetch_spans_from_cloudwatch(
                session_id="session-123",
                event_log_group="/aws/bedrock-agentcore/runtimes/my-agent-ABC-DEFAULT",
                start_time=start_time,
                end_time=end_time,
            )

        assert len(spans) == 2  # Called twice (aws/spans + event logs)
        assert all(_is_valid_adot_document(span) for span in spans)

    def test_fetch_spans_end_time_defaults_to_now(self):
        """Test that end_time defaults to datetime.now() when not provided."""
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "query-123"}
        mock_client.get_query_results.return_value = {"status": "Complete", "results": []}

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

        with patch("boto3.client", return_value=mock_client):
            fetch_spans_from_cloudwatch(
                session_id="session-123",
                event_log_group="/aws/logs/my-agent",
                start_time=start_time,
            )

        # Should have called start_query twice (aws/spans + event log group)
        assert mock_client.start_query.call_count == 2
        # end_time should be a recent timestamp, not None
        for c in mock_client.start_query.call_args_list:
            assert c[1]["endTime"] > int(start_time.timestamp())

    def test_fetch_spans_from_cloudwatch_combines_both_log_groups(self):
        """Test that results from both log groups are combined.

        Filtering of invalid documents is handled server-side by the
        CloudWatch query (ispresent filters), so all parsed results
        are returned.
        """
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "query-123"}

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
                    [{"field": "@message", "value": '{"scope": {"name": "test2"}, "traceId": "789", "spanId": "012"}'}]
                ],
            },
        ]

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

        with patch("boto3.client", return_value=mock_client):
            spans = fetch_spans_from_cloudwatch(
                session_id="session-123",
                event_log_group="/aws/bedrock-agentcore/runtimes/my-agent-ABC-DEFAULT",
                start_time=start_time,
                end_time=end_time,
            )

        assert len(spans) == 2
        assert spans[0]["scope"]["name"] == "test"
        assert spans[1]["scope"]["name"] == "test2"


class TestAgentIdFiltering:
    """Test agent_id filtering on existing methods."""

    def test_query_log_group_with_agent_id(self):
        """Test that agent_id adds parse/filter clauses to default query."""
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "q-1"}
        mock_client.get_query_results.return_value = {"status": "Complete", "results": []}

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

        helper.query_log_group("aws/spans", "sess-1", start_time, end_time, agent_id="my-agent")

        query = mock_client.start_query.call_args[1]["queryString"]
        assert "parsedAgentId" in query
        assert "my-agent" in query

    def test_query_log_group_without_agent_id(self):
        """Test that omitting agent_id does not add agent filter."""
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "q-1"}
        mock_client.get_query_results.return_value = {"status": "Complete", "results": []}

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

        helper.query_log_group("aws/spans", "sess-1", start_time, end_time)

        query = mock_client.start_query.call_args[1]["queryString"]
        assert "parsedAgentId" not in query

    def test_fetch_spans_passes_agent_id(self):
        """Test that fetch_spans forwards agent_id to query_log_group."""
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "q-1"}
        mock_client.get_query_results.return_value = {"status": "Complete", "results": []}

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

        helper.fetch_spans("sess-1", "/aws/logs/agent", start_time, end_time, agent_id="my-agent")

        # Both calls (aws/spans + event log group) should include agent filter
        for call in mock_client.start_query.call_args_list:
            assert "my-agent" in call[1]["queryString"]


class TestQuerySpansByTrace:
    """Test query_spans_by_trace method."""

    def test_returns_results(self):
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "q-1"}
        mock_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [[{"field": "traceId", "value": "trace-abc"}]],
        }

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        results = helper.query_spans_by_trace("trace-abc", 1000000, 2000000)
        assert len(results) == 1
        mock_client.start_query.assert_called_once()
        query = mock_client.start_query.call_args[1]["queryString"]
        assert "trace-abc" in query
        assert mock_client.start_query.call_args[1]["logGroupName"] == "aws/spans"

    def test_empty_results(self):
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "q-1"}
        mock_client.get_query_results.return_value = {"status": "Complete", "results": []}

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        results = helper.query_spans_by_trace("trace-abc", 1000000, 2000000)
        assert results == []


class TestQueryRuntimeLogsByTraces:
    """Test query_runtime_logs_by_traces method."""

    def test_batch_query(self):
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "q-1"}
        mock_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [
                [{"field": "traceId", "value": "t1"}, {"field": "@message", "value": "log1"}],
                [{"field": "traceId", "value": "t2"}, {"field": "@message", "value": "log2"}],
            ],
        }

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        results = helper.query_runtime_logs_by_traces(["t1", "t2"], 1000000, 2000000, "agent-1")
        assert len(results) == 2
        query = mock_client.start_query.call_args[1]["queryString"]
        assert "'t1'" in query
        assert "'t2'" in query
        assert mock_client.start_query.call_args[1]["logGroupName"] == "/aws/bedrock-agentcore/runtimes/agent-1-DEFAULT"

    def test_custom_endpoint_name(self):
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "q-1"}
        mock_client.get_query_results.return_value = {"status": "Complete", "results": []}

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        helper.query_runtime_logs_by_traces(["t1"], 1000000, 2000000, "agent-1", endpoint_name="prod")
        assert mock_client.start_query.call_args[1]["logGroupName"] == "/aws/bedrock-agentcore/runtimes/agent-1-prod"

    def test_empty_trace_ids(self):
        helper = CloudWatchSpanHelper()
        helper.logs_client = Mock()

        results = helper.query_runtime_logs_by_traces([], 1000000, 2000000, "agent-1")
        assert results == []
        helper.logs_client.start_query.assert_not_called()

    def test_fallback_to_individual_queries(self):
        mock_client = Mock()
        # First call (batch) fails, individual calls succeed
        mock_client.start_query.side_effect = [
            Exception("batch failed"),
            {"queryId": "q-1"},
            {"queryId": "q-2"},
        ]
        mock_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [[{"field": "traceId", "value": "t1"}]],
        }

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        results = helper.query_runtime_logs_by_traces(["t1", "t2"], 1000000, 2000000, "agent-1")
        assert len(results) == 2
        assert mock_client.start_query.call_count == 3


class TestGetLatestSessionId:
    """Test get_latest_session_id method."""

    def test_returns_session_id(self):
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "q-1"}
        mock_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [[{"field": "attributes.session.id", "value": "sess-latest"}]],
        }

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        result = helper.get_latest_session_id(1000000, 2000000, "agent-1")
        assert result == "sess-latest"
        query = mock_client.start_query.call_args[1]["queryString"]
        assert "agent-1" in query
        assert "limit 1" in query

    def test_returns_none_when_no_results(self):
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "q-1"}
        mock_client.get_query_results.return_value = {"status": "Complete", "results": []}

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        result = helper.get_latest_session_id(1000000, 2000000, "agent-1")
        assert result is None

    def test_returns_none_when_field_missing(self):
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "q-1"}
        mock_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [[{"field": "other_field", "value": "something"}]],
        }

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        result = helper.get_latest_session_id(1000000, 2000000, "agent-1")
        assert result is None


class TestExecuteQuery:
    """Test _execute_query helper."""

    def test_successful_query(self):
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "q-1"}
        mock_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [[{"field": "f", "value": "v"}]],
        }

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        results = helper._execute_query("fields @message", "lg", 1000000, 2000000)
        assert len(results) == 1
        # Verify ms -> seconds conversion
        assert mock_client.start_query.call_args[1]["startTime"] == 1000
        assert mock_client.start_query.call_args[1]["endTime"] == 2000

    def test_failed_query_raises(self):
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "q-1"}
        mock_client.get_query_results.return_value = {"status": "Failed"}

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        with pytest.raises(RuntimeError, match="failed with status"):
            helper._execute_query("fields @message", "lg", 1000000, 2000000)

    @patch("bedrock_agentcore.evaluation.utils.cloudwatch_span_helper.time")
    def test_timeout_raises(self, mock_time):
        mock_client = Mock()
        mock_client.start_query.return_value = {"queryId": "q-1"}
        mock_client.get_query_results.return_value = {"status": "Running"}
        # Simulate time passing beyond timeout
        mock_time.time.side_effect = [0, 0, 61]
        mock_time.sleep = Mock()

        helper = CloudWatchSpanHelper()
        helper.logs_client = mock_client

        with pytest.raises(TimeoutError):
            helper._execute_query("fields @message", "lg", 1000000, 2000000)
