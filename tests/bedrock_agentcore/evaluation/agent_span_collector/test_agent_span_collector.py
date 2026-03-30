"""Unit tests for CloudWatchAgentSpanCollector."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from bedrock_agentcore.evaluation.agent_span_collector import (
    AgentSpanCollector,
    CloudWatchAgentSpanCollector,
)

SAMPLE_SPANS = [
    {
        "scope": {"name": "agent"},
        "traceId": "trace-1",
        "spanId": "span-1",
        "name": "Agent.invoke",
    },
    {
        "scope": {"name": "agent"},
        "traceId": "trace-1",
        "spanId": "span-2",
        "name": "Tool:search",
    },
]

START_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)
END_TIME = datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc)
WIDENED_END_TIME = END_TIME + timedelta(seconds=60)
HELPER_PATCH = "bedrock_agentcore.evaluation.agent_span_collector.agent_span_collector.CloudWatchSpanHelper"
TIME_PATCH = "bedrock_agentcore.evaluation.agent_span_collector.agent_span_collector.time.monotonic"


@pytest.fixture
def collector():
    """Create a CloudWatchAgentSpanCollector with mocked helper."""
    with patch(HELPER_PATCH) as mock_helper_cls:
        c = CloudWatchAgentSpanCollector(
            log_group_name="/my-agent/logs",
            region="us-west-2",
            max_wait_seconds=10,
            poll_interval_seconds=2,
        )
        c._helper = mock_helper_cls.return_value
        return c


class TestCloudWatchAgentSpanCollector:
    def test_isagent_span_collector(self):
        with patch(HELPER_PATCH):
            c = CloudWatchAgentSpanCollector(log_group_name="/logs", region="us-west-2")
            assert isinstance(c, AgentSpanCollector)

    def test_returns_spans_on_first_attempt(self, collector):
        collector._helper.query_log_group.return_value = SAMPLE_SPANS

        with patch("time.sleep") as mock_sleep:
            result = collector.collect("sess-1", START_TIME, END_TIME)

        assert result == SAMPLE_SPANS + SAMPLE_SPANS
        mock_sleep.assert_not_called()

    def test_no_retry_when_spans_found(self, collector):
        collector._helper.query_log_group.return_value = SAMPLE_SPANS

        with patch("time.sleep") as mock_sleep:
            collector.collect("sess-1", START_TIME, END_TIME)

        mock_sleep.assert_not_called()

    def test_retries_then_succeeds(self, collector):
        collector._helper.query_log_group.side_effect = [
            [],  # aws/spans - attempt 1
            [],  # /my-agent/logs - attempt 1
            SAMPLE_SPANS,  # aws/spans - attempt 2
            SAMPLE_SPANS,  # /my-agent/logs - attempt 2
        ]

        with patch("time.sleep") as mock_sleep:
            result = collector.collect("sess-1", START_TIME, END_TIME)

        assert len(result) == 4
        mock_sleep.assert_called_once_with(2)

    def test_returns_empty_on_timeout(self, collector):
        collector._helper.query_log_group.return_value = []

        # monotonic: 0 (deadline calc), 9 (check: 9+2=11 > 10 → timeout)
        with patch("time.sleep"), patch(TIME_PATCH, side_effect=[0, 9]):
            result = collector.collect("sess-1", START_TIME, END_TIME)

        assert result == []

    def test_retries_before_timeout(self, collector):
        collector._helper.query_log_group.return_value = []

        # monotonic: 0 (deadline), 2 (check: 2+2=4 < 10 → retry), 9 (check: 9+2=11 > 10 → timeout)
        with patch("time.sleep") as mock_sleep, patch(TIME_PATCH, side_effect=[0, 2, 9]):
            result = collector.collect("sess-1", START_TIME, END_TIME)

        assert result == []
        mock_sleep.assert_called_once_with(2)

    def test_widens_end_time_by_60_seconds(self, collector):
        collector._helper.query_log_group.return_value = SAMPLE_SPANS

        with patch("time.sleep"):
            collector.collect("sess-1", START_TIME, END_TIME)

        for c in collector._helper.query_log_group.call_args_list:
            assert c[0][3] == WIDENED_END_TIME

    def test_query_includes_all_filters(self, collector):
        collector._helper.query_log_group.return_value = []

        collector._fetch_spans("sess-1", START_TIME, END_TIME)

        query_string = collector._helper.query_log_group.call_args_list[0][1]["query_string"]
        assert 'attributes.session.id = "sess-1"' in query_string
        assert "ispresent(scope.name)" in query_string
        assert "ispresent(traceId)" in query_string
        assert "ispresent(spanId)" in query_string

    def test_queries_both_log_groups(self, collector):
        collector._helper.query_log_group.side_effect = [
            [{"scope": {"name": "a"}, "traceId": "t1", "spanId": "s1"}],
            [{"scope": {"name": "b"}, "traceId": "t2", "spanId": "s2"}],
        ]

        result = collector._fetch_spans("sess-1", START_TIME, END_TIME)

        assert len(result) == 2
        assert result[0]["scope"]["name"] == "a"
        assert result[1]["scope"]["name"] == "b"

        calls = collector._helper.query_log_group.call_args_list
        assert calls[0][0][0] == "aws/spans"
        assert calls[1][0][0] == "/my-agent/logs"
