"""Integration tests for CloudWatchSpanHelper new methods.

Requires:
    SPAN_TEST_AGENT_ID: Agent runtime ID with pre-existing spans.
    SPAN_TEST_TRACE_ID: A known trace ID from that agent.
    SPAN_TEST_SESSION_ID: A known session ID from that agent.
"""

import os
import time

import pytest

from bedrock_agentcore.evaluation.utils.cloudwatch_span_helper import CloudWatchSpanHelper


@pytest.mark.integration
class TestCloudWatchSpanHelperInteg:
    """Integration tests for CloudWatchSpanHelper query methods."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.agent_id = os.environ.get("SPAN_TEST_AGENT_ID")
        cls.trace_id = os.environ.get("SPAN_TEST_TRACE_ID")
        cls.session_id = os.environ.get("SPAN_TEST_SESSION_ID")
        if not all([cls.agent_id, cls.trace_id, cls.session_id]):
            pytest.fail("SPAN_TEST_AGENT_ID, SPAN_TEST_TRACE_ID, and SPAN_TEST_SESSION_ID must be set")
        cls.helper = CloudWatchSpanHelper(region=cls.region)
        # Wide time window to always capture pre-populated data
        cls.start_time_ms = 0
        cls.end_time_ms = int(time.time() * 1000)

    @pytest.mark.order(1)
    def test_query_spans_by_trace(self):
        results = self.helper.query_spans_by_trace(
            trace_id=self.trace_id,
            start_time_ms=self.start_time_ms,
            end_time_ms=self.end_time_ms,
        )
        assert len(results) > 0
        # Verify all results match the queried trace ID
        for row in results:
            for field in row:
                if field.get("field") == "traceId":
                    assert field.get("value") == self.trace_id

    @pytest.mark.order(2)
    def test_query_runtime_logs_by_traces(self):
        results = self.helper.query_runtime_logs_by_traces(
            trace_ids=[self.trace_id],
            start_time_ms=self.start_time_ms,
            end_time_ms=self.end_time_ms,
            agent_id=self.agent_id,
        )
        assert len(results) > 0
        # Verify all results match the queried trace ID
        for row in results:
            for field in row:
                if field.get("field") == "traceId":
                    assert field.get("value") == self.trace_id

    @pytest.mark.order(3)
    def test_get_latest_session_id(self):
        session_id = self.helper.get_latest_session_id(
            start_time_ms=self.start_time_ms,
            end_time_ms=self.end_time_ms,
            agent_id=self.agent_id,
        )
        assert session_id is not None

    @pytest.mark.order(4)
    def test_query_spans_by_trace_nonexistent(self):
        results = self.helper.query_spans_by_trace(
            trace_id="00000000000000000000000000000000",
            start_time_ms=self.start_time_ms,
            end_time_ms=self.end_time_ms,
        )
        assert results == []

    @pytest.mark.order(5)
    def test_query_runtime_logs_empty_traces(self):
        results = self.helper.query_runtime_logs_by_traces(
            trace_ids=[],
            start_time_ms=self.start_time_ms,
            end_time_ms=self.end_time_ms,
            agent_id=self.agent_id,
        )
        assert results == []
