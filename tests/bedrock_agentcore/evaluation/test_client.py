"""Unit tests for EvaluationClient."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.evaluation.client import EvaluationClient

# --- Fixtures ---

SAMPLE_SPANS = [
    {
        "scope": {"name": "agent"},
        "traceId": "trace-1",
        "spanId": "span-1",
        "name": "Agent.invoke",
        "kind": "SPAN_KIND_SERVER",
    },
    {
        "scope": {"name": "agent"},
        "traceId": "trace-1",
        "spanId": "span-2",
        "name": "Tool:search",
        "kind": "SPAN_KIND_INTERNAL",
    },
    {
        "scope": {"name": "agent"},
        "traceId": "trace-1",
        "spanId": "span-3",
        "name": "Tool:calculator",
        "kind": "SPAN_KIND_INTERNAL",
    },
    {
        "scope": {"name": "agent"},
        "traceId": "trace-2",
        "spanId": "span-4",
        "name": "Agent.invoke",
        "kind": "SPAN_KIND_SERVER",
    },
    {
        "scope": {"name": "agent"},
        "traceId": "trace-2",
        "spanId": "span-5",
        "name": "Tool:search",
        "kind": "SPAN_KIND_INTERNAL",
    },
]


@pytest.fixture
def client():
    """Create an EvaluationClient with mocked boto3 clients."""
    with patch("bedrock_agentcore.evaluation.client.boto3") as mock_boto3:
        mock_boto3.Session.return_value.region_name = "us-west-2"
        mock_dp = MagicMock()
        mock_cp = MagicMock()
        mock_boto3.client.side_effect = lambda service, **kwargs: (
            mock_dp if service == "bedrock-agentcore" else mock_cp
        )
        c = EvaluationClient(region_name="us-west-2")
        c._dp_client = mock_dp
        c._cp_client = mock_cp
        return c


# --- __init__ tests ---


class TestInit:
    def test_creates_both_clients(self):
        with patch("bedrock_agentcore.evaluation.client.boto3") as mock_boto3:
            mock_boto3.Session.return_value.region_name = "us-east-1"
            EvaluationClient(region_name="us-east-1")
            calls = mock_boto3.client.call_args_list
            service_names = [call[0][0] for call in calls]
            assert "bedrock-agentcore" in service_names
            assert "bedrock-agentcore-control" in service_names

    def test_region_fallback(self):
        with patch("bedrock_agentcore.evaluation.client.boto3") as mock_boto3:
            mock_boto3.Session.return_value.region_name = "eu-west-1"
            c = EvaluationClient()
            assert c.region_name == "eu-west-1"

    def test_region_fallback_to_default(self):
        with patch("bedrock_agentcore.evaluation.client.boto3") as mock_boto3:
            mock_boto3.Session.return_value.region_name = None
            c = EvaluationClient()
            assert c.region_name == "us-west-2"

    def test_empty_evaluator_level_cache(self):
        with patch("bedrock_agentcore.evaluation.client.boto3") as mock_boto3:
            mock_boto3.Session.return_value.region_name = "us-west-2"
            c = EvaluationClient()
            assert c._evaluator_level_cache == {}


# --- run() validation tests ---


class TestRunValidation:
    def test_raises_without_agent_id_or_log_group(self, client):
        with pytest.raises(ValueError, match="Provide either agent_id or log_group_name"):
            client.run(evaluator_ids=["accuracy"], session_id="sess-1")

    def test_derives_log_group_from_agent_id(self, client):
        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = []
            client.run(evaluator_ids=["accuracy"], session_id="sess-1", agent_id="my-agent")
            mock_collector_cls.assert_called_once_with(
                log_group_name="/aws/bedrock-agentcore/runtimes/my-agent-DEFAULT",
                region="us-west-2",
                max_wait_seconds=60,
                poll_interval_seconds=2,
            )

    def test_log_group_name_takes_precedence(self, client):
        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = []
            client.run(
                evaluator_ids=["accuracy"],
                session_id="sess-1",
                agent_id="my-agent",
                log_group_name="/custom/group",
            )
            mock_collector_cls.assert_called_once_with(
                log_group_name="/custom/group",
                region="us-west-2",
                max_wait_seconds=60,
                poll_interval_seconds=2,
            )

    def test_returns_empty_when_no_spans(self, client):
        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = []
            results = client.run(evaluator_ids=["accuracy"], session_id="sess-1", agent_id="my-agent")
            assert results == []
            client._dp_client.evaluate.assert_not_called()


# --- run() end-to-end tests ---


class TestRunEndToEnd:
    def test_session_level_evaluator(self, client):
        client._cp_client.get_evaluator.return_value = {"level": "SESSION"}
        client._dp_client.evaluate.return_value = {"evaluationResults": [{"evaluatorId": "accuracy", "value": 0.9}]}

        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = SAMPLE_SPANS
            results = client.run(evaluator_ids=["accuracy"], session_id="sess-1", agent_id="my-agent")

        assert len(results) == 1
        assert results[0]["value"] == 0.9
        # SESSION level: no evaluationTarget
        call_kwargs = client._dp_client.evaluate.call_args[1]
        assert "evaluationTarget" not in call_kwargs

    def test_trace_level_evaluator(self, client):
        client._cp_client.get_evaluator.return_value = {"level": "TRACE"}
        client._dp_client.evaluate.return_value = {"evaluationResults": [{"evaluatorId": "trace-eval", "value": 0.8}]}

        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = SAMPLE_SPANS
            results = client.run(evaluator_ids=["trace-eval"], session_id="sess-1", agent_id="my-agent")

        assert len(results) == 1
        call_kwargs = client._dp_client.evaluate.call_args[1]
        assert call_kwargs["evaluationTarget"]["traceIds"] == ["trace-1", "trace-2"]

    def test_tool_call_level_evaluator(self, client):
        client._cp_client.get_evaluator.return_value = {"level": "TOOL_CALL"}
        client._dp_client.evaluate.return_value = {"evaluationResults": [{"evaluatorId": "tool-eval", "value": 0.7}]}

        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = SAMPLE_SPANS
            results = client.run(evaluator_ids=["tool-eval"], session_id="sess-1", agent_id="my-agent")

        assert len(results) == 1
        call_kwargs = client._dp_client.evaluate.call_args[1]
        assert set(call_kwargs["evaluationTarget"]["spanIds"]) == {"span-2", "span-3", "span-5"}

    def test_multiple_evaluators(self, client):
        client._cp_client.get_evaluator.side_effect = [
            {"level": "SESSION"},
            {"level": "TRACE"},
        ]
        client._dp_client.evaluate.return_value = {"evaluationResults": [{"evaluatorId": "eval", "value": 1.0}]}

        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = SAMPLE_SPANS
            results = client.run(
                evaluator_ids=["session-eval", "trace-eval"],
                session_id="sess-1",
                agent_id="my-agent",
            )

        assert len(results) == 2
        assert client._dp_client.evaluate.call_count == 2

    def test_evaluator_api_error_is_caught(self, client):
        client._cp_client.get_evaluator.return_value = {"level": "SESSION"}
        client._dp_client.evaluate.side_effect = RuntimeError("API error")

        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = SAMPLE_SPANS
            results = client.run(evaluator_ids=["accuracy"], session_id="sess-1", agent_id="my-agent")

        assert results == []

    def test_custom_look_back_time(self, client):
        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = []
            client.run(
                evaluator_ids=["accuracy"],
                session_id="sess-1",
                agent_id="my-agent",
                look_back_time=timedelta(hours=2),
            )
            call_kwargs = mock_collector_cls.return_value.collect.call_args[1]
            duration = call_kwargs["end_time"] - call_kwargs["start_time"]
            assert duration == timedelta(hours=2)


# --- _get_evaluator_level tests ---


class TestGetEvaluatorLevel:
    def test_returns_level_from_api(self, client):
        client._cp_client.get_evaluator.return_value = {"level": "TRACE"}
        assert client._get_evaluator_level("eval-1") == "TRACE"

    def test_caches_level(self, client):
        client._cp_client.get_evaluator.return_value = {"level": "TRACE"}
        client._get_evaluator_level("eval-1")
        client._get_evaluator_level("eval-1")
        client._cp_client.get_evaluator.assert_called_once()

    def test_falls_back_to_session_on_error(self, client):
        client._cp_client.get_evaluator.side_effect = RuntimeError("not found")
        assert client._get_evaluator_level("eval-1") == "SESSION"

    def test_caches_fallback(self, client):
        client._cp_client.get_evaluator.side_effect = RuntimeError("not found")
        client._get_evaluator_level("eval-1")
        client._get_evaluator_level("eval-1")
        client._cp_client.get_evaluator.assert_called_once()


# --- _build_requests_for_level tests ---


class TestBuildRequestsForLevel:
    def test_session_level(self, client):
        base = {"evaluationInput": {"sessionSpans": SAMPLE_SPANS}}
        requests = client._build_requests_for_level("eval", "SESSION", base, SAMPLE_SPANS)
        assert len(requests) == 1
        assert requests[0] is base

    def test_trace_level(self, client):
        base = {"evaluationInput": {"sessionSpans": SAMPLE_SPANS}}
        requests = client._build_requests_for_level("eval", "TRACE", base, SAMPLE_SPANS)
        assert len(requests) == 1
        assert requests[0]["evaluationTarget"]["traceIds"] == ["trace-1", "trace-2"]

    def test_tool_call_level(self, client):
        base = {"evaluationInput": {"sessionSpans": SAMPLE_SPANS}}
        requests = client._build_requests_for_level("eval", "TOOL_CALL", base, SAMPLE_SPANS)
        assert len(requests) == 1
        assert set(requests[0]["evaluationTarget"]["spanIds"]) == {"span-2", "span-3", "span-5"}

    def test_trace_level_no_traces_raises(self, client):
        base = {"evaluationInput": {"sessionSpans": []}}
        with pytest.raises(ValueError, match="No trace IDs found"):
            client._build_requests_for_level("eval", "TRACE", base, [])

    def test_tool_call_level_no_tools_raises(self, client):
        spans = [{"name": "Agent.invoke", "kind": "SPAN_KIND_SERVER", "spanId": "s1"}]
        base = {"evaluationInput": {"sessionSpans": spans}}
        with pytest.raises(ValueError, match="No tool span IDs found"):
            client._build_requests_for_level("eval", "TOOL_CALL", base, spans)

    def test_unknown_level_raises(self, client):
        with pytest.raises(ValueError, match="Unknown evaluator level"):
            client._build_requests_for_level("eval", "UNKNOWN", {}, [])

    def test_trace_level_batching(self, client):
        # Create spans with 12 unique trace IDs to trigger batching
        spans = [{"traceId": f"trace-{i}", "spanId": f"span-{i}"} for i in range(12)]
        base = {"evaluationInput": {"sessionSpans": spans}}
        requests = client._build_requests_for_level("eval", "TRACE", base, spans)
        assert len(requests) == 2
        assert len(requests[0]["evaluationTarget"]["traceIds"]) == 10
        assert len(requests[1]["evaluationTarget"]["traceIds"]) == 2


# --- Static helper tests ---


class TestExtractTraceIds:
    def test_extracts_unique_ordered(self):
        ids = EvaluationClient._extract_trace_ids(SAMPLE_SPANS)
        assert ids == ["trace-1", "trace-2"]

    def test_empty_spans(self):
        assert EvaluationClient._extract_trace_ids([]) == []

    def test_skips_missing_trace_id(self):
        spans = [{"spanId": "s1"}, {"traceId": "t1", "spanId": "s2"}]
        assert EvaluationClient._extract_trace_ids(spans) == ["t1"]


class TestExtractToolSpanIds:
    def test_extracts_tool_spans(self):
        ids = EvaluationClient._extract_tool_span_ids(SAMPLE_SPANS)
        assert ids == ["span-2", "span-3", "span-5"]

    def test_ignores_non_tool_spans(self):
        spans = [
            {"name": "Agent.invoke", "kind": "SPAN_KIND_SERVER", "spanId": "s1"},
            {"name": "LLM.call", "kind": "SPAN_KIND_INTERNAL", "spanId": "s2"},
        ]
        assert EvaluationClient._extract_tool_span_ids(spans) == []

    def test_empty_spans(self):
        assert EvaluationClient._extract_tool_span_ids([]) == []


class TestBatch:
    def test_exact_batches(self):
        batches = list(EvaluationClient._batch([1, 2, 3, 4], 2))
        assert batches == [[1, 2], [3, 4]]

    def test_remainder(self):
        batches = list(EvaluationClient._batch([1, 2, 3], 2))
        assert batches == [[1, 2], [3]]

    def test_single_batch(self):
        batches = list(EvaluationClient._batch([1, 2], 10))
        assert batches == [[1, 2]]

    def test_empty(self):
        batches = list(EvaluationClient._batch([], 10))
        assert batches == []
