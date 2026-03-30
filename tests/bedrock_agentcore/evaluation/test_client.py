"""Unit tests for EvaluationClient."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.evaluation.client import EvaluationClient, ReferenceInputs

# --- Fixtures ---

SAMPLE_SPANS = [
    {
        "scope": {"name": "agent"},
        "traceId": "trace-1",
        "spanId": "span-1",
        "name": "Agent.invoke",
        "kind": "SPAN_KIND_SERVER",
        "attributes": {"gen_ai.operation.name": "invoke_agent"},
    },
    {
        "scope": {"name": "agent"},
        "traceId": "trace-1",
        "spanId": "span-2",
        "name": "Tool:search",
        "kind": "SPAN_KIND_INTERNAL",
        "attributes": {"gen_ai.operation.name": "execute_tool"},
    },
    {
        "scope": {"name": "agent"},
        "traceId": "trace-1",
        "spanId": "span-3",
        "name": "Tool:calculator",
        "kind": "SPAN_KIND_INTERNAL",
        "attributes": {"gen_ai.operation.name": "execute_tool"},
    },
    {
        "scope": {"name": "agent"},
        "traceId": "trace-2",
        "spanId": "span-4",
        "name": "Agent.invoke",
        "kind": "SPAN_KIND_SERVER",
        "attributes": {"gen_ai.operation.name": "invoke_agent"},
    },
    {
        "scope": {"name": "agent"},
        "traceId": "trace-2",
        "spanId": "span-5",
        "name": "Tool:search",
        "kind": "SPAN_KIND_INTERNAL",
        "attributes": {"gen_ai.operation.name": "execute_tool"},
    },
]


@pytest.fixture
def client():
    """Create an EvaluationClient with mocked boto3 clients."""
    with patch("bedrock_agentcore.evaluation.client.boto3") as mock_boto3:
        mock_boto3.Session.return_value.region_name = "us-west-2"
        mock_dp = MagicMock()
        mock_cp = MagicMock()
        mock_boto3.client.side_effect = lambda service, **kwargs: mock_dp if service == "bedrock-agentcore" else mock_cp
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

    def test_evaluator_api_error_propagates(self, client):
        client._cp_client.get_evaluator.return_value = {"level": "SESSION"}
        client._dp_client.evaluate.side_effect = RuntimeError("API error")

        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = SAMPLE_SPANS
            with pytest.raises(RuntimeError, match="API error"):
                client.run(evaluator_ids=["accuracy"], session_id="sess-1", agent_id="my-agent")

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

    def test_trace_level_no_traces_returns_empty(self, client):
        base = {"evaluationInput": {"sessionSpans": []}}
        result = client._build_requests_for_level("eval", "TRACE", base, [])
        assert result == []

    def test_tool_call_level_no_tools_returns_empty(self, client):
        spans = [{"name": "Agent.invoke", "kind": "SPAN_KIND_SERVER", "spanId": "s1"}]
        base = {"evaluationInput": {"sessionSpans": spans}}
        result = client._build_requests_for_level("eval", "TOOL_CALL", base, spans)
        assert result == []

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

    def test_filters_by_trace_id(self):
        ids = EvaluationClient._extract_tool_span_ids(SAMPLE_SPANS, trace_id="trace-1")
        assert ids == ["span-2", "span-3"]

    def test_filters_by_trace_id_no_match(self):
        ids = EvaluationClient._extract_tool_span_ids(SAMPLE_SPANS, trace_id="trace-999")
        assert ids == []

    def test_extracts_langgraph_tool_spans(self):
        spans = [
            {"spanId": "s1", "traceId": "t1", "attributes": {"openinference.span.kind": "TOOL"}},
            {"spanId": "s2", "traceId": "t1", "attributes": {"openinference.span.kind": "LLM"}},
        ]
        assert EvaluationClient._extract_tool_span_ids(spans) == ["s1"]

    def test_extracts_traceloop_tool_spans(self):
        spans = [
            {"spanId": "s1", "traceId": "t1", "attributes": {"traceloop.span.kind": "tool"}},
            {"spanId": "s2", "traceId": "t1", "attributes": {"traceloop.span.kind": "workflow"}},
        ]
        assert EvaluationClient._extract_tool_span_ids(spans) == ["s1"]

    def test_ignores_span_without_tool_attributes(self):
        spans = [
            {"spanId": "s1", "traceId": "t1", "attributes": {"gen_ai.operation.name": "invoke_agent"}},
            {"spanId": "s2", "traceId": "t1", "attributes": {"some.other.attr": "value"}},
            {"spanId": "s3", "traceId": "t1"},
        ]
        assert EvaluationClient._extract_tool_span_ids(spans) == []


# --- trace_id tests ---


class TestRunWithTraceId:
    def test_trace_level_targets_single_trace(self, client):
        client._cp_client.get_evaluator.return_value = {"level": "TRACE"}
        client._dp_client.evaluate.return_value = {"evaluationResults": [{"evaluatorId": "eval", "value": 0.9}]}

        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = SAMPLE_SPANS
            results = client.run(evaluator_ids=["eval"], session_id="sess-1", agent_id="my-agent", trace_id="trace-2")

        assert len(results) == 1
        call_kwargs = client._dp_client.evaluate.call_args[1]
        assert call_kwargs["evaluationTarget"]["traceIds"] == ["trace-2"]

    def test_session_level_ignores_trace_id(self, client):
        client._cp_client.get_evaluator.return_value = {"level": "SESSION"}
        client._dp_client.evaluate.return_value = {"evaluationResults": [{"evaluatorId": "eval", "value": 0.9}]}

        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = SAMPLE_SPANS
            client.run(evaluator_ids=["eval"], session_id="sess-1", agent_id="my-agent", trace_id="trace-1")

        call_kwargs = client._dp_client.evaluate.call_args[1]
        assert "evaluationTarget" not in call_kwargs

    def test_tool_call_level_filters_to_trace(self, client):
        client._cp_client.get_evaluator.return_value = {"level": "TOOL_CALL"}
        client._dp_client.evaluate.return_value = {"evaluationResults": [{"evaluatorId": "eval", "value": 0.7}]}

        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = SAMPLE_SPANS
            results = client.run(evaluator_ids=["eval"], session_id="sess-1", agent_id="my-agent", trace_id="trace-1")

        assert len(results) == 1
        call_kwargs = client._dp_client.evaluate.call_args[1]
        # Only tool spans from trace-1 (span-2, span-3), not span-5 from trace-2
        assert set(call_kwargs["evaluationTarget"]["spanIds"]) == {"span-2", "span-3"}


# --- _build_reference_inputs tests ---


class TestBuildReferenceInputs:
    def test_assertions_only(self):
        ref = ReferenceInputs(assertions=["Agent asked for dates", "Agent confirmed booking"])
        result = EvaluationClient._build_reference_inputs("sess-1", ref, ["t1", "t2"])
        assert len(result) == 1
        assert result[0]["context"]["spanContext"] == {"sessionId": "sess-1"}
        assert result[0]["assertions"] == [{"text": "Agent asked for dates"}, {"text": "Agent confirmed booking"}]
        assert "expectedTrajectory" not in result[0]

    def test_expected_trajectory_only(self):
        ref = ReferenceInputs(expected_trajectory=["search_flights", "book_flight"])
        result = EvaluationClient._build_reference_inputs("sess-1", ref, ["t1"])
        assert len(result) == 1
        assert result[0]["expectedTrajectory"] == {"toolNames": ["search_flights", "book_flight"]}
        assert "assertions" not in result[0]

    def test_assertions_and_trajectory_combined(self):
        ref = ReferenceInputs(
            assertions=["Agent confirmed"],
            expected_trajectory=["search", "book"],
        )
        result = EvaluationClient._build_reference_inputs("sess-1", ref, ["t1"])
        assert len(result) == 1
        assert "assertions" in result[0]
        assert "expectedTrajectory" in result[0]

    def test_expected_response_string_targets_last_trace(self):
        ref = ReferenceInputs(expected_response="The capital is Paris.")
        result = EvaluationClient._build_reference_inputs("sess-1", ref, ["t1", "t2", "t3"])
        assert len(result) == 1
        assert result[0]["context"]["spanContext"] == {"sessionId": "sess-1", "traceId": "t3"}
        assert result[0]["expectedResponse"] == {"text": "The capital is Paris."}

    def test_expected_response_string_no_traces(self):
        ref = ReferenceInputs(expected_response="The capital is Paris.")
        result = EvaluationClient._build_reference_inputs("sess-1", ref, [])
        assert len(result) == 0

    def test_expected_response_dict_targets_specific_traces(self):
        ref = ReferenceInputs(
            expected_response={"t1": "Response A", "t3": "Response C"},
        )
        result = EvaluationClient._build_reference_inputs("sess-1", ref, ["t1", "t2", "t3"])
        assert len(result) == 2
        assert result[0]["context"]["spanContext"]["traceId"] == "t1"
        assert result[0]["expectedResponse"] == {"text": "Response A"}
        assert result[1]["context"]["spanContext"]["traceId"] == "t3"
        assert result[1]["expectedResponse"] == {"text": "Response C"}

    def test_all_fields_combined(self):
        ref = ReferenceInputs(
            assertions=["Agent confirmed"],
            expected_trajectory=["search", "book"],
            expected_response={"t2": "Booked flight"},
        )
        result = EvaluationClient._build_reference_inputs("sess-1", ref, ["t1", "t2"])
        # 1 session-level (assertions + trajectory) + 1 trace-level (expected_response)
        assert len(result) == 2
        assert "assertions" in result[0]
        assert "expectedTrajectory" in result[0]
        assert result[1]["expectedResponse"] == {"text": "Booked flight"}

    def test_expected_response_string_with_target_trace_id(self):
        ref = ReferenceInputs(expected_response="The capital is Paris.")
        result = EvaluationClient._build_reference_inputs("sess-1", ref, ["t1", "t2", "t3"], target_trace_id="t2")
        assert len(result) == 1
        assert result[0]["context"]["spanContext"] == {"sessionId": "sess-1", "traceId": "t2"}
        assert result[0]["expectedResponse"] == {"text": "The capital is Paris."}

    def test_expected_response_string_with_target_trace_id_no_traces(self):
        ref = ReferenceInputs(expected_response="The capital is Paris.")
        result = EvaluationClient._build_reference_inputs("sess-1", ref, [], target_trace_id="t1")
        assert len(result) == 1
        assert result[0]["context"]["spanContext"] == {"sessionId": "sess-1", "traceId": "t1"}

    def test_empty_reference_inputs(self):
        ref = ReferenceInputs()
        result = EvaluationClient._build_reference_inputs("sess-1", ref, ["t1"])
        assert result == []


# --- End-to-end reference_inputs + trace_id tests ---


class TestRunWithReferenceInputs:
    def test_reference_inputs_passed_to_api(self, client):
        client._cp_client.get_evaluator.return_value = {"level": "SESSION"}
        client._dp_client.evaluate.return_value = {"evaluationResults": [{"evaluatorId": "eval", "value": 1.0}]}

        ref = ReferenceInputs(assertions=["Agent completed the task"])

        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = SAMPLE_SPANS
            client.run(
                evaluator_ids=["eval"],
                session_id="sess-1",
                agent_id="my-agent",
                reference_inputs=ref,
            )

        call_kwargs = client._dp_client.evaluate.call_args[1]
        ref_inputs = call_kwargs["evaluationReferenceInputs"]
        assert len(ref_inputs) == 1
        assert ref_inputs[0]["assertions"] == [{"text": "Agent completed the task"}]

    def test_trace_id_and_reference_inputs_combined(self, client):
        client._cp_client.get_evaluator.return_value = {"level": "TRACE"}
        client._dp_client.evaluate.return_value = {"evaluationResults": [{"evaluatorId": "eval", "value": 0.95}]}

        ref = ReferenceInputs(
            expected_response={"trace-2": "Show flight options"},
        )

        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = SAMPLE_SPANS
            results = client.run(
                evaluator_ids=["eval"],
                session_id="sess-1",
                agent_id="my-agent",
                trace_id="trace-2",
                reference_inputs=ref,
            )

        assert len(results) == 1
        call_kwargs = client._dp_client.evaluate.call_args[1]
        # trace_id narrows the target
        assert call_kwargs["evaluationTarget"]["traceIds"] == ["trace-2"]
        # reference_inputs included
        ref_inputs = call_kwargs["evaluationReferenceInputs"]
        assert any(r.get("expectedResponse") == {"text": "Show flight options"} for r in ref_inputs)

    def test_no_reference_inputs_omits_key(self, client):
        client._cp_client.get_evaluator.return_value = {"level": "SESSION"}
        client._dp_client.evaluate.return_value = {"evaluationResults": [{"evaluatorId": "eval", "value": 0.9}]}

        with patch("bedrock_agentcore.evaluation.client.CloudWatchAgentSpanCollector") as mock_collector_cls:
            mock_collector_cls.return_value.collect.return_value = SAMPLE_SPANS
            client.run(evaluator_ids=["eval"], session_id="sess-1", agent_id="my-agent")

        call_kwargs = client._dp_client.evaluate.call_args[1]
        assert "evaluationReferenceInputs" not in call_kwargs
