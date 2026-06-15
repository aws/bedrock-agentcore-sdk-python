"""Tests for deepeval input_mapper module."""

from unittest.mock import MagicMock

import pytest
from deepeval.test_case import LLMTestCaseParams

from bedrock_agentcore.evaluation.integrations.deepeval.input_mapper import (
    ParsedEvaluationEvent,
    _get_required_params,
    build_test_case,
)


def _make_event(
    level="TRACE",
    trace_ids=None,
    span_ids=None,
    spans=None,
    reference_inputs=None,
):
    """Build a raw Lambda event dict for testing."""
    event = {
        "schemaVersion": "1.0",
        "evaluationLevel": level,
        "evaluationInput": {
            "sessionSpans": spans
            or [
                {
                    "traceId": "abc123",
                    "spanId": "span1",
                    "attributes": {
                        "gen_ai.message.role": "user",
                        "gen_ai.message.content": "What is the capital of France?",
                    },
                },
                {
                    "traceId": "abc123",
                    "spanId": "span2",
                    "attributes": {
                        "gen_ai.message.role": "assistant",
                        "gen_ai.message.content": "The capital of France is Paris.",
                    },
                },
            ]
        },
        "evaluationTarget": {},
    }
    if trace_ids is not None:
        event["evaluationTarget"]["traceIds"] = trace_ids
    if span_ids is not None:
        event["evaluationTarget"]["spanIds"] = span_ids
    if reference_inputs is not None:
        event["evaluationReferenceInputs"] = reference_inputs
    return event


def _mock_metric(name="MockMetric", required_params=None, evaluation_params=None, threshold=0.5):
    """Create a mock DeepEval metric."""
    metric = MagicMock()
    type(metric).__name__ = name
    metric.threshold = threshold

    if required_params is not None:
        metric._required_params = required_params
    else:
        del metric._required_params

    if evaluation_params is not None:
        metric.evaluation_params = evaluation_params
    else:
        del metric.evaluation_params

    return metric


class TestParsedEvaluationEvent:
    def test_from_lambda_event_trace_level(self):
        event = _make_event(level="TRACE", trace_ids=["trace-1"])
        parsed = ParsedEvaluationEvent.from_lambda_event(event)

        assert parsed.evaluation_level == "TRACE"
        assert parsed.target_trace_id == "trace-1"
        assert parsed.target_span_id is None
        assert len(parsed.session_spans) == 2

    def test_from_lambda_event_tool_call_level(self):
        event = _make_event(level="TOOL_CALL", span_ids=["span-42"])
        parsed = ParsedEvaluationEvent.from_lambda_event(event)

        assert parsed.evaluation_level == "TOOL_CALL"
        assert parsed.target_span_id == "span-42"
        assert parsed.target_trace_id is None

    def test_from_lambda_event_session_level(self):
        event = _make_event(level="SESSION")
        parsed = ParsedEvaluationEvent.from_lambda_event(event)

        assert parsed.evaluation_level == "SESSION"
        assert parsed.target_trace_id is None
        assert parsed.target_span_id is None

    def test_from_lambda_event_with_reference_inputs(self):
        refs = [{"expectedResponse": "Paris is the capital of France."}]
        event = _make_event(reference_inputs=refs)
        parsed = ParsedEvaluationEvent.from_lambda_event(event)

        assert parsed.reference_inputs == refs

    def test_from_lambda_event_missing_reference_inputs(self):
        event = _make_event()
        parsed = ParsedEvaluationEvent.from_lambda_event(event)

        assert parsed.reference_inputs == []

    def test_from_lambda_event_missing_evaluation_level_raises(self):
        event = _make_event()
        del event["evaluationLevel"]

        with pytest.raises(KeyError):
            ParsedEvaluationEvent.from_lambda_event(event)

    def test_from_lambda_event_missing_evaluation_input_raises(self):
        event = _make_event()
        del event["evaluationInput"]

        with pytest.raises(KeyError):
            ParsedEvaluationEvent.from_lambda_event(event)

    def test_from_lambda_event_missing_target_key_defaults(self):
        event = _make_event()
        del event["evaluationTarget"]
        parsed = ParsedEvaluationEvent.from_lambda_event(event)

        assert parsed.target_trace_id is None
        assert parsed.target_span_id is None


class TestGetRequiredParams:
    def test_uses_required_params_attribute(self):
        metric = _mock_metric(
            required_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        )
        result = _get_required_params(metric)

        assert result == ["input", "actual_output"]

    def test_falls_back_to_static_registry(self):
        metric = _mock_metric(name="FaithfulnessMetric")
        result = _get_required_params(metric)

        assert result == ["input", "actual_output", "retrieval_context"]

    def test_falls_back_to_evaluation_params(self):
        metric = _mock_metric(
            name="UnknownMetric",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        )
        result = _get_required_params(metric)

        assert result == ["input", "retrieval_context"]

    def test_defaults_to_input_and_actual_output(self):
        metric = _mock_metric(name="UnknownMetric")
        result = _get_required_params(metric)

        assert result == ["input", "actual_output"]

    def test_empty_required_params_falls_through(self):
        metric = _mock_metric(name="UnknownMetric", required_params=[])
        result = _get_required_params(metric)

        assert result == ["input", "actual_output"]


class TestBuildTestCase:
    def test_basic_span_extraction(self):
        event = _make_event()
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="AnswerRelevancyMetric")

        test_case = build_test_case(parsed, metric)

        assert test_case.input == "What is the capital of France?"
        assert test_case.actual_output == "The capital of France is Paris."

    def test_retrieval_context_from_tool_spans(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"gen_ai.message.role": "user", "gen_ai.message.content": "query"},
            },
            {
                "traceId": "t1",
                "spanId": "s2",
                "attributes": {"gen_ai.message.role": "tool", "gen_ai.message.content": "doc chunk 1"},
            },
            {
                "traceId": "t1",
                "spanId": "s3",
                "attributes": {"gen_ai.message.role": "tool", "gen_ai.message.content": "doc chunk 2"},
            },
            {
                "traceId": "t1",
                "spanId": "s4",
                "attributes": {"gen_ai.message.role": "assistant", "gen_ai.message.content": "answer"},
            },
        ]
        event = _make_event(spans=spans)
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="FaithfulnessMetric")

        test_case = build_test_case(parsed, metric)

        assert test_case.input == "query"
        assert test_case.actual_output == "answer"
        assert test_case.retrieval_context == ["doc chunk 1", "doc chunk 2"]

    def test_expected_output_from_reference_inputs(self):
        refs = [{"expectedResponse": "Paris"}]
        event = _make_event(reference_inputs=refs)
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="AnswerRelevancyMetric")

        test_case = build_test_case(parsed, metric)

        assert test_case.expected_output == "Paris"

    def test_missing_required_field_raises_value_error(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"gen_ai.message.role": "user", "gen_ai.message.content": "query"},
            },
            {
                "traceId": "t1",
                "spanId": "s2",
                "attributes": {"gen_ai.message.role": "assistant", "gen_ai.message.content": "answer"},
            },
        ]
        event = _make_event(spans=spans)
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="FaithfulnessMetric")

        with pytest.raises(ValueError, match="retrieval_context"):
            build_test_case(parsed, metric)

    def test_custom_field_mapper_bypasses_extraction(self):
        event = _make_event()
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="AnswerRelevancyMetric")

        def custom_mapper(raw_event):
            return {
                "input": "custom input",
                "actual_output": "custom output",
            }

        test_case = build_test_case(parsed, metric, field_mapper=custom_mapper)

        assert test_case.input == "custom input"
        assert test_case.actual_output == "custom output"

    def test_field_mapper_receives_reconstructed_event(self):
        refs = [{"expectedResponse": "expected"}]
        event = _make_event(level="TRACE", trace_ids=["t1"], reference_inputs=refs)
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="AnswerRelevancyMetric")

        received_events = []

        def capture_mapper(raw_event):
            received_events.append(raw_event)
            return {"input": "x", "actual_output": "y"}

        build_test_case(parsed, metric, field_mapper=capture_mapper)

        raw = received_events[0]
        assert raw["evaluationLevel"] == "TRACE"
        assert raw["evaluationTarget"]["traceIds"] == ["t1"]
        assert raw["evaluationReferenceInputs"] == refs

    def test_multiple_user_messages_concatenated(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"gen_ai.message.role": "user", "gen_ai.message.content": "hello"},
            },
            {
                "traceId": "t1",
                "spanId": "s2",
                "attributes": {"gen_ai.message.role": "user", "gen_ai.message.content": "world"},
            },
            {
                "traceId": "t1",
                "spanId": "s3",
                "attributes": {"gen_ai.message.role": "assistant", "gen_ai.message.content": "hi"},
            },
        ]
        event = _make_event(spans=spans)
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="AnswerRelevancyMetric")

        test_case = build_test_case(parsed, metric)

        assert test_case.input == "hello\nworld"

    def test_gen_ai_completion_fallback(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"gen_ai.message.role": "user", "gen_ai.completion": "fallback input"},
            },
            {
                "traceId": "t1",
                "spanId": "s2",
                "attributes": {"gen_ai.message.role": "assistant", "gen_ai.completion": "fallback output"},
            },
        ]
        event = _make_event(spans=spans)
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="AnswerRelevancyMetric")

        test_case = build_test_case(parsed, metric)

        assert test_case.input == "fallback input"
        assert test_case.actual_output == "fallback output"
