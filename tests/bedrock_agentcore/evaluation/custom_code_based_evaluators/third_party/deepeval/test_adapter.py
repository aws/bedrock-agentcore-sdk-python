"""Tests for DeepEvalAdapter."""

from unittest.mock import MagicMock

import pytest

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.deepeval.adapter import DeepEvalAdapter


def _make_evaluator_input(spans=None):
    """Build an EvaluatorInput with agent-level spans (CloudWatch split format)."""
    if spans is None:
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "name": "invoke_agent",
                "kind": "INTERNAL",
                "startTimeUnixNano": 1000000000,
                "endTimeUnixNano": 2000000000,
                "attributes": {"gen_ai.operation.name": "invoke_agent", "session.id": "test-session"},
                "status": {"code": "UNSET"},
            },
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "timeUnixNano": 2000000000,
                "observedTimeUnixNano": 2000000001,
                "severityNumber": 9,
                "body": {
                    "input": {"messages": [{"role": "user", "content": {"content": '[{"text": "What is AI?"}]'}}]},
                    "output": {"messages": [{"role": "assistant", "content": {"message": "AI is artificial intelligence."}}]},
                },
            },
        ]
    return EvaluatorInput(
        evaluation_level="TRACE",
        session_spans=spans,
        target_trace_id="t1",
    )


def _mock_metric(score=0.85, reason="Looks good", threshold=0.7, name="MockMetric"):
    """Create a mock metric that returns a fixed score on measure()."""
    metric = MagicMock()
    type(metric).__name__ = name
    metric.threshold = threshold
    metric.score = score
    metric.reason = reason
    del metric.success

    def measure_side_effect(test_case):
        metric.score = score
        metric.reason = reason

    metric.measure = MagicMock(side_effect=measure_side_effect)
    return metric


class TestDeepEvalAdapterSuccess:
    def test_returns_pass_when_score_above_threshold(self):
        metric = _mock_metric(score=0.9, threshold=0.7)
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.value == 0.9
        assert result.label == "Pass"
        assert result.explanation == "Looks good"

    def test_returns_fail_when_score_below_threshold(self):
        metric = _mock_metric(score=0.3, threshold=0.7)
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.3
        assert result.label == "Fail"

    def test_returns_pass_at_exact_threshold(self):
        metric = _mock_metric(score=0.7, threshold=0.7)
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.label == "Pass"

    def test_metric_measure_called_with_test_case(self):
        metric = _mock_metric()
        adapter = DeepEvalAdapter(metric=metric)

        adapter(_make_evaluator_input())

        metric.measure.assert_called_once()
        test_case = metric.measure.call_args[0][0]
        assert test_case.input == "What is AI?"
        assert test_case.actual_output == "AI is artificial intelligence."

    def test_custom_custom_mapper(self):
        from deepeval.test_case import LLMTestCase

        metric = _mock_metric()
        adapter = DeepEvalAdapter(
            metric=metric,
            custom_mapper=lambda ev: LLMTestCase(
                input="mapped input",
                actual_output="mapped output",
            ),
        )

        result = adapter(_make_evaluator_input())

        assert result.value == 0.85
        test_case = metric.measure.call_args[0][0]
        assert test_case.input == "mapped input"
        assert test_case.actual_output == "mapped output"

    def test_reference_inputs_populates_expected_output(self):
        metric = _mock_metric()
        adapter = DeepEvalAdapter(metric=metric)

        evaluator_input = EvaluatorInput(
            evaluation_level="TRACE",
            session_spans=[
                {
                    "traceId": "t1",
                    "spanId": "s1",
                    "scope": {"name": "strands.telemetry.tracer"},
                    "name": "invoke_agent",
                    "kind": "INTERNAL",
                    "startTimeUnixNano": 1000000000,
                    "endTimeUnixNano": 2000000000,
                    "attributes": {"gen_ai.operation.name": "invoke_agent", "session.id": "test-session"},
                    "status": {"code": "UNSET"},
                },
                {
                    "traceId": "t1",
                    "spanId": "s1",
                    "scope": {"name": "strands.telemetry.tracer"},
                    "timeUnixNano": 2000000000,
                    "observedTimeUnixNano": 2000000001,
                    "severityNumber": 9,
                    "body": {
                        "input": {"messages": [{"role": "user", "content": {"content": '[{"text": "What is AI?"}]'}}]},
                        "output": {"messages": [{"role": "assistant", "content": {"message": "AI is artificial intelligence."}}]},
                    },
                }
            ],
            target_trace_id="t1",
            reference_inputs=[{"expectedResponse": {"text": "AI stands for artificial intelligence."}}],
        )

        result = adapter(evaluator_input)

        test_case = metric.measure.call_args[0][0]
        assert test_case.expected_output == "AI stands for artificial intelligence."

    def test_label_uses_metric_success_true(self):
        metric = _mock_metric(score=0.3, threshold=0.7)
        metric.success = True
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.3
        assert result.label == "Pass"

    def test_label_uses_metric_success_false(self):
        metric = _mock_metric(score=0.9, threshold=0.7)
        metric.success = False
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.9
        assert result.label == "Fail"


class TestDeepEvalAdapterErrors:
    def test_no_agent_spans_returns_error(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"gen_ai.operation.name": "chat"},
                "span_events": [],
            }
        ]
        metric = _mock_metric()
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input(spans=spans))

        assert isinstance(result, EvaluatorOutput)
        assert result.errorCode == "FIELD_EXTRACTION_ERROR"
        assert result.label == "Error"

    def test_missing_input_returns_error(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer", "version": ""},
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
                "span_events": [
                    {
                        "body": {
                            "output": {"messages": [{"role": "assistant", "content": "answer"}]},
                        }
                    }
                ],
            }
        ]
        metric = _mock_metric()
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input(spans=spans))

        assert result.errorCode in ("MISSING_REQUIRED_FIELD", "FIELD_EXTRACTION_ERROR")
        assert result.errorMessage  # error message present
        assert "custom_mapper" in result.errorMessage
        metric.measure.assert_not_called()

    def test_metric_measure_exception_returns_error(self):
        metric = _mock_metric()
        metric.measure = MagicMock(side_effect=RuntimeError("LLM timeout"))
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "METRIC_ERROR"
        assert "LLM timeout" in result.errorMessage

    def test_missing_params_error_caught(self):
        from deepeval.errors import MissingTestCaseParamsError

        metric = _mock_metric()
        metric.measure = MagicMock(
            side_effect=MissingTestCaseParamsError("retrieval_context is required")
        )
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "MISSING_REQUIRED_FIELD"
        assert "retrieval_context" in result.errorMessage
        assert "custom_mapper" in result.errorMessage

    def test_never_raises(self):
        metric = _mock_metric()
        metric.measure = MagicMock(side_effect=Exception("unexpected"))
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.errorCode is not None


class TestDeepEvalAdapterEdgeCases:
    def test_metric_with_no_reason(self):
        metric = _mock_metric(score=0.8, reason=None)
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.explanation == ""

    def test_metric_score_zero(self):
        metric = _mock_metric(score=0.0, threshold=0.5)
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.0
        assert result.label == "Fail"

    def test_default_threshold_when_missing(self):
        metric = _mock_metric(score=0.6)
        del metric.threshold
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.label == "Pass"
