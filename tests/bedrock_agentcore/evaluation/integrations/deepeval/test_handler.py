"""Tests for DeepEvalHandler."""

import json
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.evaluation.integrations.deepeval.handler import DeepEvalHandler


def _make_event(
    level="TRACE",
    trace_ids=None,
    spans=None,
    reference_inputs=None,
):
    """Build a raw Lambda event dict for testing."""
    if spans is None:
        log_records = [
            {
                "body": {
                    "input": {"messages": [{"role": "user", "content": "What is AI?"}]},
                    "output": {"messages": [{"role": "assistant", "content": "AI is artificial intelligence."}]},
                }
            }
        ]
        spans = [
            {
                "traceId": "abc123",
                "spanId": "span1",
                "attributes": {"_eval_log_records": json.dumps(log_records)},
            }
        ]

    event = {
        "schemaVersion": "1.0",
        "evaluationLevel": level,
        "evaluationInput": {"sessionSpans": spans},
        "evaluationTarget": {},
    }
    if trace_ids is not None:
        event["evaluationTarget"]["traceIds"] = trace_ids
    if reference_inputs is not None:
        event["evaluationReferenceInputs"] = reference_inputs
    return event


def _mock_metric(score=0.85, reason="Looks good", threshold=0.7, name="MockMetric"):
    """Create a mock metric that returns a fixed score on measure()."""
    metric = MagicMock()
    type(metric).__name__ = name
    metric.threshold = threshold
    metric.score = score
    metric.reason = reason
    metric._required_params = None
    del metric._required_params
    del metric.evaluation_params

    def measure_side_effect(test_case):
        metric.score = score
        metric.reason = reason

    metric.measure = MagicMock(side_effect=measure_side_effect)
    return metric


class TestDeepEvalHandlerSuccess:
    def test_returns_pass_when_score_above_threshold(self):
        metric = _mock_metric(score=0.9, threshold=0.7)
        handler = DeepEvalHandler(metric=metric)

        result = handler(_make_event())

        assert result["value"] == 0.9
        assert result["label"] == "Pass"
        assert result["explanation"] == "Looks good"

    def test_returns_fail_when_score_below_threshold(self):
        metric = _mock_metric(score=0.3, threshold=0.7)
        handler = DeepEvalHandler(metric=metric)

        result = handler(_make_event())

        assert result["value"] == 0.3
        assert result["label"] == "Fail"

    def test_returns_pass_at_exact_threshold(self):
        metric = _mock_metric(score=0.7, threshold=0.7)
        handler = DeepEvalHandler(metric=metric)

        result = handler(_make_event())

        assert result["label"] == "Pass"

    def test_metric_measure_called_with_test_case(self):
        metric = _mock_metric()
        handler = DeepEvalHandler(metric=metric)

        handler(_make_event())

        metric.measure.assert_called_once()
        test_case = metric.measure.call_args[0][0]
        assert test_case.input == "What is AI?"
        assert test_case.actual_output == "AI is artificial intelligence."

    def test_context_parameter_ignored(self):
        metric = _mock_metric()
        handler = DeepEvalHandler(metric=metric)
        mock_context = {"function_name": "my-lambda"}

        result = handler(_make_event(), mock_context)

        assert result["value"] == 0.85

    def test_custom_field_mapper(self):
        metric = _mock_metric()
        handler = DeepEvalHandler(
            metric=metric,
            field_mapper=lambda event: {
                "input": "mapped input",
                "actual_output": "mapped output",
            },
        )

        result = handler(_make_event())

        assert result["value"] == 0.85
        test_case = metric.measure.call_args[0][0]
        assert test_case.input == "mapped input"
        assert test_case.actual_output == "mapped output"


class TestDeepEvalHandlerErrors:
    def test_invalid_event_returns_error(self):
        metric = _mock_metric()
        handler = DeepEvalHandler(metric=metric)

        result = handler({})

        assert result["errorCode"] == "INVALID_EVENT"
        assert "errorMessage" in result
        assert "value" not in result

    def test_missing_evaluation_input_returns_error(self):
        metric = _mock_metric()
        handler = DeepEvalHandler(metric=metric)

        event = {"evaluationLevel": "TRACE", "evaluationTarget": {}}
        result = handler(event)

        assert result["errorCode"] == "INVALID_EVENT"

    def test_missing_required_field_returns_error(self):
        log_records = [
            {
                "body": {
                    "input": {"messages": [{"role": "user", "content": "q"}]},
                    "output": {"messages": [{"role": "assistant", "content": "a"}]},
                }
            }
        ]
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"_eval_log_records": json.dumps(log_records)},
            }
        ]
        metric = _mock_metric(name="FaithfulnessMetric")
        handler = DeepEvalHandler(metric=metric)

        event = _make_event(spans=spans)
        result = handler(event)

        assert result["errorCode"] == "MISSING_REQUIRED_FIELD"
        assert "retrieval_context" in result["errorMessage"]

    def test_metric_measure_exception_returns_error(self):
        metric = _mock_metric()
        metric.measure = MagicMock(side_effect=RuntimeError("LLM timeout"))
        handler = DeepEvalHandler(metric=metric)

        result = handler(_make_event())

        assert result["errorCode"] == "METRIC_ERROR"
        assert "LLM timeout" in result["errorMessage"]

    def test_never_raises_on_any_input(self):
        metric = _mock_metric()
        handler = DeepEvalHandler(metric=metric)

        for bad_input in [None, [], "string", 42, {"random": "keys"}]:
            result = handler(bad_input)
            assert "errorCode" in result or "value" in result


class TestDeepEvalHandlerEdgeCases:
    def test_metric_with_no_reason(self):
        metric = _mock_metric(score=0.8, reason=None)
        handler = DeepEvalHandler(metric=metric)

        result = handler(_make_event())

        assert result["explanation"] == ""

    def test_metric_score_zero(self):
        metric = _mock_metric(score=0.0, threshold=0.5)
        handler = DeepEvalHandler(metric=metric)

        result = handler(_make_event())

        assert result["value"] == 0.0
        assert result["label"] == "Fail"

    def test_metric_score_one(self):
        metric = _mock_metric(score=1.0, threshold=0.5)
        handler = DeepEvalHandler(metric=metric)

        result = handler(_make_event())

        assert result["value"] == 1.0
        assert result["label"] == "Pass"

    def test_default_threshold_when_missing(self):
        metric = _mock_metric(score=0.6)
        del metric.threshold
        handler = DeepEvalHandler(metric=metric)

        result = handler(_make_event())

        assert result["label"] == "Pass"
