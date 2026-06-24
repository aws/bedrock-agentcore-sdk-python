"""Tests for AutoevalsAdapter."""

import json
import time
from unittest.mock import MagicMock

import pytest

from bedrock_agentcore.evaluation.integrations.autoevals.adapter import AutoevalsAdapter


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


def _mock_scorer(score=0.9, rationale="Good answer"):
    """Create a mock Autoevals scorer."""
    scorer = MagicMock()
    type(scorer).__name__ = "MockScorer"

    result = MagicMock()
    result.score = score
    result.metadata = {"rationale": rationale}

    scorer.eval = MagicMock(return_value=result)
    return scorer


class TestAutoevalsAdapterSuccess:
    def test_returns_pass_when_score_above_half(self):
        scorer = _mock_scorer(score=0.8)
        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_event())

        assert result["value"] == 0.8
        assert result["label"] == "Pass"
        assert result["explanation"] == "Good answer"

    def test_returns_fail_when_score_below_half(self):
        scorer = _mock_scorer(score=0.3)
        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_event())

        assert result["value"] == 0.3
        assert result["label"] == "Fail"

    def test_scorer_eval_called_with_input_and_output(self):
        scorer = _mock_scorer()
        adapter = AutoevalsAdapter(scorer=scorer)

        adapter(_make_event())

        scorer.eval.assert_called_once()
        call_kwargs = scorer.eval.call_args[1]
        assert call_kwargs["input"] == "What is AI?"
        assert call_kwargs["output"] == "AI is artificial intelligence."

    def test_expected_output_passed_as_expected(self):
        scorer = _mock_scorer()
        adapter = AutoevalsAdapter(scorer=scorer)

        refs = [{"expectedResponse": "AI stands for artificial intelligence."}]
        result = adapter(_make_event(reference_inputs=refs))

        call_kwargs = scorer.eval.call_args[1]
        assert call_kwargs["expected"] == "AI stands for artificial intelligence."

    def test_no_expected_output_omits_expected_kwarg(self):
        scorer = _mock_scorer()
        adapter = AutoevalsAdapter(scorer=scorer)

        adapter(_make_event())

        call_kwargs = scorer.eval.call_args[1]
        assert "expected" not in call_kwargs

    def test_custom_field_mapper(self):
        scorer = _mock_scorer()
        adapter = AutoevalsAdapter(
            scorer=scorer,
            field_mapper=lambda event: {
                "input": "custom input",
                "actual_output": "custom output",
            },
        )

        result = adapter(_make_event())

        call_kwargs = scorer.eval.call_args[1]
        assert call_kwargs["input"] == "custom input"
        assert call_kwargs["output"] == "custom output"


class TestAutoevalsAdapterErrors:
    def test_invalid_event_returns_error(self):
        scorer = _mock_scorer()
        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter({})

        assert result["errorCode"] == "INVALID_EVENT"

    def test_missing_input_returns_error(self):
        log_records = [
            {
                "body": {
                    "output": {"messages": [{"role": "assistant", "content": "answer"}]},
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
        scorer = _mock_scorer()
        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_event(spans=spans))

        assert result["errorCode"] == "MISSING_REQUIRED_FIELD"
        assert "input" in result["errorMessage"]

    def test_scorer_exception_returns_error(self):
        scorer = _mock_scorer()
        scorer.eval = MagicMock(side_effect=RuntimeError("API error"))
        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_event())

        assert result["errorCode"] == "METRIC_ERROR"
        assert "API error" in result["errorMessage"]

    def test_never_raises_on_bad_input(self):
        scorer = _mock_scorer()
        adapter = AutoevalsAdapter(scorer=scorer)

        for bad_input in [None, [], "string", 42]:
            result = adapter(bad_input)
            assert "errorCode" in result


class TestAutoevalsAdapterTimeout:
    def test_timeout_returns_error(self):
        scorer = _mock_scorer()
        scorer.eval = MagicMock(side_effect=lambda **kw: time.sleep(5))
        adapter = AutoevalsAdapter(scorer=scorer, timeout=1)

        result = adapter(_make_event())

        assert result["errorCode"] == "METRIC_TIMEOUT"

    def test_default_timeout_is_290(self):
        scorer = _mock_scorer()
        adapter = AutoevalsAdapter(scorer=scorer)

        assert adapter.timeout == 290


class TestAutoevalsAdapterEdgeCases:
    def test_score_none_returns_fail(self):
        scorer = _mock_scorer(score=None)
        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_event())

        assert result["label"] == "Fail"

    def test_no_metadata_returns_empty_explanation(self):
        scorer = MagicMock()
        type(scorer).__name__ = "MockScorer"
        result_obj = MagicMock(spec=[])
        result_obj.score = 0.9
        scorer.eval = MagicMock(return_value=result_obj)

        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_event())

        assert result["explanation"] == ""
