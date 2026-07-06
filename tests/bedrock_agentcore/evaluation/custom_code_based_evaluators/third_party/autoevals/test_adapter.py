"""Tests for AutoevalsAdapter."""

from unittest.mock import MagicMock

import pytest

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.autoevals.adapter import AutoevalsAdapter


def _make_evaluator_input(spans=None):
    """Build an EvaluatorInput with agent-level spans."""
    if spans is None:
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer", "version": ""},
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
                "span_events": [
                    {
                        "body": {
                            "input": {"messages": [{"role": "user", "content": "What is AI?"}]},
                            "output": {"messages": [{"role": "assistant", "content": "AI is artificial intelligence."}]},
                        }
                    }
                ],
            }
        ]
    return EvaluatorInput(
        evaluation_level="TRACE",
        session_spans=spans,
        target_trace_id="t1",
    )


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
    def test_returns_pass_when_score_above_threshold(self):
        scorer = _mock_scorer(score=0.8)
        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.value == 0.8
        assert result.label == "Pass"
        assert result.explanation == "Good answer"

    def test_returns_fail_when_score_below_threshold(self):
        scorer = _mock_scorer(score=0.3)
        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.3
        assert result.label == "Fail"

    def test_custom_threshold(self):
        scorer = _mock_scorer(score=0.6)
        adapter = AutoevalsAdapter(scorer=scorer, threshold=0.7)

        result = adapter(_make_evaluator_input())

        assert result.label == "Fail"

    def test_custom_threshold_pass(self):
        scorer = _mock_scorer(score=0.8)
        adapter = AutoevalsAdapter(scorer=scorer, threshold=0.7)

        result = adapter(_make_evaluator_input())

        assert result.label == "Pass"

    def test_default_threshold_is_half(self):
        scorer = _mock_scorer()
        adapter = AutoevalsAdapter(scorer=scorer)

        assert adapter.threshold == 0.5

    def test_scorer_eval_called_with_input_and_output(self):
        scorer = _mock_scorer()
        adapter = AutoevalsAdapter(scorer=scorer)

        adapter(_make_evaluator_input())

        scorer.eval.assert_called_once()
        call_kwargs = scorer.eval.call_args[1]
        assert call_kwargs["input"] == "What is AI?"
        assert call_kwargs["output"] == "AI is artificial intelligence."

    def test_custom_customer_mapper(self):
        scorer = _mock_scorer()
        adapter = AutoevalsAdapter(
            scorer=scorer,
            customer_mapper=lambda ev: {
                "input": "custom input",
                "output": "custom output",
            },
        )

        result = adapter(_make_evaluator_input())

        call_kwargs = scorer.eval.call_args[1]
        assert call_kwargs["input"] == "custom input"
        assert call_kwargs["output"] == "custom output"


class TestAutoevalsAdapterErrors:
    def test_no_agent_spans_returns_error(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"gen_ai.operation.name": "chat"},
                "span_events": [],
            }
        ]
        scorer = _mock_scorer()
        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_evaluator_input(spans=spans))

        assert result.errorCode == "FIELD_EXTRACTION_ERROR"

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
        scorer = _mock_scorer()
        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_evaluator_input(spans=spans))

        assert result.errorCode == "MISSING_REQUIRED_FIELD"
        assert "input" in result.errorMessage

    def test_scorer_exception_returns_error(self):
        scorer = _mock_scorer()
        scorer.eval = MagicMock(side_effect=RuntimeError("API error"))
        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "METRIC_ERROR"
        assert "API error" in result.errorMessage

    def test_never_raises(self):
        scorer = _mock_scorer()
        scorer.eval = MagicMock(side_effect=Exception("unexpected"))
        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.errorCode is not None


class TestAutoevalsAdapterEdgeCases:
    def test_score_none_returns_fail(self):
        scorer = _mock_scorer(score=None)
        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_evaluator_input())

        assert result.label == "Fail"

    def test_no_metadata_returns_empty_explanation(self):
        scorer = MagicMock()
        type(scorer).__name__ = "MockScorer"
        result_obj = MagicMock(spec=[])
        result_obj.score = 0.9
        scorer.eval = MagicMock(return_value=result_obj)

        adapter = AutoevalsAdapter(scorer=scorer)

        result = adapter(_make_evaluator_input())

        assert result.explanation == ""
