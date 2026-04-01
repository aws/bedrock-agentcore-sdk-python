"""Tests for EvaluatorInput and EvaluatorOutput dataclasses."""

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput


class TestEvaluatorInput:
    def test_all_fields(self):
        inp = EvaluatorInput(
            evaluation_level="TRACE",
            session_spans=[{"traceId": "t1", "spanId": "s1"}],
            target_trace_id="t1",
            target_span_id=None,
            schema_version="1.0",
        )
        assert inp.evaluation_level == "TRACE"
        assert inp.session_spans == [{"traceId": "t1", "spanId": "s1"}]
        assert inp.target_trace_id == "t1"
        assert inp.target_span_id is None
        assert inp.schema_version == "1.0"

    def test_session_level_no_targets(self):
        inp = EvaluatorInput(
            evaluation_level="SESSION",
            session_spans=[],
            target_trace_id=None,
            target_span_id=None,
            schema_version="1.0",
        )
        assert inp.target_trace_id is None
        assert inp.target_span_id is None


class TestEvaluatorOutput:
    def test_defaults(self):
        out = EvaluatorOutput(label="Pass")
        assert out.value is None
        assert out.label == "Pass"
        assert out.explanation is None

    def test_all_fields(self):
        out = EvaluatorOutput(value=0.85, label="Pass", explanation="Looks good")
        assert out.value == 0.85
        assert out.label == "Pass"
        assert out.explanation == "Looks good"

    def test_label_required(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EvaluatorOutput(value=1.0)

    def test_label_only(self):
        out = EvaluatorOutput(label="Fail")
        assert out.label == "Fail"
        assert out.value is None
