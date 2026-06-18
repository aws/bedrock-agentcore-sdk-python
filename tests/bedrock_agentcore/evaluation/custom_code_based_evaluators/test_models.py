"""Tests for EvaluatorInput and EvaluatorOutput dataclasses."""

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import (
    EvaluatorInput,
    EvaluatorOutput,
    ReferenceInput,
)


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

    def test_reference_inputs_default_empty(self):
        inp = EvaluatorInput(evaluation_level="SESSION", session_spans=[])
        assert inp.reference_inputs == []

    def test_evaluator_id_and_name_default_none(self):
        inp = EvaluatorInput(evaluation_level="SESSION", session_spans=[])
        assert inp.evaluator_id is None
        assert inp.evaluator_name is None

    def test_reference_inputs_coerced_from_dicts(self):
        # The service sends camelCase dicts; pydantic coerces them via aliases.
        inp = EvaluatorInput(
            evaluation_level="TRACE",
            session_spans=[],
            reference_inputs=[
                {
                    "context": {"spanContext": {"sessionId": "sess", "traceId": "t1"}},
                    "expectedResponse": {"text": "Paris"},
                }
            ],
        )
        assert len(inp.reference_inputs) == 1
        ref = inp.reference_inputs[0]
        assert isinstance(ref, ReferenceInput)
        assert ref.expected_response_text == "Paris"
        assert ref.context["spanContext"]["traceId"] == "t1"

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


class TestReferenceInput:
    def test_defaults(self):
        ref = ReferenceInput()
        assert ref.context == {}
        assert ref.expected_response is None
        assert ref.assertions == []
        assert ref.expected_trajectory is None
        assert ref.expected_response_text is None

    def test_expected_response_text(self):
        ref = ReferenceInput(expected_response={"text": "Paris"})
        assert ref.expected_response_text == "Paris"

    def test_alias_and_field_name_both_accepted(self):
        by_alias = ReferenceInput(expectedResponse={"text": "x"}, expectedTrajectory={"toolNames": ["a"]})
        by_name = ReferenceInput(expected_response={"text": "x"}, expected_trajectory={"toolNames": ["a"]})
        assert by_alias.expected_response_text == "x"
        assert by_name.expected_trajectory == {"toolNames": ["a"]}

    def test_extra_keys_preserved(self):
        ref = ReferenceInput.model_validate({"expectedResponse": {"text": "x"}, "futureField": 42})
        assert ref.model_extra["futureField"] == 42

    def test_assertions(self):
        ref = ReferenceInput(assertions=[{"text": "must be polite"}])
        assert ref.assertions == [{"text": "must be polite"}]
