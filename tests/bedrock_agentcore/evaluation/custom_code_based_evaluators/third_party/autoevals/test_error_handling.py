"""Error-handling tests for AutoEvalsAdapter — covers all error paths.

Validates that error responses return only errorCode + errorMessage (no label, no explanation)
per the service contract: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-based-evaluators.html
"""

from unittest.mock import MagicMock

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.autoevals.adapter import AutoEvalsAdapter

# --- Helpers ---


def _make_evaluator_input(
    spans=None,
    evaluation_level="TRACE",
    target_trace_id="t1",
    target_span_id=None,
    reference_inputs=None,
):
    return EvaluatorInput(
        evaluation_level=evaluation_level,
        session_spans=spans if spans is not None else [],
        target_trace_id=target_trace_id,
        target_span_id=target_span_id,
        reference_inputs=reference_inputs or [],
    )


def _valid_strands_spans():
    """Single-turn Strands spans (valid for extraction)."""
    return [
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
                "output": {
                    "messages": [{"role": "assistant", "content": {"message": "AI is artificial intelligence."}}]
                },
            },
        },
    ]


def _mock_scorer(name="Factuality"):
    scorer = MagicMock()
    type(scorer).__name__ = name
    result = MagicMock()
    result.score = 0.9
    result.metadata = {"rationale": "Good"}
    scorer.eval = MagicMock(return_value=result)
    return scorer


def _assert_error_response(result, error_code):
    """Assert the response follows the error contract."""
    assert isinstance(result, EvaluatorOutput)
    assert result.errorCode == error_code
    assert result.errorMessage is not None and len(result.errorMessage) > 0
    assert result.label is None
    assert result.explanation is None


# =============================================================================
# FIELD_EXTRACTION_ERROR — Span parsing failures (6 scenarios)
# =============================================================================


class TestFieldExtractionError:
    """Scenarios where spans cannot be parsed into evaluation fields."""

    def test_01_empty_spans(self):
        adapter = AutoEvalsAdapter(metric=_mock_scorer())
        result = adapter(_make_evaluator_input(spans=[]))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_02_unrecognized_scope(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "unknown.framework"},
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
                "name": "invoke_agent",
                "kind": "INTERNAL",
                "startTimeUnixNano": 1000000000,
                "endTimeUnixNano": 2000000000,
                "status": {"code": "UNSET"},
            }
        ]
        adapter = AutoEvalsAdapter(metric=_mock_scorer())
        result = adapter(_make_evaluator_input(spans=spans))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_03_spans_missing_body_input(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "name": "invoke_agent",
                "kind": "INTERNAL",
                "startTimeUnixNano": 1000000000,
                "endTimeUnixNano": 2000000000,
                "attributes": {"gen_ai.operation.name": "invoke_agent", "session.id": "s"},
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
                    "output": {"messages": [{"role": "assistant", "content": {"message": "answer"}}]},
                },
            },
        ]
        adapter = AutoEvalsAdapter(metric=_mock_scorer())
        result = adapter(_make_evaluator_input(spans=spans))
        assert result.errorCode in ("MISSING_REQUIRED_FIELD", "FIELD_EXTRACTION_ERROR")
        assert result.label is None

    def test_04_spans_missing_body_output(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "name": "invoke_agent",
                "kind": "INTERNAL",
                "startTimeUnixNano": 1000000000,
                "endTimeUnixNano": 2000000000,
                "attributes": {"gen_ai.operation.name": "invoke_agent", "session.id": "s"},
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
                    "input": {"messages": [{"role": "user", "content": {"content": '[{"text": "hi"}]'}}]},
                },
            },
        ]
        adapter = AutoEvalsAdapter(metric=_mock_scorer())
        result = adapter(_make_evaluator_input(spans=spans))
        assert result.errorCode in ("MISSING_REQUIRED_FIELD", "FIELD_EXTRACTION_ERROR")
        assert result.label is None

    def test_05_tool_call_invalid_span_id(self):
        spans = _valid_strands_spans()
        adapter = AutoEvalsAdapter(metric=_mock_scorer())
        result = adapter(
            _make_evaluator_input(
                spans=spans,
                evaluation_level="TOOL_CALL",
                target_span_id="nonexistent",
            )
        )
        assert result.errorCode in ("FIELD_EXTRACTION_ERROR", "MISSING_REQUIRED_FIELD")
        assert result.label is None

    def test_06_trace_mismatched_trace_id(self):
        spans = _valid_strands_spans()
        adapter = AutoEvalsAdapter(metric=_mock_scorer())
        result = adapter(
            _make_evaluator_input(
                spans=spans,
                evaluation_level="TRACE",
                target_trace_id="nonexistent",
            )
        )
        assert result.errorCode in ("FIELD_EXTRACTION_ERROR", "MISSING_REQUIRED_FIELD")
        assert result.label is None


# =============================================================================
# MISSING_REQUIRED_FIELD — Scorer needs expected but not provided (4 scenarios)
# =============================================================================


class TestMissingRequiredField:
    """Autoevals metrics that need expected_output from reference_inputs."""

    def test_07_factuality_no_expected(self):
        scorer = _mock_scorer("Factuality")
        scorer.eval = MagicMock(side_effect=TypeError("missing required argument: 'expected'"))
        adapter = AutoEvalsAdapter(metric=scorer)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_08_closed_qa_no_expected(self):
        scorer = _mock_scorer("ClosedQA")
        scorer.eval = MagicMock(side_effect=TypeError("missing required argument: 'expected'"))
        adapter = AutoEvalsAdapter(metric=scorer)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_09_answer_correctness_no_expected(self):
        scorer = _mock_scorer("AnswerCorrectness")
        scorer.eval = MagicMock(side_effect=TypeError("missing required argument: 'expected'"))
        adapter = AutoEvalsAdapter(metric=scorer)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_10_exact_match_no_expected(self):
        scorer = _mock_scorer("ExactMatch")
        scorer.eval = MagicMock(side_effect=TypeError("missing required argument: 'expected'"))
        adapter = AutoEvalsAdapter(metric=scorer)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")


# =============================================================================
# METRIC_ERROR — Runtime failures (8 scenarios)
# =============================================================================


class TestMetricError:
    """Scenarios where scorer execution fails at runtime."""

    def test_11_bedrock_throttling(self):
        scorer = _mock_scorer()
        scorer.eval = MagicMock(side_effect=Exception("ThrottlingException: Rate exceeded"))
        adapter = AutoEvalsAdapter(metric=scorer)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")
        assert "ThrottlingException" in result.errorMessage

    def test_12_bedrock_access_denied(self):
        scorer = _mock_scorer()
        scorer.eval = MagicMock(
            side_effect=Exception("AccessDeniedException: not authorized to perform bedrock:InvokeModel")
        )
        adapter = AutoEvalsAdapter(metric=scorer)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_13_bedrock_model_not_found(self):
        scorer = _mock_scorer()
        scorer.eval = MagicMock(side_effect=Exception("ResourceNotFoundException: Model not found"))
        adapter = AutoEvalsAdapter(metric=scorer)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_14_custom_mapper_raises_exception(self):
        def bad_mapper(ev):
            raise ValueError("custom mapper crashed")

        scorer = _mock_scorer()
        adapter = AutoEvalsAdapter(metric=scorer, custom_mapper=bad_mapper)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")
        assert "custom mapper crashed" in result.errorMessage

    def test_15_custom_mapper_returns_none(self):
        scorer = _mock_scorer()
        scorer.eval = MagicMock(side_effect=TypeError("argument of type 'NoneType'"))
        adapter = AutoEvalsAdapter(metric=scorer, custom_mapper=lambda ev: None)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_16_custom_mapper_returns_wrong_type(self):
        scorer = _mock_scorer()
        scorer.eval = MagicMock(side_effect=TypeError("expected dict kwargs"))
        adapter = AutoEvalsAdapter(metric=scorer, custom_mapper=lambda ev: "not a dict")
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_17_scorer_eval_network_error(self):
        scorer = _mock_scorer()
        scorer.eval = MagicMock(side_effect=ConnectionError("network unreachable"))
        adapter = AutoEvalsAdapter(metric=scorer)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_18_scorer_eval_unexpected_exception(self):
        scorer = _mock_scorer()
        scorer.eval = MagicMock(side_effect=RuntimeError("unexpected internal error"))
        adapter = AutoEvalsAdapter(metric=scorer)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")
        assert "unexpected internal error" in result.errorMessage
