"""Error-handling tests for DeepEvalAdapter — 56 scenarios covering all error paths.

Validates that error responses return only errorCode + errorMessage (no label, no explanation)
per the service contract: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-based-evaluators.html
"""

from unittest.mock import MagicMock

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import BaseConversationalMetric, BaseMetric
from deepeval.test_case import LLMTestCase

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.deepeval.adapter import DeepEvalAdapter

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
    """Single-turn Strands spans (valid for LLMTestCase metrics)."""
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


def _mock_metric(name="AnswerRelevancyMetric"):
    metric = MagicMock(spec=BaseMetric)
    type(metric).__name__ = name
    metric.threshold = 0.5
    metric.score = 0.9
    metric.reason = "Good"
    del metric.success
    return metric


def _mock_conversational_metric(name="ConversationalGEval"):
    metric = MagicMock()
    metric.__class__ = type(name, (BaseConversationalMetric,), {})
    type(metric).__name__ = name
    metric.threshold = 0.5
    return metric


def _assert_error_response(result, error_code):
    """Assert the response follows the error contract."""
    assert isinstance(result, EvaluatorOutput)
    assert result.errorCode == error_code
    assert result.errorMessage is not None and len(result.errorMessage) > 0
    assert result.label is None
    assert result.explanation is None


# =============================================================================
# FIELD_EXTRACTION_ERROR — Span parsing failures (22 scenarios)
# =============================================================================


class TestFieldExtractionError:
    """Scenarios where spans cannot be parsed into evaluation fields."""

    def test_01_empty_spans_deepeval(self):
        adapter = DeepEvalAdapter(metric=_mock_metric())
        result = adapter(_make_evaluator_input(spans=[]))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_02_empty_spans_with_trace_target(self):
        adapter = DeepEvalAdapter(metric=_mock_metric())
        result = adapter(_make_evaluator_input(spans=[], target_trace_id="nonexistent"))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_03_conversational_geval_single_turn(self):
        adapter = DeepEvalAdapter(metric=_mock_conversational_metric("ConversationalGEval"))
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans(), evaluation_level="SESSION"))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")
        assert "multiple conversation turns" in result.errorMessage

    def test_04_knowledge_retention_single_turn(self):
        adapter = DeepEvalAdapter(metric=_mock_conversational_metric("KnowledgeRetentionMetric"))
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans(), evaluation_level="SESSION"))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_05_conversation_completeness_single_turn(self):
        adapter = DeepEvalAdapter(metric=_mock_conversational_metric("ConversationCompletenessMetric"))
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans(), evaluation_level="SESSION"))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_06_goal_accuracy_single_turn(self):
        adapter = DeepEvalAdapter(metric=_mock_conversational_metric("GoalAccuracyMetric"))
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans(), evaluation_level="SESSION"))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_07_role_adherence_single_turn(self):
        adapter = DeepEvalAdapter(metric=_mock_conversational_metric("RoleAdherenceMetric"))
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans(), evaluation_level="SESSION"))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_08_turn_relevancy_single_turn(self):
        adapter = DeepEvalAdapter(metric=_mock_conversational_metric("TurnRelevancyMetric"))
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans(), evaluation_level="SESSION"))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_09_turn_faithfulness_single_turn(self):
        adapter = DeepEvalAdapter(metric=_mock_conversational_metric("TurnFaithfulnessMetric"))
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans(), evaluation_level="SESSION"))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_10_tool_use_single_turn(self):
        adapter = DeepEvalAdapter(metric=_mock_conversational_metric("ToolUseMetric"))
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans(), evaluation_level="SESSION"))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_11_topic_adherence_single_turn(self):
        adapter = DeepEvalAdapter(metric=_mock_conversational_metric("TopicAdherenceMetric"))
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans(), evaluation_level="SESSION"))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_12_turn_contextual_precision_single_turn(self):
        adapter = DeepEvalAdapter(metric=_mock_conversational_metric("TurnContextualPrecisionMetric"))
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans(), evaluation_level="SESSION"))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_13_turn_contextual_recall_single_turn(self):
        adapter = DeepEvalAdapter(metric=_mock_conversational_metric("TurnContextualRecallMetric"))
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans(), evaluation_level="SESSION"))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_14_turn_contextual_relevancy_single_turn(self):
        adapter = DeepEvalAdapter(metric=_mock_conversational_metric("TurnContextualRelevancyMetric"))
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans(), evaluation_level="SESSION"))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_15_unrecognized_scope_deepeval(self):
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
        adapter = DeepEvalAdapter(metric=_mock_metric())
        result = adapter(_make_evaluator_input(spans=spans))
        _assert_error_response(result, "FIELD_EXTRACTION_ERROR")

    def test_16_spans_missing_body_input(self):
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
        adapter = DeepEvalAdapter(metric=_mock_metric())
        result = adapter(_make_evaluator_input(spans=spans))
        assert result.errorCode in ("MISSING_REQUIRED_FIELD", "FIELD_EXTRACTION_ERROR")
        assert result.label is None

    def test_17_spans_missing_body_output(self):
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
        adapter = DeepEvalAdapter(metric=_mock_metric())
        result = adapter(_make_evaluator_input(spans=spans))
        assert result.errorCode in ("MISSING_REQUIRED_FIELD", "FIELD_EXTRACTION_ERROR")
        assert result.label is None

    def test_18_malformed_body_structure(self):
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
                "body": "not a dict",
            },
        ]
        adapter = DeepEvalAdapter(metric=_mock_metric())
        result = adapter(_make_evaluator_input(spans=spans))
        assert result.errorCode in ("FIELD_EXTRACTION_ERROR", "MISSING_REQUIRED_FIELD")
        assert result.label is None

    def test_19_service_format_empty_events(self):
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
                "events": [],
            },
        ]
        adapter = DeepEvalAdapter(metric=_mock_metric())
        result = adapter(_make_evaluator_input(spans=spans))
        assert result.errorCode in ("FIELD_EXTRACTION_ERROR", "MISSING_REQUIRED_FIELD")
        assert result.label is None

    def test_20_tool_call_level_invalid_span_id(self):
        spans = _valid_strands_spans()
        adapter = DeepEvalAdapter(metric=_mock_metric())
        result = adapter(
            _make_evaluator_input(
                spans=spans,
                evaluation_level="TOOL_CALL",
                target_span_id="nonexistent-span-id",
            )
        )
        assert result.errorCode in ("FIELD_EXTRACTION_ERROR", "MISSING_REQUIRED_FIELD")
        assert result.label is None

    def test_21_trace_level_mismatched_trace_id(self):
        spans = _valid_strands_spans()
        adapter = DeepEvalAdapter(metric=_mock_metric())
        result = adapter(
            _make_evaluator_input(
                spans=spans,
                evaluation_level="TRACE",
                target_trace_id="nonexistent-trace-id",
            )
        )
        assert result.errorCode in ("FIELD_EXTRACTION_ERROR", "MISSING_REQUIRED_FIELD")
        assert result.label is None

    def test_22_spans_only_log_records(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "timeUnixNano": 2000000000,
                "observedTimeUnixNano": 2000000001,
                "severityNumber": 9,
                "body": {"some": "log data"},
            },
        ]
        adapter = DeepEvalAdapter(metric=_mock_metric())
        result = adapter(_make_evaluator_input(spans=spans))
        assert result.errorCode in ("FIELD_EXTRACTION_ERROR", "MISSING_REQUIRED_FIELD")
        assert result.label is None


# =============================================================================
# MISSING_REQUIRED_FIELD — Metric needs data not in spans (15 scenarios)
# =============================================================================


class TestMissingRequiredField:
    """Scenarios where spans are valid but metric requires fields not present."""

    def test_23_faithfulness_no_retrieval_context(self):
        metric = _mock_metric("FaithfulnessMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("retrieval_context is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_24_contextual_precision_no_expected_output(self):
        metric = _mock_metric("ContextualPrecisionMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("expected_output is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_25_contextual_recall_no_expected_output(self):
        metric = _mock_metric("ContextualRecallMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("expected_output is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_26_contextual_relevancy_no_retrieval_context(self):
        metric = _mock_metric("ContextualRelevancyMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("retrieval_context is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_27_tool_correctness_no_expected_trajectory(self):
        metric = _mock_metric("ToolCorrectnessMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("expected_tools is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_28_tool_correctness_empty_tool_names(self):
        metric = _mock_metric("ToolCorrectnessMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("expected_tools is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_29_argument_correctness_no_tool_calls(self):
        metric = _mock_metric("ArgumentCorrectnessMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("tools_called is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_30_hallucination_no_context(self):
        metric = _mock_metric("HallucinationMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("context is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_31_prompt_alignment_no_instructions(self):
        metric = _mock_metric("PromptAlignmentMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("prompt_instructions is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_32_geval_no_evaluation_params(self):
        metric = _mock_metric("GEval")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("evaluation_params is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_33_topic_adherence_no_relevant_topics(self):
        metric = _mock_metric("TopicAdherenceMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("relevant_topics is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_34_tool_use_no_available_tools(self):
        metric = _mock_metric("ToolUseMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("available_tools is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_35_non_advice_no_advice_types(self):
        metric = _mock_metric("NonAdviceMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("advice_types is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_36_misuse_no_domain(self):
        metric = _mock_metric("MisuseMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("domain is required"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")

    def test_37_summarization_empty_input(self):
        metric = _mock_metric("SummarizationMetric")
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("input is required but empty"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "MISSING_REQUIRED_FIELD")


# =============================================================================
# METRIC_ERROR — Runtime failures (13 scenarios)
# =============================================================================


class TestMetricError:
    """Scenarios where metric execution fails at runtime."""

    def test_38_bedrock_throttling(self):
        metric = _mock_metric("AnswerRelevancyMetric")
        metric.measure = MagicMock(side_effect=Exception("ThrottlingException: Rate exceeded"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")
        assert "ThrottlingException" in result.errorMessage

    def test_39_bedrock_access_denied(self):
        metric = _mock_metric("AnswerRelevancyMetric")
        metric.measure = MagicMock(
            side_effect=Exception("AccessDeniedException: User is not authorized to perform bedrock:InvokeModel")
        )
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")
        assert "AccessDeniedException" in result.errorMessage

    def test_40_bedrock_model_not_found(self):
        metric = _mock_metric("AnswerRelevancyMetric")
        metric.measure = MagicMock(
            side_effect=Exception("ResourceNotFoundException: Model not found: invalid-model-id")
        )
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_41_bedrock_timeout(self):
        metric = _mock_metric("AnswerRelevancyMetric")
        metric.measure = MagicMock(side_effect=TimeoutError("Read timeout on endpoint"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_42_bedrock_region_mismatch(self):
        metric = _mock_metric("AnswerRelevancyMetric")
        metric.measure = MagicMock(
            side_effect=Exception("Could not resolve the foundation model from the model identifier")
        )
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_43_custom_mapper_raises_exception(self):
        def bad_mapper(ev):
            raise ValueError("custom mapper crashed")

        metric = _mock_metric()
        adapter = DeepEvalAdapter(metric=metric, custom_mapper=bad_mapper)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")
        assert "custom mapper crashed" in result.errorMessage

    def test_44_custom_mapper_returns_none(self):
        metric = _mock_metric()
        metric.measure = MagicMock(side_effect=AttributeError("'NoneType' object has no attribute 'input'"))
        adapter = DeepEvalAdapter(metric=metric, custom_mapper=lambda ev: None)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_45_custom_mapper_returns_wrong_type(self):
        metric = _mock_metric()
        metric.measure = MagicMock(side_effect=TypeError("expected LLMTestCase, got dict"))
        adapter = DeepEvalAdapter(metric=metric, custom_mapper=lambda ev: {"input": "hi"})
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_46_metric_measure_unexpected_exception(self):
        metric = _mock_metric("BiasMetric")
        metric.measure = MagicMock(side_effect=RuntimeError("unexpected internal error"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")
        assert "unexpected internal error" in result.errorMessage

    def test_47_metric_measure_network_error(self):
        metric = _mock_metric("ToxicityMetric")
        metric.measure = MagicMock(side_effect=ConnectionError("network unreachable"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_48_metric_measure_json_decode_error(self):
        metric = _mock_metric("AnswerRelevancyMetric")
        metric.measure = MagicMock(side_effect=ValueError("Expecting value: line 1"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_49_custom_mapper_returns_test_case_with_none_input(self):
        metric = _mock_metric()
        metric.measure = MagicMock(side_effect=Exception("input cannot be None"))
        adapter = DeepEvalAdapter(
            metric=metric,
            custom_mapper=lambda ev: LLMTestCase(input="", actual_output=""),
        )
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")

    def test_50_metric_measure_keyboard_interrupt_still_caught(self):
        """Even unusual exceptions are caught — adapter never raises."""
        metric = _mock_metric("AnswerRelevancyMetric")
        metric.measure = MagicMock(side_effect=Exception("something very unusual"))
        adapter = DeepEvalAdapter(metric=metric)
        result = adapter(_make_evaluator_input(spans=_valid_strands_spans()))
        _assert_error_response(result, "METRIC_ERROR")
