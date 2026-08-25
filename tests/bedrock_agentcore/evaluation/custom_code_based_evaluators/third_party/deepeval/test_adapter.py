"""Tests for DeepEvalAdapter."""

from unittest.mock import MagicMock

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
                    "output": {
                        "messages": [{"role": "assistant", "content": {"message": "AI is artificial intelligence."}}]
                    },
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
                        "output": {
                            "messages": [
                                {"role": "assistant", "content": {"message": "AI is artificial intelligence."}}
                            ]
                        },
                    },
                },
            ],
            target_trace_id="t1",
            reference_inputs=[{"expectedResponse": {"text": "AI stands for artificial intelligence."}}],
        )

        adapter(evaluator_input)  # result unused; we check test_case

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
        assert result.label is None

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
        metric.measure = MagicMock(side_effect=MissingTestCaseParamsError("retrieval_context is required"))
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


class TestDeepEvalAdapterConversational:
    """Tests for ConversationalTestCase support."""

    def test_conversational_metric_with_multi_turn_spans(self):
        """Test that BaseConversationalMetric gets ConversationalTestCase."""
        from deepeval.metrics import BaseConversationalMetric
        from deepeval.test_case import ConversationalTestCase

        metric = MagicMock(spec=BaseConversationalMetric)
        type(metric).__name__ = "KnowledgeRetentionMetric"
        metric.threshold = 0.5
        metric.score = 0.8
        metric.reason = "Good retention"
        del metric.success

        def measure_side_effect(test_case):
            assert isinstance(test_case, ConversationalTestCase)
            assert len(test_case.turns) == 4
            metric.score = 0.8
            metric.reason = "Good retention"

        metric.measure = MagicMock(side_effect=measure_side_effect)

        adapter = DeepEvalAdapter(metric=metric)

        # Multi-turn session: 2 traces, each with an invoke_agent span
        spans = [
            # Trace 1 - metadata
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "name": "invoke_agent",
                "kind": "INTERNAL",
                "startTimeUnixNano": 1000000000,
                "endTimeUnixNano": 2000000000,
                "attributes": {"gen_ai.operation.name": "invoke_agent", "session.id": "test"},
                "status": {"code": "UNSET"},
            },
            # Trace 1 - body
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer"},
                "timeUnixNano": 2000000000,
                "observedTimeUnixNano": 2000000001,
                "severityNumber": 9,
                "body": {
                    "input": {"messages": [{"role": "user", "content": {"content": '[{"text": "Hello"}]'}}]},
                    "output": {"messages": [{"role": "assistant", "content": {"message": "Hi there!"}}]},
                },
            },
            # Trace 2 - metadata
            {
                "traceId": "t2",
                "spanId": "s2",
                "scope": {"name": "strands.telemetry.tracer"},
                "name": "invoke_agent",
                "kind": "INTERNAL",
                "startTimeUnixNano": 3000000000,
                "endTimeUnixNano": 4000000000,
                "attributes": {"gen_ai.operation.name": "invoke_agent", "session.id": "test"},
                "status": {"code": "UNSET"},
            },
            # Trace 2 - body
            {
                "traceId": "t2",
                "spanId": "s2",
                "scope": {"name": "strands.telemetry.tracer"},
                "timeUnixNano": 4000000000,
                "observedTimeUnixNano": 4000000001,
                "severityNumber": 9,
                "body": {
                    "input": {"messages": [{"role": "user", "content": {"content": '[{"text": "What is AI?"}]'}}]},
                    "output": {
                        "messages": [{"role": "assistant", "content": {"message": "AI is artificial intelligence."}}]
                    },
                },
            },
        ]

        evaluator_input = EvaluatorInput(
            evaluation_level="SESSION",
            session_spans=spans,
        )

        result = adapter(evaluator_input)

        assert result.value == 0.8
        assert result.label == "Pass"
        metric.measure.assert_called_once()

    def test_conversational_metric_single_turn_returns_error(self):
        """Test that single-turn spans return error for conversational metric."""
        from deepeval.metrics import BaseConversationalMetric

        metric = MagicMock(spec=BaseConversationalMetric)
        type(metric).__name__ = "KnowledgeRetentionMetric"

        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "FIELD_EXTRACTION_ERROR"
        assert "multi-turn" in result.errorMessage.lower() or "Multiple" in result.errorMessage


class TestDeepEvalAdapterServiceNormalizedMultiTurn:
    """Tests for conversational metrics with service-normalized SESSION format.

    The AgentCore service collapses multi-turn ADOT docs into one span with
    span_events[*].body. These tests verify the adapter correctly extracts
    all turns and passes a ConversationalTestCase to the metric.
    """

    def _make_session_evaluator_input(self, num_turns=3):
        """Build EvaluatorInput in service-normalized SESSION format.

        Uses the REAL Strands body shape: message content is double-encoded JSON
        under ``content`` / ``message`` (a JSON string, matching what the service
        actually sends), so this exercises the CloudWatch-consistent decoding path
        rather than a pre-parsed convenience shape.
        """
        import json

        span_events = []
        for i in range(num_turns):
            span_events.append({
                "event_name": "strands.telemetry.tracer",
                "body": {
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": {"content": json.dumps([{"text": f"User turn {i + 1}"}])},
                            }
                        ]
                    },
                    "output": {
                        "messages": [
                            {
                                "role": "assistant",
                                "content": {"message": json.dumps([{"text": f"Bot turn {i + 1}"}])},
                            }
                        ]
                    },
                },
            })
        spans = [
            {
                "traceId": "t-session",
                "spanId": "s-session",
                "source": "adot_cw",
                "scope": {"name": "strands.telemetry.tracer"},
                "attributes": {"session.id": "multi-turn-session"},
                "span_events": span_events,
            }
        ]
        return EvaluatorInput(
            evaluation_level="SESSION",
            session_spans=spans,
        )

    def test_conversational_metric_receives_all_turns(self):
        """Multi-turn metric gets ConversationalTestCase with correct turn count."""
        from deepeval.metrics import BaseConversationalMetric
        from deepeval.test_case import ConversationalTestCase

        metric = MagicMock(spec=BaseConversationalMetric)
        type(metric).__name__ = "GoalAccuracyMetric"
        metric.threshold = 0.5
        metric.score = 0.9
        metric.reason = "Goal achieved"
        del metric.success

        captured_test_case = {}

        def measure_side_effect(test_case):
            captured_test_case["tc"] = test_case
            metric.score = 0.9
            metric.reason = "Goal achieved"

        metric.measure = MagicMock(side_effect=measure_side_effect)
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(self._make_session_evaluator_input(num_turns=4))

        assert result.value == 0.9
        assert result.label == "Pass"
        tc = captured_test_case["tc"]
        assert isinstance(tc, ConversationalTestCase)
        assert len(tc.turns) == 8  # 4 user + 4 assistant turns

    def test_conversational_metric_turn_content_correct(self):
        """Verify turn content is correctly extracted from nested message format."""
        from deepeval.metrics import BaseConversationalMetric

        metric = MagicMock(spec=BaseConversationalMetric)
        type(metric).__name__ = "RoleAdherenceMetric"
        metric.threshold = 0.5
        metric.score = 1.0
        metric.reason = "No violations"
        del metric.success

        captured_test_case = {}

        def measure_side_effect(test_case):
            captured_test_case["tc"] = test_case
            metric.score = 1.0

        metric.measure = MagicMock(side_effect=measure_side_effect)
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(self._make_session_evaluator_input(num_turns=2))

        assert result.value == 1.0
        tc = captured_test_case["tc"]
        assert tc.turns[0].role == "user"
        assert tc.turns[0].content == "User turn 1"
        assert tc.turns[1].role == "assistant"
        assert tc.turns[1].content == "Bot turn 1"
        assert tc.turns[2].role == "user"
        assert tc.turns[2].content == "User turn 2"
        assert tc.turns[3].role == "assistant"
        assert tc.turns[3].content == "Bot turn 2"

    def test_five_turn_session_evaluation(self):
        """Realistic 5-turn session evaluation (matches typical MACE migration)."""
        from deepeval.metrics import BaseConversationalMetric

        metric = MagicMock(spec=BaseConversationalMetric)
        type(metric).__name__ = "ConversationCompletenessMetric"
        metric.threshold = 0.5
        metric.score = 0.75
        metric.reason = "Mostly complete"
        del metric.success

        metric.measure = MagicMock(side_effect=lambda tc: None)
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(self._make_session_evaluator_input(num_turns=5))

        assert result.value == 0.75
        assert result.label == "Pass"
        metric.measure.assert_called_once()
        tc = metric.measure.call_args[0][0]
        assert len(tc.turns) == 10  # 5 user + 5 assistant
