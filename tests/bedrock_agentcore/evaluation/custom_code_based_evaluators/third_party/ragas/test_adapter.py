"""Tests for RAGASAdapter."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas.adapter import RAGASAdapter


def _make_spans(user_content='[{"text": "What is AI?"}]', assistant_message="AI is artificial intelligence."):
    """Build CloudWatch split format spans with the given user/assistant content."""
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
                "input": {"messages": [{"role": "user", "content": {"content": user_content}}]},
                "output": {"messages": [{"role": "assistant", "content": {"message": assistant_message}}]},
            },
        },
    ]


def _make_evaluator_input(spans=None, reference_inputs=None):
    """Build an EvaluatorInput with agent-level spans (CloudWatch split format)."""
    if spans is None:
        spans = _make_spans()
    return EvaluatorInput(
        evaluation_level="TRACE",
        session_spans=spans,
        target_trace_id="t1",
        reference_inputs=reference_inputs or [],
    )


def _mock_legacy_metric(name="faithfulness", threshold=0.5, score=0.85, required=None):
    """Create a mock legacy RAGAS metric scored via single_turn_score()."""
    metric = MagicMock()
    metric.name = name
    metric.threshold = threshold
    metric.required_columns = {"SINGLE_TURN": required if required is not None else {"user_input", "response"}}
    metric.single_turn_score = MagicMock(return_value=score)
    return metric


class _FakeCollectionsMetric:
    """Fake ragas.metrics.collections-style metric (no single_turn_score)."""

    name = "exact_match"

    def __init__(self, value=1.0, reason=None):
        self._value = value
        self._reason = reason
        self.score_calls = []

    async def ascore(self, reference: str, response: str):
        """Typed signature used by the adapter for field filtering/validation."""

    def score(self, **kwargs):
        self.score_calls.append(kwargs)
        return SimpleNamespace(value=self._value, reason=self._reason)


class _FakeDecoratorMetric:
    """Fake @discrete_metric/@numeric_metric-style metric with an untyped signature."""

    name = "my_check"

    def __init__(self, value="pass", reason="looks good"):
        self._value = value
        self._reason = reason
        self.score_calls = []

    async def ascore(self, *args, **kwargs):
        """Decorator metrics expose an untyped passthrough signature."""

    def score(self, **kwargs):
        self.score_calls.append(kwargs)
        return SimpleNamespace(value=self._value, reason=self._reason)


class TestRAGASAdapterSuccess:
    def test_returns_pass_when_score_above_threshold(self):
        metric = _mock_legacy_metric(threshold=0.7, score=0.9)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.value == 0.9
        assert result.label == "Pass"

    def test_returns_fail_when_score_below_threshold(self):
        metric = _mock_legacy_metric(threshold=0.7, score=0.3)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.3
        assert result.label == "Fail"

    def test_returns_pass_at_exact_threshold(self):
        metric = _mock_legacy_metric(threshold=0.7, score=0.7)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.label == "Pass"

    def test_sample_built_from_extracted_fields(self):
        metric = _mock_legacy_metric(score=0.9)
        adapter = RAGASAdapter(metric=metric)

        adapter(_make_evaluator_input())

        metric.single_turn_score.assert_called_once()
        sample = metric.single_turn_score.call_args[0][0]
        assert sample.user_input == "What is AI?"
        assert sample.response == "AI is artificial intelligence."

    def test_custom_mapper(self):
        metric = _mock_legacy_metric(score=0.9)
        adapter = RAGASAdapter(
            metric=metric,
            custom_mapper=lambda ev: {
                "user_input": "mapped input",
                "response": "mapped output",
                "retrieved_contexts": ["some context"],
            },
        )

        result = adapter(_make_evaluator_input())

        assert result.value == 0.9
        sample = metric.single_turn_score.call_args[0][0]
        assert sample.user_input == "mapped input"
        assert sample.response == "mapped output"
        assert sample.retrieved_contexts == ["some context"]

    def test_reference_inputs_populates_reference(self):
        metric = _mock_legacy_metric(name="answer_correctness", score=0.8)
        adapter = RAGASAdapter(metric=metric)

        evaluator_input = _make_evaluator_input(
            reference_inputs=[{"expectedResponse": {"text": "AI stands for artificial intelligence."}}],
        )

        result = adapter(evaluator_input)

        assert result.value == 0.8
        sample = metric.single_turn_score.call_args[0][0]
        assert sample.reference == "AI stands for artificial intelligence."

    def test_llm_override_sets_metric_llm(self):
        metric = _mock_legacy_metric()
        mock_llm = MagicMock()
        RAGASAdapter(metric=metric, llm=mock_llm)

        assert metric.llm == mock_llm

    def test_embeddings_override_sets_metric_embeddings(self):
        metric = _mock_legacy_metric()
        metric.embeddings = None
        mock_embeddings = MagicMock()
        RAGASAdapter(metric=metric, embeddings=mock_embeddings)

        assert metric.embeddings == mock_embeddings


class TestRAGASAdapterCollectionsMetrics:
    def test_scores_via_score_kwargs(self):
        metric = _FakeCollectionsMetric(value=1.0)
        adapter = RAGASAdapter(
            metric=metric,
            custom_mapper=lambda ev: {"user_input": "q", "response": "4", "reference": "4"},
        )

        result = adapter(_make_evaluator_input())

        assert result.value == 1.0
        assert result.label == "Pass"
        # Fields not in the ascore signature are filtered out
        assert metric.score_calls == [{"response": "4", "reference": "4"}]

    def test_missing_required_kwarg_returns_error(self):
        metric = _FakeCollectionsMetric()
        adapter = RAGASAdapter(metric=metric)

        # Default span extraction produces user_input/response but no reference
        result = adapter(_make_evaluator_input())

        assert result.errorCode == "MISSING_REQUIRED_FIELD"
        assert "reference" in result.errorMessage
        assert metric.score_calls == []

    def test_embedded_reference_satisfies_required_kwarg(self):
        metric = _FakeCollectionsMetric(value=0.0)
        adapter = RAGASAdapter(metric=metric)

        spans = _make_spans(user_content='[{"text": "What is 2+2?\\n\\nReference Answer:\\n4"}]')
        result = adapter(_make_evaluator_input(spans=spans))

        assert result.value == 0.0
        assert result.label == "Fail"
        assert metric.score_calls[0]["reference"] == "4"

    def test_metric_reason_used_as_explanation(self):
        metric = _FakeCollectionsMetric(value=0.8, reason="The response covers 4/5 key points")
        adapter = RAGASAdapter(
            metric=metric,
            custom_mapper=lambda ev: {"response": "a", "reference": "b"},
        )

        result = adapter(_make_evaluator_input())

        assert result.value == 0.8
        assert result.explanation == "The response covers 4/5 key points"

    def test_no_reason_falls_back_to_synthetic_explanation(self):
        metric = _FakeCollectionsMetric(value=0.8, reason=None)
        adapter = RAGASAdapter(
            metric=metric,
            custom_mapper=lambda ev: {"response": "a", "reference": "b"},
        )

        result = adapter(_make_evaluator_input())

        assert "exact_match" in result.explanation
        assert "0.8" in result.explanation

    def test_discrete_string_value_returns_categorical_label(self):
        metric = _FakeDecoratorMetric(value="excellent", reason="Meets all criteria")
        adapter = RAGASAdapter(
            metric=metric,
            custom_mapper=lambda ev: {"user_input": "q", "response": "a"},
        )

        result = adapter(_make_evaluator_input())

        assert result.label == "excellent"
        assert result.value is None
        assert result.explanation == "Meets all criteria"
        assert result.errorCode is None

    def test_decorator_metric_receives_all_fields(self):
        """Untyped ascore(*args, **kwargs) signatures get all fields passed through."""
        metric = _FakeDecoratorMetric(value=0.9, reason=None)
        adapter = RAGASAdapter(
            metric=metric,
            custom_mapper=lambda ev: {"user_input": "q", "response": "a", "my_custom_field": "x"},
        )

        result = adapter(_make_evaluator_input())

        assert result.value == 0.9
        assert metric.score_calls == [{"user_input": "q", "response": "a", "my_custom_field": "x"}]


class TestRAGASAdapterValidation:
    def test_missing_required_column_returns_error(self):
        metric = _mock_legacy_metric(name="context_precision", required={"user_input", "response", "reference"})
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "MISSING_REQUIRED_FIELD"
        assert "reference" in result.errorMessage
        assert "evaluationReferenceInputs" in result.errorMessage
        metric.single_turn_score.assert_not_called()

    def test_missing_retrieved_contexts_returns_error(self):
        metric = _mock_legacy_metric(required={"user_input", "response", "retrieved_contexts"})
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "MISSING_REQUIRED_FIELD"
        assert "retrieved_contexts" in result.errorMessage
        metric.single_turn_score.assert_not_called()

    def test_unsupported_metric_type_returns_error(self):
        class NotAMetric:
            name = "mystery"

        adapter = RAGASAdapter(metric=NotAMetric())

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "UNSUPPORTED_METRIC"


def _mock_multi_turn_metric(name="agent_goal_accuracy", score=0.8, required=None):
    """Create a mock legacy multi-turn RAGAS metric (no single_turn_score)."""
    metric = MagicMock()
    metric.name = name
    metric.threshold = 0.5
    metric.required_columns = {"MULTI_TURN": required if required is not None else {"user_input"}}
    del metric.single_turn_score
    metric.multi_turn_score = MagicMock(return_value=score)
    return metric


class TestRAGASAdapterMultiTurn:
    def test_multi_turn_metric_scores_from_extracted_turns(self):
        metric = _mock_multi_turn_metric(score=0.8)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.8
        assert result.label == "Pass"
        sample = metric.multi_turn_score.call_args[0][0]
        # Single invocation falls back to a user/assistant message pair
        assert len(sample.user_input) == 2
        assert sample.user_input[0].content == "What is AI?"
        assert sample.user_input[1].content == "AI is artificial intelligence."

    def test_expected_trajectory_builds_reference_tool_calls(self):
        metric = _mock_multi_turn_metric(required={"user_input", "reference_tool_calls"})
        adapter = RAGASAdapter(metric=metric)

        evaluator_input = _make_evaluator_input(
            reference_inputs=[{"expectedTrajectory": {"toolNames": ["search", "calculate"]}}],
        )

        result = adapter(evaluator_input)

        assert result.value == 0.8
        sample = metric.multi_turn_score.call_args[0][0]
        assert [tc.name for tc in sample.reference_tool_calls] == ["search", "calculate"]
        assert sample.reference_tool_calls[0].args == {}

    def test_missing_multi_turn_required_field_returns_error(self):
        metric = _mock_multi_turn_metric(required={"user_input", "reference_tool_calls"})
        adapter = RAGASAdapter(metric=metric)

        # No expectedTrajectory provided → reference_tool_calls unavailable
        result = adapter(_make_evaluator_input())

        assert result.errorCode == "MISSING_REQUIRED_FIELD"
        assert "reference_tool_calls" in result.errorMessage
        metric.multi_turn_score.assert_not_called()

    def test_real_tool_call_accuracy_with_custom_mapper(self):
        """ToolCallAccuracy is deterministic — validate end-to-end with the real metric."""
        from ragas.messages import AIMessage, HumanMessage, ToolCall
        from ragas.metrics import ToolCallAccuracy

        adapter = RAGASAdapter(
            metric=ToolCallAccuracy(),
            custom_mapper=lambda ev: {
                "user_input": [
                    HumanMessage(content="Book a flight to NYC"),
                    AIMessage(content="Booking", tool_calls=[ToolCall(name="book_flight", args={"to": "NYC"})]),
                ],
                "reference_tool_calls": [ToolCall(name="book_flight", args={"to": "NYC"})],
            },
        )

        result = adapter(_make_evaluator_input())

        assert result.value == 1.0
        assert result.label == "Pass"

    def test_real_tool_call_accuracy_mismatch_fails(self):
        from ragas.messages import AIMessage, HumanMessage, ToolCall
        from ragas.metrics import ToolCallAccuracy

        adapter = RAGASAdapter(
            metric=ToolCallAccuracy(),
            custom_mapper=lambda ev: {
                "user_input": [
                    HumanMessage(content="Book a flight to NYC"),
                    AIMessage(content="Booking", tool_calls=[ToolCall(name="wrong_tool", args={})]),
                ],
                "reference_tool_calls": [ToolCall(name="book_flight", args={"to": "NYC"})],
            },
        )

        result = adapter(_make_evaluator_input())

        assert result.value == 0.0
        assert result.label == "Fail"


class TestRAGASAdapterErrors:
    def test_no_agent_spans_returns_error(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"gen_ai.operation.name": "chat"},
                "span_events": [],
            }
        ]
        metric = _mock_legacy_metric()
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input(spans=spans))

        assert isinstance(result, EvaluatorOutput)
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
        metric = _mock_legacy_metric()
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input(spans=spans))

        assert result.errorCode in ("MISSING_REQUIRED_FIELD", "FIELD_EXTRACTION_ERROR")
        assert result.errorMessage

    def test_metric_execution_exception_returns_error(self):
        metric = _mock_legacy_metric()
        metric.single_turn_score = MagicMock(side_effect=RuntimeError("RAGAS timeout"))
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "METRIC_ERROR"
        assert "RAGAS timeout" in result.errorMessage

    def test_llm_not_set_returns_hint(self):
        metric = _mock_legacy_metric()
        metric.single_turn_score = MagicMock(side_effect=AssertionError("LLM is not set"))
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "METRIC_ERROR"
        assert "llm=" in result.errorMessage

    def test_import_error_returns_missing_dependency(self):
        metric = _mock_legacy_metric()
        metric.single_turn_score = MagicMock(side_effect=ImportError("No module named 'pandas'"))
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "MISSING_DEPENDENCY"
        assert "pandas" in result.errorMessage

    def test_nan_score_returns_error(self):
        metric = _mock_legacy_metric(score=float("nan"))
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "INVALID_SCORE"
        assert "NaN" in result.errorMessage

    def test_never_raises(self):
        metric = _mock_legacy_metric()
        metric.single_turn_score = MagicMock(side_effect=Exception("unexpected catastrophic failure"))
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.errorCode is not None


class TestRAGASAdapterEdgeCases:
    def test_metric_score_zero(self):
        metric = _mock_legacy_metric(score=0.0)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.0
        assert result.label == "Fail"

    def test_metric_score_one(self):
        metric = _mock_legacy_metric(score=1.0)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 1.0
        assert result.label == "Pass"

    def test_default_threshold_when_missing(self):
        metric = _mock_legacy_metric(score=0.6)
        del metric.threshold
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.label == "Pass"

    def test_math_isnan_not_triggered_by_valid_score(self):
        metric = _mock_legacy_metric(score=0.5)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert not math.isnan(result.value)


class TestRAGASAdapterEmbeddedParsing:
    """Tests for parsing reference/context embedded in the user message.

    Trace formats have no dedicated reference/context fields, so dataset
    recipes embed them in the user message with known separators.
    """

    def test_parses_embedded_reference_from_user_input(self):
        metric = _mock_legacy_metric(name="exact_match", score=1.0)
        adapter = RAGASAdapter(metric=metric)

        spans = _make_spans(user_content='[{"text": "What is 2+2?\\n\\nReference Answer:\\n4"}]')
        result = adapter(_make_evaluator_input(spans=spans))

        assert result.value == 1.0
        sample = metric.single_turn_score.call_args[0][0]
        assert sample.user_input == "What is 2+2?"
        assert sample.reference == "4"

    def test_parses_embedded_context_from_user_input(self):
        metric = _mock_legacy_metric(score=0.9)
        adapter = RAGASAdapter(metric=metric)

        spans = _make_spans(
            user_content='[{"text": "What is AI?\\n\\nContext:\\nAI is a branch of computer science."}]'
        )
        result = adapter(_make_evaluator_input(spans=spans))

        assert result.value == 0.9
        sample = metric.single_turn_score.call_args[0][0]
        assert sample.user_input == "What is AI?"
        assert sample.retrieved_contexts == ["AI is a branch of computer science."]
        assert sample.reference_contexts == ["AI is a branch of computer science."]

    def test_json_list_context_recovers_ranked_chunks(self):
        """Contexts embedded as json.dumps([...]) recover chunk boundaries and rank order."""
        import json

        metric = _mock_legacy_metric(name="llm_context_precision", score=0.67)
        adapter = RAGASAdapter(metric=metric)

        inner = "What is AI?\n\nContext:\n" + json.dumps(["chunk1", "chunk2", "chunk3"])
        spans = _make_spans(user_content=json.dumps([{"text": inner}]))
        result = adapter(_make_evaluator_input(spans=spans))

        assert result.value == 0.67
        sample = metric.single_turn_score.call_args[0][0]
        assert sample.retrieved_contexts == ["chunk1", "chunk2", "chunk3"]
        assert sample.reference_contexts == ["chunk1", "chunk2", "chunk3"]

    def test_json_non_list_context_treated_as_single_chunk(self):
        """JSON that isn't a list of strings stays a single chunk."""
        import json

        metric = _mock_legacy_metric(score=0.9)
        adapter = RAGASAdapter(metric=metric)

        inner = 'What is AI?\n\nContext:\n{"not": "a list"}'
        spans = _make_spans(user_content=json.dumps([{"text": inner}]))
        result = adapter(_make_evaluator_input(spans=spans))

        assert result.value == 0.9
        sample = metric.single_turn_score.call_args[0][0]
        assert sample.retrieved_contexts == ['{"not": "a list"}']

    def test_parses_combined_context_and_reference(self):
        metric = _mock_legacy_metric(name="context_precision", score=0.75)
        adapter = RAGASAdapter(metric=metric)

        spans = _make_spans(
            user_content=(
                '[{"text": "What is AI?\\n\\nContext:\\nAI is computer science.'
                '\\n\\nReference Answer:\\nArtificial Intelligence"}]'
            )
        )
        result = adapter(_make_evaluator_input(spans=spans))

        assert result.value == 0.75
        sample = metric.single_turn_score.call_args[0][0]
        assert sample.user_input == "What is AI?"
        assert sample.retrieved_contexts == ["AI is computer science."]
        assert sample.reference == "Artificial Intelligence"

    def test_no_embedded_markers_leaves_input_unchanged(self):
        metric = _mock_legacy_metric(score=0.8)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.8
        sample = metric.single_turn_score.call_args[0][0]
        assert sample.user_input == "What is AI?"
        assert sample.reference is None

    def test_reference_inputs_takes_precedence_over_embedded(self):
        metric = _mock_legacy_metric(name="answer_correctness", score=0.9)
        adapter = RAGASAdapter(metric=metric)

        spans = _make_spans(user_content='[{"text": "Q\\n\\nReference Answer:\\nembedded ref"}]')
        evaluator_input = _make_evaluator_input(
            spans=spans,
            reference_inputs=[{"expectedResponse": {"text": "service-provided ref"}}],
        )

        result = adapter(evaluator_input)

        assert result.value == 0.9
        sample = metric.single_turn_score.call_args[0][0]
        assert sample.reference == "service-provided ref"


class TestRAGASAdapterThresholdNone:
    """Tests for metrics where threshold is explicitly None (e.g. SemanticSimilarity)."""

    def test_threshold_none_defaults_to_half(self):
        metric = _mock_legacy_metric(name="semantic_similarity", score=0.6)
        metric.threshold = None
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.6
        assert result.label == "Pass"

    def test_threshold_none_score_below_default(self):
        metric = _mock_legacy_metric(name="semantic_similarity", score=0.3)
        metric.threshold = None
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.3
        assert result.label == "Fail"

    def test_threshold_none_does_not_crash(self):
        """No TypeError when comparing score >= None."""
        metric = _mock_legacy_metric(name="semantic_similarity", score=0.85)
        metric.threshold = None
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.value == 0.85
        assert result.label == "Pass"
        assert "threshold=0.5" in result.explanation
