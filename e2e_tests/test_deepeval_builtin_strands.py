"""E2E tests: DeepEval metrics × Built-in Strands mapper (Tests 1-10).

Tests all DeepEval metric categories against real Strands agent spans
using the on-demand evaluate() API.

Usage:
    .venv/bin/python -m pytest e2e_tests/test_deepeval_builtin_strands.py -v
"""

import pytest

from e2e_tests.conftest import EVALUATOR_IDS, assert_success, build_reference_input, run_evaluation


class TestDeepEvalRAGMetrics:
    """Tests 1-3: RAG metrics (AnswerRelevancy, Faithfulness, Hallucination)."""

    def test_answer_relevancy(self, dp_client):
        """Test 1: AnswerRelevancyMetric with basic QA spans."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-answer-relevancy"],
            fixture_name="strands_qa_spans.json",
        )
        assert_success(result)

    def test_faithfulness(self, dp_client):
        """Test 2: FaithfulnessMetric with RAG spans (needs retrieval_context)."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-faithfulness"],
            fixture_name="strands_rag_spans.json",
        )
        assert_success(result)

    def test_hallucination(self, dp_client):
        """Test 3: HallucinationMetric with RAG spans (needs context)."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-hallucination"],
            fixture_name="strands_rag_spans.json",
        )
        assert_success(result)


class TestDeepEvalAgenticMetrics:
    """Tests 4-5: Agentic metrics (ToolCorrectness, ArgumentCorrectness)."""

    def test_tool_correctness(self, dp_client):
        """Test 4: ToolCorrectnessMetric with tool call spans + expectedTrajectory."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-tool-correctness"],
            fixture_name="strands_qa_spans.json",
            reference_inputs=build_reference_input(
                "strands_qa_spans.json",
                expectedTrajectory={"toolNames": ["calculate_bmi", "calculate_daily_calories"]},
            ),
        )
        assert_success(result)

    def test_argument_correctness(self, dp_client):
        """Test 5: ArgumentCorrectnessMetric with tool call spans + expectedTrajectory."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-argument-correctness"],
            fixture_name="strands_qa_spans.json",
            reference_inputs=build_reference_input(
                "strands_qa_spans.json",
                expectedTrajectory={"toolNames": ["calculate_bmi", "calculate_daily_calories"]},
            ),
        )
        assert_success(result)


class TestDeepEvalCustomMetrics:
    """Test 6: GEval with customer-defined criteria."""

    def test_geval_custom_criteria(self, dp_client):
        """Test 6: GEval metric with custom evaluation criteria."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-geval"],
            fixture_name="strands_qa_spans.json",
        )
        assert_success(result)


class TestDeepEvalContextualMetrics:
    """Test 7: ContextualPrecision (P1)."""

    @pytest.mark.p1
    def test_contextual_precision(self, dp_client):
        """Test 7: ContextualPrecisionMetric with Strands QA spans + expected_output."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-contextual-precision"],
            fixture_name="strands_qa_spans.json",
            reference_inputs=build_reference_input(
                "strands_qa_spans.json",
                expectedResponse={"text": "BMI is 24.7, which is in the normal weight range."},
            ),
        )
        assert_success(result)


class TestDeepEvalNonLLMMetrics:
    """Test 8: JsonCorrectness (P1) — deterministic, no LLM call."""

    @pytest.mark.p1
    def test_json_correctness(self, dp_client):
        """Test 8: JsonCorrectnessMetric — no LLM judge, deterministic scoring."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-json-correctness"],
            fixture_name="strands_qa_spans.json",
        )
        assert_success(result, expect_explanation=False)


class TestDeepEvalSafetyMetrics:
    """Tests 9-10: Safety metrics (Bias, Toxicity) — P1."""

    @pytest.mark.p1
    def test_bias(self, dp_client):
        """Test 9: BiasMetric on QA spans."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-bias"],
            fixture_name="strands_qa_spans.json",
        )
        assert_success(result)

    @pytest.mark.p1
    def test_toxicity(self, dp_client):
        """Test 10: ToxicityMetric on QA spans."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-toxicity"],
            fixture_name="strands_qa_spans.json",
        )
        assert_success(result)
