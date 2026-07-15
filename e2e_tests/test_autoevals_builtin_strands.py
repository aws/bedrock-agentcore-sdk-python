"""E2E tests: Autoevals metrics × Built-in Strands mapper (Tests 13-16).

Tests all Autoevals metric types (LLM-Judge, LLM-Scorer, Deterministic)
against real Strands agent spans.

Usage:
    .venv/bin/python -m pytest e2e_tests/test_autoevals_builtin_strands.py -v
"""

import pytest

from e2e_tests.conftest import EVALUATOR_IDS, assert_success, build_reference_input, run_evaluation


class TestAutoevalsLLMJudge:
    """Tests 13-14: LLM-Judge type metrics (Factuality, ClosedQA)."""

    def test_factuality(self, dp_client):
        """Test 13: Factuality with Strands RAG spans + expected_output."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["autoevals-factuality"],
            fixture_name="strands_rag_spans.json",
            reference_inputs=build_reference_input(
                "strands_rag_spans.json",
                expectedResponse={"text": "The patient should take medication twice daily with food."},
            ),
        )
        assert_success(result)

    @pytest.mark.p1
    def test_closed_qa(self, dp_client):
        """Test 14: ClosedQA with Strands spans + expected_output."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["autoevals-closedqa"],
            fixture_name="strands_qa_spans.json",
            reference_inputs=build_reference_input(
                "strands_qa_spans.json",
                expectedResponse={"text": "BMI is calculated as weight divided by height squared."},
            ),
        )
        assert_success(result)


class TestAutoevalsLLMScorer:
    """Test 15: LLM-Scorer type (AnswerCorrectness)."""

    def test_answer_correctness(self, dp_client):
        """Test 15: Autoevals AnswerCorrectness with Strands QA spans + expected."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["autoevals-answer-correctness"],
            fixture_name="strands_qa_spans.json",
            reference_inputs=build_reference_input(
                "strands_qa_spans.json",
                expectedResponse={"text": "BMI is 24.7, which is in the normal weight range."},
            ),
        )
        assert_success(result, expect_explanation=False)


class TestAutoevalsDeterministic:
    """Test 16: Deterministic type (ExactMatch) — no LLM call."""

    def test_exact_match(self, dp_client):
        """Test 16: ExactMatch with Strands spans + expected_output."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["autoevals-exact-match"],
            fixture_name="strands_qa_spans.json",
            reference_inputs=build_reference_input(
                "strands_qa_spans.json",
                expectedResponse={"text": "Your BMI is 22.9"},
            ),
        )
        assert_success(result, expect_explanation=False)
