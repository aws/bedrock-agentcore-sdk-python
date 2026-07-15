"""E2E tests: DeepEval metrics × LangChain mapper auto-detection (Tests 11-12).

Proves that the built-in OpenInference and OpenTelemetry LangChain mappers
correctly extract fields from LangGraph agent spans.

Usage:
    .venv/bin/python -m pytest e2e_tests/test_deepeval_langgraph.py -v
"""

import pytest

from e2e_tests.conftest import EVALUATOR_IDS, assert_success, run_evaluation


class TestDeepEvalLangGraph:
    """Tests 11-12: DeepEval with LangChain mapper auto-detection."""

    def test_answer_relevancy_openinference(self, dp_client):
        """Test 11: AnswerRelevancyMetric with OpenInference LangChain spans."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-answer-relevancy"],
            fixture_name="openinference_langchain_spans.json",
        )
        assert_success(result)

    def test_answer_relevancy_opentelemetry(self, dp_client):
        """Test 12: AnswerRelevancyMetric with OpenTelemetry LangChain spans (adot v18)."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-answer-relevancy"],
            fixture_name="opentelemetry_langchain_spans.json",
        )
        assert_success(result)
