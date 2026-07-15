"""E2E tests: Autoevals metrics × LangChain mapper auto-detection (Tests 17-18).

Proves that Autoevals metrics work with LangGraph agent spans
auto-detected by the built-in mapper registry.

Usage:
    .venv/bin/python -m pytest e2e_tests/test_autoevals_langgraph.py -v
"""

import pytest

from e2e_tests.conftest import EVALUATOR_IDS, assert_success, build_reference_input, run_evaluation


class TestAutoevalsLangGraph:
    """Tests 17-18: Autoevals with LangChain mapper auto-detection."""

    def test_factuality_openinference(self, dp_client):
        """Test 17: Factuality with OpenInference LangChain spans."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["autoevals-factuality"],
            fixture_name="openinference_langchain_spans.json",
            reference_inputs=build_reference_input(
                "openinference_langchain_spans.json",
                expectedResponse={"text": "The weather in Seattle is typically rainy."},
            ),
        )
        assert_success(result)

    @pytest.mark.p1
    def test_factuality_opentelemetry(self, dp_client):
        """Test 18: Factuality with OpenTelemetry LangChain spans (adot v18)."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["autoevals-factuality"],
            fixture_name="opentelemetry_langchain_spans.json",
            reference_inputs=build_reference_input(
                "opentelemetry_langchain_spans.json",
                expectedResponse={"text": "There are flights from Seattle to NYC on March 15."},
            ),
        )
        assert_success(result)
