"""E2E tests: Custom mapper path (Tests 19-21).

Proves that customers using unsupported frameworks (Google ADK, OpenAI Agents)
can write a custom_mapper and get evaluation working end-to-end.

Usage:
    .venv/bin/python -m pytest e2e_tests/test_custom_mappers.py -v
"""

import pytest

from e2e_tests.conftest import EVALUATOR_IDS, assert_success, build_reference_input, run_evaluation


class TestDeepEvalCustomMappers:
    """Tests 19-20: DeepEval with custom_mapper for unsupported frameworks."""

    def test_custom_flat_google_adk(self, dp_client):
        """Test 19: Custom mapper extracting from Google ADK spans (flat attributes)."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-custom-flat"],
            fixture_name="custom_flat_spans.json",
        )
        assert_success(result)

    def test_custom_nested_openai_agents(self, dp_client):
        """Test 20: Custom mapper extracting from OpenAI Agents spans (nested JSON)."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["deepeval-custom-nested"],
            fixture_name="custom_nested_spans.json",
        )
        assert_success(result)


class TestAutoevalsCustomMapper:
    """Test 21: Autoevals with custom_mapper returning dict."""

    def test_custom_autoevals(self, dp_client):
        """Test 21: Custom mapper returning Autoevals kwargs dict."""
        result = run_evaluation(
            dp_client,
            evaluator_id=EVALUATOR_IDS["autoevals-custom"],
            fixture_name="custom_flat_spans.json",
            reference_inputs=build_reference_input(
                "custom_flat_spans.json",
                expectedResponse={"text": "The travel plan includes flights and hotels."},
            ),
        )
        assert_success(result)
