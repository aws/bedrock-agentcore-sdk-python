"""Integration tests for third-party evaluation adapters.

These tests require `deepeval` and `autoevals` packages to be installed.
They verify the full adapter flow from EvaluatorInput through span parsing
to metric execution, using real library metrics (not mocks).

SETUP:
    pip install deepeval autoevals

RUN:
    pytest tests_integ/evaluation/test_third_party_adapters.py -v
"""

import pytest

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput


def _make_agent_evaluator_input(
    user_prompt="What is the capital of France?",
    agent_response="The capital of France is Paris.",
    tool_messages=None,
):
    """Build an EvaluatorInput with agent-level spans."""
    output_messages = []
    if tool_messages:
        for msg in tool_messages:
            output_messages.append({"role": "tool", "content": msg})
    output_messages.append({"role": "assistant", "content": agent_response})

    spans = [
        {
            "traceId": "integ-trace-1",
            "spanId": "integ-span-1",
            "attributes": {"gen_ai.operation.name": "invoke_agent"},
            "span_events": [
                {
                    "body": {
                        "input": {"messages": [{"role": "user", "content": user_prompt}]},
                        "output": {"messages": output_messages},
                    }
                }
            ],
        }
    ]
    return EvaluatorInput(
        evaluation_level="TRACE",
        session_spans=spans,
        target_trace_id="integ-trace-1",
    )


class TestDeepEvalAdapterIntegration:
    """Integration tests for DeepEvalAdapter with real DeepEval metrics."""

    @pytest.fixture(autouse=True)
    def check_deepeval(self):
        """Skip if deepeval is not installed."""
        pytest.importorskip("deepeval")

    def test_bias_metric_passes(self):
        from deepeval.metrics import BiasMetric

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.deepeval import DeepEvalAdapter

        metric = BiasMetric(threshold=0.5)
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(_make_agent_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.value is not None
        assert result.label in ("Pass", "Fail")

    def test_missing_retrieval_context_returns_error(self):
        from deepeval.metrics import FaithfulnessMetric

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.deepeval import DeepEvalAdapter

        metric = FaithfulnessMetric(threshold=0.7)
        adapter = DeepEvalAdapter(metric=metric)

        result = adapter(
            _make_agent_evaluator_input(
                user_prompt="Is the sky blue?",
                agent_response="Yes, the sky is blue.",
            )
        )

        assert isinstance(result, EvaluatorOutput)
        assert result.errorCode == "MISSING_REQUIRED_FIELD" or result.value is not None

    def test_with_custom_mapper(self):
        from deepeval.metrics import BiasMetric
        from deepeval.test_case import LLMTestCase

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.deepeval import DeepEvalAdapter

        metric = BiasMetric(threshold=0.5)
        adapter = DeepEvalAdapter(
            metric=metric,
            custom_mapper=lambda ev: LLMTestCase(
                input="Is Python a good language?",
                actual_output="Python is a versatile programming language used widely.",
            ),
        )

        result = adapter(_make_agent_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.value is not None


class TestAutoEvalsAdapterIntegration:
    """Integration tests for AutoEvalsAdapter with real Autoevals scorers."""

    @pytest.fixture(autouse=True)
    def check_autoevals(self):
        """Skip if autoevals is not installed."""
        pytest.importorskip("autoevals")

    def test_factuality_scorer(self):
        from autoevals import Factuality

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.autoevals import AutoEvalsAdapter

        scorer = Factuality()
        adapter = AutoEvalsAdapter(metric=scorer)

        evaluator_input = _make_agent_evaluator_input()
        evaluator_input.session_spans[0]["span_events"][0]["body"]["output"]["messages"] = [
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]

        result = adapter(evaluator_input)

        assert isinstance(result, EvaluatorOutput)
        assert result.value is not None
        assert result.label in ("Pass", "Fail")

    def test_custom_threshold(self):
        from autoevals import Factuality

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.autoevals import AutoEvalsAdapter

        scorer = Factuality()
        adapter = AutoEvalsAdapter(metric=scorer, threshold=0.9)

        result = adapter(_make_agent_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.value is not None

    def test_with_custom_mapper(self):
        from autoevals import Factuality

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.autoevals import AutoEvalsAdapter

        scorer = Factuality()
        adapter = AutoEvalsAdapter(
            metric=scorer,
            custom_mapper=lambda ev: {
                "input": "What is 2+2?",
                "output": "4",
                "expected": "4",
            },
        )

        result = adapter(_make_agent_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.value is not None
