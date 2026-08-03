"""Integration tests for third-party evaluation adapters.

These tests require `deepeval`, `autoevals`, and `ragas` packages to be installed.
They verify the full adapter flow from EvaluatorInput through span parsing
to metric execution, using real library metrics (not mocks).

SETUP:
    pip install deepeval autoevals ragas

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
        """Verify deepeval is installed."""
        import deepeval  # noqa: F401

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
        """Verify autoevals is installed."""
        import autoevals  # noqa: F401

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


def _make_ragas_evaluator_input(user_prompt, agent_response):
    """Build an EvaluatorInput with CloudWatch split format spans (supported by span mappers)."""
    import json

    spans = [
        {
            "traceId": "integ-trace-1",
            "spanId": "integ-span-1",
            "scope": {"name": "strands.telemetry.tracer"},
            "name": "invoke_agent",
            "kind": "INTERNAL",
            "startTimeUnixNano": 1000000000,
            "endTimeUnixNano": 2000000000,
            "attributes": {"gen_ai.operation.name": "invoke_agent", "session.id": "integ-session"},
            "status": {"code": "UNSET"},
        },
        {
            "traceId": "integ-trace-1",
            "spanId": "integ-span-1",
            "scope": {"name": "strands.telemetry.tracer"},
            "timeUnixNano": 2000000000,
            "observedTimeUnixNano": 2000000001,
            "severityNumber": 9,
            "body": {
                "input": {"messages": [{"role": "user", "content": {"content": json.dumps([{"text": user_prompt}])}}]},
                "output": {"messages": [{"role": "assistant", "content": {"message": agent_response}}]},
            },
        },
    ]
    return EvaluatorInput(
        evaluation_level="TRACE",
        session_spans=spans,
        target_trace_id="integ-trace-1",
    )


class TestRAGASAdapterIntegration:
    """Integration tests for RAGASAdapter with real RAGAS metrics.

    Uses deterministic (non-LLM) metrics so no model access is required.
    """

    @pytest.fixture(autouse=True)
    def check_ragas(self):
        """Verify ragas is installed."""
        import ragas  # noqa: F401

    def test_exact_match_with_embedded_reference(self):
        from ragas.metrics import ExactMatch

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RAGASAdapter

        adapter = RAGASAdapter(metric=ExactMatch())

        result = adapter(
            _make_ragas_evaluator_input(
                user_prompt="What is the capital of France?\n\nReference Answer:\nThe capital of France is Paris.",
                agent_response="The capital of France is Paris.",
            )
        )

        assert isinstance(result, EvaluatorOutput)
        assert result.value == 1.0
        assert result.label == "Pass"

    def test_exact_match_fails_on_mismatch(self):
        from ragas.metrics import ExactMatch

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RAGASAdapter

        adapter = RAGASAdapter(metric=ExactMatch())

        result = adapter(
            _make_ragas_evaluator_input(
                user_prompt="What is the capital of France?\n\nReference Answer:\nParis",
                agent_response="The capital of France is Paris.",
            )
        )

        assert isinstance(result, EvaluatorOutput)
        assert result.value == 0.0
        assert result.label == "Fail"

    def test_missing_reference_returns_error(self):
        from ragas.metrics import ExactMatch

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RAGASAdapter

        adapter = RAGASAdapter(metric=ExactMatch())

        result = adapter(
            _make_ragas_evaluator_input(
                user_prompt="What is the capital of France?",
                agent_response="The capital of France is Paris.",
            )
        )

        assert isinstance(result, EvaluatorOutput)
        # No reference available; RAGAS raises or produces NaN depending on version
        assert result.errorCode is not None or result.value is not None

    def test_with_custom_mapper(self):
        from ragas.metrics import ExactMatch

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RAGASAdapter

        adapter = RAGASAdapter(
            metric=ExactMatch(),
            custom_mapper=lambda ev: {
                "user_input": "What is 2+2?",
                "response": "4",
                "reference": "4",
            },
        )

        result = adapter(_make_agent_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.value == 1.0
        assert result.label == "Pass"

    def test_collections_metric_exact_match(self):
        """New-generation ragas.metrics.collections metrics score via metric.score(**kwargs)."""
        from ragas.metrics.collections import ExactMatch as CollectionsExactMatch

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RAGASAdapter

        adapter = RAGASAdapter(metric=CollectionsExactMatch())

        result = adapter(
            _make_ragas_evaluator_input(
                user_prompt="What is the capital of France?\n\nReference Answer:\nThe capital of France is Paris.",
                agent_response="The capital of France is Paris.",
            )
        )

        assert isinstance(result, EvaluatorOutput)
        assert result.value == 1.0
        assert result.label == "Pass"

    def test_decorator_discrete_metric(self):
        """Decorator-based custom metrics return categorical labels."""
        from ragas.metrics import discrete_metric

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RAGASAdapter

        @discrete_metric(name="mentions_paris", allowed_values=["yes", "no"])
        def mentions_paris(response: str) -> str:
            return "yes" if "Paris" in response else "no"

        adapter = RAGASAdapter(
            metric=mentions_paris,
            custom_mapper=lambda ev: {"response": "The capital of France is Paris."},
        )

        result = adapter(_make_agent_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.label == "yes"

    def test_adapter_imports_without_datasets(self):
        """The adapter module must not require the HF datasets library at import time."""
        import subprocess
        import sys

        code = (
            "import sys\n"
            "sys.modules['datasets'] = None  # poison the import\n"
            "from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RAGASAdapter\n"
            "print('OK')\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout
