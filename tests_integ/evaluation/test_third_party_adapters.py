"""Integration tests for third-party evaluation adapters.

These tests require `deepeval`, `autoevals`, and `ragas` packages to be installed.
They verify the full adapter flow from EvaluatorInput through span parsing
to metric execution, using real library metrics (not mocks).

Both libraries judge with an LLM, so every test here requires OPENAI_API_KEY.
In CI the key is fetched from the central DevX Secrets Manager account as a
repo-specific workflow secret.

SETUP:
    pip install deepeval autoevals ragas
    export OPENAI_API_KEY=...

RUN:
    pytest tests_integ/evaluation/test_third_party_adapters.py -v
"""

import json
import os

import pytest

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput


def _text_content(text):
    """Wrap text in the nested, double-encoded shape the CloudWatch mapper expects.

    strands-evals reads message content via `content.content` / `content.message`,
    where the inner value is a JSON-encoded list of content blocks. Passing a bare
    string yields no AgentInvocationSpan and the adapters fail extraction.
    """
    return {"content": json.dumps([{"text": text}])}


def _tool_use_content(name, args, tool_use_id):
    """Wrap a tool call in the toolUse content block the CloudWatch mapper reads."""
    return {"content": json.dumps([{"toolUse": {"name": name, "input": args, "toolUseId": tool_use_id}}])}


def _tool_result_content(tool_use_id, result):
    """Wrap a tool result in the toolResult content block paired to a toolUse id."""
    return {"content": json.dumps([{"toolResult": {"toolUseId": tool_use_id, "content": [{"text": result}]}}])}


def _make_agent_evaluator_input(
    user_prompt="What is the capital of France?",
    agent_response="The capital of France is Paris.",
    tool_messages=None,
    tool_calls=None,
    reference_tool_names=None,
):
    """Build an EvaluatorInput with agent-level spans.

    Args:
        user_prompt: The user message content.
        agent_response: The final assistant message content.
        tool_messages: Optional plain-text tool outputs (become retrieval context).
        tool_calls: Optional list of ``{"name": ..., "args": {...}}`` dicts, emitted
            as paired toolUse/toolResult blocks so span mapping populates
            ``tools_called``.
        reference_tool_names: Optional expected tool trajectory, supplied as
            ``evaluationReferenceInputs`` the way the service would.
    """
    output_messages = []
    if tool_messages:
        for msg in tool_messages:
            output_messages.append({"role": "tool", "content": _text_content(msg)})
    if tool_calls:
        for index, call in enumerate(tool_calls):
            tool_use_id = f"integ-tool-{index}"
            output_messages.append(
                {"role": "assistant", "content": _tool_use_content(call["name"], call.get("args", {}), tool_use_id)}
            )
            output_messages.append(
                {"role": "user", "content": _tool_result_content(tool_use_id, call.get("result", "ok"))}
            )
    output_messages.append({"role": "assistant", "content": _text_content(agent_response)})

    spans = [
        {
            "traceId": "integ-trace-1",
            "spanId": "integ-span-1",
            # The mapper keys off scope.name to pick the CloudWatch span format.
            "scope": {"name": "strands.telemetry.tracer"},
            "attributes": {"gen_ai.operation.name": "invoke_agent"},
            "span_events": [
                {
                    "body": {
                        "input": {"messages": [{"role": "user", "content": _text_content(user_prompt)}]},
                        "output": {"messages": output_messages},
                    }
                }
            ],
        }
    ]
    reference_inputs = []
    if reference_tool_names:
        reference_inputs = [{"expectedTrajectory": {"toolNames": list(reference_tool_names)}}]
    return EvaluatorInput(
        evaluation_level="TRACE",
        session_spans=spans,
        target_trace_id="integ-trace-1",
        reference_inputs=reference_inputs,
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

    @pytest.fixture
    def openai_client(self):
        """An OpenAI client pinned to the OpenAI API.

        Left to its own defaults, autoevals sends requests to the Braintrust AI
        gateway authenticated with BRAINTRUST_API_KEY, which rejects an OpenAI
        key with 401. Passing an explicit client is the documented way to choose
        the provider (the api_key/base_url arguments are deprecated).
        """
        from openai import OpenAI

        return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1")

    def test_factuality_scorer(self, openai_client):
        from autoevals import Factuality

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.autoevals import AutoEvalsAdapter

        scorer = Factuality(client=openai_client)
        adapter = AutoEvalsAdapter(metric=scorer)

        # Assistant-only output (no tool messages), which is what the helper
        # already builds; keep it explicit here to document the intent.
        evaluator_input = _make_agent_evaluator_input(agent_response="The capital of France is Paris.")

        result = adapter(evaluator_input)

        assert isinstance(result, EvaluatorOutput)
        assert result.value is not None
        assert result.label in ("Pass", "Fail")

    def test_custom_threshold(self, openai_client):
        from autoevals import Factuality

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.autoevals import AutoEvalsAdapter

        scorer = Factuality(client=openai_client)
        adapter = AutoEvalsAdapter(metric=scorer, threshold=0.9)

        result = adapter(_make_agent_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.value is not None

    def test_with_custom_mapper(self, openai_client):
        from autoevals import Factuality

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.autoevals import AutoEvalsAdapter

        scorer = Factuality(client=openai_client)
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
            _make_agent_evaluator_input(
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
            _make_agent_evaluator_input(
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
            _make_agent_evaluator_input(
                user_prompt="What is the capital of France?",
                agent_response="The capital of France is Paris.",
            )
        )

        assert isinstance(result, EvaluatorOutput)
        # ExactMatch declares reference as required; the adapter validates
        # before scoring rather than letting ragas return a silent 0.0
        assert result.errorCode == "MISSING_REQUIRED_FIELD"
        assert "reference" in result.errorMessage

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
            _make_agent_evaluator_input(
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

    def test_collections_tool_call_accuracy_default_mapping(self):
        """Real collections ToolCallAccuracy via default span mapping (no custom mapper).

        Collections metrics take a conversation-shaped user_input and
        reference_tool_calls, so the adapter must supply the converted messages
        and tool calls rather than the flat single-turn fields.
        """
        from ragas.metrics.collections import ToolCallAccuracy

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RAGASAdapter

        adapter = RAGASAdapter(metric=ToolCallAccuracy())

        result = adapter(
            _make_agent_evaluator_input(
                user_prompt="Book a flight to NYC",
                agent_response="Booked your flight.",
                tool_calls=[{"name": "book_flight", "args": {"destination": "NYC"}}],
                reference_tool_names=["book_flight"],
            )
        )

        assert isinstance(result, EvaluatorOutput)
        assert result.errorCode is None
        assert result.value == 1.0
        assert result.label == "Pass"

    def test_collections_tool_call_accuracy_wrong_tool_fails(self):
        from ragas.metrics.collections import ToolCallAccuracy

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RAGASAdapter

        adapter = RAGASAdapter(metric=ToolCallAccuracy())

        result = adapter(
            _make_agent_evaluator_input(
                user_prompt="Book a flight to NYC",
                agent_response="Looked up the weather instead.",
                tool_calls=[{"name": "get_weather", "args": {"city": "NYC"}}],
                reference_tool_names=["book_flight"],
            )
        )

        assert isinstance(result, EvaluatorOutput)
        assert result.value == 0.0
        assert result.label == "Fail"

    def test_legacy_tool_call_accuracy_default_mapping(self):
        """The legacy multi-turn class scores identically through default mapping."""
        from ragas.metrics import ToolCallAccuracy

        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RAGASAdapter

        adapter = RAGASAdapter(metric=ToolCallAccuracy())

        result = adapter(
            _make_agent_evaluator_input(
                user_prompt="Book a flight to NYC",
                agent_response="Booked your flight.",
                tool_calls=[{"name": "book_flight", "args": {"destination": "NYC"}}],
                reference_tool_names=["book_flight"],
            )
        )

        assert isinstance(result, EvaluatorOutput)
        assert result.errorCode is None
        assert result.value == 1.0

    def test_adapter_imports_without_datasets(self):
        """The adapter module itself must not require the HF datasets library at import time.

        Scope: this verifies the adapter adds no datasets dependency of its
        own. ragas <1.0 still imports datasets when the ragas package is
        imported; slim deployments handle that separately with a trimmed
        ragas build.
        """
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
