"""DeepEval Lambda handler with custom_mapper for unsupported span formats.

Demonstrates the custom_mapper path for frameworks not supported by the
built-in mapper registry (e.g., Google ADK, OpenAI Agents, custom agents).

MAPPER_TYPE env var selects which custom_mapper to use:
    flat   — extracts from span attributes directly (Google ADK style)
    nested — extracts from span_events[].body nested JSON (OpenAI Agents style)
"""

import os

os.environ.setdefault("DEEPEVAL_RESULTS_FOLDER", "/tmp/.deepeval")
os.chdir("/tmp")

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from bedrock_agentcore.evaluation.custom_code_based_evaluators import (
    EvaluatorInput,
    EvaluatorOutput,
    custom_code_based_evaluator,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.deepeval import DeepEvalAdapter

MAPPER_TYPE = os.environ.get("MAPPER_TYPE", "flat")
THRESHOLD = float(os.environ.get("METRIC_THRESHOLD", "0.5"))


def flat_mapper(evaluator_input: EvaluatorInput) -> LLMTestCase:
    """Extract from OpenInference-style span attributes (Google ADK, OpenAI Agents).

    Looks for input.value and output.value in the agent/chain span attributes.
    """
    spans = evaluator_input.session_spans
    agent_span = None
    for span in spans:
        attrs = span.get("attributes", {})
        kind = attrs.get("openinference.span.kind", "")
        if kind in ("AGENT", "CHAIN"):
            agent_span = span
    if agent_span is None:
        agent_span = spans[0]

    attrs = agent_span.get("attributes", {})
    input_text = attrs.get("input.value", "")
    output_text = attrs.get("output.value", "")

    return LLMTestCase(
        input=input_text,
        actual_output=output_text,
    )


def nested_mapper(evaluator_input: EvaluatorInput) -> LLMTestCase:
    """Extract from nested span_events body (OpenAI Agents style).

    Parses span_events[].body for input/output messages.
    """
    spans = evaluator_input.session_spans
    agent_span = None
    for span in spans:
        attrs = span.get("attributes", {})
        kind = attrs.get("openinference.span.kind", "")
        if kind in ("AGENT", "CHAIN"):
            agent_span = span
    if agent_span is None:
        agent_span = spans[0]

    input_text = ""
    output_text = ""

    attrs = agent_span.get("attributes", {})
    input_text = attrs.get("input.value", "")
    output_text = attrs.get("output.value", "")

    return LLMTestCase(
        input=input_text,
        actual_output=output_text,
    )


MAPPER_REGISTRY = {
    "flat": flat_mapper,
    "nested": nested_mapper,
}

custom_mapper = MAPPER_REGISTRY[MAPPER_TYPE]
adapter = DeepEvalAdapter(metric=AnswerRelevancyMetric(threshold=THRESHOLD), custom_mapper=custom_mapper)


@custom_code_based_evaluator()
def handler(evaluator_input: EvaluatorInput, context) -> EvaluatorOutput:
    return adapter(evaluator_input, context)
