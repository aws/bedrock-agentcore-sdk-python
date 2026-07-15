"""Autoevals Lambda handler with custom_mapper for unsupported span formats.

Demonstrates the custom_mapper path for Autoevals — customer returns a dict
of kwargs for scorer.eval() instead of relying on built-in span extraction.
"""

import os

from autoevals import Factuality

from bedrock_agentcore.evaluation.custom_code_based_evaluators import (
    EvaluatorInput,
    EvaluatorOutput,
    custom_code_based_evaluator,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.autoevals import AutoevalsAdapter

THRESHOLD = os.environ.get("METRIC_THRESHOLD")


def custom_mapper(evaluator_input: EvaluatorInput) -> dict:
    """Extract from OpenInference-style span attributes and return Autoevals kwargs.

    Returns dict with keys: input, output, expected (optional).
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

    result = {
        "input": input_text,
        "output": output_text,
    }

    # Pass expected from reference_inputs if available
    if evaluator_input.reference_inputs:
        ref = evaluator_input.reference_inputs[0]
        expected = getattr(ref, "expected_response_text", None)
        if expected:
            result["expected"] = expected

    return result


threshold = float(THRESHOLD) if THRESHOLD else None
adapter = AutoevalsAdapter(metric=Factuality(), custom_mapper=custom_mapper, threshold=threshold)


@custom_code_based_evaluator()
def handler(evaluator_input: EvaluatorInput, context) -> EvaluatorOutput:
    return adapter(evaluator_input, context)
