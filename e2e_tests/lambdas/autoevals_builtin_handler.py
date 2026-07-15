"""Parameterized Autoevals Lambda handler.

Reads METRIC_NAME from environment variable to select which scorer to run.
Deploy once per metric with different METRIC_NAME values.

Supported METRIC_NAME values:
    Factuality, ClosedQA, AnswerRelevancy, ExactMatch
"""

import os

from autoevals import ClosedQA, ExactMatch, Factuality
from autoevals import AnswerRelevancy as AutoevalsAnswerRelevancy

from bedrock_agentcore.evaluation.custom_code_based_evaluators import (
    EvaluatorInput,
    EvaluatorOutput,
    custom_code_based_evaluator,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.autoevals import AutoevalsAdapter

METRIC_NAME = os.environ.get("METRIC_NAME", "Factuality")
THRESHOLD = os.environ.get("METRIC_THRESHOLD")

METRIC_REGISTRY = {
    "Factuality": lambda: Factuality(),
    "ClosedQA": lambda: ClosedQA(),
    "AnswerRelevancy": lambda: AutoevalsAnswerRelevancy(),
    "ExactMatch": lambda: ExactMatch(),
}

metric = METRIC_REGISTRY[METRIC_NAME]()
threshold = float(THRESHOLD) if THRESHOLD else None
adapter = AutoevalsAdapter(metric=metric, threshold=threshold)


@custom_code_based_evaluator()
def handler(evaluator_input: EvaluatorInput, context) -> EvaluatorOutput:
    return adapter(evaluator_input, context)
