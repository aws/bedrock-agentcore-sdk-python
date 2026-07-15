"""Parameterized DeepEval Lambda handler.

Reads METRIC_NAME from environment variable to select which metric to run.
Deploy once, register multiple times with different METRIC_NAME values.

Supported METRIC_NAME values:
    AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric,
    ToolCorrectnessMetric, ArgumentCorrectnessMetric, BiasMetric,
    ToxicityMetric, ContextualPrecisionMetric, JsonCorrectnessMetric, GEval
"""

import os

os.environ.setdefault("DEEPEVAL_RESULTS_FOLDER", "/tmp/.deepeval")
os.chdir("/tmp")

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
    JsonCorrectnessMetric,
    ToolCorrectnessMetric,
)

from bedrock_agentcore.evaluation.custom_code_based_evaluators import (
    EvaluatorInput,
    EvaluatorOutput,
    custom_code_based_evaluator,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.deepeval import DeepEvalAdapter

METRIC_NAME = os.environ.get("METRIC_NAME", "AnswerRelevancyMetric")
THRESHOLD = float(os.environ.get("METRIC_THRESHOLD", "0.5"))

METRIC_REGISTRY = {
    "AnswerRelevancyMetric": lambda: AnswerRelevancyMetric(threshold=THRESHOLD),
    "FaithfulnessMetric": lambda: FaithfulnessMetric(threshold=THRESHOLD),
    "HallucinationMetric": lambda: HallucinationMetric(threshold=THRESHOLD),
    "ToolCorrectnessMetric": lambda: ToolCorrectnessMetric(threshold=THRESHOLD),
    "ArgumentCorrectnessMetric": lambda: ToolCorrectnessMetric(threshold=THRESHOLD),
    "BiasMetric": lambda: BiasMetric(threshold=THRESHOLD),
    "ToxicityMetric": lambda: ToxicityMetric(threshold=THRESHOLD),
    "ContextualPrecisionMetric": lambda: ContextualPrecisionMetric(threshold=THRESHOLD),
    "JsonCorrectnessMetric": lambda: JsonCorrectnessMetric(),
    "GEval": lambda: GEval(
        name=os.environ.get("GEVAL_NAME", "helpfulness"),
        criteria=os.environ.get("GEVAL_CRITERIA", "Determine whether the response is helpful and addresses the user's question."),
        threshold=THRESHOLD,
    ),
}

metric = METRIC_REGISTRY[METRIC_NAME]()
adapter = DeepEvalAdapter(metric=metric)


@custom_code_based_evaluator()
def handler(evaluator_input: EvaluatorInput, context) -> EvaluatorOutput:
    return adapter(evaluator_input, context)
