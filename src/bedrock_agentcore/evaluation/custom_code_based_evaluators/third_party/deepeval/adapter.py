"""DeepEval adapter for AgentCore code-based evaluators."""

import logging
from typing import Any, Callable, Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.base import BaseAdapter

logger = logging.getLogger(__name__)


class DeepEvalAdapter(BaseAdapter):
    """Adapter that runs a DeepEval metric against AgentCore evaluation events.

    Example (default span mapping)::

        from deepeval.metrics import AnswerRelevancyMetric
        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.deepeval import DeepEvalAdapter

        metric = AnswerRelevancyMetric(threshold=0.7)
        adapter = DeepEvalAdapter(metric=metric)

    Example (customer mapper returning LLMTestCase)::

        adapter = DeepEvalAdapter(
            metric=AnswerRelevancyMetric(threshold=0.7),
            customer_mapper=lambda ev: LLMTestCase(
                input=ev.session_spans[0]["attributes"]["user_query"],
                actual_output=ev.session_spans[0]["attributes"]["response"],
            ),
        )
    """

    def __init__(
        self,
        metric: BaseMetric,
        customer_mapper: Optional[Callable[[EvaluatorInput], LLMTestCase]] = None,
        model: Optional[Any] = None,
    ):
        """Initialize the adapter.

        Args:
            metric: A DeepEval BaseMetric instance (e.g. AnswerRelevancyMetric).
            customer_mapper: Optional callable that receives the EvaluatorInput and
                returns a LLMTestCase. Bypasses default span mapping when provided.
            model: Optional model override for the metric's LLM.
        """
        self.metric = metric
        self.customer_mapper = customer_mapper
        if model is not None:
            self.metric.model = model

    def _run(self, evaluator_input: EvaluatorInput) -> EvaluatorOutput:
        """Run the DeepEval metric pipeline."""
        if self.customer_mapper is not None:
            test_case = self.customer_mapper(evaluator_input)
        else:
            result = self._default_extract(evaluator_input)
            if not result.input or not result.actual_output:
                missing = []
                if not result.input:
                    missing.append("input")
                if not result.actual_output:
                    missing.append("actual_output")
                metric_name = type(self.metric).__name__
                return EvaluatorOutput(
                    label="Error",
                    errorCode="MISSING_REQUIRED_FIELD",
                    errorMessage=f"Field(s) {missing} required by {metric_name} but not found in evaluation event. "
                    f"Provide a customer_mapper or ensure spans contain the necessary data.",
                )
            test_case = LLMTestCase(
                input=result.input,
                actual_output=result.actual_output,
                expected_output=result.expected_output,
                context=result.context,
                retrieval_context=result.retrieval_context,
            )

        try:
            self.metric.measure(test_case)
        except Exception as e:
            if type(e).__name__ == "MissingTestCaseParamsError":
                return EvaluatorOutput(
                    label="Error",
                    errorCode="MISSING_REQUIRED_FIELD",
                    errorMessage=f"{type(self.metric).__name__} requires fields not extracted from spans: {e}. "
                    f"Provide a customer_mapper to supply custom fields from your trace data.",
                )
            raise

        score = self.metric.score
        reason = getattr(self.metric, "reason", None) or ""
        threshold = getattr(self.metric, "threshold", 0.5)
        success = getattr(self.metric, "success", score is not None and score >= threshold)
        label = "Pass" if success else "Fail"

        return EvaluatorOutput(value=score, label=label, explanation=reason)
