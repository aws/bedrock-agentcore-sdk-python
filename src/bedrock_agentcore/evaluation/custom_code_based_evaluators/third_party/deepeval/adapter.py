"""DeepEval adapter for AgentCore code-based evaluators."""

import logging
from typing import Any, Callable, Dict, Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.base import BaseAdapter

logger = logging.getLogger(__name__)


class DeepEvalAdapter(BaseAdapter):
    """Adapter that runs a DeepEval metric against AgentCore evaluation events.

    Example::

        from deepeval.metrics import AnswerRelevancyMetric
        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.deepeval import DeepEvalAdapter

        metric = AnswerRelevancyMetric(threshold=0.7)
        adapter = DeepEvalAdapter(metric=metric)
    """

    def __init__(
        self,
        metric: BaseMetric,
        field_mapper: Optional[Callable[[EvaluatorInput], Dict[str, Any]]] = None,
        model: Optional[Any] = None,
    ):
        """Initialize the adapter.

        Args:
            metric: A DeepEval BaseMetric instance (e.g. AnswerRelevancyMetric).
            field_mapper: Optional callable that receives the EvaluatorInput and
                returns a dict of LLMTestCase field values. Bypasses default span
                parsing when provided.
            model: Optional model override for the metric's LLM.
        """
        super().__init__(field_mapper=field_mapper)
        self.metric = metric
        if model is not None:
            self.metric.model = model

    def validate_fields(self, fields: Dict[str, Any]) -> None:
        """No pre-validation; let DeepEval raise on missing params."""

    def execute(self, fields: Dict[str, Any]) -> EvaluatorOutput:
        """Run the DeepEval metric and return formatted results."""
        test_case = LLMTestCase(
            input=fields.get("input", ""),
            actual_output=fields.get("actual_output", ""),
            expected_output=fields.get("expected_output"),
            context=fields.get("context"),
            retrieval_context=fields.get("retrieval_context"),
        )

        try:
            self.metric.measure(test_case)
        except Exception as e:
            error_type = type(e).__name__
            if "MissingTestCaseParams" in error_type or "missing" in str(e).lower():
                return EvaluatorOutput(
                    label="Error",
                    errorCode="MISSING_REQUIRED_FIELD",
                    errorMessage=f"{type(self.metric).__name__} requires fields not available: {e}",
                )
            raise

        score = self.metric.score
        reason = getattr(self.metric, "reason", None) or ""
        threshold = getattr(self.metric, "threshold", 0.5)
        success = getattr(self.metric, "success", score is not None and score >= threshold)
        label = "Pass" if success else "Fail"

        return EvaluatorOutput(value=score, label=label, explanation=reason)
