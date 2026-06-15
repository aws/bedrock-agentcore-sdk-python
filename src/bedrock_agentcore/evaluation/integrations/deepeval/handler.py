"""DeepEval handler that adapts AgentCore Lambda evaluation events to DeepEval metrics."""

import logging
from typing import Any, Callable, Dict, Optional

from deepeval.metrics import BaseMetric

from bedrock_agentcore.evaluation.integrations.deepeval.input_mapper import (
    ParsedEvaluationEvent,
    build_test_case,
)

logger = logging.getLogger(__name__)


class DeepEvalHandler:
    """Lambda handler that runs a DeepEval metric against AgentCore evaluation events.

    Never raises unhandled exceptions — always returns a valid response dict.

    Example::

        from deepeval.metrics import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric(threshold=0.7)
        handler = DeepEvalHandler(metric=metric)

        # Use as Lambda handler
        def lambda_handler(event, context):
            return handler(event, context)
    """

    def __init__(
        self,
        metric: BaseMetric,
        field_mapper: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """Initialize the handler.

        Args:
            metric: A DeepEval BaseMetric instance (e.g. AnswerRelevancyMetric).
            field_mapper: Optional callable that receives the raw Lambda event and
                returns a dict of LLMTestCase field values. Bypasses default span
                extraction when provided.
        """
        self.metric = metric
        self.field_mapper = field_mapper

    def __call__(self, event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        """Handle a Lambda invocation.

        Args:
            event: Raw Lambda event dict from the evaluation service.
            context: Lambda context object (unused).

        Returns:
            Success: {"value": float, "label": str, "explanation": str}
            Error: {"errorCode": str, "errorMessage": str}
        """
        try:
            parsed = ParsedEvaluationEvent.from_lambda_event(event)
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Failed to parse evaluation event: %s", e)
            return _error_response("INVALID_EVENT", f"Failed to parse evaluation event: {e}")

        try:
            test_case = build_test_case(parsed, self.metric, self.field_mapper)
        except ValueError as e:
            logger.error("Missing required fields: %s", e)
            return _error_response("MISSING_REQUIRED_FIELD", str(e))

        try:
            self.metric.measure(test_case)
        except Exception as e:
            logger.error("Metric measurement failed: %s", e, exc_info=True)
            return _error_response("METRIC_ERROR", f"{type(self.metric).__name__} failed: {e}")

        score = self.metric.score
        reason = getattr(self.metric, "reason", None) or ""
        threshold = getattr(self.metric, "threshold", 0.5)
        success = getattr(self.metric, "success", score is not None and score >= threshold)
        label = "Pass" if success else "Fail"

        return {"value": score, "label": label, "explanation": reason}


def _error_response(code: str, message: str) -> Dict[str, str]:
    """Build a standardized error response dict."""
    return {"errorCode": code, "errorMessage": message}
