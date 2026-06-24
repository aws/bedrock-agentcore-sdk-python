"""DeepEval adapter for AgentCore evaluation integrations."""

import logging
from typing import Any, Callable, Dict, List, Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams

from bedrock_agentcore.evaluation.integrations.base import (
    BaseAdapter,
    ParsedEvaluationEvent,
    extract_fields_from_spans,
)

logger = logging.getLogger(__name__)

_PARAM_TO_FIELD: Dict[SingleTurnParams, str] = {
    SingleTurnParams.INPUT: "input",
    SingleTurnParams.ACTUAL_OUTPUT: "actual_output",
    SingleTurnParams.EXPECTED_OUTPUT: "expected_output",
    SingleTurnParams.CONTEXT: "context",
    SingleTurnParams.RETRIEVAL_CONTEXT: "retrieval_context",
}

_METRIC_REQUIRED_PARAMS: Dict[str, List[str]] = {
    "AnswerRelevancyMetric": ["input", "actual_output"],
    "FaithfulnessMetric": ["input", "actual_output", "retrieval_context"],
    "ContextualRelevancyMetric": ["input", "actual_output", "retrieval_context"],
    "ContextualPrecisionMetric": ["input", "actual_output", "expected_output", "retrieval_context"],
    "ContextualRecallMetric": ["input", "actual_output", "expected_output", "retrieval_context"],
    "HallucinationMetric": ["input", "actual_output", "context"],
    "BiasMetric": ["input", "actual_output"],
    "ToxicityMetric": ["input", "actual_output"],
    "GEval": ["input", "actual_output"],
    "SummarizationMetric": ["input", "actual_output"],
}


def _get_required_params(metric: BaseMetric) -> List[str]:
    """Determine which LLMTestCase fields a metric requires.

    Fallback chain:
        1. metric._required_params (DeepEval internal attribute)
        2. Static registry _METRIC_REQUIRED_PARAMS keyed by class name
        3. metric.evaluation_params (GEval special case)
        4. Default: ["input", "actual_output"]
    """
    if hasattr(metric, "_required_params") and metric._required_params:
        params = metric._required_params
        if all(p in _PARAM_TO_FIELD for p in params):
            return [_PARAM_TO_FIELD[p] for p in params]

    class_name = type(metric).__name__
    if class_name in _METRIC_REQUIRED_PARAMS:
        return _METRIC_REQUIRED_PARAMS[class_name]

    if hasattr(metric, "evaluation_params") and metric.evaluation_params:
        params = metric.evaluation_params
        return [_PARAM_TO_FIELD.get(p, str(p).lower()) for p in params]

    return ["input", "actual_output"]


class DeepEvalAdapter(BaseAdapter):
    """Adapter that runs a DeepEval metric against AgentCore evaluation events.

    Example::

        from deepeval.metrics import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric(threshold=0.7)
        handler = DeepEvalAdapter(metric=metric)

        # Use as Lambda handler
        def lambda_handler(event, context):
            return handler(event, context)
    """

    def __init__(
        self,
        metric: BaseMetric,
        field_mapper: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        model: Optional[Any] = None,
        timeout: Optional[int] = None,
    ):
        """Initialize the adapter.

        Args:
            metric: A DeepEval BaseMetric instance (e.g. AnswerRelevancyMetric).
            field_mapper: Optional callable that receives the raw Lambda event and
                returns a dict of LLMTestCase field values. Bypasses default span
                extraction when provided.
            model: Optional model override for the metric's LLM. Can be a string
                model ID (e.g. "bedrock/anthropic.claude-3") or a DeepEvalBaseLLM
                subclass instance.
            timeout: Maximum seconds to allow for metric.measure(). Defaults to 290
                (slightly under Lambda's 300s max).
        """
        super().__init__(field_mapper=field_mapper, timeout=timeout)
        self.metric = metric
        if model is not None:
            self.metric.model = model

    def validate_fields(self, fields: Dict[str, Any]) -> None:
        """Validate that fields required by the metric are present."""
        required = _get_required_params(self.metric)
        missing = [f for f in required if f not in fields or not fields[f]]
        if missing:
            metric_name = type(self.metric).__name__
            raise ValueError(
                f"Field(s) {missing} required by {metric_name} but not found in evaluation event. "
                f"Provide a field_mapper or ensure spans contain the necessary data."
            )

    def execute(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Run the DeepEval metric and return formatted results."""
        test_case = LLMTestCase(
            input=fields.get("input", ""),
            actual_output=fields.get("actual_output", ""),
            expected_output=fields.get("expected_output"),
            context=fields.get("context"),
            retrieval_context=fields.get("retrieval_context"),
        )

        self.metric.measure(test_case)

        score = self.metric.score
        reason = getattr(self.metric, "reason", None) or ""
        threshold = getattr(self.metric, "threshold", 0.5)
        success = getattr(self.metric, "success", score is not None and score >= threshold)
        label = "Pass" if success else "Fail"

        return {"value": score, "label": label, "explanation": reason}


def build_test_case(
    parsed: ParsedEvaluationEvent,
    metric: BaseMetric,
    field_mapper: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> LLMTestCase:
    """Build a DeepEval LLMTestCase from a parsed evaluation event.

    Args:
        parsed: The parsed Lambda event.
        metric: The DeepEval metric (used to determine required fields).
        field_mapper: Optional callable that receives the raw Lambda event fields
            and returns a dict of LLMTestCase field values. Bypasses default
            span extraction when provided.

    Returns:
        An LLMTestCase ready for metric.measure().

    Raises:
        ValueError: If required fields for the metric cannot be populated.
    """
    if field_mapper is not None:
        raw_event = {
            "evaluationLevel": parsed.evaluation_level,
            "evaluationInput": {"sessionSpans": parsed.session_spans},
            "evaluationTarget": {
                "traceIds": [parsed.target_trace_id] if parsed.target_trace_id else [],
                "spanIds": [parsed.target_span_id] if parsed.target_span_id else [],
            },
            "evaluationReferenceInputs": parsed.reference_inputs,
        }
        fields = field_mapper(raw_event)
    else:
        fields = extract_fields_from_spans(parsed)

    required = _get_required_params(metric)
    missing = [f for f in required if f not in fields or not fields[f]]
    if missing:
        metric_name = type(metric).__name__
        raise ValueError(
            f"Field(s) {missing} required by {metric_name} but not found in evaluation event. "
            f"Provide a field_mapper or ensure spans contain the necessary data."
        )

    return LLMTestCase(
        input=fields.get("input", ""),
        actual_output=fields.get("actual_output", ""),
        expected_output=fields.get("expected_output"),
        context=fields.get("context"),
        retrieval_context=fields.get("retrieval_context"),
    )


# Backward-compatible alias
DeepEvalHandler = DeepEvalAdapter
