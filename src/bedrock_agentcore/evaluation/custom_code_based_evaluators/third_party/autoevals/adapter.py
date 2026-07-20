"""Autoevals adapter for AgentCore code-based evaluators."""

import logging
from typing import Any, Callable, Dict, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.base import BaseAdapter

logger = logging.getLogger(__name__)


class AutoEvalsAdapter(BaseAdapter):
    """Adapter that runs an Autoevals scorer against AgentCore evaluation events.

    Example (default span mapping)::

        from autoevals import Factuality
        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.autoevals import AutoEvalsAdapter

        scorer = Factuality()
        adapter = AutoEvalsAdapter(metric=scorer)

    Example (custom mapper returning eval kwargs)::

        from typing import Dict, Any

        def my_mapper(ev: EvaluatorInput) -> Dict[str, Any]:
            return {
                "input": ev.session_spans[0]["attributes"]["question"],
                "output": ev.session_spans[0]["attributes"]["answer"],
                "expected": "the expected answer",
            }

        adapter = AutoEvalsAdapter(
            metric=Factuality(),
            custom_mapper=my_mapper,
        )
    """

    def __init__(
        self,
        metric: Any,
        custom_mapper: Optional[Callable[[EvaluatorInput], Dict[str, Any]]] = None,
        threshold: Optional[float] = None,
    ):
        """Initialize the adapter.

        Args:
            metric: An Autoevals scorer instance (e.g. Factuality(), ClosedQA()).
            custom_mapper: Optional callable that receives the EvaluatorInput and
                returns a dict of kwargs for metric.eval(). Bypasses default span
                mapping when provided. Expected keys: input, output, expected (optional).
            threshold: Optional score threshold for Pass/Fail label. If None, label
                is omitted from the output.
        """
        self.metric = metric
        self.custom_mapper = custom_mapper
        self.threshold = threshold

    def _run(self, evaluator_input: EvaluatorInput) -> EvaluatorOutput:
        """Run the Autoevals scorer pipeline."""
        if self.custom_mapper is not None:
            kwargs = self.custom_mapper(evaluator_input)
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
                    f"Provide a custom_mapper or ensure spans contain the necessary data.",
                )
            kwargs: Dict[str, Any] = {
                "input": result.input,
                "output": result.actual_output,
            }
            if result.expected_output:
                kwargs["expected"] = result.expected_output
            elif result.assertions:
                kwargs["expected"] = "\n".join(result.assertions)
            if result.retrieval_context:
                kwargs["context"] = "\n".join(result.retrieval_context)

        eval_result = self.metric.eval(**kwargs)

        score = eval_result.score
        label = None
        if self.threshold is not None:
            label = "Pass" if score is not None and score >= self.threshold else "Fail"
        metadata = getattr(eval_result, "metadata", None)
        explanation = metadata.get("rationale", "") if isinstance(metadata, dict) else ""

        return EvaluatorOutput(value=score, label=label, explanation=explanation)
