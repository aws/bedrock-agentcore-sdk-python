"""Autoevals adapter for AgentCore code-based evaluators."""

import logging
from typing import Any, Callable, Dict, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.base import BaseAdapter

logger = logging.getLogger(__name__)


class AutoevalsAdapter(BaseAdapter):
    """Adapter that runs an Autoevals scorer against AgentCore evaluation events.

    Example (default span mapping)::

        from autoevals import Factuality
        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.autoevals import AutoevalsAdapter

        scorer = Factuality()
        adapter = AutoevalsAdapter(scorer=scorer)

    Example (customer mapper returning eval kwargs)::

        adapter = AutoevalsAdapter(
            scorer=Factuality(),
            customer_mapper=lambda ev: {
                "input": ev.session_spans[0]["attributes"]["question"],
                "output": ev.session_spans[0]["attributes"]["answer"],
                "expected": "the expected answer",
            },
        )
    """

    def __init__(
        self,
        scorer: Any,
        customer_mapper: Optional[Callable[[EvaluatorInput], Dict[str, Any]]] = None,
        threshold: float = 0.5,
    ):
        """Initialize the adapter.

        Args:
            scorer: An Autoevals scorer instance (e.g. Factuality(), ClosedQA()).
            customer_mapper: Optional callable that receives the EvaluatorInput and
                returns a dict of kwargs for scorer.eval(). Bypasses default span
                mapping when provided. Expected keys: input, output, expected (optional).
            threshold: Score threshold for Pass/Fail determination. Defaults to 0.5.
        """
        self.scorer = scorer
        self.customer_mapper = customer_mapper
        self.threshold = threshold

    def _run(self, evaluator_input: EvaluatorInput) -> EvaluatorOutput:
        """Run the Autoevals scorer pipeline."""
        if self.customer_mapper is not None:
            kwargs = self.customer_mapper(evaluator_input)
        else:
            result = self._default_extract(evaluator_input)
            if not result.input or not result.actual_output:
                missing = []
                if not result.input:
                    missing.append("input")
                if not result.actual_output:
                    missing.append("actual_output")
                scorer_name = type(self.scorer).__name__
                return EvaluatorOutput(
                    label="Error",
                    errorCode="MISSING_REQUIRED_FIELD",
                    errorMessage=f"Field(s) {missing} required by {scorer_name} but not found in evaluation event. "
                    f"Provide a customer_mapper or ensure spans contain the necessary data.",
                )
            kwargs: Dict[str, Any] = {
                "input": result.input,
                "output": result.actual_output,
            }
            if result.expected_output:
                kwargs["expected"] = result.expected_output

        eval_result = self.scorer.eval(**kwargs)

        score = eval_result.score
        label = "Pass" if score is not None and score >= self.threshold else "Fail"
        explanation = getattr(eval_result, "metadata", {}).get("rationale", "") if hasattr(eval_result, "metadata") else ""

        return EvaluatorOutput(value=score, label=label, explanation=explanation)
