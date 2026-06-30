"""Autoevals adapter for AgentCore code-based evaluators."""

import logging
from typing import Any, Callable, Dict, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.base import BaseAdapter

logger = logging.getLogger(__name__)


class AutoevalsAdapter(BaseAdapter):
    """Adapter that runs an Autoevals scorer against AgentCore evaluation events.

    Example::

        from autoevals import Factuality
        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.autoevals import AutoevalsAdapter

        scorer = Factuality()
        adapter = AutoevalsAdapter(scorer=scorer)
    """

    def __init__(
        self,
        scorer: Any,
        field_mapper: Optional[Callable[[EvaluatorInput], Dict[str, Any]]] = None,
        threshold: float = 0.5,
    ):
        """Initialize the adapter.

        Args:
            scorer: An Autoevals scorer instance (e.g. Factuality(), ClosedQA()).
            field_mapper: Optional callable that receives the EvaluatorInput and
                returns a dict of field values. Bypasses default span parsing.
            threshold: Score threshold for Pass/Fail determination. Defaults to 0.5.
        """
        super().__init__(field_mapper=field_mapper)
        self.scorer = scorer
        self.threshold = threshold

    def validate_fields(self, fields: Dict[str, Any]) -> None:
        """Validate minimum required fields; scorer raises on additional missing params."""
        missing = []
        if not fields.get("input"):
            missing.append("input")
        if not fields.get("actual_output"):
            missing.append("actual_output")
        if missing:
            scorer_name = type(self.scorer).__name__
            raise ValueError(
                f"Field(s) {missing} required by {scorer_name} but not found in evaluation event. "
                f"Provide a field_mapper or ensure spans contain the necessary data."
            )

    def execute(self, fields: Dict[str, Any]) -> EvaluatorOutput:
        """Run the Autoevals scorer and return formatted results."""
        kwargs: Dict[str, Any] = {
            "input": fields.get("input", ""),
            "output": fields.get("actual_output", ""),
        }
        if fields.get("expected_output"):
            kwargs["expected"] = fields["expected_output"]

        result = self.scorer.eval(**kwargs)

        score = result.score
        label = "Pass" if score is not None and score >= self.threshold else "Fail"
        explanation = getattr(result, "metadata", {}).get("rationale", "") if hasattr(result, "metadata") else ""

        return EvaluatorOutput(value=score, label=label, explanation=explanation)
