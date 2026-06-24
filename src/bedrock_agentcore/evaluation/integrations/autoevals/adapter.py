"""Autoevals adapter for AgentCore evaluation integrations."""

import logging
from typing import Any, Callable, Dict, Optional

from bedrock_agentcore.evaluation.integrations.base import BaseAdapter

logger = logging.getLogger(__name__)


class AutoevalsAdapter(BaseAdapter):
    """Adapter that runs an Autoevals scorer against AgentCore evaluation events.

    Example::

        from autoevals import Factuality

        scorer = Factuality()
        handler = AutoevalsAdapter(scorer=scorer)

        # Use as Lambda handler
        def lambda_handler(event, context):
            return handler(event, context)
    """

    def __init__(
        self,
        scorer: Any,
        field_mapper: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
    ):
        """Initialize the adapter.

        Args:
            scorer: An Autoevals scorer instance (e.g. Factuality(), ClosedQA()).
            field_mapper: Optional callable that receives the raw Lambda event and
                returns a dict of field values. Bypasses default span extraction.
            timeout: Maximum seconds to allow for scorer.eval(). Defaults to 290.
        """
        super().__init__(field_mapper=field_mapper, timeout=timeout)
        self.scorer = scorer

    def validate_fields(self, fields: Dict[str, Any]) -> None:
        """Validate that input and actual_output are present."""
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

    def execute(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Run the Autoevals scorer and return formatted results."""
        kwargs: Dict[str, Any] = {
            "input": fields.get("input", ""),
            "output": fields.get("actual_output", ""),
        }
        if fields.get("expected_output"):
            kwargs["expected"] = fields["expected_output"]

        result = self.scorer.eval(**kwargs)

        score = result.score
        label = "Pass" if score is not None and score >= 0.5 else "Fail"
        explanation = getattr(result, "metadata", {}).get("rationale", "") if hasattr(result, "metadata") else ""

        return {"value": score, "label": label, "explanation": explanation}
