"""Base adapter for third-party evaluation framework integrations."""

import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_parsers import (
    SpanParseResult,
    parse_spans,
)

logger = logging.getLogger(__name__)


class BaseAdapter(abc.ABC):
    """Base adapter for third-party evaluation framework integrations.

    Accepts an EvaluatorInput (from the code_based_evaluators flow),
    extracts fields from spans using the built-in parser layer, runs the
    evaluation via execute(), and returns an EvaluatorOutput.

    Never raises unhandled exceptions — always returns a valid EvaluatorOutput.
    """

    def __init__(
        self,
        field_mapper: Optional[Callable[[EvaluatorInput], Dict[str, Any]]] = None,
    ):
        """Initialize the adapter.

        Args:
            field_mapper: Optional callable that receives the EvaluatorInput and
                returns a dict of field values. Bypasses default span parsing
                when provided.
        """
        self.field_mapper = field_mapper

    def __call__(self, evaluator_input: EvaluatorInput, context: Any = None) -> EvaluatorOutput:
        """Handle an evaluation invocation.

        Args:
            evaluator_input: Parsed EvaluatorInput from the code-based evaluator flow.
            context: Lambda context object (unused).

        Returns:
            EvaluatorOutput with score, label, and explanation or error fields.
        """
        try:
            fields = self._extract_fields(evaluator_input)
        except ValueError as e:
            logger.error("Field extraction failed: %s", e)
            return EvaluatorOutput(
                label="Error",
                errorCode="FIELD_EXTRACTION_ERROR",
                errorMessage=str(e),
            )

        try:
            self.validate_fields(fields)
        except ValueError as e:
            logger.error("Validation failed: %s", e)
            return EvaluatorOutput(
                label="Error",
                errorCode="MISSING_REQUIRED_FIELD",
                errorMessage=str(e),
            )

        try:
            return self.execute(fields)
        except Exception as e:
            logger.error("Execution failed: %s", e, exc_info=True)
            return EvaluatorOutput(
                label="Error",
                errorCode="METRIC_ERROR",
                errorMessage=f"{type(self).__name__} failed: {e}",
            )

    def _extract_fields(self, evaluator_input: EvaluatorInput) -> Dict[str, Any]:
        """Extract fields from the EvaluatorInput."""
        if self.field_mapper is not None:
            return self.field_mapper(evaluator_input)

        reference_inputs = getattr(evaluator_input, "reference_inputs", None)
        result = parse_spans(evaluator_input.session_spans, reference_inputs)
        return result.to_dict()

    @abc.abstractmethod
    def validate_fields(self, fields: Dict[str, Any]) -> None:
        """Validate that required fields are present.

        Each adapter must explicitly declare its validation behavior.

        Args:
            fields: Extracted field dict.

        Raises:
            ValueError: If required fields are missing.
        """

    @abc.abstractmethod
    def execute(self, fields: Dict[str, Any]) -> EvaluatorOutput:
        """Run the evaluation and return an EvaluatorOutput.

        Args:
            fields: Extracted field dict with keys like "input", "actual_output", etc.

        Returns:
            EvaluatorOutput with evaluation results.
        """
