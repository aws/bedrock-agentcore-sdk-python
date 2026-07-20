"""Base adapter for third-party evaluation framework integrations."""

import abc
import logging
from typing import Any, Dict, List, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers import (
    SpanMapResult,
    map_spans,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
    FieldExtractionError,
)

logger = logging.getLogger(__name__)


class BaseAdapter(abc.ABC):
    """Base adapter for third-party evaluation framework integrations.

    Accepts an EvaluatorInput (from the code_based_evaluators flow),
    extracts fields from spans using the built-in mapper layer, runs the
    evaluation via execute(), and returns an EvaluatorOutput.

    Never raises unhandled exceptions — always returns a valid EvaluatorOutput.
    """

    def __call__(self, evaluator_input: EvaluatorInput, context: Any = None) -> EvaluatorOutput:
        """Handle an evaluation invocation.

        Args:
            evaluator_input: Parsed EvaluatorInput from the code-based evaluator flow.
            context: Lambda context object (unused).

        Returns:
            EvaluatorOutput with score, label, and explanation or error fields.
        """
        try:
            return self._run(evaluator_input)
        except FieldExtractionError as e:
            logger.error("Field extraction failed: %s", e)
            return EvaluatorOutput(
                label="Error",
                errorCode="FIELD_EXTRACTION_ERROR",
                errorMessage=str(e),
            )
        except Exception as e:
            logger.error("Execution failed: %s", e, exc_info=True)
            return EvaluatorOutput(
                label="Error",
                errorCode="METRIC_ERROR",
                errorMessage=f"{type(self).__name__} failed: {e}",
            )

    @abc.abstractmethod
    def _run(self, evaluator_input: EvaluatorInput) -> EvaluatorOutput:
        """Run the full evaluation pipeline. Subclasses implement this."""

    def _default_extract(self, evaluator_input: EvaluatorInput) -> SpanMapResult:
        """Extract fields using the built-in span mapper layer."""
        spans = self._filter_spans_by_target(evaluator_input)
        return map_spans(spans, evaluator_input.reference_inputs)

    def _filter_spans_by_target(self, evaluator_input: EvaluatorInput) -> List[Dict]:
        """Filter session spans based on evaluationLevel and evaluationTarget.

        The service passes ALL session spans in every Lambda invocation without
        pre-filtering. It fans out one Lambda call per evaluation target (one per
        trace at TRACE level, one per span at TOOL_CALL level) and provides the
        target ID so the Lambda can scope its evaluation. We filter here because
        the service-side _invoke_single_eval_target passes session_spans directly
        into the payload without filtering.

        Levels:
        - SESSION: all spans (no filtering) — one call for the entire session
        - TRACE: only spans matching target_trace_id (service sends exactly one)
        - TOOL_CALL: only the span matching target_span_id (service sends exactly one)
        """
        spans = evaluator_input.session_spans

        if evaluator_input.evaluation_level == "TRACE" and evaluator_input.target_trace_id:
            spans = [s for s in spans if s.get("traceId") == evaluator_input.target_trace_id]
        elif evaluator_input.evaluation_level == "TOOL_CALL" and evaluator_input.target_span_id:
            spans = [s for s in spans if s.get("spanId") == evaluator_input.target_span_id]

        return spans
