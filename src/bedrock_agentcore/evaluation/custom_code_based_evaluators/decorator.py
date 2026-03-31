"""Decorator that adapts a typed evaluator function into a Lambda handler."""

import functools
import logging

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput

logger = logging.getLogger(__name__)


def custom_code_based_evaluator():
    """Decorator that wraps a typed evaluator function as a Lambda handler.

    The decorated function receives an ``EvaluatorInput`` and the Lambda
    ``context``, and returns an ``EvaluatorOutput``. The decorator handles
    parsing the raw Lambda event dict into ``EvaluatorInput`` and serializing
    the ``EvaluatorOutput`` into the response contract expected by the
    evaluation service.

    Must be called with parentheses: ``@custom_code_based_evaluator()``.

    Example::

        @custom_code_based_evaluator()
        def handler(input: EvaluatorInput, context) -> EvaluatorOutput:
            return EvaluatorOutput(value=1.0, label="Pass")
    """

    def decorator(fn):
        @functools.wraps(fn)
        def lambda_handler(event, context=None):
            logger.debug("Raw Lambda event: %s", event)

            target = event.get("evaluationTarget") or {}
            trace_ids = target.get("traceIds") or []
            span_ids = target.get("spanIds") or []

            evaluator_input = EvaluatorInput(
                evaluation_level=event["evaluationLevel"],
                session_spans=event["evaluationInput"]["sessionSpans"],
                target_trace_id=trace_ids[0] if trace_ids else None,
                target_span_id=span_ids[0] if span_ids else None,
                schema_version=event.get("schemaVersion", "1.0"),
            )

            result = fn(evaluator_input, context)

            if not isinstance(result, EvaluatorOutput):
                raise TypeError(f"Evaluator must return an EvaluatorOutput, got {type(result).__name__}")

            return result.model_dump()

        lambda_handler.unwrapped = fn
        return lambda_handler

    return decorator
