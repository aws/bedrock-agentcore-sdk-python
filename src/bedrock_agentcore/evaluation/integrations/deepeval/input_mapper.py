"""Map AgentCore Lambda evaluation events to DeepEval LLMTestCase objects."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

logger = logging.getLogger(__name__)

_PARAM_TO_FIELD: Dict[LLMTestCaseParams, str] = {
    LLMTestCaseParams.INPUT: "input",
    LLMTestCaseParams.ACTUAL_OUTPUT: "actual_output",
    LLMTestCaseParams.EXPECTED_OUTPUT: "expected_output",
    LLMTestCaseParams.CONTEXT: "context",
    LLMTestCaseParams.RETRIEVAL_CONTEXT: "retrieval_context",
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


@dataclass
class ParsedEvaluationEvent:
    """Parsed representation of the AgentCore Lambda evaluation event."""

    evaluation_level: str
    session_spans: List[Dict[str, Any]]
    target_trace_id: Optional[str] = None
    target_span_id: Optional[str] = None
    reference_inputs: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_lambda_event(cls, event: Dict[str, Any]) -> "ParsedEvaluationEvent":
        """Parse a raw Lambda event dict into a structured object.

        Args:
            event: Raw Lambda event payload from the evaluation service.

        Returns:
            ParsedEvaluationEvent with extracted fields.

        Raises:
            KeyError: If required top-level fields are missing.
        """
        evaluation_input = event["evaluationInput"]
        target = event.get("evaluationTarget") or {}
        trace_ids = target.get("traceIds") or []
        span_ids = target.get("spanIds") or []

        return cls(
            evaluation_level=event["evaluationLevel"],
            session_spans=evaluation_input["sessionSpans"],
            target_trace_id=trace_ids[0] if trace_ids else None,
            target_span_id=span_ids[0] if span_ids else None,
            reference_inputs=event.get("evaluationReferenceInputs") or [],
        )


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
        return [_PARAM_TO_FIELD.get(p, str(p).lower()) for p in params]

    class_name = type(metric).__name__
    if class_name in _METRIC_REQUIRED_PARAMS:
        return _METRIC_REQUIRED_PARAMS[class_name]

    if hasattr(metric, "evaluation_params") and metric.evaluation_params:
        params = metric.evaluation_params
        return [_PARAM_TO_FIELD.get(p, str(p).lower()) for p in params]

    return ["input", "actual_output"]


def _get_message_content(message: Any) -> str:
    """Extract text content from a message object.

    Message content can be a dict with a "content" or "message" key, or a plain string.
    Handles one level of nesting (e.g. {"content": {"content": "text"}}).
    """
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        for key in ("content", "message"):
            if key in message:
                val = message[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    return _get_message_content(val)
                return str(val)
    return ""


def _extract_fields_from_spans(
    parsed: ParsedEvaluationEvent,
) -> Dict[str, Any]:
    """Extract LLMTestCase fields from AgentCore session spans.

    Parses _eval_log_records from span attributes, filters by target_trace_id,
    and extracts messages by role:
        - input ← input messages where role=="user"
        - actual_output ← output messages where role=="assistant"
        - retrieval_context ← output messages where role=="tool"
        - expected_output ← evaluationReferenceInputs[0].expectedResponse
    """
    user_messages: List[str] = []
    assistant_messages: List[str] = []
    tool_messages: List[str] = []

    for span in parsed.session_spans:
        attributes = span.get("attributes", {})
        log_records_raw = attributes.get("_eval_log_records")
        if not log_records_raw:
            continue

        if isinstance(log_records_raw, str):
            try:
                log_records = json.loads(log_records_raw)
            except (json.JSONDecodeError, TypeError):
                logger.debug("Failed to parse _eval_log_records as JSON")
                continue
        else:
            log_records = log_records_raw

        if not isinstance(log_records, list):
            continue

        for record in log_records:
            if not isinstance(record, dict):
                continue

            if parsed.target_trace_id:
                record_trace_id = record.get("traceId") or record.get("trace_id")
                if record_trace_id and record_trace_id != parsed.target_trace_id:
                    continue

            body = record.get("body", {})
            if not isinstance(body, dict):
                continue

            input_data = body.get("input", {})
            if isinstance(input_data, dict):
                for msg in input_data.get("messages", []):
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "")
                    content = _get_message_content(msg)
                    if role == "user" and content:
                        user_messages.append(content)

            output_data = body.get("output", {})
            if isinstance(output_data, dict):
                for msg in output_data.get("messages", []):
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "")
                    content = _get_message_content(msg)
                    if role == "assistant" and content:
                        assistant_messages.append(content)
                    elif role == "tool" and content:
                        tool_messages.append(content)

    fields: Dict[str, Any] = {}

    if user_messages:
        fields["input"] = "\n".join(user_messages)
    if assistant_messages:
        fields["actual_output"] = "\n".join(assistant_messages)
    if tool_messages:
        fields["retrieval_context"] = tool_messages

    if parsed.reference_inputs:
        expected = parsed.reference_inputs[0].get("expectedResponse")
        if expected:
            fields["expected_output"] = expected

    return fields


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
        fields = _extract_fields_from_spans(parsed)

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
