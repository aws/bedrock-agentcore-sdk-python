"""Base adapter for AgentCore evaluation integrations."""

import abc
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput

logger = logging.getLogger(__name__)


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


def extract_fields_from_spans(
    parsed: ParsedEvaluationEvent,
) -> Dict[str, Any]:
    """Extract evaluation fields from AgentCore session spans.

    Parses _eval_log_records from span attributes, filters by target_trace_id,
    and extracts messages by role:
        - input ← input messages where role=="user"
        - actual_output ← output messages where role=="assistant"
        - retrieval_context ← output messages where role=="tool"
        - context ← same as retrieval_context
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
        fields["context"] = tool_messages

    if parsed.reference_inputs:
        expected = parsed.reference_inputs[0].get("expectedResponse")
        if expected:
            fields["expected_output"] = expected

    return fields


class _ExecutionTimeout(Exception):
    """Raised when execution exceeds the configured timeout."""


def _error_response(code: str, message: str) -> Dict[str, str]:
    """Build a standardized error response dict."""
    return {"errorCode": code, "errorMessage": message}


class BaseAdapter(abc.ABC):
    """Base adapter for evaluation framework integrations.

    Subclasses only need to implement execute(fields) which runs the actual
    evaluation logic and returns (score, label, explanation).

    Never raises unhandled exceptions — always returns a valid response dict.
    """

    DEFAULT_TIMEOUT = 290

    def __init__(
        self,
        field_mapper: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
    ):
        """Initialize the adapter.

        Args:
            field_mapper: Optional callable that receives the raw Lambda event and
                returns a dict of field values. Bypasses default span extraction.
            timeout: Maximum seconds to allow for execute(). Defaults to 290
                (slightly under Lambda's 300s max).
        """
        self.field_mapper = field_mapper
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT

    def __call__(self, event: Union[Dict[str, Any], EvaluatorInput], context: Any = None) -> Dict[str, Any]:
        """Handle a Lambda invocation.

        Args:
            event: Either a raw Lambda event dict or an EvaluatorInput instance
                from bedrock_agentcore.evaluation.custom_code_based_evaluators.models.
            context: Lambda context object (unused).

        Returns:
            Success: {"value": float, "label": str, "explanation": str}
            Error: {"errorCode": str, "errorMessage": str}
        """
        try:
            if isinstance(event, EvaluatorInput):
                parsed = ParsedEvaluationEvent(
                    evaluation_level=event.evaluation_level,
                    session_spans=event.session_spans,
                    target_trace_id=event.target_trace_id,
                    target_span_id=event.target_span_id,
                    reference_inputs=getattr(event, "reference_inputs", []) or [],
                )
            else:
                parsed = ParsedEvaluationEvent.from_lambda_event(event)
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Failed to parse evaluation event: %s", e)
            return _error_response("INVALID_EVENT", f"Failed to parse evaluation event: {e}")

        try:
            fields = self._extract_fields(parsed)
        except ValueError as e:
            logger.error("Missing required fields: %s", e)
            return _error_response("MISSING_REQUIRED_FIELD", str(e))

        try:
            result = self._execute_with_timeout(fields)
        except _ExecutionTimeout:
            return _error_response(
                "METRIC_TIMEOUT",
                f"{type(self).__name__} exceeded {self.timeout}s timeout.",
            )
        except Exception as e:
            logger.error("Execution failed: %s", e, exc_info=True)
            return _error_response("METRIC_ERROR", f"{type(self).__name__} failed: {e}")

        return result

    def _extract_fields(self, parsed: ParsedEvaluationEvent) -> Dict[str, Any]:
        """Extract fields from event, using field_mapper if provided."""
        if self.field_mapper is not None:
            raw_event = {
                "evaluationLevel": parsed.evaluation_level,
                "evaluationInput": {"sessionSpans": parsed.session_spans},
                "evaluationTarget": {
                    "traceIds": [parsed.target_trace_id] if parsed.target_trace_id else [],
                    "spanIds": [parsed.target_span_id] if parsed.target_span_id else [],
                },
                "evaluationReferenceInputs": parsed.reference_inputs,
            }
            return self.field_mapper(raw_event)

        fields = extract_fields_from_spans(parsed)
        self.validate_fields(fields)
        return fields

    def validate_fields(self, fields: Dict[str, Any]) -> None:
        """Validate that required fields are present.

        Override in subclasses to enforce field requirements.
        Default implementation does nothing.
        """

    @abc.abstractmethod
    def execute(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Run the evaluation and return the response dict.

        Args:
            fields: Extracted field dict with keys like "input", "actual_output", etc.

        Returns:
            {"value": float, "label": str, "explanation": str}
        """

    def _execute_with_timeout(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Run execute() with a thread-based timeout."""
        if self.timeout <= 0:
            return self.execute(fields)

        result_holder: list = []
        exception_holder: list = []

        def target():
            try:
                result_holder.append(self.execute(fields))
            except Exception as e:
                exception_holder.append(e)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            raise _ExecutionTimeout()

        if exception_holder:
            raise exception_holder[0]

        return result_holder[0]
