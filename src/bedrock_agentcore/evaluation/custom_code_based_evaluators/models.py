"""Typed models for code-based evaluator Lambda input and output."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReferenceInput(BaseModel):
    """A single ground-truth entry from the event's ``evaluationReferenceInputs`` list.

    Field shapes follow the AgentCore code-based-evaluator contract. ``extra="allow"``
    keeps unknown/future keys instead of dropping them.

    Attributes:
        context: Span context for the entry, e.g. {"spanContext": {"sessionId", "traceId"}}.
        expected_response: Expected response object, e.g. {"text": "..."} (NOT a bare string).
        assertions: Assertion-style ground truth, e.g. [{"text": "..."}].
        expected_trajectory: Expected tool trajectory, e.g. {"toolNames": [...]}.
    """

    context: Dict[str, Any] = Field(default_factory=dict)
    expected_response: Optional[Dict[str, Any]] = Field(default=None, alias="expectedResponse")
    assertions: List[Dict[str, Any]] = Field(default_factory=list)
    expected_trajectory: Optional[Dict[str, Any]] = Field(default=None, alias="expectedTrajectory")

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    @property
    def expected_response_text(self) -> Optional[str]:
        """The ``expected_response.text`` value, or None if not present."""
        return (self.expected_response or {}).get("text")


class EvaluatorInput(BaseModel):
    """Parsed input for a code-based evaluator Lambda function.

    Attributes:
        evaluation_level: The evaluation granularity - "SESSION", "TRACE", or "TOOL_CALL".
        session_spans: Raw ADOT span dicts from the evaluation service.
        target_trace_id: The target trace ID (set for TRACE level, None otherwise).
        target_span_id: The target span ID (set for TOOL_CALL level, None otherwise).
        schema_version: Schema version of the Lambda contract.
        evaluator_id: The ID of the code-based evaluator that was invoked.
        evaluator_name: The name of the code-based evaluator that was invoked.
        reference_inputs: Ground-truth reference inputs (from evaluationReferenceInputs),
            filtered by the service according to evaluation level. Empty when no ground
            truth is configured.
    """

    evaluation_level: str
    session_spans: List[Dict]
    target_trace_id: Optional[str] = None
    target_span_id: Optional[str] = None
    schema_version: str = "1.0"
    evaluator_id: Optional[str] = None
    evaluator_name: Optional[str] = None
    reference_inputs: List[ReferenceInput] = Field(default_factory=list)


class EvaluatorOutput(BaseModel):
    """Result returned by a code-based evaluator function.

    For **success** responses, ``label`` is required and ``errorCode`` / ``errorMessage``
    should be omitted.  For **error** responses, set ``errorCode`` (and optionally
    ``errorMessage``); ``label`` may be omitted.

    Attributes:
        value: Numerical score for the evaluation (success responses).
        label: Categorical label (e.g. "Pass", "Fail"). Required unless errorCode is set.
        explanation: Optional explanation of the evaluation result.
        errorCode: Error code for error responses (e.g. "VALIDATION_FAILED").
        errorMessage: Human-readable error description for error responses.
    """

    value: Optional[float] = None
    label: Optional[str] = None
    explanation: Optional[str] = None
    errorCode: Optional[str] = None
    errorMessage: Optional[str] = None

    @model_validator(mode="after")
    def _require_label_or_error_code(self) -> "EvaluatorOutput":
        if not self.errorCode and self.label is None and self.value is None:
            raise ValueError(
                "Either label, value, or errorCode must be set; "
                "set errorCode to return an error response without a label"
            )
        return self
