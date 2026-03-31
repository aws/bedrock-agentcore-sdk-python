"""Typed models for code-based evaluator Lambda input and output."""

from typing import Dict, List, Optional

from pydantic import BaseModel


class EvaluatorInput(BaseModel):
    """Parsed input for a code-based evaluator Lambda function.

    Attributes:
        evaluation_level: The evaluation granularity - "SESSION", "TRACE", or "TOOL_CALL".
        session_spans: Raw ADOT span dicts from the evaluation service.
        target_trace_id: The target trace ID (set for TRACE level, None otherwise).
        target_span_id: The target span ID (set for TOOL_CALL level, None otherwise).
        schema_version: Schema version of the Lambda contract.
    """

    evaluation_level: str
    session_spans: List[Dict]
    target_trace_id: Optional[str] = None
    target_span_id: Optional[str] = None
    schema_version: str = "1.0"


class EvaluatorOutput(BaseModel):
    """Result returned by a code-based evaluator function.

    Attributes:
        value: Numerical score for the evaluation.
        label: Categorical label (e.g. "Pass", "Fail"). Required.
        explanation: Optional explanation of the evaluation result.
    """

    value: Optional[float] = None
    label: str
    explanation: Optional[str] = None
    errorCode: Optional[str] = None
    errorMessage: Optional[str] = None
