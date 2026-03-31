"""Result types for the evaluation runner."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvaluatorResult(BaseModel):
    """Results from a single evaluator, grouped.

    Attributes:
        evaluator_id: The evaluator that produced these results.
        results: List of raw response dicts from the Evaluate API.
    """

    evaluator_id: str
    results: List[Dict[str, Any]]


class ScenarioResult(BaseModel):
    """Evaluation results for a single scenario.

    Attributes:
        scenario_id: The scenario that was evaluated.
        session_id: Framework-generated session ID.
        status: "COMPLETED" or "FAILED".
        error: Error message if scenario failed, None otherwise.
        evaluator_results: Results grouped by evaluator.
    """

    scenario_id: str
    session_id: str
    status: str = "COMPLETED"
    error: Optional[str] = None
    evaluator_results: List[EvaluatorResult] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Aggregate results for an entire evaluation run.

    Attributes:
        scenario_results: Results for each scenario in the dataset.
    """

    scenario_results: List[ScenarioResult] = Field(default_factory=list)
