"""Configuration for the evaluation runner."""

from typing import List

from pydantic import BaseModel


class EvaluatorConfig(BaseModel):
    """Configuration for evaluators.

    Attributes:
        evaluator_ids: List of evaluator IDs (built-in names or custom ARNs).
    """

    evaluator_ids: List[str]


class EvaluationRunConfig(BaseModel):
    """Top-level configuration for an on-demand evaluation run.

    Attributes:
        evaluator_config: Evaluator settings.
        evaluation_delay_seconds: Seconds to wait for CloudWatch span ingestion.
        max_concurrent_scenarios: Thread pool size for concurrent scenario execution.
    """

    evaluator_config: EvaluatorConfig
    evaluation_delay_seconds: int = 180
    max_concurrent_scenarios: int = 5
