"""Configuration for the evaluation runner."""

from typing import List, Optional

from pydantic import BaseModel

from bedrock_agentcore.evaluation.runner.dataset_types import SimulationConfig


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
        simulation_config: Actor simulation settings. Required when the dataset
            contains SimulatedScenario entries.
    """

    evaluator_config: EvaluatorConfig
    evaluation_delay_seconds: int = 180
    max_concurrent_scenarios: int = 5
    simulation_config: Optional[SimulationConfig] = None
