"""Runner package: shared types and evaluation runner."""

from .dataset_providers import DatasetProvider, FileDatasetProvider
from .dataset_types import (
    Dataset,
    Input,
    PredefinedScenario,
    Scenario,
    Turn,
)
from .invoker_types import (
    AgentInvokerFn,
    AgentInvokerInput,
    AgentInvokerOutput,
)
from .on_demand import (
    AgentSpanCollector,
    CloudWatchAgentSpanCollector,
    EvaluationResult,
    EvaluationRunConfig,
    EvaluatorConfig,
    EvaluatorResult,
    OnDemandEvaluationDatasetRunner,
    ScenarioResult,
)
from .scenario_executor import (
    PredefinedScenarioExecutor,
    ScenarioExecutionResult,
    ScenarioExecutor,
)

__all__ = [
    "AgentInvokerFn",
    "AgentInvokerInput",
    "AgentInvokerOutput",
    "AgentSpanCollector",
    "CloudWatchAgentSpanCollector",
    "Dataset",
    "DatasetProvider",
    "EvaluationResult",
    "EvaluationRunConfig",
    "EvaluatorConfig",
    "EvaluatorResult",
    "FileDatasetProvider",
    "Input",
    "OnDemandEvaluationDatasetRunner",
    "PredefinedScenario",
    "PredefinedScenarioExecutor",
    "Scenario",
    "ScenarioExecutionResult",
    "ScenarioExecutor",
    "ScenarioResult",
    "Turn",
]
