"""AgentCore Evaluation: EvaluationClient, OnDemandEvaluationDatasetRunner, and Strands integration."""

from bedrock_agentcore.evaluation.client import EvaluationClient, ReferenceInputs
from bedrock_agentcore.evaluation.custom_code_based_evaluators import (
    EvaluatorInput,
    EvaluatorOutput,
    custom_code_based_evaluator,
)
from bedrock_agentcore.evaluation.runner.batch.batch_evaluation_models import (
    BatchEvaluationResult,
    BatchEvaluationRunConfig,
    BatchEvaluationSummary,
    BatchEvaluatorConfig,
    CloudWatchOutputDataConfig,
    CloudWatchSessionSourceConfig,
    EvaluatorStatistics,
    EvaluatorSummary,
    FailedScenario,
    TokenUsageSummary,
)
from bedrock_agentcore.evaluation.runner.batch.batch_evaluation_runner import (
    BatchEvaluationRunner,
)
from bedrock_agentcore.evaluation.runner.dataset_providers import (
    DatasetProvider,
    FileDatasetProvider,
)
from bedrock_agentcore.evaluation.runner.dataset_types import (
    Dataset,
    Input,
    PredefinedScenario,
    Scenario,
    Turn,
)
from bedrock_agentcore.evaluation.runner.invoker_types import (
    AgentInvokerFn,
    AgentInvokerInput,
    AgentInvokerOutput,
)
from bedrock_agentcore.evaluation.runner.on_demand import (
    AgentSpanCollector,
    CloudWatchAgentSpanCollector,
    EvaluationResult,
    EvaluationRunConfig,
    EvaluatorConfig,
    EvaluatorResult,
    OnDemandEvaluationDatasetRunner,
    PredefinedScenarioExecutor,
    ScenarioExecutionResult,
    ScenarioExecutor,
    ScenarioResult,
)
from bedrock_agentcore.evaluation.span_to_adot_serializer import (
    convert_strands_to_adot,
)
from bedrock_agentcore.evaluation.utils.cloudwatch_span_helper import (
    fetch_spans_from_cloudwatch,
)

__all__ = [
    "AgentInvokerFn",
    "BatchEvaluationRunner",
    "BatchEvaluationResult",
    "BatchEvaluationRunConfig",
    "CloudWatchOutputDataConfig",
    "CloudWatchSessionSourceConfig",
    "BatchEvaluatorConfig",
    "BatchEvaluationSummary",
    "EvaluatorStatistics",
    "EvaluatorSummary",
    "FailedScenario",
    "TokenUsageSummary",
    "AgentInvokerInput",
    "AgentInvokerOutput",
    "CloudWatchAgentSpanCollector",
    "Dataset",
    "DatasetProvider",
    "EvaluationClient",
    "EvaluationResult",
    "EvaluationRunConfig",
    "EvaluatorConfig",
    "EvaluatorInput",
    "EvaluatorOutput",
    "EvaluatorResult",
    "FileDatasetProvider",
    "Input",
    "OnDemandEvaluationDatasetRunner",
    "ReferenceInputs",
    "Scenario",
    "ScenarioExecutionResult",
    "ScenarioExecutor",
    "ScenarioResult",
    "AgentSpanCollector",
    "StrandsEvalsAgentCoreEvaluator",
    "Turn",
    "PredefinedScenario",
    "PredefinedScenarioExecutor",
    "custom_code_based_evaluator",
    "convert_strands_to_adot",
    "create_strands_evaluator",
    "fetch_spans_from_cloudwatch",
]

_STRANDS_EVALS_EXTRAS = {
    "StrandsEvalsAgentCoreEvaluator",
    "create_strands_evaluator",
}


def __getattr__(name: str):
    """Lazy import for optional strands-agents-evals dependencies."""
    if name in _STRANDS_EVALS_EXTRAS:
        try:
            from bedrock_agentcore.evaluation.integrations.strands_agents_evals.evaluator import (
                StrandsEvalsAgentCoreEvaluator,
                create_strands_evaluator,
            )
        except ImportError as e:
            raise ImportError(
                f"'{name}' requires the 'strands-agents-evals' extra. "
                "Install it with: pip install 'bedrock-agentcore[strands-agents-evals]'"
            ) from e

        _lazy = {
            "StrandsEvalsAgentCoreEvaluator": StrandsEvalsAgentCoreEvaluator,
            "create_strands_evaluator": create_strands_evaluator,
        }
        return _lazy[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
