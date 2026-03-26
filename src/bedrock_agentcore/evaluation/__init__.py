"""AgentCore Evaluation: EvaluationClient and Strands integration."""

from bedrock_agentcore.evaluation.client import EvaluationClient
from bedrock_agentcore.evaluation.span_to_adot_serializer import (
    convert_strands_to_adot,
)
from bedrock_agentcore.evaluation.utils.cloudwatch_span_helper import (
    fetch_spans_from_cloudwatch,
)

__all__ = [
    "EvaluationClient",
    "StrandsEvalsAgentCoreEvaluator",
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
