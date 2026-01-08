"""Strands integration for Bedrock AgentCore Evaluation."""

from bedrock_agentcore.evaluation.integrations.strands_agents_evals.evaluator import (
    StrandsEvalsAgentCoreEvaluator,
    create_strands_evaluator,
)

__all__ = [
    "create_strands_evaluator",
    "StrandsEvalsAgentCoreEvaluator",
]
