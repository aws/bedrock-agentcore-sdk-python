"""Strands evaluator wrapper for AgentCore Evaluation API."""

import asyncio
import logging
from typing import Any, List, Optional

import boto3
from botocore.config import Config
from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput
from typing_extensions import TypeVar

from bedrock_agentcore._utils.endpoints import DEFAULT_REGION
from bedrock_agentcore.evaluation.span_to_adot_serializer import convert_strands_to_adot

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def _is_valid_adot_document(item: Any) -> bool:
    """Check if item is a valid ADOT document.

    Args:
        item: Potential ADOT document

    Returns:
        True if item has required ADOT fields
    """
    return isinstance(item, dict) and "scope" in item and "traceId" in item and "spanId" in item


def _validate_spans(spans):
    """Validate spans are OpenTelemetry Span objects."""
    if not spans:
        return False
    # Check first span has required OTel attributes
    first_span = spans[0]
    return hasattr(first_span, "context") and hasattr(first_span, "instrumentation_scope")


def _is_adot_format(spans: List[Any]) -> bool:
    """Check if spans are already in ADOT format.

    ADOT format is detected by presence of 'scope' dict with 'name' field.
    This indicates spans were exported via ADOT (e.g., from CloudWatch) rather
    than raw OTel spans from in-memory exporter.

    Args:
        spans: List of span objects (either raw OTel or ADOT JSON dicts)

    Returns:
        True if spans are in ADOT format, False if raw OTel spans
    """
    if not spans:
        logger.warning("Empty spans list provided to format detector")
        return False

    first_span = spans[0]

    # ADOT format: dict with required fields
    if _is_valid_adot_document(first_span):
        scope = first_span.get("scope", {})
        if isinstance(scope, dict) and "name" in scope:
            logger.debug("Detected ADOT format with scope.name=%s", scope.get("name"))
        return True

    # Raw OTel: object with attributes
    logger.debug("Detected raw OTel format (type=%s)", type(first_span).__name__)
    return False


class StrandsEvalsAgentCoreEvaluator(Evaluator[str, str]):
    """Wraps AgentCore Evaluation API as Strands Evaluator.

    Automatically converts Strands OTel spans to AgentCore format.
    """

    def __init__(
        self,
        evaluator_id: str,
        region: str = DEFAULT_REGION,
        test_pass_score: float = 0.7,
        config: Optional[Config] = None,
    ):
        """Initialize the evaluator.

        Args:
            evaluator_id: Built-in evaluator name or custom evaluator ARN
            region: AWS region for the evaluation API
            test_pass_score: Minimum score threshold for test to pass
            config: Optional boto3 Config for client configuration
        """
        super().__init__()
        self.evaluator_id = evaluator_id
        self.test_pass_score = test_pass_score

        # Create client with provided or default config
        client_config = config or self._get_default_config()
        self.client = boto3.client("agentcore-evaluation-dataplane", region_name=region, config=client_config)

    @staticmethod
    def _get_default_config() -> Config:
        """Get default boto3 client configuration."""
        return Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=300,
        )

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> List[EvaluationOutput]:
        """Evaluate agent output using AgentCore Evaluation API.

        Args:
            evaluation_case: Evaluation case with input, expected output, and trajectory

        Returns:
            List of evaluation outputs with scores and explanations
        """
        # Handle empty trajectory (e.g., agent failed to execute)
        if not evaluation_case.actual_trajectory:
            return [
                EvaluationOutput(
                    score=0.0, test_pass=False, reason="No trajectory data available - agent may have failed to execute"
                )
            ]

        # Check if spans are already in ADOT format or need conversion
        if _is_adot_format(evaluation_case.actual_trajectory):
            # Already in ADOT format (fetched from CloudWatch), use as-is
            spans = evaluation_case.actual_trajectory
        else:
            # Raw OTel spans from in-memory exporter, validate and convert
            if not _validate_spans(evaluation_case.actual_trajectory):
                return [EvaluationOutput(score=0.0, test_pass=False, reason="Invalid span objects")]
            spans = convert_strands_to_adot(evaluation_case.actual_trajectory)

        request_payload = {"evaluatorId": self.evaluator_id, "evaluationInput": {"sessionSpans": spans}}

        try:
            response = self.client.evaluate(**request_payload)
        except Exception as e:
            logger.warning("AgentCore Evaluation API error: %s", e, exc_info=True)
            return [EvaluationOutput(score=0.0, test_pass=False, reason=f"API error: {str(e)}")]

        return [
            EvaluationOutput(
                score=r.get("value", 0.0),
                test_pass=r.get("value", 0.0) >= self.test_pass_score,
                reason=r.get("explanation", ""),
            )
            for r in response["evaluationResults"]
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> List[EvaluationOutput]:
        """Evaluate agent output asynchronously using AgentCore Evaluation API.

        Args:
            evaluation_case: Evaluation case with input, expected output, and trajectory

        Returns:
            List of evaluation outputs with scores and explanations
        """
        return await asyncio.to_thread(self.evaluate, evaluation_case)


def create_strands_evaluator(evaluator_id: str, **kwargs) -> StrandsEvalsAgentCoreEvaluator:
    """Create Strands-compatible evaluator backed by AgentCore Evaluation API.

    Args:
        evaluator_id: "Builtin.Helpfulness" or custom evaluator ARN
        **kwargs: Additional arguments passed to StrandsEvalsAgentCoreEvaluator
            region (str): AWS region (default: us-west-2)
            test_pass_score (float): Minimum score for test to pass (default: 0.7)

    Returns:
        StrandsEvalsAgentCoreEvaluator instance

    Example:
        evaluator = create_strands_evaluator("Builtin.Helpfulness")
        dataset = Dataset(cases=cases, evaluator=evaluator)
        report = dataset.run_evaluations(task_fn)
    """
    return StrandsEvalsAgentCoreEvaluator(evaluator_id, **kwargs)
