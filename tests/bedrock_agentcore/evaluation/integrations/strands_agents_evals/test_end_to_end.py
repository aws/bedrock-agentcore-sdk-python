"""End-to-end integration tests for Strands AgentCore Evaluation."""

from unittest.mock import Mock, patch

import pytest
from strands import tool
from strands_evals import Case, Experiment

from bedrock_agentcore.evaluation import create_strands_evaluator

# Suppress Pydantic serialization warnings for OTel spans
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning:pydantic.main")


@pytest.fixture
def mock_boto_client():
    """Create a mock boto3 client."""
    client = Mock()
    client.evaluate.return_value = {"evaluationResults": [{"value": 0.85, "explanation": "Good response"}]}
    return client


@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression."""
    return str(eval(expression))


class TestEndToEndIntegration:
    """Test end-to-end integration matching real developer experience."""

    def test_evaluation_with_adot_format(self, mock_boto_client):
        """Test evaluation with pre-formatted ADOT spans from CloudWatch."""
        # Simulate ADOT spans from CloudWatch
        adot_spans = [
            {
                "scope": {"name": "strands.agent"},
                "traceId": "1234567890abcdef",
                "spanId": "abcdef123456",
                "name": "test-span",
            }
        ]

        cases = [Case(input="Test", expected_output="Response")]

        def task_fn(case):
            return {"output": "Response", "trajectory": adot_spans}

        with patch("boto3.client", return_value=mock_boto_client):
            evaluator = create_strands_evaluator("Builtin.Helpfulness")
            experiment = Experiment(cases=cases, evaluators=[evaluator])
            experiment.run_evaluations(task_fn)

            # Verify ADOT spans passed through without conversion
            call_args = mock_boto_client.evaluate.call_args[1]
            assert call_args["evaluationInput"]["sessionSpans"] == adot_spans

    def test_evaluation_with_empty_trajectory(self, mock_boto_client):
        """Test evaluation handles empty trajectory gracefully."""
        cases = [Case(input="Test", expected_output="Response")]

        def task_fn(case):
            return {"output": "Response", "trajectory": []}

        with patch("boto3.client", return_value=mock_boto_client):
            evaluator = create_strands_evaluator("Builtin.Helpfulness")
            experiment = Experiment(cases=cases, evaluators=[evaluator])
            reports = experiment.run_evaluations(task_fn)
            report = reports[0]

            # Should return 0 score for empty trajectory
            assert report.overall_score == 0.0
            assert not any(report.test_passes)  # All tests failed
