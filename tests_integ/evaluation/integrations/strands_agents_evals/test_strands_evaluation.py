"""
Integration tests for Strands AgentCore Evaluation.

Run with: python -m pytest tests_integ/evaluation/integrations/strands/test_strands_evaluation.py -v
"""

import logging
import os

import pytest
from strands import Agent, tool
from strands_evals import Case, Experiment
from strands_evals.telemetry import StrandsEvalsTelemetry

from bedrock_agentcore.evaluation import create_strands_evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REGION = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")

# Suppress Pydantic serialization warnings for OTel spans
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning:pydantic.main")


@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"


@pytest.mark.integration
class TestStrandsEvaluationIntegration:
    """Real integration tests for Strands AgentCore Evaluation."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")

    def test_real_evaluation_with_builtin_helpfulness(self):
        """Test real evaluation with Builtin.Helpfulness evaluator."""
        telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()

        def task_fn(case):
            agent = Agent(
                tools=[calculator],
                system_prompt="You are a helpful math assistant. Use the calculator tool to solve problems.",
            )
            agent_response = agent(case.input)

            # Convert tuple to list to avoid Pydantic warning
            raw_spans = list(telemetry.in_memory_exporter.get_finished_spans())
            return {"output": str(agent_response), "trajectory": raw_spans}

        cases = [Case(input="What is 2+2?", expected_output="4")]

        evaluator = create_strands_evaluator("Builtin.Helpfulness", region=REGION)
        experiment = Experiment(cases=cases, evaluators=[evaluator])
        reports = experiment.run_evaluations(task_fn)
        report = reports[0]

        # Verify results
        assert report.overall_score >= 0.0
        assert report.overall_score <= 1.0
        logger.info("Evaluation score: %s", report.overall_score)

    def test_real_evaluation_with_builtin_accuracy(self):
        """Test real evaluation with Builtin.Accuracy evaluator."""
        telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()

        def task_fn(case):
            agent = Agent(
                tools=[calculator],
                system_prompt="You are a helpful math assistant. Use the calculator tool to solve problems.",
            )
            agent_response = agent(case.input)

            raw_spans = list(telemetry.in_memory_exporter.get_finished_spans())
            return {"output": str(agent_response), "trajectory": raw_spans}

        cases = [Case(input="Calculate 5 + 3", expected_output="8")]

        evaluator = create_strands_evaluator("Builtin.Accuracy", region=REGION)
        experiment = Experiment(cases=cases, evaluators=[evaluator])
        reports = experiment.run_evaluations(task_fn)
        report = reports[0]

        assert report.overall_score >= 0.0
        assert report.overall_score <= 1.0
        logger.info("Accuracy score: %s", report.overall_score)

    def test_real_evaluation_with_multiple_cases(self):
        """Test real evaluation with multiple test cases."""
        telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()

        def task_fn(case):
            agent = Agent(
                tools=[calculator],
                system_prompt="You are a helpful math assistant. Use the calculator tool to solve problems.",
            )
            agent_response = agent(case.input)

            raw_spans = list(telemetry.in_memory_exporter.get_finished_spans())
            return {"output": str(agent_response), "trajectory": raw_spans}

        cases = [
            Case(input="What is 5 + 3?", expected_output="8"),
            Case(input="Calculate 10 + 7", expected_output="17"),
            Case(input="What is 100 - 25?", expected_output="75"),
        ]

        evaluator = create_strands_evaluator("Builtin.Helpfulness", region=REGION, test_pass_score=0.6)
        experiment = Experiment(cases=cases, evaluators=[evaluator])
        reports = experiment.run_evaluations(task_fn)
        report = reports[0]

        assert report.overall_score >= 0.0
        assert report.overall_score <= 1.0
        assert len(report.test_passes) == 3
        pass_rate = sum(report.test_passes) / len(report.test_passes)
        logger.info("Average score: %.2f", report.overall_score)
        logger.info("Pass rate: %.1f%%", pass_rate * 100)

    def test_evaluation_with_empty_trajectory(self):
        """Test evaluation handles empty trajectory gracefully."""

        def task_fn(case):
            return {"output": "Response", "trajectory": []}

        cases = [Case(input="Test", expected_output="Response")]

        evaluator = create_strands_evaluator("Builtin.Helpfulness", region=REGION)
        experiment = Experiment(cases=cases, evaluators=[evaluator])
        reports = experiment.run_evaluations(task_fn)
        report = reports[0]

        # Should return 0 score for empty trajectory
        assert report.overall_score == 0.0
        assert not any(report.test_passes)

    def test_evaluation_with_custom_pass_score(self):
        """Test evaluation with custom test pass score threshold."""
        telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()

        def task_fn(case):
            agent = Agent(
                tools=[calculator],
                system_prompt="You are a helpful math assistant. Use the calculator tool to solve problems.",
            )
            agent_response = agent(case.input)

            raw_spans = list(telemetry.in_memory_exporter.get_finished_spans())
            return {"output": str(agent_response), "trajectory": raw_spans}

        cases = [Case(input="What is 2+2?", expected_output="4")]

        # Test with high threshold
        evaluator = create_strands_evaluator("Builtin.Helpfulness", region=REGION, test_pass_score=0.9)
        experiment = Experiment(cases=cases, evaluators=[evaluator])
        reports = experiment.run_evaluations(task_fn)
        report = reports[0]

        assert report.overall_score >= 0.0
        assert report.overall_score <= 1.0
        logger.info("Score with 0.9 threshold: %s", report.overall_score)
