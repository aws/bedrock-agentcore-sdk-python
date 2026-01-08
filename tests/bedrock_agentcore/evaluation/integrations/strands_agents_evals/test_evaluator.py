"""Tests for Strands AgentCore Evaluator."""

from unittest.mock import Mock, patch

import pytest
from botocore.config import Config as BotocoreConfig
from strands_evals.types import EvaluationData

from bedrock_agentcore.evaluation.integrations.strands_agents_evals.evaluator import (
    StrandsEvalsAgentCoreEvaluator,
    _is_adot_format,
    _is_valid_adot_document,
    _validate_spans,
    create_strands_evaluator,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_boto_client():
    """Create a mock boto3 client."""
    client = Mock()
    client.evaluate.return_value = {"evaluationResults": [{"value": 0.85, "explanation": "Good response"}]}
    return client


@pytest.fixture
def mock_otel_span():
    """Create a mock OTel span."""
    span = Mock()
    span.context = Mock()
    span.context.trace_id = 0x1234567890ABCDEF
    span.context.span_id = 0x1234567890ABCDEF
    span.context.trace_flags = 1
    span.instrumentation_scope = Mock()
    span.instrumentation_scope.name = "strands.agent"
    span.instrumentation_scope.version = "1.0.0"
    span.resource = Mock()
    span.resource.attributes = {}
    span.status = Mock()
    span.status.status_code = Mock(__str__=Mock(return_value="StatusCode.OK"))
    span.parent = None
    span.name = "test-span"
    span.start_time = 1000
    span.end_time = 2000
    span.kind = Mock(__str__=Mock(return_value="SpanKind.INTERNAL"))
    span.attributes = {}
    span.events = []
    return span


@pytest.fixture
def adot_span():
    """Create an ADOT-formatted span."""
    return {
        "scope": {"name": "strands.agent"},
        "traceId": "1234567890abcdef",
        "spanId": "abcdef123456",
        "name": "test-span",
    }


@pytest.fixture
def evaluator(mock_boto_client):
    """Create an evaluator with mocked client."""
    with patch("boto3.client", return_value=mock_boto_client):
        evaluator = StrandsEvalsAgentCoreEvaluator(
            evaluator_id="Builtin.Helpfulness",
            region="us-west-2",
            test_pass_score=0.7,
        )
        return evaluator


# ==============================================================================
# Helper Function Tests
# ==============================================================================


class TestValidateSpans:
    """Test _validate_spans helper function."""

    def test_valid_otel_spans(self, mock_otel_span):
        """Test validation passes for valid OTel spans."""
        assert _validate_spans([mock_otel_span]) is True

    def test_empty_spans(self):
        """Test validation fails for empty list."""
        assert _validate_spans([]) is False

    def test_invalid_span_no_context(self):
        """Test validation fails when span has no context."""
        span = Mock(spec=[])
        assert _validate_spans([span]) is False

    def test_invalid_span_no_instrumentation_scope(self):
        """Test validation fails when span has no instrumentation_scope."""
        span = Mock()
        span.context = Mock()
        del span.instrumentation_scope
        assert _validate_spans([span]) is False


class TestIsAdotFormat:
    """Test _is_adot_format helper function."""

    def test_adot_format_detected(self, adot_span):
        """Test ADOT format is correctly detected."""
        assert _is_adot_format([adot_span]) is True

    def test_otel_format_detected(self, mock_otel_span):
        """Test OTel format is correctly detected."""
        assert _is_adot_format([mock_otel_span]) is False

    def test_empty_list(self):
        """Test empty list returns False."""
        assert _is_adot_format([]) is False

    def test_dict_without_scope(self):
        """Test dict without scope returns False."""
        assert _is_adot_format([{"traceId": "123"}]) is False

    def test_dict_with_scope_no_name(self):
        """Test dict with scope but no name returns False."""
        assert _is_adot_format([{"scope": {}}]) is False


# ==============================================================================
# StrandsEvalsAgentCoreEvaluator Tests
# ==============================================================================


class TestStrandsEvalsAgentCoreEvaluator:
    """Test StrandsEvalsAgentCoreEvaluator class."""

    def test_init_basic(self, mock_boto_client):
        """Test basic initialization."""
        with patch("boto3.client", return_value=mock_boto_client) as mock_client_call:
            evaluator = StrandsEvalsAgentCoreEvaluator(
                evaluator_id="Builtin.Helpfulness",
                region="us-west-2",
            )

            assert evaluator.evaluator_id == "Builtin.Helpfulness"
            assert evaluator.test_pass_score == 0.7  # default
            mock_client_call.assert_called_once()

    def test_init_custom_pass_score(self, mock_boto_client):
        """Test initialization with custom pass score."""
        with patch("boto3.client", return_value=mock_boto_client):
            evaluator = StrandsEvalsAgentCoreEvaluator(
                evaluator_id="Builtin.Accuracy",
                region="us-east-1",
                test_pass_score=0.9,
            )

            assert evaluator.test_pass_score == 0.9

    def test_init_custom_config(self, mock_boto_client):
        """Test initialization with custom boto config."""
        custom_config = BotocoreConfig(connect_timeout=10)

        with patch("boto3.client", return_value=mock_boto_client) as mock_client_call:
            StrandsEvalsAgentCoreEvaluator(
                evaluator_id="Builtin.Helpfulness",
                region="us-west-2",
                config=custom_config,
            )

            call_kwargs = mock_client_call.call_args[1]
            assert call_kwargs["config"] == custom_config

    def test_evaluate_success(self, evaluator, mock_boto_client, mock_otel_span):
        """Test successful evaluation."""
        evaluation_case = EvaluationData(
            input="What is 2+2?",
            actual_output="4",
            actual_trajectory=[mock_otel_span],
        )

        results = evaluator.evaluate(evaluation_case)

        assert len(results) == 1
        assert results[0].score == 0.85
        assert results[0].test_pass is True
        assert results[0].reason == "Good response"
        mock_boto_client.evaluate.assert_called_once()

    def test_evaluate_empty_trajectory(self, evaluator):
        """Test evaluation with empty trajectory."""
        evaluation_case = EvaluationData(
            input="What is 2+2?",
            actual_output="4",
            actual_trajectory=[],
        )

        results = evaluator.evaluate(evaluation_case)

        assert len(results) == 1
        assert results[0].score == 0.0
        assert results[0].test_pass is False
        assert "No trajectory data" in results[0].reason

    def test_evaluate_none_trajectory(self, evaluator):
        """Test evaluation with None trajectory."""
        evaluation_case = EvaluationData(
            input="What is 2+2?",
            actual_output="4",
            actual_trajectory=None,
        )

        results = evaluator.evaluate(evaluation_case)

        assert len(results) == 1
        assert results[0].score == 0.0
        assert results[0].test_pass is False

    def test_evaluate_invalid_spans(self, evaluator):
        """Test evaluation with invalid span objects."""
        invalid_span = Mock(spec=[])  # No context or instrumentation_scope

        evaluation_case = EvaluationData(
            input="What is 2+2?",
            actual_output="4",
            actual_trajectory=[invalid_span],
        )

        results = evaluator.evaluate(evaluation_case)

        assert len(results) == 1
        assert results[0].score == 0.0
        assert "Invalid span objects" in results[0].reason

    def test_evaluate_adot_format_passthrough(self, evaluator, mock_boto_client, adot_span):
        """Test ADOT format spans are passed through without conversion."""
        evaluation_case = EvaluationData(
            input="What is 2+2?",
            actual_output="4",
            actual_trajectory=[adot_span],
        )

        evaluator.evaluate(evaluation_case)

        # Verify the ADOT span was passed directly
        call_args = mock_boto_client.evaluate.call_args
        assert call_args[1]["evaluationInput"]["sessionSpans"] == [adot_span]

    def test_evaluate_api_error(self, evaluator, mock_boto_client, mock_otel_span):
        """Test evaluation handles API errors."""
        mock_boto_client.evaluate.side_effect = Exception("API Error")

        evaluation_case = EvaluationData(
            input="What is 2+2?",
            actual_output="4",
            actual_trajectory=[mock_otel_span],
        )

        results = evaluator.evaluate(evaluation_case)

        assert len(results) == 1
        assert results[0].score == 0.0
        assert results[0].test_pass is False
        assert "API error" in results[0].reason

    def test_evaluate_below_pass_threshold(self, mock_boto_client, mock_otel_span):
        """Test evaluation below pass threshold."""
        mock_boto_client.evaluate.return_value = {
            "evaluationResults": [{"value": 0.5, "explanation": "Needs improvement"}]
        }

        with patch("boto3.client", return_value=mock_boto_client):
            evaluator = StrandsEvalsAgentCoreEvaluator(
                evaluator_id="Builtin.Helpfulness",
                region="us-west-2",
                test_pass_score=0.7,
            )

        evaluation_case = EvaluationData(
            input="What is 2+2?",
            actual_output="4",
            actual_trajectory=[mock_otel_span],
        )

        results = evaluator.evaluate(evaluation_case)

        assert results[0].score == 0.5
        assert results[0].test_pass is False

    def test_evaluate_multiple_results(self, mock_boto_client, mock_otel_span):
        """Test evaluation with multiple results."""
        mock_boto_client.evaluate.return_value = {
            "evaluationResults": [
                {"value": 0.9, "explanation": "Great"},
                {"value": 0.6, "explanation": "OK"},
            ]
        }

        with patch("boto3.client", return_value=mock_boto_client):
            evaluator = StrandsEvalsAgentCoreEvaluator(
                evaluator_id="Builtin.Helpfulness",
                region="us-west-2",
                test_pass_score=0.7,
            )

        evaluation_case = EvaluationData(
            input="Test",
            actual_output="Response",
            actual_trajectory=[mock_otel_span],
        )

        results = evaluator.evaluate(evaluation_case)

        assert len(results) == 2
        assert results[0].test_pass is True
        assert results[1].test_pass is False


class TestEvaluateAsync:
    """Test async evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_async(self, evaluator, mock_boto_client, mock_otel_span):
        """Test async evaluation delegates to sync."""
        evaluation_case = EvaluationData(
            input="What is 2+2?",
            actual_output="4",
            actual_trajectory=[mock_otel_span],
        )

        results = await evaluator.evaluate_async(evaluation_case)

        assert len(results) == 1
        assert results[0].score == 0.85


# ==============================================================================
# Factory Function Tests
# ==============================================================================


class TestCreateStrandsEvaluator:
    """Test create_strands_evaluator factory function."""

    def test_create_basic(self, mock_boto_client):
        """Test basic evaluator creation."""
        with patch("boto3.client", return_value=mock_boto_client):
            evaluator = create_strands_evaluator("Builtin.Helpfulness")

            assert isinstance(evaluator, StrandsEvalsAgentCoreEvaluator)
            assert evaluator.evaluator_id == "Builtin.Helpfulness"

    def test_create_with_region(self, mock_boto_client):
        """Test evaluator creation with region."""
        with patch("boto3.client", return_value=mock_boto_client) as mock_client_call:
            create_strands_evaluator("Builtin.Accuracy", region="eu-west-1")

            call_kwargs = mock_client_call.call_args[1]
            assert call_kwargs["region_name"] == "eu-west-1"

    def test_create_with_pass_score(self, mock_boto_client):
        """Test evaluator creation with custom pass score."""
        with patch("boto3.client", return_value=mock_boto_client):
            evaluator = create_strands_evaluator(
                "Builtin.Helpfulness",
                test_pass_score=0.8,
            )

            assert evaluator.test_pass_score == 0.8

    def test_create_with_custom_arn(self, mock_boto_client):
        """Test evaluator creation with custom ARN."""
        custom_arn = "arn:aws:bedrock:us-west-2:123456789012:evaluator/my-evaluator"

        with patch("boto3.client", return_value=mock_boto_client):
            evaluator = create_strands_evaluator(custom_arn)

            assert evaluator.evaluator_id == custom_arn


class TestIsValidAdotDocument:
    """Test _is_valid_adot_document helper."""

    def test_valid_adot_document(self):
        """Test valid ADOT document is recognized."""
        doc = {"scope": {"name": "test"}, "traceId": "123", "spanId": "456"}
        assert _is_valid_adot_document(doc) is True

    def test_missing_scope(self):
        """Test document missing scope is invalid."""
        doc = {"traceId": "123", "spanId": "456"}
        assert _is_valid_adot_document(doc) is False

    def test_missing_trace_id(self):
        """Test document missing traceId is invalid."""
        doc = {"scope": {"name": "test"}, "spanId": "456"}
        assert _is_valid_adot_document(doc) is False

    def test_missing_span_id(self):
        """Test document missing spanId is invalid."""
        doc = {"scope": {"name": "test"}, "traceId": "123"}
        assert _is_valid_adot_document(doc) is False

    def test_not_a_dict(self):
        """Test non-dict is invalid."""
        assert _is_valid_adot_document("not a dict") is False
        assert _is_valid_adot_document(None) is False
