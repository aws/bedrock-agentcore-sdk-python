"""Tests for RecommendationClient."""

from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore._utils.config import WaitConfig
from bedrock_agentcore.evaluation.recommendation_client import RecommendationClient


@pytest.fixture
def mock_boto3():
    with patch("bedrock_agentcore.evaluation.recommendation_client.boto3") as m:
        yield m


@pytest.fixture
def client_and_dp(mock_boto3):
    """Return a (RecommendationClient, mock_dp_client) tuple."""
    mock_session = MagicMock()
    mock_boto3.Session.return_value = mock_session
    mock_dp = mock_session.client.return_value
    client = RecommendationClient()
    return client, mock_dp


_FAST = WaitConfig(max_wait=5, poll_interval=1)


class TestRecommendationClientInit:
    def test_uses_data_plane_service(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        RecommendationClient(region_name="eu-west-1")
        args, kwargs = mock_session.client.call_args
        assert args == ("bedrock-agentcore",)
        assert kwargs["region_name"] == "eu-west-1"

    def test_default_region_from_session(self, mock_boto3):
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_boto3.Session.return_value = mock_session
        assert RecommendationClient().region_name == "us-east-1"

    def test_custom_session(self, mock_boto3):
        custom_session = MagicMock()
        custom_session.region_name = "ap-southeast-1"
        client = RecommendationClient(boto3_session=custom_session)
        assert client.region_name == "ap-southeast-1"
        custom_session.client.assert_called_once()

    def test_endpoint_override(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        with patch(
            "bedrock_agentcore.evaluation.recommendation_client.DP_ENDPOINT_OVERRIDE",
            "https://example.test",
        ):
            RecommendationClient(region_name="us-west-2")
        assert mock_session.client.call_args.kwargs["endpoint_url"] == "https://example.test"


class TestRecommendationClientPassthrough:
    def test_allowed_method_forwarded(self, client_and_dp):
        client, mock_dp = client_and_dp
        mock_dp.list_recommendations.return_value = {"recommendations": []}
        client.list_recommendations(maxResults=10)
        mock_dp.list_recommendations.assert_called_once_with(maxResults=10)

    def test_snake_case_kwargs_converted(self, client_and_dp):
        client, mock_dp = client_and_dp
        client.get_recommendation(recommendation_id="rec-1")
        mock_dp.get_recommendation.assert_called_once_with(recommendationId="rec-1")

    def test_disallowed_method_raises(self, client_and_dp):
        client, _ = client_and_dp
        with pytest.raises(AttributeError, match="no attribute 'start_batch_evaluation'"):
            client.start_batch_evaluation()


class TestStartRecommendationAndWait:
    def test_returns_completed_response(self, client_and_dp):
        client, mock_dp = client_and_dp
        mock_dp.start_recommendation.return_value = {"recommendationId": "rec-1", "status": "PENDING"}
        mock_dp.get_recommendation.side_effect = [
            {"recommendationId": "rec-1", "status": "IN_PROGRESS"},
            {
                "recommendationId": "rec-1",
                "status": "COMPLETED",
                "recommendationResult": {"systemPromptRecommendationResult": {"recommendedSystemPrompt": "new"}},
            },
        ]
        with patch("bedrock_agentcore._utils.polling.time.sleep"):
            result = client.start_recommendation_and_wait(
                wait_config=_FAST, name="r", type="SYSTEM_PROMPT_RECOMMENDATION", recommendation_config={}
            )
        assert result["recommendationResult"]["systemPromptRecommendationResult"]["recommendedSystemPrompt"] == "new"
        mock_dp.start_recommendation.assert_called_once_with(
            name="r", type="SYSTEM_PROMPT_RECOMMENDATION", recommendationConfig={}
        )
        mock_dp.get_recommendation.assert_called_with(recommendationId="rec-1")

    def test_failed_raises_with_result(self, client_and_dp):
        client, mock_dp = client_and_dp
        mock_dp.start_recommendation.return_value = {"recommendationId": "rec-1"}
        mock_dp.get_recommendation.return_value = {
            "status": "FAILED",
            "recommendationResult": {"systemPromptRecommendationResult": {"errorMessage": "no traces found"}},
        }
        with pytest.raises(RuntimeError, match="no traces found"):
            client.start_recommendation_and_wait(wait_config=_FAST, name="r")

    def test_timeout(self, client_and_dp):
        client, mock_dp = client_and_dp
        mock_dp.start_recommendation.return_value = {"recommendationId": "rec-1"}
        mock_dp.get_recommendation.return_value = {"status": "IN_PROGRESS"}
        with (
            patch("bedrock_agentcore._utils.polling.time.sleep"),
            patch("bedrock_agentcore._utils.polling.time.time", side_effect=[0, 0, 10]),
            pytest.raises(TimeoutError),
        ):
            client.start_recommendation_and_wait(wait_config=_FAST, name="r")
