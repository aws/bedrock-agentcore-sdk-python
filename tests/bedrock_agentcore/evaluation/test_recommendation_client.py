"""Tests for RecommendationClient."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore._utils.config import WaitConfig
from bedrock_agentcore.config_bundle.bundle import ConfigBundleRef
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
_LOGGER = "bedrock_agentcore.evaluation.recommendation_client"


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


_BUNDLE_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:configuration-bundle/orders_bundle-A1B2C3D4E5"
_BUNDLE_ID = "orders_bundle-A1B2C3D4E5"
_INLINE = {"inlineHarnessConfig": {"systemPrompt": {"text": "You help customers track their orders."}}}
_HARNESS_SOURCES = {
    "batch": {"batchEvaluationSource": {"batchEvaluationId": "batch_eval-A1B2C3D4E5"}},
    "online": {
        "onlineEvaluationConfigSource": {
            "onlineEvaluationConfigId": "orders_online-A1B2C3D4E5",
            "timeRange": {"startTime": "2026-10-01T00:00:00Z", "endTime": "2026-10-08T00:00:00Z"},
        }
    },
    "insights": {"insightsSource": {"applicationName": "orders_app.DEFAULT", "agentInsightIds": ["466f5319f6906f9f"]}},
}


def _harness_config(source, harness_configuration=None):
    return {
        "harnessRecommendationConfig": {
            "evaluationSource": _HARNESS_SOURCES[source],
            "harnessConfiguration": harness_configuration or _INLINE,
        }
    }


def _bundle_entry(version_id):
    return {
        "explanation": "Changed.",
        "recommendedConfig": {"configurationBundle": {"bundleArn": _BUNDLE_ARN, "versionId": version_id}},
    }


def _harness_rec(*entries, status="COMPLETED"):
    return {
        "recommendationId": "rec-1",
        "type": "HARNESS_RECOMMENDATION",
        "status": status,
        "recommendationResult": {"harnessRecommendationResult": {"recommendations": list(entries)}},
    }


_V1 = "28954cb8-d6ee-40c2-9da7-76c1c362a8b1"
_V2 = "3733e065-b9b8-4b2b-a05a-568cfafd816b"
_NO_CHANGE = {"explanation": "The traces provide no evidence for a change."}
_INLINE_ENTRY = {"explanation": "Changed.", "recommendedConfig": _INLINE}
_FAILED_ENTRY = {"errorCode": "ValidationException", "errorMessage": "Pass agentInsightIds to select at most 10."}


class TestHarnessRecommendationPassthrough:
    @pytest.mark.parametrize("source", ["batch", "online", "insights"])
    def test_start_forwards_typed_wrapped_config(self, client_and_dp, source):
        client, mock_dp = client_and_dp
        config = _harness_config(source)
        client.start_recommendation(name="r", type="HARNESS_RECOMMENDATION", recommendation_config=config)
        mock_dp.start_recommendation.assert_called_once_with(
            name="r", type="HARNESS_RECOMMENDATION", recommendationConfig=config
        )

    @pytest.mark.parametrize("source", ["batch", "online", "insights"])
    def test_start_and_wait_forwards_typed_wrapped_config(self, client_and_dp, source):
        client, mock_dp = client_and_dp
        config = _harness_config(source, {"configurationBundle": {"bundleArn": _BUNDLE_ARN, "versionId": _V2}})
        mock_dp.start_recommendation.return_value = {"recommendationId": "rec-1"}
        mock_dp.get_recommendation.return_value = _harness_rec(_bundle_entry(_V1))
        with patch("bedrock_agentcore._utils.polling.time.sleep"):
            result = client.start_recommendation_and_wait(
                wait_config=_FAST, name="r", type="HARNESS_RECOMMENDATION", recommendation_config=config
            )
        assert result["status"] == "COMPLETED"
        mock_dp.start_recommendation.assert_called_once_with(
            name="r", type="HARNESS_RECOMMENDATION", recommendationConfig=config
        )

    def test_list_forwards_agent_insight_id(self, client_and_dp):
        client, mock_dp = client_and_dp
        client.list_recommendations(max_results=10, agent_insight_id="466f5319f6906f9f")
        mock_dp.list_recommendations.assert_called_once_with(maxResults=10, agentInsightId="466f5319f6906f9f")


class TestGetRecommendedBundleVersions:
    def test_changed_bundle_returns_ref(self, client_and_dp):
        client, mock_dp = client_and_dp
        result = client.get_recommended_bundle_versions(_harness_rec(_bundle_entry(_V1)))
        assert result == [ConfigBundleRef(bundle_arn=_BUNDLE_ARN, bundle_version=_V1)]
        assert result[0].bundle_id == _BUNDLE_ID
        assert mock_dp.method_calls == []

    def test_makes_no_network_calls(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        client = RecommendationClient()
        mock_session.reset_mock()
        client.get_recommended_bundle_versions(_harness_rec(_bundle_entry(_V1), _bundle_entry(_V2)))
        assert mock_session.mock_calls == []

    def test_only_bundle_entries_returned_in_order(self, client_and_dp):
        client, _ = client_and_dp
        rec = _harness_rec(_bundle_entry(_V2), _NO_CHANGE, _INLINE_ENTRY, _bundle_entry(_V1))
        assert [ref.bundle_version for ref in client.get_recommended_bundle_versions(rec)] == [_V2, _V1]

    @pytest.mark.parametrize(
        "entries",
        [[_NO_CHANGE], [_INLINE_ENTRY], [_NO_CHANGE, _INLINE_ENTRY], []],
        ids=["no-change", "inline", "no-change-and-inline", "empty"],
    )
    def test_no_bundle_result_returns_empty_list(self, client_and_dp, caplog, entries):
        client, _ = client_and_dp
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            assert client.get_recommended_bundle_versions(_harness_rec(*entries)) == []
        assert caplog.records == []

    def test_failed_entries_skipped_and_logged(self, client_and_dp, caplog):
        client, _ = client_and_dp
        rec = _harness_rec(_FAILED_ENTRY, _bundle_entry(_V1), _NO_CHANGE)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = client.get_recommended_bundle_versions(rec)
        assert result == [ConfigBundleRef(bundle_arn=_BUNDLE_ARN, bundle_version=_V1)]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "ValidationException" in warnings[0].getMessage()
        assert _FAILED_ENTRY["errorMessage"] in warnings[0].getMessage()

    def test_all_failed_raises(self, client_and_dp):
        client, _ = client_and_dp
        other = {"errorCode": "InternalServerException", "errorMessage": "boom"}
        with pytest.raises(
            RuntimeError,
            match=r"all 2 recommendations failed: ValidationException: .*; InternalServerException: boom",
        ):
            client.get_recommended_bundle_versions(_harness_rec(_FAILED_ENTRY, other))

    def test_non_harness_raises(self, client_and_dp):
        client, _ = client_and_dp
        rec = {
            "type": "SYSTEM_PROMPT_RECOMMENDATION",
            "status": "COMPLETED",
            "recommendationResult": {"systemPromptRecommendationResult": {"recommendedSystemPrompt": "new"}},
        }
        with pytest.raises(ValueError, match="is a SYSTEM_PROMPT_RECOMMENDATION, not a HARNESS_RECOMMENDATION"):
            client.get_recommended_bundle_versions(rec)

    @pytest.mark.parametrize("status", ["FAILED", "IN_PROGRESS"])
    def test_not_completed_raises(self, client_and_dp, status):
        client, _ = client_and_dp
        with pytest.raises(ValueError, match=f"status is {status}; bundle versions are only available when COMPLETED"):
            client.get_recommended_bundle_versions(_harness_rec(_FAILED_ENTRY, status=status))

    def test_missing_result_raises(self, client_and_dp):
        client, _ = client_and_dp
        summary = {"recommendationId": "rec-1", "type": "HARNESS_RECOMMENDATION", "status": "COMPLETED"}
        with pytest.raises(ValueError, match="has no harnessRecommendationResult; pass a GetRecommendation response"):
            client.get_recommended_bundle_versions(summary)
