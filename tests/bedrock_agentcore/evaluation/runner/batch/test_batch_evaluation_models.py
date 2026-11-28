"""Unit tests for session source config to_data_source_config methods."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from bedrock_agentcore.evaluation.runner.batch.batch_evaluation_models import (
    BatchEvaluationSummary,
    CloudWatchDataSourceConfig,
    OnlineEvaluationDataSourceConfig,
)

_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
_T1 = datetime(2024, 1, 2, tzinfo=timezone.utc)

_CW_SOURCE = CloudWatchDataSourceConfig(
    service_names=["my-service"],
    log_group_names=["/aws/my-log-group"],
    ingestion_delay_seconds=0,
)

_ONLINE_CONFIG_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:online-evaluation-config/oec-1"


def test_cloudwatch_to_data_source_config_returns_session_ids():
    result = _CW_SOURCE.to_data_source_config(["s1", "s2"], _T0, _T1)
    cw = result["cloudWatchLogs"]
    assert cw["serviceNames"] == ["my-service"]
    assert cw["logGroupNames"] == ["/aws/my-log-group"]
    assert cw["filterConfig"]["sessionIds"] == ["s1", "s2"]
    assert cw["filterConfig"]["timeRange"]["startTime"] == _T0
    assert cw["filterConfig"]["timeRange"]["endTime"] == _T1


def test_online_to_data_source_config_includes_time_range_by_default():
    source = OnlineEvaluationDataSourceConfig(online_evaluation_config_arn=_ONLINE_CONFIG_ARN)
    result = source.to_data_source_config(["s1", "s2"], _T0, _T1)
    online = result["onlineEvaluationConfigSource"]
    assert online["onlineEvaluationConfigArn"] == _ONLINE_CONFIG_ARN
    assert online["sessionFilterConfig"]["startTime"] == _T0
    assert online["sessionFilterConfig"]["endTime"] == _T1
    # Online source does not filter by session IDs.
    assert "sessionIds" not in online


def test_online_to_data_source_config_omits_time_range_when_disabled():
    source = OnlineEvaluationDataSourceConfig(
        online_evaluation_config_arn=_ONLINE_CONFIG_ARN,
        use_invocation_time_range=False,
    )
    result = source.to_data_source_config(["s1"], _T0, _T1)
    online = result["onlineEvaluationConfigSource"]
    assert online == {"onlineEvaluationConfigArn": _ONLINE_CONFIG_ARN}


def test_online_config_arn_must_be_non_empty():
    with pytest.raises(ValidationError):
        OnlineEvaluationDataSourceConfig(online_evaluation_config_arn="")


def test_evaluator_summaries_keep_the_dimension_of_each_multi_dimension_summary():
    summary = BatchEvaluationSummary.model_validate(
        {
            "evaluatorSummaries": [
                {
                    "evaluatorId": "SupportQuality-a1b2c3",
                    "evaluationDimensionName": "Accuracy",
                    "statistics": {"averageScore": 4.2},
                },
                {"evaluatorId": "SupportQuality-a1b2c3", "evaluationDimensionName": "Tone", "statistics": {}},
                {"evaluatorId": "SupportQuality-a1b2c3", "totalEvaluated": 0, "totalFailed": 1},
                {"evaluatorId": "Builtin.Helpfulness", "statistics": {"averageScore": 0.8}},
            ]
        }
    )

    assert [(s.evaluator_id, s.evaluation_dimension_name) for s in summary.evaluator_summaries] == [
        ("SupportQuality-a1b2c3", "Accuracy"),
        ("SupportQuality-a1b2c3", "Tone"),
        ("SupportQuality-a1b2c3", None),
        ("Builtin.Helpfulness", None),
    ]
    assert summary.evaluator_summaries[2].total_failed == 1
