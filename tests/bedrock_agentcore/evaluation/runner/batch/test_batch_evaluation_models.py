"""Unit tests for session source config to_data_source_config methods."""

from datetime import datetime, timezone

from bedrock_agentcore.evaluation.runner.batch.batch_evaluation_models import (
    CloudWatchDataSourceConfig,
)

_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
_T1 = datetime(2024, 1, 2, tzinfo=timezone.utc)

_CW_SOURCE = CloudWatchDataSourceConfig(
    service_names=["my-service"],
    log_group_names=["/aws/my-log-group"],
    ingestion_delay_seconds=0,
)


def test_cloudwatch_to_data_source_config_returns_session_ids():
    result = _CW_SOURCE.to_data_source_config(["s1", "s2"], _T0, _T1)
    cw = result["cloudWatchLogs"]
    assert cw["serviceNames"] == ["my-service"]
    assert cw["logGroupNames"] == ["/aws/my-log-group"]
    assert cw["filterConfig"]["sessionIds"] == ["s1", "s2"]
    assert cw["filterConfig"]["timeRange"]["startTime"] == _T0
    assert cw["filterConfig"]["timeRange"]["endTime"] == _T1
