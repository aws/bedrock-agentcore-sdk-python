"""Unit tests for session source config to_api_source methods."""

from datetime import datetime, timezone

from bedrock_agentcore.evaluation.runner.batch.batch_evaluation_models import (
    CloudWatchSessionSourceConfig,
)

_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
_T1 = datetime(2024, 1, 2, tzinfo=timezone.utc)

_CW_SOURCE = CloudWatchSessionSourceConfig(
    service_names=["my-service"],
    log_group_names=["/aws/my-log-group"],
    ingestion_delay_seconds=0,
)


def test_cloudwatch_to_api_source_returns_session_ids():
    result = _CW_SOURCE.to_api_source(["s1", "s2"], _T0, _T1)
    cw = result["cloudWatchSource"]
    assert cw["serviceNames"] == ["my-service"]
    assert cw["logGroupNames"] == ["/aws/my-log-group"]
    assert cw["sessionInput"] == {"sessionIds": ["s1", "s2"]}
