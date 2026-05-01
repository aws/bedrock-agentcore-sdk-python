"""Tests for shared _utils: pagination and polling."""

from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore._utils.namespace import build_namespace_params
from bedrock_agentcore._utils.polling import wait_until, wait_until_deleted


class TestWaitUntil:
    def test_immediate_success(self):
        poll_fn = Mock(return_value={"status": "ACTIVE"})
        result = wait_until(poll_fn, "ACTIVE", {"FAILED"})
        assert result["status"] == "ACTIVE"
        poll_fn.assert_called_once()

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch(
        "bedrock_agentcore._utils.polling.time.time",
        side_effect=[0, 0, 0, 1, 1],
    )
    def test_polls_until_target(self, _mock_time, _mock_sleep):
        poll_fn = Mock(
            side_effect=[{"status": "CREATING"}, {"status": "ACTIVE"}],
        )
        result = wait_until(poll_fn, "ACTIVE", {"FAILED"})
        assert result["status"] == "ACTIVE"
        assert poll_fn.call_count == 2

    def test_raises_on_failed_status(self):
        poll_fn = Mock(
            return_value={"status": "FAILED", "statusReasons": ["broke"]},
        )
        with pytest.raises(RuntimeError, match="FAILED"):
            wait_until(poll_fn, "ACTIVE", {"FAILED"})

    def test_custom_error_field(self):
        poll_fn = Mock(
            return_value={
                "status": "CREATE_FAILED",
                "failureReason": "bad config",
            },
        )
        with pytest.raises(RuntimeError, match="bad config"):
            wait_until(
                poll_fn,
                "ACTIVE",
                {"CREATE_FAILED"},
                error_field="failureReason",
            )

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch(
        "bedrock_agentcore._utils.polling.time.time",
        side_effect=[0, 0, 0, 301],
    )
    def test_timeout(self, _mock_time, _mock_sleep):
        poll_fn = Mock(return_value={"status": "CREATING"})
        with pytest.raises(TimeoutError):
            wait_until(poll_fn, "ACTIVE", {"FAILED"})


class TestWaitUntilDeleted:
    def test_immediate_not_found(self):
        poll_fn = Mock(
            side_effect=ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": ""}},
                "Get",
            ),
        )
        wait_until_deleted(poll_fn)
        poll_fn.assert_called_once()

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch(
        "bedrock_agentcore._utils.polling.time.time",
        side_effect=[0, 0, 0, 1, 1],
    )
    def test_polls_then_deleted(self, _mock_time, _mock_sleep):
        poll_fn = Mock(
            side_effect=[
                {"status": "DELETING"},
                ClientError(
                    {"Error": {"Code": "ResourceNotFoundException", "Message": ""}},
                    "Get",
                ),
            ],
        )
        wait_until_deleted(poll_fn)
        assert poll_fn.call_count == 2

    def test_raises_on_failed_status(self):
        poll_fn = Mock(
            return_value={
                "status": "DELETE_FAILED",
                "statusReasons": ["stuck"],
            },
        )
        with pytest.raises(RuntimeError, match="DELETE_FAILED"):
            wait_until_deleted(
                poll_fn,
                failed={"DELETE_FAILED"},
            )

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch(
        "bedrock_agentcore._utils.polling.time.time",
        side_effect=[0, 0, 0, 301],
    )
    def test_timeout(self, _mock_time, _mock_sleep):
        poll_fn = Mock(return_value={"status": "DELETING"})
        with pytest.raises(TimeoutError):
            wait_until_deleted(poll_fn)


class TestBuildNamespaceParams:
    """Tests for build_namespace_params utility."""

    def test_namespace_only(self):
        assert build_namespace_params(namespace="/actor/Jane/") == {"namespace": "/actor/Jane/"}

    def test_namespace_path_only(self):
        assert build_namespace_params(namespace_path="/org/team/") == {"namespacePath": "/org/team/"}

    def test_both_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_namespace_params(namespace="/a/", namespace_path="/b/")

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            build_namespace_params()

    def test_wildcard_in_namespace_raises(self):
        with pytest.raises(ValueError, match="[Ww]ildcard"):
            build_namespace_params(namespace="/actor/*/")

    def test_wildcard_in_namespace_path_raises(self):
        with pytest.raises(ValueError, match="[Ww]ildcard"):
            build_namespace_params(namespace_path="/org/*/team/")
