"""Tests for shared _utils: pagination and polling."""

from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore._utils.config import ListConfig
from bedrock_agentcore._utils.pagination import list_all
from bedrock_agentcore._utils.polling import wait_until, wait_until_deleted


class TestListAll:
    def test_single_page(self):
        client = Mock()
        client.list_things.return_value = {"items": [{"id": "1"}, {"id": "2"}]}
        result = list_all(client, "list_things", "items")
        assert len(result) == 2

    def test_pagination(self):
        client = Mock()
        client.list_things.side_effect = [
            {"items": [{"id": "1"}], "nextToken": "tok1"},
            {"items": [{"id": "2"}]},
        ]
        result = list_all(client, "list_things", "items")
        assert len(result) == 2
        assert client.list_things.call_count == 2

    def test_respects_total_items(self):
        client = Mock()
        client.list_things.return_value = {
            "items": [{"id": str(i)} for i in range(50)],
        }
        result = list_all(
            client,
            "list_things",
            "items",
            ListConfig(total_items=10),
        )
        assert len(result) == 10

    def test_strips_incoming_next_token(self):
        client = Mock()
        client.list_things.return_value = {"items": [{"id": "1"}]}
        list_all(client, "list_things", "items", nextToken="stale")
        call_kwargs = client.list_things.call_args[1]
        assert "nextToken" not in call_kwargs

    def test_forwards_kwargs(self):
        client = Mock()
        client.list_things.return_value = {"items": []}
        list_all(client, "list_things", "items", someFilter="value")
        client.list_things.assert_called_once_with(someFilter="value")


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
