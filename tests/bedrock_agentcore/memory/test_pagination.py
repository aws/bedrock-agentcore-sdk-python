"""Unit tests for pagination utilities."""

from unittest.mock import MagicMock

import pytest

from bedrock_agentcore.memory.pagination import paginate_turns


class TestPaginateTurns:
    """Tests for paginate_turns function."""

    def test_basic_pagination_single_page(self):
        """Test basic pagination with all turns in single page."""
        mock_list_events = MagicMock(return_value={
            "events": [
                {"payload": [{"conversational": {"role": "USER", "text": "Hello"}}]},
                {"payload": [{"conversational": {"role": "ASSISTANT", "text": "Hi"}}]},
                {"payload": [{"conversational": {"role": "USER", "text": "How are you?"}}]},
                {"payload": [{"conversational": {"role": "ASSISTANT", "text": "Good"}}]},
            ]
        })

        result = paginate_turns(
            list_events_fn=mock_list_events,
            base_params={"memoryId": "mem-1", "actorId": "actor-1", "sessionId": "sess-1"},
            k=2,
            max_results=None,
            user_role_value="USER",
            wrap_message=lambda x: x,
        )

        assert len(result) == 2
        mock_list_events.assert_called_once()

    def test_pagination_continues_until_k_turns(self):
        """Test that pagination continues across pages until k turns found."""
        mock_list_events = MagicMock(side_effect=[
            {
                "events": [
                    {"payload": [{"conversational": {"role": "USER", "text": "msg1"}}]},
                    {"payload": [{"conversational": {"role": "ASSISTANT", "text": "resp1"}}]},
                ],
                "nextToken": "token1"
            },
            {
                "events": [
                    {"payload": [{"conversational": {"role": "USER", "text": "msg2"}}]},
                    {"payload": [{"conversational": {"role": "ASSISTANT", "text": "resp2"}}]},
                ],
                "nextToken": "token2"
            },
            {
                "events": [
                    {"payload": [{"conversational": {"role": "USER", "text": "msg3"}}]},
                    {"payload": [{"conversational": {"role": "ASSISTANT", "text": "resp3"}}]},
                ]
            },
        ])

        result = paginate_turns(
            list_events_fn=mock_list_events,
            base_params={"memoryId": "mem-1", "actorId": "actor-1", "sessionId": "sess-1"},
            k=3,
            max_results=None,
            user_role_value="USER",
            wrap_message=lambda x: x,
        )

        assert len(result) == 3
        assert mock_list_events.call_count == 3

    def test_max_results_limits_fetching(self):
        """Test that max_results limits total events fetched."""
        mock_list_events = MagicMock(return_value={
            "events": [
                {"payload": [{"conversational": {"role": "USER", "text": "msg"}}]},
            ] * 50,
            "nextToken": "token"
        })

        result = paginate_turns(
            list_events_fn=mock_list_events,
            base_params={"memoryId": "mem-1", "actorId": "actor-1", "sessionId": "sess-1"},
            k=100,
            max_results=50,
            user_role_value="USER",
            wrap_message=lambda x: x,
        )

        # Should stop after fetching max_results events
        mock_list_events.assert_called_once()
        call_kwargs = mock_list_events.call_args[1]
        assert call_kwargs["maxResults"] == 50

    def test_empty_events_stops_pagination(self):
        """Test that empty events response stops pagination."""
        mock_list_events = MagicMock(return_value={"events": []})

        result = paginate_turns(
            list_events_fn=mock_list_events,
            base_params={"memoryId": "mem-1", "actorId": "actor-1", "sessionId": "sess-1"},
            k=5,
            max_results=None,
            user_role_value="USER",
            wrap_message=lambda x: x,
        )

        assert result == []
        mock_list_events.assert_called_once()

    def test_wrap_message_applied(self):
        """Test that wrap_message function is applied to each message."""
        mock_list_events = MagicMock(return_value={
            "events": [
                {"payload": [{"conversational": {"role": "USER", "text": "Hello"}}]},
                {"payload": [{"conversational": {"role": "ASSISTANT", "text": "Hi"}}]},
            ]
        })

        class Wrapper:
            def __init__(self, data):
                self.data = data

        result = paginate_turns(
            list_events_fn=mock_list_events,
            base_params={"memoryId": "mem-1", "actorId": "actor-1", "sessionId": "sess-1"},
            k=1,
            max_results=None,
            user_role_value="USER",
            wrap_message=Wrapper,
        )

        assert len(result) == 1
        assert all(isinstance(msg, Wrapper) for msg in result[0])

    def test_no_next_token_stops_pagination(self):
        """Test that missing nextToken stops pagination."""
        mock_list_events = MagicMock(return_value={
            "events": [
                {"payload": [{"conversational": {"role": "USER", "text": "msg"}}]},
            ]
        })

        result = paginate_turns(
            list_events_fn=mock_list_events,
            base_params={"memoryId": "mem-1", "actorId": "actor-1", "sessionId": "sess-1"},
            k=10,
            max_results=None,
            user_role_value="USER",
            wrap_message=lambda x: x,
        )

        # Should stop after first page since no nextToken
        mock_list_events.assert_called_once()

    def test_last_turn_included(self):
        """Test that incomplete last turn is included."""
        mock_list_events = MagicMock(return_value={
            "events": [
                {"payload": [{"conversational": {"role": "USER", "text": "Hello"}}]},
            ]
        })

        result = paginate_turns(
            list_events_fn=mock_list_events,
            base_params={"memoryId": "mem-1", "actorId": "actor-1", "sessionId": "sess-1"},
            k=5,
            max_results=None,
            user_role_value="USER",
            wrap_message=lambda x: x,
        )

        assert len(result) == 1
        assert result[0][0]["text"] == "Hello"

    def test_skips_non_conversational_payload(self):
        """Test that non-conversational payloads are skipped."""
        mock_list_events = MagicMock(return_value={
            "events": [
                {"payload": [{"blob": "some data"}]},
                {"payload": [{"conversational": {"role": "USER", "text": "Hello"}}]},
            ]
        })

        result = paginate_turns(
            list_events_fn=mock_list_events,
            base_params={"memoryId": "mem-1", "actorId": "actor-1", "sessionId": "sess-1"},
            k=5,
            max_results=None,
            user_role_value="USER",
            wrap_message=lambda x: x,
        )

        assert len(result) == 1
        assert len(result[0]) == 1
