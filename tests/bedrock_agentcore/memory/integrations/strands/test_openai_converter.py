"""Tests for OpenAIConverseConverter."""

import json
from unittest.mock import patch

import pytest
from strands.types.session import SessionMessage

from bedrock_agentcore.memory.integrations.strands.converters import OpenAIConverseConverter


class TestOpenAIConverseConverterMessageToPayload:
    """Test converting Strands SessionMessages (Bedrock Converse) to OpenAI-format STM payloads."""

    def test_user_text_message(self):
        """Convert a simple user text message to OpenAI format payload."""
        msg = SessionMessage(
            message={"role": "user", "content": [{"text": "Hello"}]},
            message_id=1,
        )
        result = OpenAIConverseConverter.message_to_payload(msg)

        assert len(result) == 1
        payload_json, role = result[0]
        assert role == "user"

        payload = json.loads(payload_json)
        assert payload["role"] == "user"
        assert payload["content"] == "Hello"

    def test_assistant_text_message(self):
        """Convert a simple assistant text message to OpenAI format payload."""
        msg = SessionMessage(
            message={"role": "assistant", "content": [{"text": "Hi there"}]},
            message_id=2,
        )
        result = OpenAIConverseConverter.message_to_payload(msg)

        assert len(result) == 1
        payload_json, role = result[0]
        assert role == "assistant"

        payload = json.loads(payload_json)
        assert payload["role"] == "assistant"
        assert payload["content"] == "Hi there"

    def test_assistant_tool_use_message(self):
        """Convert an assistant message with toolUse to OpenAI tool_calls format."""
        msg = SessionMessage(
            message={
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "call_123",
                            "name": "get_weather",
                            "input": {"city": "Seattle"},
                        }
                    }
                ],
            },
            message_id=3,
        )
        result = OpenAIConverseConverter.message_to_payload(msg)

        assert len(result) == 1
        payload_json, role = result[0]
        assert role == "assistant"

        payload = json.loads(payload_json)
        assert payload["role"] == "assistant"
        assert payload.get("content") is None
        assert len(payload["tool_calls"]) == 1

        tc = payload["tool_calls"][0]
        assert tc["id"] == "call_123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Seattle"}

    def test_assistant_text_and_tool_use(self):
        """Convert an assistant message with both text and toolUse."""
        msg = SessionMessage(
            message={
                "role": "assistant",
                "content": [
                    {"text": "Let me check that for you."},
                    {
                        "toolUse": {
                            "toolUseId": "call_456",
                            "name": "search",
                            "input": {"q": "test"},
                        }
                    },
                ],
            },
            message_id=4,
        )
        result = OpenAIConverseConverter.message_to_payload(msg)

        assert len(result) == 1
        payload_json, role = result[0]
        payload = json.loads(payload_json)

        assert payload["role"] == "assistant"
        assert payload["content"] == "Let me check that for you."
        assert len(payload["tool_calls"]) == 1
        assert payload["tool_calls"][0]["function"]["name"] == "search"

    def test_tool_result_message(self):
        """Convert a toolResult message to OpenAI tool response format."""
        msg = SessionMessage(
            message={
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_123",
                            "content": [{"text": "72°F and sunny"}],
                            "status": "success",
                        }
                    }
                ],
            },
            message_id=5,
        )
        result = OpenAIConverseConverter.message_to_payload(msg)

        assert len(result) == 1
        payload_json, role = result[0]
        payload = json.loads(payload_json)

        assert payload["role"] == "tool"
        assert payload["tool_call_id"] == "call_123"
        assert payload["content"] == "72°F and sunny"

    def test_empty_content_returns_empty(self):
        """A message with empty content list returns empty payload."""
        msg = SessionMessage(
            message={"role": "user", "content": []},
            message_id=6,
        )
        result = OpenAIConverseConverter.message_to_payload(msg)
        assert result == []

    def test_empty_text_filtered(self):
        """A message with only empty/whitespace text returns empty payload."""
        msg = SessionMessage(
            message={"role": "user", "content": [{"text": "   "}]},
            message_id=7,
        )
        result = OpenAIConverseConverter.message_to_payload(msg)
        assert result == []

    def test_multiple_tool_calls(self):
        """Convert an assistant message with multiple tool calls."""
        msg = SessionMessage(
            message={
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "c1", "name": "fn_a", "input": {}}},
                    {"toolUse": {"toolUseId": "c2", "name": "fn_b", "input": {"x": 1}}},
                ],
            },
            message_id=8,
        )
        result = OpenAIConverseConverter.message_to_payload(msg)

        payload = json.loads(result[0][0])
        assert len(payload["tool_calls"]) == 2
        assert payload["tool_calls"][0]["id"] == "c1"
        assert payload["tool_calls"][1]["id"] == "c2"


class TestOpenAIConverseConverterEventsToMessages:
    """Test converting STM events (OpenAI format) to Strands SessionMessages (Bedrock Converse)."""

    def _make_conversational_event(self, openai_msg: dict, role: str = "USER") -> dict:
        """Helper to create an STM event with a conversational payload."""
        return {
            "eventId": "event-1",
            "payload": [
                {
                    "conversational": {
                        "content": {"text": json.dumps(openai_msg)},
                        "role": role,
                    }
                }
            ],
        }

    def _make_blob_event(self, openai_msg: dict, role: str = "user") -> dict:
        """Helper to create an STM event with a blob payload."""
        return {
            "eventId": "event-1",
            "payload": [
                {"blob": json.dumps((json.dumps(openai_msg), role))}
            ],
        }

    def test_user_text_event(self):
        """Convert an OpenAI user text event to Bedrock Converse SessionMessage."""
        events = [self._make_conversational_event({"role": "user", "content": "Hello"}, "USER")]

        messages = OpenAIConverseConverter.events_to_messages(events)

        assert len(messages) == 1
        msg = messages[0].message
        assert msg["role"] == "user"
        assert msg["content"] == [{"text": "Hello"}]

    def test_assistant_text_event(self):
        """Convert an OpenAI assistant text event to Bedrock Converse SessionMessage."""
        events = [self._make_conversational_event({"role": "assistant", "content": "Hi"}, "ASSISTANT")]

        messages = OpenAIConverseConverter.events_to_messages(events)

        assert len(messages) == 1
        msg = messages[0].message
        assert msg["role"] == "assistant"
        assert msg["content"] == [{"text": "Hi"}]

    def test_assistant_tool_calls_event(self):
        """Convert OpenAI assistant tool_calls event to Bedrock Converse toolUse."""
        openai_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "NYC"}',
                    },
                }
            ],
        }
        events = [self._make_conversational_event(openai_msg, "ASSISTANT")]

        messages = OpenAIConverseConverter.events_to_messages(events)

        assert len(messages) == 1
        msg = messages[0].message
        assert msg["role"] == "assistant"
        content = msg["content"]
        assert len(content) == 1
        assert content[0]["toolUse"]["toolUseId"] == "call_abc"
        assert content[0]["toolUse"]["name"] == "get_weather"
        assert content[0]["toolUse"]["input"] == {"city": "NYC"}

    def test_assistant_text_and_tool_calls_event(self):
        """Convert OpenAI assistant with both text and tool_calls."""
        openai_msg = {
            "role": "assistant",
            "content": "Let me look that up.",
            "tool_calls": [
                {
                    "id": "call_xyz",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "test"}'},
                }
            ],
        }
        events = [self._make_conversational_event(openai_msg, "ASSISTANT")]

        messages = OpenAIConverseConverter.events_to_messages(events)

        msg = messages[0].message
        assert msg["role"] == "assistant"
        # Should have text + toolUse
        assert msg["content"][0] == {"text": "Let me look that up."}
        assert msg["content"][1]["toolUse"]["name"] == "search"

    def test_tool_response_event(self):
        """Convert OpenAI tool response to Bedrock Converse toolResult."""
        openai_msg = {
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": "72°F and sunny",
        }
        events = [self._make_conversational_event(openai_msg, "USER")]

        messages = OpenAIConverseConverter.events_to_messages(events)

        assert len(messages) == 1
        msg = messages[0].message
        assert msg["role"] == "user"
        tool_result = msg["content"][0]["toolResult"]
        assert tool_result["toolUseId"] == "call_abc"
        assert tool_result["content"] == [{"text": "72°F and sunny"}]

    def test_system_message_event(self):
        """Convert OpenAI system message to user message in Bedrock Converse."""
        openai_msg = {"role": "system", "content": "You are helpful."}
        events = [self._make_conversational_event(openai_msg, "USER")]

        messages = OpenAIConverseConverter.events_to_messages(events)

        assert len(messages) == 1
        msg = messages[0].message
        assert msg["role"] == "user"
        assert msg["content"] == [{"text": "You are helpful."}]

    def test_multiple_events_in_order(self):
        """Multiple events are returned in correct order (reversed from STM)."""
        events = [
            self._make_conversational_event({"role": "user", "content": "First"}, "USER"),
            self._make_conversational_event({"role": "assistant", "content": "Second"}, "ASSISTANT"),
        ]

        messages = OpenAIConverseConverter.events_to_messages(events)

        assert len(messages) == 2
        # STM returns newest first; converter reverses
        assert messages[0].message["content"][0]["text"] == "Second"
        assert messages[1].message["content"][0]["text"] == "First"

    def test_empty_events(self):
        """Empty event list returns empty message list."""
        assert OpenAIConverseConverter.events_to_messages([]) == []

    def test_event_with_empty_payload(self):
        """Event with no payload is skipped."""
        events = [{"eventId": "event-1"}]
        assert OpenAIConverseConverter.events_to_messages(events) == []

    def test_blob_event(self):
        """Convert a blob-format STM event with OpenAI data."""
        openai_msg = {"role": "user", "content": "From blob"}
        events = [self._make_blob_event(openai_msg)]

        messages = OpenAIConverseConverter.events_to_messages(events)

        assert len(messages) == 1
        assert messages[0].message["role"] == "user"
        assert messages[0].message["content"] == [{"text": "From blob"}]

    def test_multiple_tool_calls_event(self):
        """Convert OpenAI message with multiple tool calls."""
        openai_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "fn_a", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "fn_b", "arguments": '{"x":1}'}},
            ],
        }
        events = [self._make_conversational_event(openai_msg, "ASSISTANT")]

        messages = OpenAIConverseConverter.events_to_messages(events)

        content = messages[0].message["content"]
        assert len(content) == 2
        assert content[0]["toolUse"]["toolUseId"] == "c1"
        assert content[1]["toolUse"]["toolUseId"] == "c2"

    @patch("bedrock_agentcore.memory.integrations.strands.converters.openai.logger")
    def test_blob_invalid_json(self, mock_logger):
        """Invalid JSON in blob payload is handled gracefully."""
        events = [{"eventId": "e1", "payload": [{"blob": "not valid json"}]}]

        messages = OpenAIConverseConverter.events_to_messages(events)

        assert messages == []
        mock_logger.error.assert_called_once()

    @patch("bedrock_agentcore.memory.integrations.strands.converters.openai.logger")
    def test_conversational_invalid_json(self, mock_logger):
        """Invalid JSON in conversational payload is handled gracefully."""
        events = [
            {
                "eventId": "e1",
                "payload": [
                    {"conversational": {"content": {"text": "not valid json"}, "role": "USER"}}
                ],
            }
        ]

        messages = OpenAIConverseConverter.events_to_messages(events)

        assert messages == []
        mock_logger.error.assert_called_once()

    def test_tool_calls_with_malformed_arguments(self):
        """Tool calls with non-JSON arguments default to empty dict."""
        openai_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {"name": "broken_fn", "arguments": "not json"},
                }
            ],
        }
        events = [self._make_conversational_event(openai_msg, "ASSISTANT")]

        messages = OpenAIConverseConverter.events_to_messages(events)

        assert len(messages) == 1
        tool_use = messages[0].message["content"][0]["toolUse"]
        assert tool_use["name"] == "broken_fn"
        assert tool_use["input"] == {}

    def test_assistant_null_content_no_tool_calls(self):
        """Assistant message with null content and no tool_calls produces empty content."""
        openai_msg = {"role": "assistant", "content": None}
        events = [self._make_conversational_event(openai_msg, "ASSISTANT")]

        messages = OpenAIConverseConverter.events_to_messages(events)

        # No content items → filtered out
        assert messages == []


class TestOpenAIConverseConverterExceedsLimit:
    """Test conversational size limit check."""

    def test_small_message_does_not_exceed(self):
        result = OpenAIConverseConverter.exceeds_conversational_limit(("short", "user"))
        assert result is False

    def test_large_message_exceeds(self):
        big = "x" * 9000
        result = OpenAIConverseConverter.exceeds_conversational_limit((big, "user"))
        assert result is True


class TestOpenAIConverseConverterRoundTrip:
    """Test that converting Bedrock→OpenAI→Bedrock preserves message semantics."""

    def test_roundtrip_user_text(self):
        """User text survives Bedrock→OpenAI→Bedrock round trip."""
        original = SessionMessage(
            message={"role": "user", "content": [{"text": "Hello world"}]},
            message_id=1,
        )
        payload = OpenAIConverseConverter.message_to_payload(original)
        openai_json = payload[0][0]

        # Simulate STM storage and retrieval
        event = {
            "eventId": "e1",
            "payload": [{"conversational": {"content": {"text": openai_json}, "role": "USER"}}],
        }
        restored = OpenAIConverseConverter.events_to_messages([event])

        assert len(restored) == 1
        assert restored[0].message["role"] == "user"
        assert restored[0].message["content"] == [{"text": "Hello world"}]

    def test_roundtrip_tool_use(self):
        """Tool use survives Bedrock→OpenAI→Bedrock round trip."""
        original = SessionMessage(
            message={
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "call_1", "name": "calc", "input": {"expr": "2+2"}}},
                ],
            },
            message_id=2,
        )
        payload = OpenAIConverseConverter.message_to_payload(original)
        openai_json = payload[0][0]

        event = {
            "eventId": "e2",
            "payload": [{"conversational": {"content": {"text": openai_json}, "role": "ASSISTANT"}}],
        }
        restored = OpenAIConverseConverter.events_to_messages([event])

        assert len(restored) == 1
        tu = restored[0].message["content"][0]["toolUse"]
        assert tu["toolUseId"] == "call_1"
        assert tu["name"] == "calc"
        assert tu["input"] == {"expr": "2+2"}

    def test_roundtrip_tool_result(self):
        """Tool result survives Bedrock→OpenAI→Bedrock round trip."""
        original = SessionMessage(
            message={
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call_1", "content": [{"text": "4"}], "status": "success"}},
                ],
            },
            message_id=3,
        )
        payload = OpenAIConverseConverter.message_to_payload(original)
        openai_json = payload[0][0]

        event = {
            "eventId": "e3",
            "payload": [{"conversational": {"content": {"text": openai_json}, "role": "USER"}}],
        }
        restored = OpenAIConverseConverter.events_to_messages([event])

        assert len(restored) == 1
        tr = restored[0].message["content"][0]["toolResult"]
        assert tr["toolUseId"] == "call_1"
        assert tr["content"] == [{"text": "4"}]
