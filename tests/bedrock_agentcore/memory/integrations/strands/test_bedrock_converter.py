"""Tests for AgentCoreMemoryConverter."""

import json
from unittest.mock import patch

from strands.types.session import SessionMessage

from bedrock_agentcore.memory.integrations.strands.bedrock_converter import AgentCoreMemoryConverter


class TestAgentCoreMemoryConverter:
    """Test cases for AgentCoreMemoryConverter."""

    def test_message_to_payload(self):
        """Test converting SessionMessage to payload format."""
        message = SessionMessage(
            message_id=1, message={"role": "user", "content": [{"text": "Hello"}]}, created_at="2023-01-01T00:00:00Z"
        )

        result = AgentCoreMemoryConverter.message_to_payload(message)

        assert len(result) == 1
        assert result[0][1] == "user"
        parsed_content = json.loads(result[0][0])
        assert parsed_content["message"]["content"][0]["text"] == "Hello"

    def test_events_to_messages_conversational(self):
        """Test converting conversational events to SessionMessages."""
        session_message = SessionMessage(
            message_id=1, message={"role": "user", "content": [{"text": "Hello"}]}, created_at="2023-01-01T00:00:00Z"
        )

        events = [
            {
                "payload": [
                    {"conversational": {"content": {"text": json.dumps(session_message.to_dict())}, "role": "USER"}}
                ]
            }
        ]

        result = AgentCoreMemoryConverter.events_to_messages(events)

        assert len(result) == 1
        assert result[0].message["role"] == "user"

    def test_events_to_messages_blob_valid(self):
        """Test converting blob events to SessionMessages."""
        session_message = SessionMessage(
            message_id=1, message={"role": "user", "content": [{"text": "Hello"}]}, created_at="2023-01-01T00:00:00Z"
        )

        blob_data = [json.dumps(session_message.to_dict()), "user"]
        events = [{"payload": [{"blob": json.dumps(blob_data)}]}]

        result = AgentCoreMemoryConverter.events_to_messages(events)

        assert len(result) == 1
        assert result[0].message["role"] == "user"

    @patch("bedrock_agentcore.memory.integrations.strands.bedrock_converter.logger")
    def test_events_to_messages_blob_invalid_json(self, mock_logger):
        """Test handling invalid JSON in blob events."""
        events = [{"payload": [{"blob": "invalid json"}]}]

        result = AgentCoreMemoryConverter.events_to_messages(events)

        assert len(result) == 0
        mock_logger.error.assert_called()

    @patch("bedrock_agentcore.memory.integrations.strands.bedrock_converter.logger")
    def test_events_to_messages_blob_invalid_session_message(self, mock_logger):
        """Test handling invalid SessionMessage in blob events."""
        blob_data = ["invalid", "user"]
        events = [{"payload": [{"blob": json.dumps(blob_data)}]}]

        result = AgentCoreMemoryConverter.events_to_messages(events)

        assert len(result) == 0
        mock_logger.error.assert_called()

    def test_total_length(self):
        """Test calculating total length of message tuple."""
        message = ("hello", "world")
        result = AgentCoreMemoryConverter.total_length(message)
        assert result == 10

    def test_exceeds_conversational_limit_false(self):
        """Test message under conversational limit."""
        message = ("short", "message")
        result = AgentCoreMemoryConverter.exceeds_conversational_limit(message)
        assert result is False

    def test_exceeds_conversational_limit_true(self):
        """Test message over conversational limit."""
        long_text = "x" * 5000
        message = (long_text, long_text)
        result = AgentCoreMemoryConverter.exceeds_conversational_limit(message)
        assert result is True

    def test_filter_empty_text_removes_empty_string(self):
        """Test filtering removes empty text items."""
        message = {"role": "user", "content": [{"text": ""}, {"text": "hello"}]}
        result = AgentCoreMemoryConverter._filter_empty_text(message)
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "hello"

    def test_filter_empty_text_removes_whitespace_only(self):
        """Test filtering removes whitespace-only text items."""
        message = {"role": "user", "content": [{"text": "   "}, {"text": "hello"}]}
        result = AgentCoreMemoryConverter._filter_empty_text(message)
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "hello"

    def test_filter_empty_text_keeps_non_text_items(self):
        """Test filtering keeps non-text items like toolUse."""
        message = {"role": "user", "content": [{"text": ""}, {"toolUse": {"name": "test"}}]}
        result = AgentCoreMemoryConverter._filter_empty_text(message)
        assert len(result["content"]) == 1
        assert "toolUse" in result["content"][0]

    def test_filter_empty_text_all_empty_returns_empty_content(self):
        """Test filtering all empty text returns empty content array."""
        message = {"role": "user", "content": [{"text": ""}]}
        result = AgentCoreMemoryConverter._filter_empty_text(message)
        assert result["content"] == []

    def test_message_to_payload_skips_all_empty_text(self):
        """Test message_to_payload returns empty list when all text is empty."""
        message = SessionMessage(
            message_id=1, message={"role": "user", "content": [{"text": ""}]}, created_at="2023-01-01T00:00:00Z"
        )
        result = AgentCoreMemoryConverter.message_to_payload(message)
        assert result == []

    def test_message_to_payload_filters_empty_text_items(self):
        """Test message_to_payload filters out empty text but keeps valid content."""
        message = SessionMessage(
            message_id=1,
            message={"role": "user", "content": [{"text": ""}, {"text": "hello"}]},
            created_at="2023-01-01T00:00:00Z",
        )
        result = AgentCoreMemoryConverter.message_to_payload(message)
        assert len(result) == 1
        parsed = json.loads(result[0][0])
        assert len(parsed["message"]["content"]) == 1
        assert parsed["message"]["content"][0]["text"] == "hello"

    def test_events_to_messages_filters_empty_text_conversational(self):
        """Test events_to_messages filters empty text from conversational payloads."""
        msg_with_empty = SessionMessage(
            message_id=1,
            message={"role": "user", "content": [{"text": ""}, {"text": "hello"}]},
            created_at="2023-01-01T00:00:00Z",
        )
        events = [
            {
                "payload": [
                    {"conversational": {"content": {"text": json.dumps(msg_with_empty.to_dict())}, "role": "USER"}}
                ]
            }
        ]
        result = AgentCoreMemoryConverter.events_to_messages(events)
        assert len(result) == 1
        assert len(result[0].message["content"]) == 1
        assert result[0].message["content"][0]["text"] == "hello"

    def test_events_to_messages_drops_all_empty_conversational(self):
        """Test events_to_messages drops messages with only empty text."""
        empty_msg = SessionMessage(
            message_id=1, message={"role": "user", "content": [{"text": ""}]}, created_at="2023-01-01T00:00:00Z"
        )
        events = [
            {"payload": [{"conversational": {"content": {"text": json.dumps(empty_msg.to_dict())}, "role": "USER"}}]}
        ]
        result = AgentCoreMemoryConverter.events_to_messages(events)
        assert len(result) == 0

    def test_events_to_messages_filters_empty_text_blob(self):
        """Test events_to_messages filters empty text from blob payloads."""
        msg_with_empty = SessionMessage(
            message_id=1,
            message={"role": "user", "content": [{"text": ""}, {"text": "hello"}]},
            created_at="2023-01-01T00:00:00Z",
        )
        events = [{"payload": [{"blob": json.dumps([json.dumps(msg_with_empty.to_dict()), "user"])}]}]
        result = AgentCoreMemoryConverter.events_to_messages(events)
        assert len(result) == 1
        assert len(result[0].message["content"]) == 1
        assert result[0].message["content"][0]["text"] == "hello"

    def test_message_to_payload_with_bytes_encodes_before_filtering(self):
        """Test message_to_payload encodes bytes to base64 before filtering empty text.

        This test verifies the fix for issue #198 where json.dumps() failed with
        'Object of type bytes is not JSON serializable' when messages contained
        image data with raw bytes. The fix ensures to_dict() (which encodes bytes
        to base64) is called before _filter_empty_text.
        """
        message = SessionMessage(
            message_id=1,
            message={
                "role": "user",
                "content": [
                    {"text": ""},  # Empty text that will be filtered out
                    {"image": {"source": {"bytes": b"fake image data"}}},
                ],
            },
            created_at="2023-01-01T00:00:00Z",
        )

        # This should not raise "Object of type bytes is not JSON serializable"
        result = AgentCoreMemoryConverter.message_to_payload(message)

        assert len(result) == 1
        # Verify json.dumps succeeded and bytes were encoded
        parsed = json.loads(result[0][0])
        assert len(parsed["message"]["content"]) == 1
        assert "image" in parsed["message"]["content"][0]
        # Verify bytes were encoded (strands uses __bytes_encoded__ format)
        encoded_bytes = parsed["message"]["content"][0]["image"]["source"]["bytes"]
        assert isinstance(encoded_bytes, dict)
        assert encoded_bytes.get("__bytes_encoded__") is True
        assert "data" in encoded_bytes
