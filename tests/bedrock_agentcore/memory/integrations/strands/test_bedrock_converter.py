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
