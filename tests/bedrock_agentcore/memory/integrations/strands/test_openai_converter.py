"""Tests for OpenAIConverseConverter."""

import json

from strands.types.session import SessionMessage

from bedrock_agentcore.memory.integrations.strands.converters import OpenAIConverseConverter


def test_tool_result_message_serializes_as_openai_tool_role():
    """toolResult messages should be saved with role='tool' for STM."""
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
    assert role == "tool"

    payload = json.loads(payload_json)
    assert payload["role"] == "tool"
    assert payload["tool_call_id"] == "call_123"
    assert payload["content"] == "72°F and sunny"


def test_openai_tool_role_deserializes_to_strands_user_tool_result_message():
    """OpenAI role='tool' should restore as Strands toolResult content on user role."""
    events = [
        {
            "payload": [
                {
                    "conversational": {
                        "content": {
                            "text": json.dumps(
                                {
                                    "role": "tool",
                                    "tool_call_id": "call_321",
                                    "content": "done",
                                    "status": "success",
                                }
                            )
                        }
                    }
                }
            ]
        }
    ]

    messages = OpenAIConverseConverter.events_to_messages(events)
    assert len(messages) == 1
    assert messages[0].message["role"] == "user"
    assert messages[0].message["content"][0]["toolResult"]["toolUseId"] == "call_321"
