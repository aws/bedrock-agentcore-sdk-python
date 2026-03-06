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


def test_reasoning_content_round_trips_in_openai_converter():
    """reasoningContent blocks should be preserved via storage extension field."""
    msg = SessionMessage(
        message={
            "role": "assistant",
            "content": [
                {"text": "answer"},
                {"reasoningContent": {"reasoningText": {"text": "chain", "signature": "abc"}}},
            ],
        },
        message_id=6,
    )

    payload_json, _ = OpenAIConverseConverter.message_to_payload(msg)[0]
    payload = json.loads(payload_json)
    assert "_strands_reasoning_content" in payload

    events = [{"payload": [{"conversational": {"content": {"text": payload_json}}}]}]
    restored = OpenAIConverseConverter.events_to_messages(events)
    assert len(restored) == 1
    restored_content = restored[0].message["content"]
    assert any("reasoningContent" in block for block in restored_content)


def test_openai_converter_round_trips_from_blob_payload():
    msg = SessionMessage(
        message={
            "role": "assistant",
            "content": [
                {"text": "hello from blob"},
                {"reasoningContent": {"reasoningText": {"text": "trace", "signature": "sig"}}},
            ],
        },
        message_id=7,
    )

    payload_json, role = OpenAIConverseConverter.message_to_payload(msg)[0]
    events = [{"payload": [{"blob": json.dumps((payload_json, role))}]}]

    restored = OpenAIConverseConverter.events_to_messages(events)
    assert len(restored) == 1
    assert restored[0].message["role"] == "assistant"
    assert any(block.get("text") == "hello from blob" for block in restored[0].message["content"])
    assert any("reasoningContent" in block for block in restored[0].message["content"])


def test_openai_converter_returns_empty_for_empty_or_none_text_content():
    empty_msg = SessionMessage(message={"role": "assistant", "content": []}, message_id=8)
    none_msg = SessionMessage(message={"role": "assistant", "content": [{"text": None}]}, message_id=9)

    assert OpenAIConverseConverter.message_to_payload(empty_msg) == []
    assert OpenAIConverseConverter.message_to_payload(none_msg) == []


def test_openai_converter_detects_oversized_payload():
    oversized_payload = ("x" * 100000, "assistant")
    assert OpenAIConverseConverter.exceeds_conversational_limit(oversized_payload) is True
