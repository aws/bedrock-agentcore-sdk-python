"""Tests for GeminiConverseConverter."""

import json

from strands.types.session import SessionMessage

from bedrock_agentcore.memory.integrations.strands.converters import GeminiConverseConverter


def test_gemini_converter_round_trips_tool_result_and_reasoning():
    msg = SessionMessage(
        message={
            "role": "user",
            "content": [
                {"text": "tool output follows"},
                {"toolResult": {"toolUseId": "call_2", "status": "success", "content": [{"text": "42"}]}},
                {"reasoningContent": {"reasoningText": {"text": "thought", "signature": "sig"}}},
            ],
        },
        message_id=2,
    )

    payload_json, role = GeminiConverseConverter.message_to_payload(msg)[0]
    assert role == "user"

    payload = json.loads(payload_json)
    assert payload["role"] == "user"
    assert any("functionResponse" in part for part in payload["parts"])
    assert any("thought" in part for part in payload["parts"])

    events = [{"payload": [{"conversational": {"content": {"text": payload_json}}}]}]
    restored = GeminiConverseConverter.events_to_messages(events)

    assert len(restored) == 1
    restored_content = restored[0].message["content"]
    assert any("toolResult" in block for block in restored_content)
    assert any("reasoningContent" in block for block in restored_content)


def test_gemini_converter_ignores_none_parts_on_restore():
    payload = json.dumps({"role": "model", "parts": [None, {"text": "kept"}]})
    events = [{"payload": [{"conversational": {"content": {"text": payload}}}]}]

    restored = GeminiConverseConverter.events_to_messages(events)

    assert len(restored) == 1
    assert restored[0].message["role"] == "assistant"
    assert restored[0].message["content"] == [{"text": "kept"}]
