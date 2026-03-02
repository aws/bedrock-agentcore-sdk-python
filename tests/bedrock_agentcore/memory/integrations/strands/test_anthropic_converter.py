"""Tests for AnthropicConverseConverter."""

import json

from strands.types.session import SessionMessage

from bedrock_agentcore.memory.integrations.strands.converters import AnthropicConverseConverter


def test_anthropic_converter_round_trips_tool_and_reasoning_blocks():
    msg = SessionMessage(
        message={
            "role": "assistant",
            "content": [
                {"text": "thinking and calling tool"},
                {"reasoningContent": {"reasoningText": {"text": "private chain", "signature": "sig"}}},
                {"toolUse": {"toolUseId": "call_1", "name": "search", "input": {"q": "x"}}},
            ],
        },
        message_id=1,
    )

    payload_json, role = AnthropicConverseConverter.message_to_payload(msg)[0]
    assert role == "assistant"

    payload = json.loads(payload_json)
    assert payload["role"] == "assistant"
    assert any(block.get("type") == "thinking" for block in payload["content"])
    assert any(block.get("type") == "tool_use" for block in payload["content"])

    events = [{"payload": [{"conversational": {"content": {"text": payload_json}}}]}]
    restored = AnthropicConverseConverter.events_to_messages(events)

    assert len(restored) == 1
    restored_content = restored[0].message["content"]
    assert any("reasoningContent" in block for block in restored_content)
    assert any("toolUse" in block for block in restored_content)
