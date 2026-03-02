"""Tests for AutoConverseConverter."""

import json

from strands.types.session import SessionMessage

from bedrock_agentcore.memory.integrations.strands.converters import (
    AnthropicConverseConverter,
    AutoConverseConverter,
    GeminiConverseConverter,
    OpenAIConverseConverter,
)


class _OpenAIModel:
    __module__ = "nflx_strands.models.openai"


class _AnthropicModel:
    __module__ = "nflx_strands.models.anthropic"


class _GeminiModel:
    __module__ = "nflx_strands.models.gemini"


def test_auto_converter_selects_provider_specific_writer():
    assert AutoConverseConverter.select_write_converter_for_model(_OpenAIModel()) is OpenAIConverseConverter
    assert AutoConverseConverter.select_write_converter_for_model(_AnthropicModel()) is AnthropicConverseConverter
    assert AutoConverseConverter.select_write_converter_for_model(_GeminiModel()) is GeminiConverseConverter


def test_auto_converter_restores_mixed_event_payloads():
    openai_payload = json.dumps({"role": "tool", "tool_call_id": "c1", "content": "ok"})
    anthropic_payload = json.dumps(
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "trace", "signature": "sig"}]}
    )
    gemini_payload = json.dumps({"role": "model", "parts": [{"thought": {"text": "thought", "signature": "sig2"}}]})

    events = [
        {"payload": [{"conversational": {"content": {"text": openai_payload}}}]},
        {"payload": [{"conversational": {"content": {"text": anthropic_payload}}}]},
        {"payload": [{"conversational": {"content": {"text": gemini_payload}}}]},
    ]

    messages = AutoConverseConverter.events_to_messages(events)
    assert len(messages) == 3
    for message in messages:
        assert message.message["role"] in {"user", "assistant"}

    all_content = [block for message in messages for block in message.message["content"]]
    assert any("toolResult" in block for block in all_content)
    assert sum(1 for block in all_content if "reasoningContent" in block) >= 2


def test_auto_converter_uses_selected_writer_for_payload_shape():
    msg = SessionMessage(message={"role": "assistant", "content": [{"text": "hi"}]}, message_id=1)
    AutoConverseConverter.set_write_converter(OpenAIConverseConverter)
    payload_json, _ = AutoConverseConverter.message_to_payload(msg)[0]
    parsed = json.loads(payload_json)
    assert "role" in parsed and "content" in parsed


def test_auto_converter_skips_malformed_json_and_missing_required_fields():
    events = [
        {"payload": [{"conversational": {"content": {"text": "{bad-json"}}}]},
        {"payload": [{"conversational": {"content": {"text": json.dumps({"role": "assistant"})}}}]},
        {"payload": [{"conversational": {"content": {"text": json.dumps({"role": "assistant", "content": "ok"})}}}]},
    ]

    messages = AutoConverseConverter.events_to_messages(events)

    assert len(messages) == 1
    assert messages[0].message["role"] == "assistant"
    assert messages[0].message["content"] == [{"text": "ok"}]
