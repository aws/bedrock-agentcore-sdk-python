"""Automatic converter selection for mixed-model sessions."""

from __future__ import annotations

import logging
from typing import Any, Tuple

from strands.types.session import SessionMessage

from .anthropic import AnthropicConverseConverter
from .bedrock import BedrockConverseConverter
from .gemini import GeminiConverseConverter
from .openai import OpenAIConverseConverter
from .protocol import MemoryConverter, exceeds_conversational_limit

logger = logging.getLogger(__name__)


class AutoConverseConverter:
    """Auto-selects a write converter and can restore mixed payload formats."""

    _write_converter: type[MemoryConverter] = BedrockConverseConverter
    _read_converters: tuple[type[MemoryConverter], ...] = (
        BedrockConverseConverter,
        OpenAIConverseConverter,
        AnthropicConverseConverter,
        GeminiConverseConverter,
    )

    @classmethod
    def set_write_converter(cls, converter: type[MemoryConverter]) -> None:
        cls._write_converter = converter

    @classmethod
    def select_write_converter_for_model(cls, model: Any) -> type[MemoryConverter]:
        """Pick a converter by model class/module name."""
        model_cls_name = model.__class__.__name__.lower()
        model_mod = model.__class__.__module__.lower()
        full = f"{model_mod}.{model_cls_name}"

        if "anthropic" in full:
            return AnthropicConverseConverter
        if "gemini" in full:
            return GeminiConverseConverter
        if "openai" in full:
            return OpenAIConverseConverter
        return BedrockConverseConverter

    @classmethod
    def message_to_payload(cls, session_message: SessionMessage) -> list[Tuple[str, str]]:
        return cls._write_converter.message_to_payload(session_message)

    @classmethod
    def events_to_messages(cls, events: list[dict[str, Any]]) -> list[SessionMessage]:
        """Decode each payload item using the first converter that succeeds.

        This allows sessions to remain readable after switching model providers.
        """
        restored: list[SessionMessage] = []

        # Oldest to newest
        for event in reversed(events):
            payload_items = event.get("payload", [])
            for payload_item in payload_items:
                fake_event = {"payload": [payload_item]}
                parsed = None
                for converter in cls._read_converters:
                    try:
                        candidate = converter.events_to_messages([fake_event])
                    except Exception:
                        continue
                    if candidate:
                        parsed = candidate
                        break
                if parsed:
                    restored.extend(parsed)
                else:
                    logger.debug("Skipping undecodable payload item: %s", payload_item.keys())

        return restored

    @staticmethod
    def exceeds_conversational_limit(message: tuple[str, str]) -> bool:
        return exceeds_conversational_limit(message)

