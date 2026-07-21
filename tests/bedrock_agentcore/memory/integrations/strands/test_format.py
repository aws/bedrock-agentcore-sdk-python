"""Tests for AgentCore event message formatting."""

import pytest
from strands.types.content import Message

from bedrock_agentcore.memory.integrations.strands._format import (
    extract_text,
    is_user_or_assistant_with_text,
    map_role,
)


def message(role: str, content: list[dict[str, object]]) -> Message:
    """Build a minimally typed Strands message."""
    return {"role": role, "content": content}  # type: ignore[typeddict-item]


@pytest.mark.parametrize(("role", "expected"), [("user", "USER"), ("assistant", "ASSISTANT")])
def test_map_role(role: str, expected: str) -> None:
    """Map the two Strands conversation roles."""
    assert map_role(message(role, [])) == expected


def test_extract_text_concatenates_blocks_and_ignores_non_text() -> None:
    """Trim and join only non-empty text blocks."""
    actual = extract_text(
        message(
            "user",
            [
                {"text": " hello "},
                {"toolUse": {"toolUseId": "t1", "name": "noop", "input": {}}},
                {"text": "   "},
                {"text": "world"},
            ],
        )
    )
    assert actual == "hello\nworld"


def test_extract_text_matches_ecmascript_trim_boundaries() -> None:
    """Retain Python-only U+0085 whitespace and remove ECMAScript BOM."""
    actual = extract_text(
        message(
            "user",
            [
                {"text": " \u0085 "},
                {"text": "\ufeff"},
            ],
        )
    )
    assert actual == "\u0085"


def test_extract_text_returns_empty_for_tool_only_message() -> None:
    """Tool-only messages have no AgentCore conversational text."""
    assert extract_text(message("assistant", [{"toolUse": {}}])) == ""


@pytest.mark.parametrize(
    ("role", "content", "expected"),
    [
        ("user", [{"text": "hi"}], True),
        ("assistant", [{"text": "hi"}], True),
        ("user", [{"toolUse": {}}], False),
        ("assistant", [{"text": "   "}], False),
    ],
)
def test_is_user_or_assistant_with_text(role: str, content: list[dict[str, object]], expected: bool) -> None:
    """Accept only supported roles with non-blank text."""
    assert is_user_or_assistant_with_text(message(role, content)) is expected
