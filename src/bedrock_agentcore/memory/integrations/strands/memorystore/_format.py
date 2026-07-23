"""Internal Strands-message formatting helpers."""

from typing import Literal

from strands.types.content import Message

AgentCoreRole = Literal["USER", "ASSISTANT"]


def map_role(message: Message) -> AgentCoreRole:
    """Map a Strands role to the AgentCore conversational-role subset.

    Args:
        message: Strands message to map.

    Returns:
        ``USER`` for a user message, otherwise ``ASSISTANT``.
    """
    return "USER" if message["role"] == "user" else "ASSISTANT"


def extract_text(message: Message) -> str:
    """Join non-empty text blocks and ignore all other block kinds.

    Args:
        message: Strands message whose text should be extracted.

    Returns:
        Trimmed blocks joined by newlines.
    """
    # Drop blank blocks before joining so an empty middle block does not
    # leave a stray blank line in the concatenated event text.
    parts = []
    for block in message["content"]:
        if "text" not in block:
            continue
        text = block["text"].strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def is_user_or_assistant_with_text(message: Message) -> bool:
    """Return whether a user/assistant message contains extractable text.

    Args:
        message: Strands message to inspect.

    Returns:
        ``True`` only for a supported role with non-blank text.
    """
    return message["role"] in ("user", "assistant") and bool(extract_text(message))
