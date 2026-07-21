"""Internal Strands-message formatting helpers."""

from typing import Literal

from strands.types.content import Message

AgentCoreRole = Literal["USER", "ASSISTANT"]

# ECMAScript WhiteSpace + LineTerminator code points used by String.prototype.trim.
_ECMASCRIPT_TRIM_CHARACTERS = (
    "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def _ecmascript_trim(value: str) -> str:
    """Trim exactly the code points removed by ECMAScript ``String.trim()``."""
    return value.strip(_ECMASCRIPT_TRIM_CHARACTERS)


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
    parts = []
    for block in message["content"]:
        if "text" not in block:
            continue
        text = _ecmascript_trim(block["text"])
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
