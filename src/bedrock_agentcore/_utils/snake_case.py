"""Utilities for wrapping boto3 methods to accept snake_case kwargs."""

import functools
import re
from typing import Any, Callable, Dict

_VALID_SNAKE_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


def snake_to_camel(name: str) -> str:
    """Convert a snake_case string to camelCase.

    Already-camelCase strings pass through unchanged (no underscores to split on).
    Raises ValueError for malformed snake_case (e.g. leading/trailing underscores,
    consecutive underscores, uppercase characters).
    """
    if "_" not in name:
        return name
    if not _VALID_SNAKE_RE.match(name):
        raise ValueError(f"Invalid parameter name: '{name}'")
    parts = name.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def accept_snake_case_kwargs(method: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a boto3 method to accept both snake_case and camelCase kwargs.

    Converts all snake_case kwargs to camelCase before forwarding.
    Raises TypeError if both forms are provided (e.g. memory_id and memoryId).
    """

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        converted: Dict[str, Any] = {}
        original_keys: Dict[str, str] = {}
        for key, value in kwargs.items():
            camel_key = snake_to_camel(key)
            if camel_key in converted:
                raise TypeError(
                    f"Got both '{original_keys[camel_key]}' and '{key}' for the same parameter. Use one or the other."
                )
            original_keys[camel_key] = key
            converted[camel_key] = value
        return method(*args, **converted)

    return wrapper
