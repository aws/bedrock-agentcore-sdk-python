"""Bedrock AgentCore runtime utilities for object conversion and serialization."""

import base64
from dataclasses import asdict, is_dataclass
from typing import Any


def convert_complex_objects(obj: Any, _depth: int = 0) -> Any:
    """Recursively convert complex objects to serializable dictionaries."""
    # Prevent infinite recursion
    if _depth > 50:
        return f"<too_deep:{type(obj).__name__}>"

    # Handle Pydantic models (like AIMessage). The dump can still hold bytes or
    # sets, so it is converted too.
    if hasattr(obj, "model_dump"):
        return convert_complex_objects(obj.model_dump(), _depth + 1)

    # Handle dataclasses (like AgentResult), converting the dump for the same reason
    elif is_dataclass(obj):
        return convert_complex_objects(asdict(obj), _depth + 1)

    # Handle dictionaries recursively
    elif isinstance(obj, dict):
        return {k: convert_complex_objects(v, _depth + 1) for k, v in obj.items()}

    # Handle lists and tuples recursively
    elif isinstance(obj, (list, tuple)):
        return [convert_complex_objects(item, _depth + 1) for item in obj]

    # Handle sets (convert to list)
    elif isinstance(obj, set):
        return [convert_complex_objects(item, _depth + 1) for item in obj]

    # Handle binary data (base64, since JSON has no bytes type)
    elif isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(obj).decode("ascii")

    # Return primitives as-is
    else:
        return obj


def is_valid_partition(partition: str) -> bool:
    """Returns if parsed-arn partition is valid."""
    return partition in ("aws", "aws-us-gov")
