"""Bedrock AgentCore runtime utilities for object conversion and serialization."""

from dataclasses import asdict, is_dataclass
from typing import Any


def convert_complex_objects(obj: Any, _depth: int = 0) -> Any:
    """Recursively convert complex objects to serializable dictionaries."""
    # Prevent infinite recursion
    if _depth > 50:
        return f"<too_deep:{type(obj).__name__}>"

    # Handle Pydantic models (like AIMessage)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    # Handle dataclasses (like AgentResult)
    elif is_dataclass(obj):
        return asdict(obj)

    # Handle dictionaries recursively
    elif isinstance(obj, dict):
        return {k: convert_complex_objects(v, _depth + 1) for k, v in obj.items()}

    # Handle lists and tuples recursively
    elif isinstance(obj, (list, tuple)):
        return [convert_complex_objects(item, _depth + 1) for item in obj]

    # Handle sets (convert to list)
    elif isinstance(obj, set):
        return [convert_complex_objects(item, _depth + 1) for item in obj]

    # Return primitives as-is
    else:
        return obj
