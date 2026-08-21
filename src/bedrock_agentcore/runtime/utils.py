"""Bedrock AgentCore runtime utilities for object conversion and serialization."""

import base64
import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


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


def is_valid_partition(partition: str) -> bool:
    """Returns if parsed-arn partition is valid."""
    return partition in ("aws", "aws-us-gov")


def extract_sub_from_bearer(authorization: Optional[str]) -> Optional[str]:
    """Return the 'sub' claim from a Bearer JWT without signature validation.

    Intended only for populating the OTel ``enduser.id`` span attribute.
    The token is NOT validated — its signature, expiry, and issuer are not
    checked.  Trust decisions must be made by the inbound auth layer before
    the request reaches agent code.

    Returns ``None`` when the header is absent, malformed, or has no 'sub'.
    """
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    segments = parts[1].strip().split(".")
    if len(segments) < 2:
        return None
    payload = segments[1]
    # JWT base64url uses no padding; add it back for Python's decoder.
    padding = (4 - len(payload) % 4) % 4
    try:
        decoded = base64.urlsafe_b64decode(payload + "=" * padding)
        sub = json.loads(decoded).get("sub")
        return str(sub) if sub is not None else None
    except Exception:
        logger.debug("Could not decode JWT payload for enduser.id extraction", exc_info=True)
        return None
