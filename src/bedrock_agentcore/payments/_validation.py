"""Shared validation for payments SDK inputs."""

import re
from typing import Optional

_PERMIT2_ALLOWANCE_LIMIT_PATTERN = re.compile(r"[0-9]{1,78}\Z")


def validate_permit2_allowance_limit(value: Optional[str]) -> None:
    """Validate a Permit2 allowance against the service model constraints."""
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(
            f"permit2_allowance_limit must be a string in the asset's smallest denomination, got {type(value).__name__}"
        )
    if not _PERMIT2_ALLOWANCE_LIMIT_PATTERN.fullmatch(value) or int(value) <= 0:
        raise ValueError(f"permit2_allowance_limit must be a positive ASCII integer of 1-78 digits, got {value!r}")
