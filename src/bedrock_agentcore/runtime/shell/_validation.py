"""Validation helpers for shell ARNs and shell IDs."""

import re
from typing import Dict

from ..utils import is_valid_partition

# shell_id: alphanumeric start, then alphanumeric/hyphen/underscore.
# \Z (not $) so a trailing \n is rejected — $ matches before \n in Python.
_SHELL_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}\Z")


def parse_runtime_arn(runtime_arn: str) -> Dict[str, str]:
    """Parse and validate a runtime ARN, returning its components.

    Args:
        runtime_arn: Full runtime ARN in the form
            ``arn:{partition}:bedrock-agentcore:{region}:{account}:runtime/{id}``.

    Returns:
        Dict with ``region``, ``account_id``, and ``runtime_id``.

    Raises:
        ValueError: If the ARN format is invalid or any component is empty.
    """
    parts = runtime_arn.split(":")
    if len(parts) != 6:
        raise ValueError(f"Invalid runtime ARN format: {runtime_arn}")
    if parts[0] != "arn" or not is_valid_partition(parts[1]) or parts[2] != "bedrock-agentcore":
        raise ValueError(f"Invalid runtime ARN format: {runtime_arn}")
    resource = parts[5]
    if not resource.startswith("runtime/"):
        raise ValueError(f"Invalid runtime ARN format: {runtime_arn}")
    runtime_id = resource.split("/", 1)[1]
    region = parts[3]
    account_id = parts[4]
    if not region or not account_id or not runtime_id:
        raise ValueError("ARN components cannot be empty")
    return {"region": region, "account_id": account_id, "runtime_id": runtime_id}


def validate_shell_id(shell_id: str) -> None:
    """Validate shell_id against the API's constraint set.

    Args:
        shell_id: The value to validate.

    Raises:
        ValueError: If the value is not a str, is empty, is too long, or does not
            match the pattern ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$.
    """
    if not isinstance(shell_id, str):
        raise ValueError(f"shell_id must be str, got {type(shell_id).__name__!r}")
    if not _SHELL_ID_RE.match(shell_id):
        raise ValueError(
            f"Invalid shell_id {shell_id!r}. "
            "Must be 1–128 characters, start with alphanumeric, "
            "and contain only alphanumeric, hyphen, or underscore."
        )
