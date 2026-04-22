"""Endpoint utilities for BedrockAgentCore services."""

import os
import re
from urllib.parse import urlparse

# Environment-configurable constants with fallback defaults
DP_ENDPOINT_OVERRIDE = os.getenv("BEDROCK_AGENTCORE_DP_ENDPOINT")
CP_ENDPOINT_OVERRIDE = os.getenv("BEDROCK_AGENTCORE_CP_ENDPOINT")
DEFAULT_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"

# Regex for valid AWS region names (e.g., us-east-1, eu-west-2, cn-north-1, us-gov-west-1).
# Uses \A and \Z anchors to prevent newline injection bypass that $ allows.
_VALID_REGION_PATTERN = re.compile(r"\A[a-z]{2}(-[a-z]+)+-\d+\Z")


class InvalidRegionError(ValueError):
    """Raised when an invalid AWS region string is provided.

    This prevents SSRF attacks where a crafted region value
    (e.g., ``x@attacker.com:443/#``) could redirect SDK API calls
    to non-AWS hosts.
    """


def validate_region(region: str) -> str:
    """Validate that a region string is a well-formed AWS region name.

    Args:
        region: The region string to validate.

    Returns:
        The validated region string (unchanged).

    Raises:
        InvalidRegionError: If the region does not match the expected pattern.
    """
    if not isinstance(region, str) or not _VALID_REGION_PATTERN.match(region):
        raise InvalidRegionError(
            f"Invalid AWS region: {region!r}. Region must match pattern like 'us-east-1', 'eu-west-2', 'cn-north-1'."
        )
    return region


def _validate_endpoint_url(url: str) -> str:
    """Validate that a constructed endpoint URL resolves to an AWS host.

    This is a defense-in-depth check that catches URL manipulation even if
    the region regex is somehow bypassed.

    Args:
        url: The constructed endpoint URL.

    Returns:
        The validated URL (unchanged).

    Raises:
        InvalidRegionError: If the URL hostname does not end with an AWS domain.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    _AWS_DOMAINS = (".amazonaws.com", ".amazonaws.com.cn", ".api.aws")
    if not any(hostname.endswith(d) for d in _AWS_DOMAINS):
        raise InvalidRegionError(f"Constructed endpoint resolves to non-AWS host: {hostname!r}")
    return url


def get_data_plane_endpoint(region: str = DEFAULT_REGION) -> str:
    if DP_ENDPOINT_OVERRIDE:
        return _validate_endpoint_url(DP_ENDPOINT_OVERRIDE)
    validate_region(region)
    url = f"https://bedrock-agentcore.{region}.amazonaws.com"
    return _validate_endpoint_url(url)


def get_control_plane_endpoint(region: str = DEFAULT_REGION) -> str:
    if CP_ENDPOINT_OVERRIDE:
        return _validate_endpoint_url(CP_ENDPOINT_OVERRIDE)
    validate_region(region)
    url = f"https://bedrock-agentcore-control.{region}.amazonaws.com"
    return _validate_endpoint_url(url)
