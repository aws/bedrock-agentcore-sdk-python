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

# A gateway identifier becomes a DNS label in the gateway's MCP endpoint, so it is
# constrained to the characters a label allows. Anchored with \A and \Z for the same
# reason as the region pattern.
_VALID_GATEWAY_ID_PATTERN = re.compile(r"\A[a-zA-Z0-9][a-zA-Z0-9-]{0,62}\Z")


class InvalidGatewayIdentifierError(ValueError):
    """Raised when a gateway identifier is not a valid DNS label.

    The identifier is interpolated into the endpoint hostname, so an
    unvalidated value could redirect requests to a non-AWS host.
    """


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


def get_gateway_mcp_endpoint(gateway_id: str, region: str = DEFAULT_REGION) -> str:
    """Build the MCP endpoint URL for a gateway.

    Args:
        gateway_id: The gateway identifier (not an ARN).
        region: The region the gateway lives in.

    Returns:
        The gateway's streamable HTTP MCP endpoint URL.

    Raises:
        InvalidGatewayIdentifierError: If the identifier is not a valid DNS label.
        InvalidRegionError: If the region is malformed or the URL resolves off-AWS.
    """
    if not isinstance(gateway_id, str) or not _VALID_GATEWAY_ID_PATTERN.match(gateway_id):
        raise InvalidGatewayIdentifierError(
            f"Invalid gateway identifier: {gateway_id!r}. Expected a gateway ID such as 'my-gateway-abc123'."
        )
    validate_region(region)
    url = f"https://{gateway_id}.gateway.bedrock-agentcore.{region}.amazonaws.com/mcp"
    return _validate_endpoint_url(url)
