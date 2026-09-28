"""Endpoint utilities for BedrockAgentCore services."""

import functools
import logging
import os
import re
from urllib.parse import urlparse

from botocore.exceptions import UnknownRegionError
from botocore.loaders import create_loader
from botocore.regions import EndpointResolver

logger = logging.getLogger(__name__)

# Environment-configurable constants with fallback defaults
DP_ENDPOINT_OVERRIDE = os.getenv("BEDROCK_AGENTCORE_DP_ENDPOINT")
CP_ENDPOINT_OVERRIDE = os.getenv("BEDROCK_AGENTCORE_CP_ENDPOINT")
DEFAULT_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"


@functools.lru_cache(maxsize=1)
def _endpoint_resolver() -> EndpointResolver:
    """Build a resolver over botocore's bundled static endpoint data (once, lazily).

    Uses only public botocore APIs and performs no network I/O — the partition
    table ships with botocore. Cached so the ``endpoints.json`` load happens at
    most once, since this module sits on the runtime hot path.
    """
    return EndpointResolver(create_loader().load_data("endpoints"))


@functools.lru_cache(maxsize=None)
def _dns_suffix_for_region(region: str) -> str:
    """Return the partition DNS suffix for a region.

    For example, ``us-west-2`` resolves to ``amazonaws.com`` while ``cn-north-1``
    resolves to ``amazonaws.com.cn`` and ``us-gov-west-1`` to ``amazonaws.com``.
    The suffix is derived from botocore's static partition data rather than a
    hardcoded table, so China (``aws-cn``), GovCloud (``aws-us-gov``), and future
    partitions are handled without further changes.

    Falls back to ``amazonaws.com`` only for regions botocore does not recognise,
    logging a warning so a wrong-partition fallback is visible rather than silent.
    """
    resolver = _endpoint_resolver()
    try:
        partition = resolver.get_partition_for_region(region)
    except UnknownRegionError:
        logger.warning(
            "Region %r is not recognised by botocore; falling back to the "
            "'amazonaws.com' DNS suffix. Endpoints may be incorrect outside the "
            "commercial partition.",
            region,
        )
        return "amazonaws.com"
    return resolver.get_partition_dns_suffix(partition) or "amazonaws.com"


@functools.lru_cache(maxsize=1)
def known_partitions() -> frozenset:
    """Return the set of AWS partition names botocore knows about (offline)."""
    return frozenset(_endpoint_resolver().get_available_partitions())


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
    url = f"https://bedrock-agentcore.{region}.{_dns_suffix_for_region(region)}"
    return _validate_endpoint_url(url)


def get_control_plane_endpoint(region: str = DEFAULT_REGION) -> str:
    if CP_ENDPOINT_OVERRIDE:
        return _validate_endpoint_url(CP_ENDPOINT_OVERRIDE)
    validate_region(region)
    url = f"https://bedrock-agentcore-control.{region}.{_dns_suffix_for_region(region)}"
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
    url = f"https://{gateway_id}.gateway.bedrock-agentcore.{region}.{_dns_suffix_for_region(region)}/mcp"
    return _validate_endpoint_url(url)
