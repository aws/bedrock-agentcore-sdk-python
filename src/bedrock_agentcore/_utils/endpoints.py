"""Endpoint utilities for BedrockAgentCore services."""

import os
import re

from .security import SecurityValidator

# Environment-configurable constants with fallback defaults
DP_ENDPOINT_OVERRIDE = os.getenv("BEDROCK_AGENTCORE_DP_ENDPOINT")
CP_ENDPOINT_OVERRIDE = os.getenv("BEDROCK_AGENTCORE_CP_ENDPOINT")
DEFAULT_REGION = os.getenv("AWS_REGION", "us-west-2")

# Valid AWS regions pattern
AWS_REGION_PATTERN = re.compile(r'^[a-z]{2}-[a-z]+-\d{1}$')


def _validate_region(region: str) -> str:
    """Validate AWS region format."""
    if not region or not AWS_REGION_PATTERN.match(region):
        raise ValueError(f"Invalid AWS region format: {region}")
    return region


def get_data_plane_endpoint(region: str = DEFAULT_REGION) -> str:
    """Get validated data plane endpoint."""
    region = _validate_region(region)
    
    if DP_ENDPOINT_OVERRIDE:
        if not SecurityValidator.validate_endpoint(DP_ENDPOINT_OVERRIDE):
            raise ValueError(f"Invalid data plane endpoint override: {DP_ENDPOINT_OVERRIDE}")
        return DP_ENDPOINT_OVERRIDE
    
    return f"https://bedrock-agentcore.{region}.amazonaws.com"


def get_control_plane_endpoint(region: str = DEFAULT_REGION) -> str:
    """Get validated control plane endpoint."""
    region = _validate_region(region)
    
    if CP_ENDPOINT_OVERRIDE:
        if not SecurityValidator.validate_endpoint(CP_ENDPOINT_OVERRIDE):
            raise ValueError(f"Invalid control plane endpoint override: {CP_ENDPOINT_OVERRIDE}")
        return CP_ENDPOINT_OVERRIDE
    
    return f"https://bedrock-agentcore-control.{region}.amazonaws.com"