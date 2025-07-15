"""Endpoint utilities for BedrockAgentCore services."""

import os

# Environment-configurable constants with fallback defaults
DP_ENDPOINT_OVERRIDE = os.getenv("BEDROCK_AGENTCORE_DP_ENDPOINT")
CP_ENDPOINT_OVERRIDE = os.getenv("BEDROCK_AGENTCORE_CP_ENDPOINT")
DEFAULT_REGION = os.getenv("AWS_REGION", "us-west-2")


def get_data_plane_endpoint(region: str = DEFAULT_REGION) -> str:
    return DP_ENDPOINT_OVERRIDE or f"https://bedrock-agentcore.{region}.amazonaws.com"


def get_control_plane_endpoint(region: str = DEFAULT_REGION) -> str:
    return CP_ENDPOINT_OVERRIDE or f"https://bedrock-agentcore-control.{region}.amazonaws.com"
