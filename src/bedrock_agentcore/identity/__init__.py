"""Bedrock AgentCore SDK identity package."""

from .auth import requires_access_token, requires_api_key

__all__ = ["requires_access_token", "requires_api_key"]
