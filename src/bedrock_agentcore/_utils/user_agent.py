"""User-Agent utilities for BedrockAgentCore SDK."""

from typing import Optional

# Get version from package metadata
try:
    from importlib.metadata import version

    SDK_VERSION = version("bedrock-agentcore")
except Exception:
    # Fallback if package isn't installed properly (e.g., during development)
    SDK_VERSION = "unknown"


def build_user_agent_suffix(integration_source: Optional[str] = None) -> str:
    """Build the suffix string to append to boto3 User-Agent header.

    This value is passed to botocore's Config(user_agent_extra=...) parameter.

    Args:
        integration_source: Optional integration framework identifier
                           (e.g., 'langchain', 'crewai', 'strands')

    Returns:
        String to append to User-Agent header

    Example:
        >>> build_user_agent_suffix("langchain")
        'bedrock-agentcore/1.0.0 (integration_source=langchain)'
        >>> build_user_agent_suffix()
        'bedrock-agentcore/1.0.0'
    """
    base = f"bedrock-agentcore/{SDK_VERSION}"

    if integration_source:
        # Sanitize to prevent header injection
        sanitized = "".join(c for c in integration_source.lower() if c.isalnum() or c in "-_")
        return f"{base} (integration_source={sanitized})"

    return base
