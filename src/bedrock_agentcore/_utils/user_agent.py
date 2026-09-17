"""User-Agent utilities for BedrockAgentCore SDK."""

from typing import Optional

# Get version from package metadata
try:
    from importlib.metadata import version

    SDK_VERSION = version("bedrock-agentcore")
except Exception:
    # Fallback if package isn't installed properly (e.g., during development)
    SDK_VERSION = "unknown"


def _sanitize_token(value: str) -> str:
    """Sanitize a User-Agent token to prevent header injection."""
    return "".join(c for c in value.lower() if c.isalnum() or c in "-_")


def build_user_agent_suffix(integration_source: Optional[str] = None, feature: Optional[str] = None) -> str:
    """Build the suffix string to append to boto3 User-Agent header.

    This value is passed to botocore's Config(user_agent_extra=...) parameter.

    Args:
        integration_source: Optional integration framework identifier
                           (e.g., 'langgraph', 'crewai', 'strands', 'raw-sdk')
        feature: Optional feature identifier used to separate a capability's
                calls from other calls (e.g., 'payments')

    Returns:
        String to append to User-Agent header

    Example:
        >>> build_user_agent_suffix("langgraph")
        'bedrock-agentcore/1.0.0 (integration_source=langgraph)'
        >>> build_user_agent_suffix("strands", feature="payments")
        'bedrock-agentcore/1.0.0 (integration_source=strands; feature=payments)'
        >>> build_user_agent_suffix()
        'bedrock-agentcore/1.0.0'
    """
    base = f"bedrock-agentcore/{SDK_VERSION}"

    tokens = []
    if integration_source:
        tokens.append(f"integration_source={_sanitize_token(integration_source)}")
    if feature:
        tokens.append(f"feature={_sanitize_token(feature)}")

    if tokens:
        return f"{base} ({'; '.join(tokens)})"

    return base
