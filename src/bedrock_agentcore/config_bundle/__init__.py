"""Configuration bundle support for BedrockAgentCore."""

from .bundle import ConfigBundleComponents, ConfigBundleRef
from .client import ConfigBundleClient

__all__ = [
    "ConfigBundleRef",
    "ConfigBundleComponents",
    "ConfigBundleClient",
]
