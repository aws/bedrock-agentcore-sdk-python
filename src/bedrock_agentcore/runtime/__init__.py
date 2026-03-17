"""BedrockAgentCore Runtime Package.

This package contains the core runtime components for Bedrock AgentCore applications:
- BedrockAgentCoreApp: Main application class
- RequestContext: HTTP request context
- BedrockAgentCoreContext: Agent identity context
"""

from .agent_core_runtime_client import AgentCoreRuntimeClient
from .app import BedrockAgentCoreApp
from .context import BedrockAgentCoreContext, RequestContext
from .models import PingStatus

__all__ = [
    "AgentCoreRuntimeClient",
    "BedrockAgentCoreApp",
    "BedrockCallContextBuilder",
    "RequestContext",
    "BedrockAgentCoreContext",
    "PingStatus",
    "build_a2a_app",
    "build_runtime_url",
    "serve_a2a",
]


def __getattr__(name: str):
    """Lazy imports for A2A symbols so the a2a-sdk optional dependency is not required at import time."""
    _a2a_exports = {"BedrockCallContextBuilder", "build_a2a_app", "build_runtime_url", "serve_a2a"}
    if name in _a2a_exports:
        from . import a2a as _a2a_module

        return getattr(_a2a_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
