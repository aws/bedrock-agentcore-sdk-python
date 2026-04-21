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
    "AGUIApp",
    "BedrockAgentCoreApp",
    "BedrockCallContextBuilder",
    "RequestContext",
    "BedrockAgentCoreContext",
    "PingStatus",
    "build_a2a_app",
    "build_ag_ui_app",
    "build_runtime_url",
    "serve_a2a",
    "serve_ag_ui",
]


def __getattr__(name: str):
    """Lazy imports for A2A and AG-UI symbols so optional dependencies are not required at import time."""
    _a2a_exports = {"BedrockCallContextBuilder", "build_a2a_app", "build_runtime_url", "serve_a2a"}
    if name in _a2a_exports:
        from . import a2a as _a2a_module

        return getattr(_a2a_module, name)

    _ag_ui_exports = {"AGUIApp", "build_ag_ui_app", "serve_ag_ui"}
    if name in _ag_ui_exports:
        from . import ag_ui as _ag_ui_module

        return getattr(_ag_ui_module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
