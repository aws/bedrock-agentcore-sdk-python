"""BedrockAgentCore Runtime Package.

This package contains the core runtime components for Bedrock AgentCore applications:
- BedrockAgentCoreApp: Main application class
- RequestContext: HTTP request context
- BedrockAgentCoreContext: Agent identity context
- Agent: High-level runtime management with YAML config
- Build strategies: PrebuiltImage, CodeBuild, LocalBuild, DirectCodeDeploy
"""

from .agent import Agent
from .agent_core_runtime_client import AgentCoreRuntimeClient
from .app import BedrockAgentCoreApp
from .build import (
    Build,
    CodeBuild,
    CodeBuildStrategy,
    DirectCodeDeploy,
    DirectCodeDeployStrategy,
    LocalBuild,
    LocalBuildStrategy,
    PrebuiltImage,
    codebuild,
    direct_code_deploy,
    local,
    prebuilt,
)
from .context import BedrockAgentCoreContext, RequestContext
from .models import PingStatus

__all__ = [
    "Agent",
    "AgentCoreRuntimeClient",
    "BedrockAgentCoreApp",
    "RequestContext",
    "BedrockAgentCoreContext",
    "PingStatus",
    # Build strategies (new names)
    "Build",
    "PrebuiltImage",
    "CodeBuild",
    "LocalBuild",
    "DirectCodeDeploy",
    # Build strategies (backwards compatibility)
    "CodeBuildStrategy",
    "LocalBuildStrategy",
    "DirectCodeDeployStrategy",
    # Factory functions
    "prebuilt",
    "codebuild",
    "local",
    "direct_code_deploy",
]
