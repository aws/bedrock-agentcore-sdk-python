"""Configuration models for Bedrock AgentCore Project.

This module provides Pydantic models for Project configuration
matching the starter-toolkit agentcore.json schema.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PythonVersion(str, Enum):
    """Supported Python versions."""

    PYTHON_3_10 = "PYTHON_3_10"
    PYTHON_3_11 = "PYTHON_3_11"
    PYTHON_3_12 = "PYTHON_3_12"
    PYTHON_3_13 = "PYTHON_3_13"


class NetworkMode(str, Enum):
    """Network mode options."""

    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"


class MemoryRelation(str, Enum):
    """Memory provider relation type."""

    OWN = "own"
    USE = "use"


class MemoryStrategyType(str, Enum):
    """Memory strategy types."""

    SEMANTIC = "SEMANTIC"
    SUMMARIZATION = "SUMMARIZATION"
    USER_PREFERENCE = "USER_PREFERENCE"
    CUSTOM = "CUSTOM"


# ==================== Runtime Config ====================


class InstrumentationConfig(BaseModel):
    """Instrumentation configuration."""

    model_config = ConfigDict(populate_by_name=True)

    enable_otel: bool = Field(default=False, alias="enableOtel")


class RuntimeConfig(BaseModel):
    """Agent runtime configuration."""

    model_config = ConfigDict(populate_by_name=True)

    artifact: str = "CodeZip"
    name: Optional[str] = None
    python_version: PythonVersion = Field(default=PythonVersion.PYTHON_3_12, alias="pythonVersion")
    entrypoint: str
    code_location: str = Field(alias="codeLocation")
    network_mode: NetworkMode = Field(default=NetworkMode.PUBLIC, alias="networkMode")
    instrumentation: Optional[InstrumentationConfig] = None


# ==================== Memory Provider Config ====================


class MemoryStrategyConfig(BaseModel):
    """Memory strategy configuration."""

    model_config = ConfigDict(populate_by_name=True)

    type: MemoryStrategyType


class MemoryProviderConfig(BaseModel):
    """Memory provider configuration within an agent."""

    model_config = ConfigDict(populate_by_name=True)

    type: str = "AgentCoreMemory"
    relation: MemoryRelation = MemoryRelation.OWN
    name: str
    event_expiry_duration: Optional[int] = Field(default=None, alias="eventExpiryDuration")
    memory_strategies: Optional[List[MemoryStrategyConfig]] = Field(default=None, alias="memoryStrategies")


# ==================== Agent Config ====================


class AgentConfig(BaseModel):
    """Agent configuration within a project."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(max_length=64)
    id: Optional[str] = None
    runtime: RuntimeConfig
    memory_providers: Optional[List[MemoryProviderConfig]] = Field(default=None, alias="memoryProviders")


# ==================== Project Config ====================


class ProjectConfig(BaseModel):
    """Complete project configuration model.

    This model represents the configuration for a Bedrock AgentCore project,
    matching the starter-toolkit agentcore.json schema.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(max_length=23)
    version: Optional[str] = None
    description: Optional[str] = None
    agents: Optional[List[AgentConfig]] = None


# ==================== AWS Targets Config ====================


class AWSTarget(BaseModel):
    """AWS deployment target."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    account: str
    region: str


# ==================== Deployed State Config ====================


class AgentDeployedState(BaseModel):
    """Deployed state for a single agent."""

    model_config = ConfigDict(populate_by_name=True)

    runtime_id: Optional[str] = Field(default=None, alias="runtimeId")
    runtime_arn: Optional[str] = Field(default=None, alias="runtimeArn")
    role_arn: Optional[str] = Field(default=None, alias="roleArn")
    session_id: Optional[str] = Field(default=None, alias="sessionId")
    memory_ids: Optional[List[str]] = Field(default=None, alias="memoryIds")


class TargetResources(BaseModel):
    """Resources deployed to a target."""

    model_config = ConfigDict(populate_by_name=True)

    agents: Optional[Dict[str, AgentDeployedState]] = None


class TargetDeployedState(BaseModel):
    """Deployed state for a target."""

    model_config = ConfigDict(populate_by_name=True)

    resources: Optional[TargetResources] = None


class DeployedState(BaseModel):
    """Complete deployed state model."""

    model_config = ConfigDict(populate_by_name=True)

    targets: Optional[Dict[str, TargetDeployedState]] = None
