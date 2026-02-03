"""Configuration models for Bedrock AgentCore Runtime.

This module provides Pydantic models for Agent runtime configuration
with YAML serialization support.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RuntimeStatus(str, Enum):
    """Runtime status values."""

    CREATING = "CREATING"
    ACTIVE = "ACTIVE"
    UPDATING = "UPDATING"
    DELETING = "DELETING"
    FAILED = "FAILED"
    NOT_FOUND = "NOT_FOUND"


class NetworkMode(str, Enum):
    """Network mode options."""

    PUBLIC = "PUBLIC"
    VPC = "VPC"


class VpcConfigModel(BaseModel):
    """VPC configuration for runtime networking."""

    model_config = ConfigDict(populate_by_name=True)

    security_groups: List[str] = Field(alias="securityGroups")
    subnets: List[str]


class NetworkConfigurationModel(BaseModel):
    """Network configuration for runtime deployment."""

    model_config = ConfigDict(populate_by_name=True)

    network_mode: NetworkMode = Field(default=NetworkMode.PUBLIC, alias="networkMode")
    vpc_config: Optional[VpcConfigModel] = Field(default=None, alias="vpcConfig")


class RuntimeArtifactModel(BaseModel):
    """Container artifact configuration."""

    model_config = ConfigDict(populate_by_name=True)

    image_uri: str = Field(alias="imageUri")


class BuildStrategyType(str, Enum):
    """Build strategy type options."""

    ECR = "ecr"
    DIRECT_CODE_DEPLOY = "direct_code_deploy"


class BuildConfigModel(BaseModel):
    """Build configuration for agent deployment.

    Attributes:
        strategy: Build strategy type (ecr, direct_code_deploy)
        image_uri: Pre-built image URI (for ECR with pre-built mode)
        source_path: Path to agent source code directory (for CodeBuild or DirectCodeDeploy)
        entrypoint: Entry point e.g. "agent.py:app" (for CodeBuild or DirectCodeDeploy)
        s3_bucket: S3 bucket for direct code deploy
    """

    model_config = ConfigDict(populate_by_name=True)

    strategy: BuildStrategyType = Field(default=BuildStrategyType.ECR)
    image_uri: Optional[str] = Field(default=None, alias="imageUri")
    source_path: Optional[str] = Field(default=None, alias="sourcePath")
    entrypoint: Optional[str] = None
    s3_bucket: Optional[str] = Field(default=None, alias="s3Bucket")


class RuntimeConfigModel(BaseModel):
    """Complete runtime configuration model.

    This model represents the configuration for a Bedrock AgentCore runtime,
    suitable for YAML serialization and deserialization.

    Attributes:
        name: Unique runtime name
        description: Optional description
        artifact: Container image configuration
        build: Build configuration for source-based deployment
        network_configuration: Network settings (PUBLIC or VPC)
        environment_variables: Environment variables for the container
        tags: Resource tags
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: Optional[str] = None
    artifact: Optional[RuntimeArtifactModel] = None
    build: Optional[BuildConfigModel] = None
    network_configuration: Optional[NetworkConfigurationModel] = Field(
        default=None, alias="networkConfiguration"
    )
    environment_variables: Optional[Dict[str, str]] = Field(default=None, alias="environmentVariables")
    tags: Optional[Dict[str, str]] = None
