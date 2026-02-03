"""Tests for runtime configuration models."""

import pytest

from bedrock_agentcore.runtime.config import (
    NetworkConfigurationModel,
    NetworkMode,
    RuntimeArtifactModel,
    RuntimeConfigModel,
    RuntimeStatus,
    VpcConfigModel,
)


class TestRuntimeStatus:
    """Tests for RuntimeStatus enum."""

    def test_status_values(self) -> None:
        """Test that all expected status values exist."""
        assert RuntimeStatus.CREATING == "CREATING"
        assert RuntimeStatus.ACTIVE == "ACTIVE"
        assert RuntimeStatus.UPDATING == "UPDATING"
        assert RuntimeStatus.DELETING == "DELETING"
        assert RuntimeStatus.FAILED == "FAILED"
        assert RuntimeStatus.NOT_FOUND == "NOT_FOUND"


class TestNetworkMode:
    """Tests for NetworkMode enum."""

    def test_network_mode_values(self) -> None:
        """Test that network mode values exist."""
        assert NetworkMode.PUBLIC == "PUBLIC"
        assert NetworkMode.VPC == "VPC"


class TestVpcConfigModel:
    """Tests for VpcConfigModel."""

    def test_create_with_alias(self) -> None:
        """Test creating VpcConfigModel with alias names."""
        config = VpcConfigModel(
            securityGroups=["sg-123", "sg-456"],
            subnets=["subnet-abc", "subnet-def"],
        )
        assert config.security_groups == ["sg-123", "sg-456"]
        assert config.subnets == ["subnet-abc", "subnet-def"]

    def test_dump_by_alias(self) -> None:
        """Test dumping config with alias names."""
        config = VpcConfigModel(
            securityGroups=["sg-123"],
            subnets=["subnet-abc"],
        )
        data = config.model_dump(by_alias=True)
        assert data["securityGroups"] == ["sg-123"]
        assert data["subnets"] == ["subnet-abc"]


class TestNetworkConfigurationModel:
    """Tests for NetworkConfigurationModel."""

    def test_default_public_mode(self) -> None:
        """Test default network mode is PUBLIC."""
        config = NetworkConfigurationModel()
        assert config.network_mode == NetworkMode.PUBLIC
        assert config.vpc_config is None

    def test_vpc_mode(self) -> None:
        """Test VPC network mode."""
        vpc = VpcConfigModel(
            securityGroups=["sg-123"],
            subnets=["subnet-abc"],
        )
        config = NetworkConfigurationModel(
            networkMode=NetworkMode.VPC,
            vpcConfig=vpc,
        )
        assert config.network_mode == NetworkMode.VPC
        assert config.vpc_config is not None


class TestRuntimeArtifactModel:
    """Tests for RuntimeArtifactModel."""

    def test_create(self) -> None:
        """Test creating RuntimeArtifactModel."""
        artifact = RuntimeArtifactModel(imageUri="123456789.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        assert artifact.image_uri == "123456789.dkr.ecr.us-west-2.amazonaws.com/test:latest"

    def test_dump_by_alias(self) -> None:
        """Test dumping with alias."""
        artifact = RuntimeArtifactModel(imageUri="test:latest")
        data = artifact.model_dump(by_alias=True)
        assert data["imageUri"] == "test:latest"


class TestRuntimeConfigModel:
    """Tests for RuntimeConfigModel."""

    def test_minimal_config(self) -> None:
        """Test minimal config with just name."""
        config = RuntimeConfigModel(name="test-agent")
        assert config.name == "test-agent"
        assert config.description is None
        assert config.artifact is None
        assert config.network_configuration is None
        assert config.environment_variables is None
        assert config.tags is None

    def test_full_config(self) -> None:
        """Test full config with all fields."""
        config = RuntimeConfigModel(
            name="test-agent",
            description="Test agent description",
            artifact=RuntimeArtifactModel(imageUri="test:latest"),
            networkConfiguration=NetworkConfigurationModel(networkMode=NetworkMode.PUBLIC),
            environmentVariables={"LOG_LEVEL": "INFO"},
            tags={"Environment": "test"},
        )
        assert config.name == "test-agent"
        assert config.description == "Test agent description"
        assert config.artifact is not None
        assert config.artifact.image_uri == "test:latest"
        assert config.network_configuration is not None
        assert config.environment_variables == {"LOG_LEVEL": "INFO"}
        assert config.tags == {"Environment": "test"}

    def test_dump_excludes_none(self) -> None:
        """Test that dump excludes None values."""
        config = RuntimeConfigModel(name="test-agent")
        data = config.model_dump(by_alias=True, exclude_none=True)
        assert "name" in data
        assert "description" not in data
        assert "artifact" not in data

    def test_validate_from_dict(self) -> None:
        """Test validating from dict (YAML-like structure)."""
        data = {
            "name": "test-agent",
            "artifact": {"imageUri": "test:latest"},
            "networkConfiguration": {"networkMode": "PUBLIC"},
        }
        config = RuntimeConfigModel.model_validate(data)
        assert config.name == "test-agent"
        assert config.artifact is not None
        assert config.artifact.image_uri == "test:latest"
