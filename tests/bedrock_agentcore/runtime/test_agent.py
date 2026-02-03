"""Tests for Agent class."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from bedrock_agentcore.runtime.agent import Agent
from bedrock_agentcore.runtime.build import DirectCodeDeploy, ECR
from bedrock_agentcore.runtime.config import NetworkMode


class TestAgentInit:
    """Tests for Agent initialization."""

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_minimal_init_with_ecr_prebuilt(self, mock_boto3: MagicMock) -> None:
        """Test minimal agent initialization with ECR prebuilt image."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        build = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(
            name="test-agent",
            build=build,
        )

        assert agent.name == "test-agent"
        assert agent.config.name == "test-agent"
        assert agent.config.artifact is not None
        assert agent.config.artifact.image_uri == "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"
        assert agent.runtime_arn is None
        assert agent.runtime_id is None
        assert agent.is_deployed is False
        assert agent.image_uri == "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_init_with_ecr_codebuild(self, mock_boto3: MagicMock) -> None:
        """Test agent initialization with ECR CodeBuild strategy."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        build = ECR(source_path="./test-src", entrypoint="main.py:app")
        agent = Agent(
            name="test-agent",
            build=build,
        )

        assert agent.name == "test-agent"
        assert agent.build_strategy is build
        assert agent.image_uri is None  # Not yet built
        assert agent.config.artifact is None  # Not yet built

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_full_init(self, mock_boto3: MagicMock) -> None:
        """Test full agent initialization with all parameters."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        build = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(
            name="test-agent",
            build=build,
            description="Test agent description",
            network_mode="PUBLIC",
            environment_variables={"LOG_LEVEL": "INFO"},
            tags={"Environment": "test"},
            region="us-east-1",
        )

        assert agent.name == "test-agent"
        assert agent.config.description == "Test agent description"
        assert agent.config.network_configuration is not None
        assert agent.config.network_configuration.network_mode == NetworkMode.PUBLIC
        assert agent.config.environment_variables == {"LOG_LEVEL": "INFO"}
        assert agent.config.tags == {"Environment": "test"}

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_vpc_mode_init(self, mock_boto3: MagicMock) -> None:
        """Test agent initialization with VPC mode."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        build = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(
            name="test-agent",
            build=build,
            network_mode="VPC",
            security_groups=["sg-123", "sg-456"],
            subnets=["subnet-abc", "subnet-def"],
        )

        assert agent.config.network_configuration is not None
        assert agent.config.network_configuration.network_mode == NetworkMode.VPC
        assert agent.config.network_configuration.vpc_config is not None
        assert agent.config.network_configuration.vpc_config.security_groups == ["sg-123", "sg-456"]
        assert agent.config.network_configuration.vpc_config.subnets == ["subnet-abc", "subnet-def"]


class TestAgentSaveLoad:
    """Tests for Agent save/load operations."""

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_save_to_yaml_ecr_prebuilt(self, mock_boto3: MagicMock) -> None:
        """Test saving agent config with ECR prebuilt image to YAML file."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        build = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(
            name="test-agent",
            build=build,
            description="Test agent",
            environment_variables={"LOG_LEVEL": "INFO"},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            file_path = f.name

        try:
            result = agent.save(file_path)
            assert result == file_path

            # Verify file contents
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)

            assert data["name"] == "test-agent"
            assert data["description"] == "Test agent"
            assert data["artifact"]["imageUri"] == "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"
            assert data["environmentVariables"]["LOG_LEVEL"] == "INFO"
            assert data["build"]["strategy"] == "ecr"
            assert data["build"]["imageUri"] == "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"
        finally:
            Path(file_path).unlink(missing_ok=True)

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_save_to_yaml_ecr_codebuild(self, mock_boto3: MagicMock) -> None:
        """Test saving agent config with ECR codebuild strategy to YAML file."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        build = ECR(source_path="./test-src", entrypoint="main.py:app")
        agent = Agent(
            name="test-agent",
            build=build,
            description="Test agent",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            file_path = f.name

        try:
            result = agent.save(file_path)
            assert result == file_path

            # Verify file contents
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)

            assert data["name"] == "test-agent"
            assert data["description"] == "Test agent"
            assert data["build"]["strategy"] == "ecr"
            assert data["build"]["sourcePath"] == "./test-src"
            assert data["build"]["entrypoint"] == "main.py:app"
        finally:
            Path(file_path).unlink(missing_ok=True)

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_from_yaml_ecr_prebuilt(self, mock_boto3: MagicMock) -> None:
        """Test loading agent with ECR prebuilt image from YAML file."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        # Create mock paginator
        mock_control_plane = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"agentRuntimeSummaries": []}]
        mock_control_plane.get_paginator.return_value = mock_paginator
        mock_boto3.client.return_value = mock_control_plane

        yaml_content = """
name: test-agent
description: Test agent from YAML
build:
  strategy: ecr
  imageUri: 123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest
networkConfiguration:
  networkMode: PUBLIC
environmentVariables:
  LOG_LEVEL: DEBUG
tags:
  Environment: staging
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            file_path = f.name

        try:
            agent = Agent.from_yaml(file_path)

            assert agent.name == "test-agent"
            assert agent.config.description == "Test agent from YAML"
            assert agent.image_uri == "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"
            assert agent.config.environment_variables == {"LOG_LEVEL": "DEBUG"}
            assert agent.config.tags == {"Environment": "staging"}
            assert isinstance(agent.build_strategy, ECR)
            assert agent.build_strategy.mode == "prebuilt"
        finally:
            Path(file_path).unlink(missing_ok=True)

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_from_yaml_ecr_codebuild(self, mock_boto3: MagicMock) -> None:
        """Test loading agent with ECR codebuild strategy from YAML file."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        # Create mock paginator
        mock_control_plane = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"agentRuntimeSummaries": []}]
        mock_control_plane.get_paginator.return_value = mock_paginator
        mock_boto3.client.return_value = mock_control_plane

        yaml_content = """
name: test-agent
description: Test agent from YAML
build:
  strategy: ecr
  sourcePath: ./my-agent
  entrypoint: agent.py:app
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            file_path = f.name

        try:
            agent = Agent.from_yaml(file_path)

            assert agent.name == "test-agent"
            assert isinstance(agent.build_strategy, ECR)
            assert agent.build_strategy.mode == "codebuild"
            assert agent.build_strategy.source_path == "./my-agent"
            assert agent.build_strategy.entrypoint == "agent.py:app"
        finally:
            Path(file_path).unlink(missing_ok=True)

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_from_yaml_backwards_compat(self, mock_boto3: MagicMock) -> None:
        """Test loading agent from YAML with only artifact.imageUri (backwards compatibility)."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        # Create mock paginator
        mock_control_plane = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"agentRuntimeSummaries": []}]
        mock_control_plane.get_paginator.return_value = mock_paginator
        mock_boto3.client.return_value = mock_control_plane

        yaml_content = """
name: test-agent
description: Test agent from YAML
artifact:
  imageUri: 123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            file_path = f.name

        try:
            agent = Agent.from_yaml(file_path)

            assert agent.name == "test-agent"
            assert isinstance(agent.build_strategy, ECR)
            assert agent.image_uri == "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"
        finally:
            Path(file_path).unlink(missing_ok=True)

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_from_yaml_file_not_found(self, mock_boto3: MagicMock) -> None:
        """Test that from_yaml raises FileNotFoundError for missing file."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Agent.from_yaml("/nonexistent/path/config.yaml")


class TestAgentLaunch:
    """Tests for Agent launch operations."""

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_launch_without_built_image_raises(self, mock_boto3: MagicMock) -> None:
        """Test that launch raises ValueError when source-based agent not built."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        build = ECR(source_path="./test-source", entrypoint="agent.py:app")
        agent = Agent(
            name="test-agent",
            build=build,
        )

        with pytest.raises(ValueError, match="Cannot launch agent without image_uri"):
            agent.launch()


class TestAgentInvoke:
    """Tests for Agent invoke operations."""

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_invoke_not_deployed_raises(self, mock_boto3: MagicMock) -> None:
        """Test that invoke raises ValueError when not deployed."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        build = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(
            name="test-agent",
            build=build,
        )

        with pytest.raises(ValueError, match="Agent is not deployed"):
            agent.invoke({"message": "Hello"})

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_invoke_with_dict_payload(self, mock_boto3: MagicMock) -> None:
        """Test invoke with dictionary payload."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        mock_data_plane = MagicMock()
        response_payload = json.dumps({"response": "Hello back!"}).encode("utf-8")
        mock_data_plane.invoke_agent_runtime.return_value = {
            "payload": response_payload,
            "sessionId": "session-123",
            "contentType": "application/json",
        }
        mock_boto3.client.return_value = mock_data_plane

        build = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(
            name="test-agent",
            build=build,
        )
        # Simulate deployed state
        agent._runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123456789012:agent-runtime/test-id"

        result = agent.invoke({"message": "Hello"})

        assert result["payload"] == {"response": "Hello back!"}
        assert result["sessionId"] == "session-123"


class TestAgentDestroy:
    """Tests for Agent destroy operations."""

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_destroy_not_deployed(self, mock_boto3: MagicMock) -> None:
        """Test destroy when not deployed returns NOT_DEPLOYED status."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        build = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(
            name="test-agent",
            build=build,
        )

        result = agent.destroy()

        assert result["status"] == "NOT_DEPLOYED"


class TestAgentBuildStrategies:
    """Tests for Agent build strategy serialization."""

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_build_config_ecr_prebuilt(self, mock_boto3: MagicMock) -> None:
        """Test that ECR prebuilt is serialized correctly."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        build = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(name="test-agent", build=build)

        assert agent.config.build is not None
        assert agent.config.build.strategy.value == "ecr"
        assert agent.config.build.image_uri == "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_build_config_ecr_codebuild(self, mock_boto3: MagicMock) -> None:
        """Test that ECR codebuild is serialized correctly."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        build = ECR(source_path="./src", entrypoint="main.py:app")
        agent = Agent(name="test-agent", build=build)

        assert agent.config.build is not None
        assert agent.config.build.strategy.value == "ecr"
        assert agent.config.build.source_path == "./src"
        assert agent.config.build.entrypoint == "main.py:app"

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_build_config_direct_code_deploy(self, mock_boto3: MagicMock) -> None:
        """Test that DirectCodeDeploy is serialized correctly."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        build = DirectCodeDeploy(source_path="./src", entrypoint="main.py:app", s3_bucket="my-bucket")
        agent = Agent(name="test-agent", build=build)

        assert agent.config.build is not None
        assert agent.config.build.strategy.value == "direct_code_deploy"
        assert agent.config.build.source_path == "./src"
        assert agent.config.build.entrypoint == "main.py:app"
        assert agent.config.build.s3_bucket == "my-bucket"
