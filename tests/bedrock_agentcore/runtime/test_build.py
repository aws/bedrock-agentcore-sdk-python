"""Tests for Build strategies."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.runtime.build import (
    Build,
    DirectCodeDeploy,
    ECR,
)


class TestECRPrebuilt:
    """Tests for ECR with pre-built image."""

    def test_strategy_name(self) -> None:
        """Test that strategy name is correct."""
        strategy = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        assert strategy.strategy_name == "ecr"

    def test_mode_is_prebuilt(self) -> None:
        """Test that mode is 'prebuilt' when image_uri is provided."""
        strategy = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        assert strategy.mode == "prebuilt"

    def test_image_uri(self) -> None:
        """Test that image_uri is returned correctly."""
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"
        strategy = ECR(image_uri=image_uri)
        assert strategy.image_uri == image_uri

    def test_source_path_is_none(self) -> None:
        """Test that source_path is None for pre-built."""
        strategy = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        assert strategy.source_path is None

    def test_entrypoint_is_none(self) -> None:
        """Test that entrypoint is None for pre-built."""
        strategy = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        assert strategy.entrypoint is None

    def test_launch_returns_image_uri(self) -> None:
        """Test that launch() returns the image URI for pre-built."""
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"
        strategy = ECR(image_uri=image_uri)
        result = strategy.launch(agent_name="test-agent")
        assert result["imageUri"] == image_uri
        assert result["status"] == "READY"


class TestECRCodeBuild:
    """Tests for ECR with CodeBuild (source-based)."""

    def test_strategy_name(self) -> None:
        """Test that strategy name is correct."""
        strategy = ECR(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.strategy_name == "ecr"

    def test_mode_is_codebuild(self) -> None:
        """Test that mode is 'codebuild' when source_path is provided."""
        strategy = ECR(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.mode == "codebuild"

    def test_source_path_and_entrypoint(self) -> None:
        """Test source_path and entrypoint are stored."""
        strategy = ECR(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.source_path == "./test-src"
        assert strategy.entrypoint == "main.py:app"

    def test_image_uri_is_none_before_launch(self) -> None:
        """Test image_uri is None before launch."""
        strategy = ECR(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.image_uri is None

    @patch("bedrock_agentcore.runtime.builder.build_and_push")
    def test_launch_calls_builder(self, mock_build_and_push: MagicMock) -> None:
        """Test that launch() delegates to builder module."""
        mock_build_and_push.return_value = {
            "imageUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
            "buildId": "build-123",
            "status": "SUCCEEDED",
        }

        strategy = ECR(source_path="/tmp/test-agent", entrypoint="main.py:app")
        result = strategy.launch(
            agent_name="test-agent",
            region_name="us-west-2",
        )

        mock_build_and_push.assert_called_once()
        assert result["status"] == "SUCCEEDED"
        assert "imageUri" in result
        assert strategy.image_uri == "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"


class TestECRValidation:
    """Tests for ECR validation."""

    def test_requires_image_uri_or_source_path(self) -> None:
        """Test that either image_uri or source_path must be provided."""
        with pytest.raises(ValueError, match="Must provide either"):
            ECR()

    def test_source_path_requires_entrypoint(self) -> None:
        """Test that source_path requires entrypoint."""
        with pytest.raises(ValueError, match="Must provide either"):
            ECR(source_path="./test-src")

    def test_entrypoint_requires_source_path(self) -> None:
        """Test that entrypoint alone is not valid."""
        with pytest.raises(ValueError, match="Must provide either"):
            ECR(entrypoint="main.py:app")


class TestDirectCodeDeploy:
    """Tests for DirectCodeDeploy."""

    def test_strategy_name(self) -> None:
        """Test that strategy name is correct."""
        strategy = DirectCodeDeploy(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.strategy_name == "direct_code_deploy"

    def test_custom_bucket(self) -> None:
        """Test custom S3 bucket specification."""
        strategy = DirectCodeDeploy(source_path="./test-src", entrypoint="main.py:app", s3_bucket="my-bucket")
        assert strategy._s3_bucket == "my-bucket"

    def test_source_path_and_entrypoint(self) -> None:
        """Test source_path and entrypoint are stored."""
        strategy = DirectCodeDeploy(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.source_path == "./test-src"
        assert strategy.entrypoint == "main.py:app"

    def test_image_uri_is_none(self) -> None:
        """Test that image_uri is always None (direct code deploy doesn't produce images)."""
        strategy = DirectCodeDeploy(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.image_uri is None

    def test_package_uri_is_none_before_launch(self) -> None:
        """Test that package_uri is None before launch."""
        strategy = DirectCodeDeploy(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.package_uri is None

    @patch("shutil.which")
    def test_validate_prerequisites_with_zip(self, mock_which: MagicMock) -> None:
        """Test validate_prerequisites passes with zip available."""
        mock_which.return_value = "/usr/bin/zip"

        strategy = DirectCodeDeploy(source_path="./test-src", entrypoint="main.py:app")
        strategy.validate_prerequisites()  # Should not raise

    @patch("shutil.which")
    def test_validate_prerequisites_without_zip(self, mock_which: MagicMock) -> None:
        """Test validate_prerequisites fails without zip."""
        mock_which.return_value = None

        strategy = DirectCodeDeploy(source_path="./test-src", entrypoint="main.py:app")
        with pytest.raises(RuntimeError, match="zip utility not found"):
            strategy.validate_prerequisites()

    def test_create_code_package(self) -> None:
        """Test _create_code_package creates proper zip."""
        strategy = DirectCodeDeploy(source_path="./test-src", entrypoint="main.py:app")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test source files
            source_dir = Path(temp_dir) / "source"
            source_dir.mkdir()
            (source_dir / "main.py").write_text("print('hello')")
            (source_dir / "requirements.txt").write_text("boto3")

            # Create package
            output_path = Path(temp_dir) / "output.zip"
            strategy._create_code_package(str(source_dir), str(output_path))

            assert output_path.exists()


class TestBuildAbstractClass:
    """Tests for Build abstract class."""

    def test_cannot_instantiate(self) -> None:
        """Test that Build cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Build()  # type: ignore

    def test_subclass_must_implement_methods(self) -> None:
        """Test that subclass must implement abstract methods."""

        class IncompleteBuild(Build):
            pass

        with pytest.raises(TypeError):
            IncompleteBuild()  # type: ignore


class TestAgentWithBuildStrategy:
    """Tests for Agent integration with Build strategies."""

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_agent_with_ecr_prebuilt(self, mock_boto3: MagicMock) -> None:
        """Test Agent with ECR pre-built image."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        from bedrock_agentcore.runtime import Agent

        build = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(
            name="test-agent",
            build=build,
        )

        assert agent.build_strategy is build
        assert agent.build_strategy.strategy_name == "ecr"
        assert agent.image_uri == "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_agent_with_ecr_codebuild(self, mock_boto3: MagicMock) -> None:
        """Test Agent with ECR CodeBuild."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        from bedrock_agentcore.runtime import Agent

        build = ECR(source_path="./test-src", entrypoint="main.py:app")
        agent = Agent(
            name="test-agent",
            build=build,
        )

        assert agent.build_strategy is build
        assert agent.build_strategy.strategy_name == "ecr"
        assert agent.image_uri is None  # Not yet built

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_agent_with_direct_code_deploy(self, mock_boto3: MagicMock) -> None:
        """Test Agent accepts DirectCodeDeploy."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        from bedrock_agentcore.runtime import Agent

        build = DirectCodeDeploy(source_path="./test-src", entrypoint="main.py:app")
        agent = Agent(
            name="test-agent",
            build=build,
        )

        assert agent.build_strategy is build
        assert agent.build_strategy.strategy_name == "direct_code_deploy"
