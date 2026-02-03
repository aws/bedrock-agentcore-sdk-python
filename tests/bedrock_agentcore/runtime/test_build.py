"""Tests for Build strategies."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.runtime.build import (
    Build,
    CodeBuildStrategy,
    DirectCodeDeployStrategy,
    LocalBuildStrategy,
    codebuild,
    direct_code_deploy,
    local,
)


class TestCodeBuildStrategy:
    """Tests for CodeBuildStrategy."""

    def test_strategy_name(self) -> None:
        """Test that strategy name is correct."""
        strategy = CodeBuildStrategy()
        assert strategy.strategy_name == "codebuild"

    @patch("bedrock_agentcore.runtime.builder.build_and_push")
    def test_build_calls_builder(self, mock_build_and_push: MagicMock) -> None:
        """Test that build delegates to builder module."""
        mock_build_and_push.return_value = {
            "imageUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
            "buildId": "build-123",
            "status": "SUCCEEDED",
        }

        strategy = CodeBuildStrategy()
        result = strategy.build(
            source_path="/tmp/test-agent",
            agent_name="test-agent",
            entrypoint="main.py:app",
            region_name="us-west-2",
        )

        mock_build_and_push.assert_called_once()
        assert result["status"] == "SUCCEEDED"
        assert "imageUri" in result

    def test_factory_function(self) -> None:
        """Test codebuild() factory function."""
        strategy = codebuild()
        assert isinstance(strategy, CodeBuildStrategy)


class TestLocalBuildStrategy:
    """Tests for LocalBuildStrategy."""

    def test_strategy_name(self) -> None:
        """Test that strategy name is correct."""
        strategy = LocalBuildStrategy()
        assert strategy.strategy_name == "local"

    def test_explicit_runtime(self) -> None:
        """Test explicit runtime specification."""
        strategy = LocalBuildStrategy(runtime="docker")
        assert strategy._runtime == "docker"

    @patch("shutil.which")
    def test_auto_detect_docker(self, mock_which: MagicMock) -> None:
        """Test auto-detection of Docker runtime."""
        mock_which.side_effect = lambda x: "/usr/bin/docker" if x == "docker" else None

        strategy = LocalBuildStrategy()
        assert strategy.runtime == "docker"

    @patch("shutil.which")
    def test_auto_detect_finch(self, mock_which: MagicMock) -> None:
        """Test auto-detection of Finch runtime."""
        mock_which.side_effect = lambda x: "/usr/local/bin/finch" if x == "finch" else None

        strategy = LocalBuildStrategy()
        assert strategy.runtime == "finch"

    @patch("shutil.which")
    def test_no_runtime_raises(self, mock_which: MagicMock) -> None:
        """Test that missing runtime raises error."""
        mock_which.return_value = None

        strategy = LocalBuildStrategy()
        with pytest.raises(RuntimeError, match="No container runtime found"):
            _ = strategy.runtime

    @patch("shutil.which")
    def test_validate_prerequisites(self, mock_which: MagicMock) -> None:
        """Test validate_prerequisites checks for runtime."""
        mock_which.return_value = None

        strategy = LocalBuildStrategy()
        with pytest.raises(RuntimeError, match="No container runtime found"):
            strategy.validate_prerequisites()

    def test_factory_function(self) -> None:
        """Test local() factory function."""
        strategy = local(runtime="docker")
        assert isinstance(strategy, LocalBuildStrategy)
        assert strategy._runtime == "docker"


class TestDirectCodeDeployStrategy:
    """Tests for DirectCodeDeployStrategy."""

    def test_strategy_name(self) -> None:
        """Test that strategy name is correct."""
        strategy = DirectCodeDeployStrategy()
        assert strategy.strategy_name == "direct_code_deploy"

    def test_custom_bucket(self) -> None:
        """Test custom S3 bucket specification."""
        strategy = DirectCodeDeployStrategy(s3_bucket="my-bucket")
        assert strategy._s3_bucket == "my-bucket"

    @patch("shutil.which")
    def test_validate_prerequisites_with_zip(self, mock_which: MagicMock) -> None:
        """Test validate_prerequisites passes with zip available."""
        mock_which.return_value = "/usr/bin/zip"

        strategy = DirectCodeDeployStrategy()
        strategy.validate_prerequisites()  # Should not raise

    @patch("shutil.which")
    def test_validate_prerequisites_without_zip(self, mock_which: MagicMock) -> None:
        """Test validate_prerequisites fails without zip."""
        mock_which.return_value = None

        strategy = DirectCodeDeployStrategy()
        with pytest.raises(RuntimeError, match="zip utility not found"):
            strategy.validate_prerequisites()

    def test_factory_function(self) -> None:
        """Test direct_code_deploy() factory function."""
        strategy = direct_code_deploy(s3_bucket="my-bucket")
        assert isinstance(strategy, DirectCodeDeployStrategy)
        assert strategy._s3_bucket == "my-bucket"

    def test_create_code_package(self) -> None:
        """Test _create_code_package creates proper zip."""
        strategy = DirectCodeDeployStrategy()

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
    def test_agent_with_codebuild_strategy(self, mock_boto3: MagicMock) -> None:
        """Test Agent accepts CodeBuildStrategy."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        from bedrock_agentcore.runtime import Agent

        strategy = CodeBuildStrategy()
        agent = Agent(
            name="test-agent",
            source_path="./test-src",
            entrypoint="main.py:app",
            build=strategy,
        )

        assert agent.build_strategy is strategy
        assert agent.build_strategy.strategy_name == "codebuild"

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_agent_with_local_strategy(self, mock_boto3: MagicMock) -> None:
        """Test Agent accepts LocalBuildStrategy."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        from bedrock_agentcore.runtime import Agent

        strategy = LocalBuildStrategy(runtime="docker")
        agent = Agent(
            name="test-agent",
            source_path="./test-src",
            entrypoint="main.py:app",
            build=strategy,
        )

        assert agent.build_strategy is strategy
        assert agent.build_strategy.strategy_name == "local"

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_agent_defaults_to_codebuild(self, mock_boto3: MagicMock) -> None:
        """Test Agent defaults to CodeBuildStrategy when source_path provided."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        from bedrock_agentcore.runtime import Agent

        agent = Agent(
            name="test-agent",
            source_path="./test-src",
            entrypoint="main.py:app",
        )

        assert agent.build_strategy is not None
        assert isinstance(agent.build_strategy, CodeBuildStrategy)

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_agent_with_image_uri_no_build_strategy(self, mock_boto3: MagicMock) -> None:
        """Test Agent with image_uri has no build strategy."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        from bedrock_agentcore.runtime import Agent

        agent = Agent(
            name="test-agent",
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
        )

        assert agent.build_strategy is None
