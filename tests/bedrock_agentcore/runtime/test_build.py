"""Tests for Build strategies."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.runtime.build import (
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


class TestPrebuiltImage:
    """Tests for PrebuiltImage."""

    def test_strategy_name(self) -> None:
        """Test that strategy name is correct."""
        strategy = PrebuiltImage(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        assert strategy.strategy_name == "prebuilt"

    def test_image_uri(self) -> None:
        """Test that image_uri is returned correctly."""
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"
        strategy = PrebuiltImage(image_uri=image_uri)
        assert strategy.image_uri == image_uri

    def test_build_returns_image_uri(self) -> None:
        """Test that build() returns the image URI."""
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"
        strategy = PrebuiltImage(image_uri=image_uri)
        result = strategy.build(agent_name="test-agent")
        assert result["imageUri"] == image_uri
        assert result["status"] == "READY"

    def test_factory_function(self) -> None:
        """Test prebuilt() factory function."""
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"
        strategy = prebuilt(image_uri=image_uri)
        assert isinstance(strategy, PrebuiltImage)
        assert strategy.image_uri == image_uri


class TestCodeBuild:
    """Tests for CodeBuild (CodeBuildStrategy)."""

    def test_strategy_name(self) -> None:
        """Test that strategy name is correct."""
        strategy = CodeBuild(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.strategy_name == "codebuild"

    def test_source_path_and_entrypoint(self) -> None:
        """Test source_path and entrypoint are stored."""
        strategy = CodeBuild(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.source_path == "./test-src"
        assert strategy.entrypoint == "main.py:app"

    def test_image_uri_is_none_before_build(self) -> None:
        """Test image_uri is None before build."""
        strategy = CodeBuild(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.image_uri is None

    @patch("bedrock_agentcore.runtime.builder.build_and_push")
    def test_build_calls_builder(self, mock_build_and_push: MagicMock) -> None:
        """Test that build delegates to builder module."""
        mock_build_and_push.return_value = {
            "imageUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
            "buildId": "build-123",
            "status": "SUCCEEDED",
        }

        strategy = CodeBuild(source_path="/tmp/test-agent", entrypoint="main.py:app")
        result = strategy.build(
            agent_name="test-agent",
            region_name="us-west-2",
        )

        mock_build_and_push.assert_called_once()
        assert result["status"] == "SUCCEEDED"
        assert "imageUri" in result
        assert strategy.image_uri == "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"

    def test_factory_function(self) -> None:
        """Test codebuild() factory function."""
        strategy = codebuild(source_path="./test-src", entrypoint="main.py:app")
        assert isinstance(strategy, CodeBuild)
        assert strategy.source_path == "./test-src"
        assert strategy.entrypoint == "main.py:app"

    def test_backwards_compatibility_alias(self) -> None:
        """Test CodeBuildStrategy alias works."""
        strategy = CodeBuildStrategy(source_path="./test-src", entrypoint="main.py:app")
        assert isinstance(strategy, CodeBuild)


class TestLocalBuild:
    """Tests for LocalBuild (LocalBuildStrategy)."""

    def test_strategy_name(self) -> None:
        """Test that strategy name is correct."""
        strategy = LocalBuild(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.strategy_name == "local"

    def test_explicit_runtime(self) -> None:
        """Test explicit runtime specification."""
        strategy = LocalBuild(source_path="./test-src", entrypoint="main.py:app", runtime="docker")
        assert strategy._runtime == "docker"

    def test_source_path_and_entrypoint(self) -> None:
        """Test source_path and entrypoint are stored."""
        strategy = LocalBuild(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.source_path == "./test-src"
        assert strategy.entrypoint == "main.py:app"

    @patch("shutil.which")
    def test_auto_detect_docker(self, mock_which: MagicMock) -> None:
        """Test auto-detection of Docker runtime."""
        mock_which.side_effect = lambda x: "/usr/bin/docker" if x == "docker" else None

        strategy = LocalBuild(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.runtime == "docker"

    @patch("shutil.which")
    def test_auto_detect_finch(self, mock_which: MagicMock) -> None:
        """Test auto-detection of Finch runtime."""
        mock_which.side_effect = lambda x: "/usr/local/bin/finch" if x == "finch" else None

        strategy = LocalBuild(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.runtime == "finch"

    @patch("shutil.which")
    def test_no_runtime_raises(self, mock_which: MagicMock) -> None:
        """Test that missing runtime raises error."""
        mock_which.return_value = None

        strategy = LocalBuild(source_path="./test-src", entrypoint="main.py:app")
        with pytest.raises(RuntimeError, match="No container runtime found"):
            _ = strategy.runtime

    @patch("shutil.which")
    def test_validate_prerequisites(self, mock_which: MagicMock) -> None:
        """Test validate_prerequisites checks for runtime."""
        mock_which.return_value = None

        strategy = LocalBuild(source_path="./test-src", entrypoint="main.py:app")
        with pytest.raises(RuntimeError, match="No container runtime found"):
            strategy.validate_prerequisites()

    def test_factory_function(self) -> None:
        """Test local() factory function."""
        strategy = local(source_path="./test-src", entrypoint="main.py:app", runtime="docker")
        assert isinstance(strategy, LocalBuild)
        assert strategy._runtime == "docker"
        assert strategy.source_path == "./test-src"
        assert strategy.entrypoint == "main.py:app"

    def test_backwards_compatibility_alias(self) -> None:
        """Test LocalBuildStrategy alias works."""
        strategy = LocalBuildStrategy(source_path="./test-src", entrypoint="main.py:app")
        assert isinstance(strategy, LocalBuild)


class TestDirectCodeDeploy:
    """Tests for DirectCodeDeploy (DirectCodeDeployStrategy)."""

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
        """Test that image_uri is None (direct code deploy doesn't produce images)."""
        strategy = DirectCodeDeploy(source_path="./test-src", entrypoint="main.py:app")
        assert strategy.image_uri is None

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

    def test_factory_function(self) -> None:
        """Test direct_code_deploy() factory function."""
        strategy = direct_code_deploy(source_path="./test-src", entrypoint="main.py:app", s3_bucket="my-bucket")
        assert isinstance(strategy, DirectCodeDeploy)
        assert strategy._s3_bucket == "my-bucket"
        assert strategy.source_path == "./test-src"
        assert strategy.entrypoint == "main.py:app"

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

    def test_backwards_compatibility_alias(self) -> None:
        """Test DirectCodeDeployStrategy alias works."""
        strategy = DirectCodeDeployStrategy(source_path="./test-src", entrypoint="main.py:app")
        assert isinstance(strategy, DirectCodeDeploy)


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
    def test_agent_with_prebuilt_image(self, mock_boto3: MagicMock) -> None:
        """Test Agent with PrebuiltImage."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        from bedrock_agentcore.runtime import Agent

        build = PrebuiltImage(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(
            name="test-agent",
            build=build,
        )

        assert agent.build_strategy is build
        assert agent.build_strategy.strategy_name == "prebuilt"
        assert agent.image_uri == "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_agent_with_codebuild_strategy(self, mock_boto3: MagicMock) -> None:
        """Test Agent accepts CodeBuild."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        from bedrock_agentcore.runtime import Agent

        build = CodeBuild(source_path="./test-src", entrypoint="main.py:app")
        agent = Agent(
            name="test-agent",
            build=build,
        )

        assert agent.build_strategy is build
        assert agent.build_strategy.strategy_name == "codebuild"
        assert agent.image_uri is None  # Not yet built

    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_agent_with_local_strategy(self, mock_boto3: MagicMock) -> None:
        """Test Agent accepts LocalBuild."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        from bedrock_agentcore.runtime import Agent

        build = LocalBuild(source_path="./test-src", entrypoint="main.py:app", runtime="docker")
        agent = Agent(
            name="test-agent",
            build=build,
        )

        assert agent.build_strategy is build
        assert agent.build_strategy.strategy_name == "local"

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
