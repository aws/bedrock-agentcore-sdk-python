"""Tests for Docker build operations."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.runtime.builder import (
    build_and_push,
    generate_dockerfile,
    get_codebuild_client,
    get_s3_client,
)


class TestGetClients:
    """Tests for client factory functions."""

    @patch("bedrock_agentcore.runtime.builder.boto3")
    def test_get_codebuild_client(self, mock_boto3: MagicMock) -> None:
        """Test CodeBuild client creation."""
        get_codebuild_client("us-west-2")
        mock_boto3.client.assert_called_once()
        call_kwargs = mock_boto3.client.call_args
        assert call_kwargs[0][0] == "codebuild"
        assert call_kwargs[1]["region_name"] == "us-west-2"

    @patch("bedrock_agentcore.runtime.builder.boto3")
    def test_get_s3_client(self, mock_boto3: MagicMock) -> None:
        """Test S3 client creation."""
        get_s3_client("us-west-2")
        mock_boto3.client.assert_called_once()
        call_kwargs = mock_boto3.client.call_args
        assert call_kwargs[0][0] == "s3"
        assert call_kwargs[1]["region_name"] == "us-west-2"


class TestGenerateDockerfile:
    """Tests for generate_dockerfile."""

    def test_generates_dockerfile(self) -> None:
        """Test Dockerfile generation with entrypoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_dockerfile(temp_dir, "agent.py:app")

            # Verify file was created
            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            assert os.path.exists(dockerfile_path)
            assert result == dockerfile_path

            # Verify content
            with open(dockerfile_path, "r") as f:
                content = f.read()
            assert "FROM python:3.12-slim" in content
            assert 'CMD ["python", "-m", "agent"]' in content

    def test_generates_dockerfile_module_only(self) -> None:
        """Test Dockerfile generation with module name only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_dockerfile(temp_dir, "myagent")

            with open(result, "r") as f:
                content = f.read()
            assert 'CMD ["python", "-m", "myagent"]' in content

    def test_generates_dockerfile_with_py_extension(self) -> None:
        """Test Dockerfile generation with .py extension."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_dockerfile(temp_dir, "agent.py")

            with open(result, "r") as f:
                content = f.read()
            assert 'CMD ["python", "-m", "agent"]' in content

    def test_generates_dockerfile_custom_output(self) -> None:
        """Test Dockerfile generation with custom output path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = os.path.join(temp_dir, "custom", "Dockerfile.custom")
            os.makedirs(os.path.dirname(custom_path), exist_ok=True)

            result = generate_dockerfile(temp_dir, "agent.py:app", output_path=custom_path)

            assert result == custom_path
            assert os.path.exists(custom_path)


class TestBuildAndPush:
    """Tests for build_and_push."""

    def test_source_path_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing source."""
        with pytest.raises(FileNotFoundError, match="Source path not found"):
            build_and_push(
                source_path="/nonexistent/path",
                agent_name="test-agent",
                entrypoint="agent.py:app",
                region_name="us-west-2",
            )

    @patch("bedrock_agentcore.runtime.builder.build_image_codebuild")
    @patch("bedrock_agentcore.runtime.builder.ensure_ecr_repository")
    @patch("bedrock_agentcore.runtime.builder.generate_dockerfile")
    def test_build_and_push_success(
        self,
        mock_generate: MagicMock,
        mock_ensure_ecr: MagicMock,
        mock_codebuild: MagicMock,
    ) -> None:
        """Test successful build and push."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a minimal source directory
            Path(temp_dir).joinpath("agent.py").touch()

            mock_ensure_ecr.return_value = {
                "repositoryUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/bedrock-agentcore/test-agent",
                "created": True,
            }

            mock_codebuild.return_value = {
                "buildId": "build-123",
                "imageUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/bedrock-agentcore/test-agent:latest",
                "status": "SUCCEEDED",
            }

            result = build_and_push(
                source_path=temp_dir,
                agent_name="test-agent",
                entrypoint="agent.py:app",
                region_name="us-west-2",
            )

            assert result["status"] == "SUCCEEDED"
            assert "imageUri" in result

            # Verify ECR repository was created with auto-generated name
            mock_ensure_ecr.assert_called_once_with(
                "bedrock-agentcore/test-agent", "us-west-2"
            )

    @patch("bedrock_agentcore.runtime.builder.build_image_codebuild")
    @patch("bedrock_agentcore.runtime.builder.ensure_ecr_repository")
    def test_build_and_push_with_existing_dockerfile(
        self,
        mock_ensure_ecr: MagicMock,
        mock_codebuild: MagicMock,
    ) -> None:
        """Test build and push with existing Dockerfile."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create source with existing Dockerfile
            Path(temp_dir).joinpath("agent.py").touch()
            dockerfile = Path(temp_dir).joinpath("Dockerfile")
            dockerfile.write_text("FROM python:3.12\nCMD ['python', 'custom.py']")

            mock_ensure_ecr.return_value = {
                "repositoryUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/bedrock-agentcore/test-agent",
                "created": False,
            }

            mock_codebuild.return_value = {
                "buildId": "build-456",
                "imageUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/bedrock-agentcore/test-agent:v1.0",
                "status": "SUCCEEDED",
            }

            result = build_and_push(
                source_path=temp_dir,
                agent_name="test-agent",
                entrypoint="custom.py",
                region_name="us-west-2",
                tag="v1.0",
            )

            assert result["status"] == "SUCCEEDED"

            # Verify CodeBuild was called with correct tag
            mock_codebuild.assert_called_once()
            call_kwargs = mock_codebuild.call_args[1]
            assert call_kwargs["tag"] == "v1.0"

    @patch("bedrock_agentcore.runtime.builder.build_image_codebuild")
    @patch("bedrock_agentcore.runtime.builder.ensure_ecr_repository")
    def test_build_and_push_wait_false(
        self,
        mock_ensure_ecr: MagicMock,
        mock_codebuild: MagicMock,
    ) -> None:
        """Test build and push without waiting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir).joinpath("agent.py").touch()
            Path(temp_dir).joinpath("Dockerfile").touch()

            mock_ensure_ecr.return_value = {
                "repositoryUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/bedrock-agentcore/test-agent",
            }

            mock_codebuild.return_value = {
                "buildId": "build-789",
                "imageUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/bedrock-agentcore/test-agent:latest",
                "status": "IN_PROGRESS",
            }

            build_and_push(
                source_path=temp_dir,
                agent_name="test-agent",
                entrypoint="agent.py:app",
                wait=False,
            )

            # Verify wait=False was passed to CodeBuild
            call_kwargs = mock_codebuild.call_args[1]
            assert call_kwargs["wait"] is False
