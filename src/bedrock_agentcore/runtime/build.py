"""Build strategies for Bedrock AgentCore agent deployments.

This module provides an abstract Build class and concrete implementations
for different build/deployment strategies:

- CodeBuildStrategy: Builds ARM64 container images using AWS CodeBuild
- LocalBuildStrategy: Builds container images locally using Docker/Finch/Podman
- DirectCodeDeployStrategy: Packages Python code as zip for direct deployment

Example:
    from bedrock_agentcore.runtime import Agent
    from bedrock_agentcore.runtime.build import CodeBuildStrategy, LocalBuildStrategy

    # Use CodeBuild (default for cloud deployments)
    agent = Agent(
        name="my-agent",
        source_path="./agent-src",
        entrypoint="main.py:app",
        build=CodeBuildStrategy(),
    )

    # Use local Docker build
    agent = Agent(
        name="my-agent",
        source_path="./agent-src",
        entrypoint="main.py:app",
        build=LocalBuildStrategy(),
    )
"""

import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Build(ABC):
    """Abstract base class for build strategies.

    Subclasses implement different methods for building and packaging
    agent code for deployment to Bedrock AgentCore.
    """

    @abstractmethod
    def build(
        self,
        source_path: str,
        agent_name: str,
        entrypoint: str,
        region_name: Optional[str] = None,
        tag: str = "latest",
        max_wait: int = 600,
    ) -> Dict[str, Any]:
        """Build and package the agent code.

        Args:
            source_path: Path to agent source code
            agent_name: Name of the agent
            entrypoint: Entry point (e.g., "main.py:app")
            region_name: AWS region name
            tag: Image/version tag
            max_wait: Maximum seconds to wait for build

        Returns:
            Dictionary with build results including:
                - imageUri or packageUri depending on strategy
                - status: Build status
                - Additional strategy-specific fields
        """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this build strategy."""
        pass

    def validate_prerequisites(self) -> None:
        """Validate that prerequisites for this build strategy are met.

        Raises:
            RuntimeError: If prerequisites are not met
        """
        pass


class CodeBuildStrategy(Build):
    """Build strategy using AWS CodeBuild for ARM64 container images.

    This is the recommended strategy for cloud deployments as it:
    - Builds ARM64 images optimized for Bedrock AgentCore
    - Doesn't require local Docker installation
    - Handles ECR repository creation automatically
    - Creates IAM roles automatically

    Example:
        build = CodeBuildStrategy()
        result = build.build(
            source_path="./my-agent",
            agent_name="my-agent",
            entrypoint="agent.py:app",
        )
        print(result["imageUri"])
    """

    @property
    def strategy_name(self) -> str:
        return "codebuild"

    def build(
        self,
        source_path: str,
        agent_name: str,
        entrypoint: str,
        region_name: Optional[str] = None,
        tag: str = "latest",
        max_wait: int = 600,
    ) -> Dict[str, Any]:
        """Build ARM64 container image using AWS CodeBuild.

        Args:
            source_path: Path to agent source code
            agent_name: Name of the agent
            entrypoint: Entry point (e.g., "main.py:app")
            region_name: AWS region name
            tag: Image tag
            max_wait: Maximum seconds to wait for build

        Returns:
            Dictionary with:
                - imageUri: ECR image URI
                - buildId: CodeBuild build ID
                - status: "SUCCEEDED"
        """
        from .builder import build_and_push

        return build_and_push(
            source_path=source_path,
            agent_name=agent_name,
            entrypoint=entrypoint,
            region_name=region_name,
            tag=tag,
            wait=True,
            max_wait=max_wait,
        )


class LocalBuildStrategy(Build):
    """Build strategy using local container runtime (Docker/Finch/Podman).

    This strategy builds container images locally and pushes to ECR.
    Useful for development and testing when you have Docker installed.

    Example:
        build = LocalBuildStrategy()
        result = build.build(
            source_path="./my-agent",
            agent_name="my-agent",
            entrypoint="agent.py:app",
        )
        print(result["imageUri"])
    """

    def __init__(self, runtime: Optional[str] = None):
        """Initialize local build strategy.

        Args:
            runtime: Container runtime to use ("docker", "finch", "podman").
                    If None, auto-detects available runtime.
        """
        self._runtime = runtime
        self._detected_runtime: Optional[str] = None

    @property
    def strategy_name(self) -> str:
        return "local"

    @property
    def runtime(self) -> str:
        """Get the container runtime to use."""
        if self._detected_runtime:
            return self._detected_runtime

        if self._runtime:
            self._detected_runtime = self._runtime
            return self._runtime

        # Auto-detect available runtime
        for rt in ["docker", "finch", "podman"]:
            if shutil.which(rt):
                self._detected_runtime = rt
                logger.info("Detected container runtime: %s", rt)
                return rt

        raise RuntimeError(
            "No container runtime found. Install Docker, Finch, or Podman."
        )

    def validate_prerequisites(self) -> None:
        """Validate that a container runtime is available."""
        _ = self.runtime  # Will raise if not found

    def build(
        self,
        source_path: str,
        agent_name: str,
        entrypoint: str,
        region_name: Optional[str] = None,
        tag: str = "latest",
        max_wait: int = 600,
    ) -> Dict[str, Any]:
        """Build container image locally and push to ECR.

        Args:
            source_path: Path to agent source code
            agent_name: Name of the agent
            entrypoint: Entry point (e.g., "main.py:app")
            region_name: AWS region name
            tag: Image tag
            max_wait: Maximum seconds to wait (unused for local builds)

        Returns:
            Dictionary with:
                - imageUri: ECR image URI
                - status: "SUCCEEDED"
        """
        import boto3

        from .builder import generate_dockerfile
        from .ecr import ensure_ecr_repository

        source_path = os.path.abspath(source_path)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path not found: {source_path}")

        region = region_name or boto3.Session().region_name or "us-west-2"
        sts_client = boto3.client("sts")
        account_id = sts_client.get_caller_identity()["Account"]

        # Ensure ECR repository
        ecr_repository = f"bedrock-agentcore/{agent_name}"
        ecr_result = ensure_ecr_repository(ecr_repository, region_name)
        ecr_repository_uri = ecr_result["repositoryUri"]
        full_image_uri = f"{ecr_repository_uri}:{tag}"

        # Generate Dockerfile if not present
        dockerfile_path = os.path.join(source_path, "Dockerfile")
        if not os.path.exists(dockerfile_path):
            logger.info("No Dockerfile found, generating one...")
            generate_dockerfile(source_path, entrypoint)

        runtime = self.runtime

        # Build image locally
        logger.info("Building image locally using %s...", runtime)
        build_cmd = [runtime, "build", "-t", full_image_uri, source_path]
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Local build failed: {result.stderr}")

        # Login to ECR
        logger.info("Logging in to ECR...")
        ecr_registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
        login_password_cmd = ["aws", "ecr", "get-login-password", "--region", region]
        password_result = subprocess.run(login_password_cmd, capture_output=True, text=True)
        if password_result.returncode != 0:
            raise RuntimeError(f"Failed to get ECR login: {password_result.stderr}")

        login_cmd = [
            runtime, "login",
            "--username", "AWS",
            "--password-stdin",
            ecr_registry,
        ]
        login_result = subprocess.run(
            login_cmd,
            input=password_result.stdout,
            capture_output=True,
            text=True,
        )
        if login_result.returncode != 0:
            raise RuntimeError(f"ECR login failed: {login_result.stderr}")

        # Push image
        logger.info("Pushing image to ECR...")
        push_cmd = [runtime, "push", full_image_uri]
        push_result = subprocess.run(push_cmd, capture_output=True, text=True)
        if push_result.returncode != 0:
            raise RuntimeError(f"Push failed: {push_result.stderr}")

        logger.info("Local build complete. Image URI: %s", full_image_uri)
        return {
            "imageUri": full_image_uri,
            "status": "SUCCEEDED",
            "runtime": runtime,
        }


class DirectCodeDeployStrategy(Build):
    """Build strategy for direct Python code deployment without containerization.

    This strategy packages Python code as a zip file and uploads to S3
    for direct deployment to Bedrock AgentCore. No container build required.

    Requires:
    - Python source code with pyproject.toml or requirements.txt
    - zip utility available

    Example:
        build = DirectCodeDeployStrategy(s3_bucket="my-bucket")
        result = build.build(
            source_path="./my-agent",
            agent_name="my-agent",
            entrypoint="agent.py:app",
        )
        print(result["packageUri"])
    """

    def __init__(
        self,
        s3_bucket: Optional[str] = None,
        auto_create_bucket: bool = True,
    ):
        """Initialize direct code deploy strategy.

        Args:
            s3_bucket: S3 bucket for code packages. If None, auto-generates.
            auto_create_bucket: Create bucket if it doesn't exist
        """
        self._s3_bucket = s3_bucket
        self._auto_create_bucket = auto_create_bucket

    @property
    def strategy_name(self) -> str:
        return "direct_code_deploy"

    def validate_prerequisites(self) -> None:
        """Validate that zip utility is available."""
        if not shutil.which("zip"):
            raise RuntimeError("zip utility not found. Install zip to use direct code deploy.")

    def build(
        self,
        source_path: str,
        agent_name: str,
        entrypoint: str,
        region_name: Optional[str] = None,
        tag: str = "latest",
        max_wait: int = 600,
    ) -> Dict[str, Any]:
        """Package Python code and upload to S3.

        Args:
            source_path: Path to agent source code
            agent_name: Name of the agent
            entrypoint: Entry point (e.g., "main.py:app")
            region_name: AWS region name
            tag: Version tag for the package
            max_wait: Maximum seconds to wait (unused)

        Returns:
            Dictionary with:
                - packageUri: S3 URI of the code package
                - s3Bucket: Bucket name
                - s3Key: Object key
                - status: "SUCCEEDED"
        """
        import boto3
        from botocore.exceptions import ClientError

        source_path = os.path.abspath(source_path)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path not found: {source_path}")

        region = region_name or boto3.Session().region_name or "us-west-2"
        sts_client = boto3.client("sts")
        account_id = sts_client.get_caller_identity()["Account"]

        # Determine bucket name
        bucket_name = self._s3_bucket
        if not bucket_name:
            bucket_name = f"bedrock-agentcore-code-{account_id}-{region}"

        # Ensure bucket exists
        s3_client = boto3.client("s3", region_name=region)
        if self._auto_create_bucket:
            try:
                s3_client.head_bucket(Bucket=bucket_name)
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    logger.info("Creating S3 bucket '%s'...", bucket_name)
                    if region == "us-east-1":
                        s3_client.create_bucket(Bucket=bucket_name)
                    else:
                        s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={"LocationConstraint": region},
                        )
                else:
                    raise

        # Create zip package
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "code.zip")
            self._create_code_package(source_path, zip_path)

            # Upload to S3
            s3_key = f"{agent_name}/{tag}/code.zip"
            logger.info("Uploading code package to s3://%s/%s", bucket_name, s3_key)
            s3_client.upload_file(zip_path, bucket_name, s3_key)

        package_uri = f"s3://{bucket_name}/{s3_key}"
        logger.info("Direct code deploy complete. Package URI: %s", package_uri)

        return {
            "packageUri": package_uri,
            "s3Bucket": bucket_name,
            "s3Key": s3_key,
            "status": "SUCCEEDED",
            "entrypoint": entrypoint,
        }

    def _create_code_package(self, source_path: str, output_path: str) -> str:
        """Create a zip package of the source code.

        Args:
            source_path: Path to source code
            output_path: Path for output zip file

        Returns:
            Path to created zip file
        """
        # Directories to exclude
        exclude_dirs = {
            ".git", "__pycache__", ".venv", "venv", "node_modules",
            ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build",
            "*.egg-info",
        }

        # File patterns to exclude
        exclude_patterns = {
            "*.pyc", "*.pyo", "*.pyd", ".DS_Store", "*.so",
            ".env", ".env.*", "*.log",
        }

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_path):
                # Filter directories
                dirs[:] = [d for d in dirs if d not in exclude_dirs]

                for file in files:
                    # Check exclude patterns
                    if any(self._matches_pattern(file, p) for p in exclude_patterns):
                        continue

                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_path)
                    zipf.write(file_path, arcname)

        logger.debug("Created code package: %s", output_path)
        return output_path

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches a glob pattern."""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)


# Convenience factory functions
def codebuild() -> CodeBuildStrategy:
    """Create a CodeBuild build strategy.

    Returns:
        CodeBuildStrategy instance
    """
    return CodeBuildStrategy()


def local(runtime: Optional[str] = None) -> LocalBuildStrategy:
    """Create a local build strategy.

    Args:
        runtime: Container runtime ("docker", "finch", "podman") or None to auto-detect

    Returns:
        LocalBuildStrategy instance
    """
    return LocalBuildStrategy(runtime=runtime)


def direct_code_deploy(
    s3_bucket: Optional[str] = None,
    auto_create_bucket: bool = True,
) -> DirectCodeDeployStrategy:
    """Create a direct code deploy strategy.

    Args:
        s3_bucket: S3 bucket for code packages
        auto_create_bucket: Create bucket if it doesn't exist

    Returns:
        DirectCodeDeployStrategy instance
    """
    return DirectCodeDeployStrategy(
        s3_bucket=s3_bucket,
        auto_create_bucket=auto_create_bucket,
    )
