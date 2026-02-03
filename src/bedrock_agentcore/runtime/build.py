"""Build strategies for Bedrock AgentCore agent deployments.

This module provides an abstract Build class and concrete implementations
for different build/deployment strategies:

- PrebuiltImage: Use a pre-built container image from ECR
- CodeBuild: Build ARM64 container images using AWS CodeBuild
- LocalBuild: Build container images locally using Docker/Finch/Podman
- DirectCodeDeploy: Package Python code as zip for direct deployment

Example:
    from bedrock_agentcore.runtime import Agent
    from bedrock_agentcore.runtime.build import PrebuiltImage, CodeBuild, LocalBuild

    # Pre-built image
    agent = Agent(
        name="my-agent",
        build=PrebuiltImage(image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-agent:latest"),
    )

    # Build from source with CodeBuild
    agent = Agent(
        name="my-agent",
        build=CodeBuild(source_path="./agent-src", entrypoint="main.py:app"),
    )

    # Build from source with local Docker
    agent = Agent(
        name="my-agent",
        build=LocalBuild(source_path="./agent-src", entrypoint="main.py:app"),
    )
"""

import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Build(ABC):
    """Abstract base class for build strategies.

    Subclasses implement different methods for building and packaging
    agent code for deployment to Bedrock AgentCore.
    """

    @abstractmethod
    def build(
        self,
        agent_name: str,
        region_name: Optional[str] = None,
        tag: str = "latest",
        max_wait: int = 600,
    ) -> Dict[str, Any]:
        """Build and package the agent code.

        Args:
            agent_name: Name of the agent
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

    @property
    @abstractmethod
    def image_uri(self) -> Optional[str]:
        """Return the image URI if available (after build or for pre-built)."""
        pass

    def validate_prerequisites(self) -> None:  # noqa: B027
        """Validate that prerequisites for this build strategy are met.

        This is a hook that subclasses can optionally override. The default
        implementation does nothing (no prerequisites required).

        Raises:
            RuntimeError: If prerequisites are not met
        """
        pass


class PrebuiltImage(Build):
    """Use a pre-built container image from ECR.

    This is the simplest strategy - just reference an existing image.

    Example:
        build = PrebuiltImage(
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-agent:latest"
        )
    """

    def __init__(self, image_uri: str):
        """Initialize with a pre-built image URI.

        Args:
            image_uri: ECR image URI
        """
        self._image_uri = image_uri

    @property
    def strategy_name(self) -> str:
        """Return the strategy name."""
        return "prebuilt"

    @property
    def image_uri(self) -> Optional[str]:
        """Return the image URI."""
        return self._image_uri

    def build(
        self,
        agent_name: str,
        region_name: Optional[str] = None,
        tag: str = "latest",
        max_wait: int = 600,
    ) -> Dict[str, Any]:
        """No-op for pre-built images - just return the image URI.

        Returns:
            Dictionary with imageUri and status
        """
        logger.info("Using pre-built image: %s", self._image_uri)
        return {
            "imageUri": self._image_uri,
            "status": "READY",
        }


class CodeBuild(Build):
    """Build ARM64 container images using AWS CodeBuild.

    This is the recommended strategy for cloud deployments as it:
    - Builds ARM64 images optimized for Bedrock AgentCore
    - Doesn't require local Docker installation
    - Handles ECR repository creation automatically
    - Creates IAM roles automatically

    Example:
        build = CodeBuild(
            source_path="./my-agent",
            entrypoint="agent.py:app",
        )
    """

    def __init__(self, source_path: str, entrypoint: str):
        """Initialize CodeBuild strategy.

        Args:
            source_path: Path to agent source code
            entrypoint: Entry point (e.g., "main.py:app")
        """
        self._source_path = source_path
        self._entrypoint = entrypoint
        self._built_image_uri: Optional[str] = None

    @property
    def strategy_name(self) -> str:
        """Return the strategy name."""
        return "codebuild"

    @property
    def source_path(self) -> str:
        """Return the source path."""
        return self._source_path

    @property
    def entrypoint(self) -> str:
        """Return the entrypoint."""
        return self._entrypoint

    @property
    def image_uri(self) -> Optional[str]:
        """Return the image URI (None until build completes)."""
        return self._built_image_uri

    def build(
        self,
        agent_name: str,
        region_name: Optional[str] = None,
        tag: str = "latest",
        max_wait: int = 600,
    ) -> Dict[str, Any]:
        """Build ARM64 container image using AWS CodeBuild.

        Args:
            agent_name: Name of the agent
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

        result = build_and_push(
            source_path=self._source_path,
            agent_name=agent_name,
            entrypoint=self._entrypoint,
            region_name=region_name,
            tag=tag,
            wait=True,
            max_wait=max_wait,
        )

        self._built_image_uri = result.get("imageUri")
        return result


class LocalBuild(Build):
    """Build container images locally using Docker/Finch/Podman.

    This strategy builds container images locally and pushes to ECR.
    Useful for development and testing when you have Docker installed.

    Example:
        build = LocalBuild(
            source_path="./my-agent",
            entrypoint="agent.py:app",
        )
    """

    def __init__(
        self,
        source_path: str,
        entrypoint: str,
        runtime: Optional[str] = None,
    ):
        """Initialize local build strategy.

        Args:
            source_path: Path to agent source code
            entrypoint: Entry point (e.g., "main.py:app")
            runtime: Container runtime ("docker", "finch", "podman").
                    If None, auto-detects available runtime.
        """
        self._source_path = source_path
        self._entrypoint = entrypoint
        self._runtime = runtime
        self._detected_runtime: Optional[str] = None
        self._built_image_uri: Optional[str] = None

    @property
    def strategy_name(self) -> str:
        """Return the strategy name."""
        return "local"

    @property
    def source_path(self) -> str:
        """Return the source path."""
        return self._source_path

    @property
    def entrypoint(self) -> str:
        """Return the entrypoint."""
        return self._entrypoint

    @property
    def image_uri(self) -> Optional[str]:
        """Return the image URI (None until build completes)."""
        return self._built_image_uri

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
        agent_name: str,
        region_name: Optional[str] = None,
        tag: str = "latest",
        max_wait: int = 600,
    ) -> Dict[str, Any]:
        """Build container image locally and push to ECR.

        Args:
            agent_name: Name of the agent
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

        source_path = os.path.abspath(self._source_path)
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
            generate_dockerfile(source_path, self._entrypoint)

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

        self._built_image_uri = full_image_uri
        logger.info("Local build complete. Image URI: %s", full_image_uri)
        return {
            "imageUri": full_image_uri,
            "status": "SUCCEEDED",
            "runtime": runtime,
        }


class DirectCodeDeploy(Build):
    """Package Python code as zip for direct deployment without containerization.

    This strategy packages Python code as a zip file and uploads to S3
    for direct deployment to Bedrock AgentCore. No container build required.

    Example:
        build = DirectCodeDeploy(
            source_path="./my-agent",
            entrypoint="agent.py:app",
        )
    """

    def __init__(
        self,
        source_path: str,
        entrypoint: str,
        s3_bucket: Optional[str] = None,
        auto_create_bucket: bool = True,
    ):
        """Initialize direct code deploy strategy.

        Args:
            source_path: Path to agent source code
            entrypoint: Entry point (e.g., "main.py:app")
            s3_bucket: S3 bucket for code packages. If None, auto-generates.
            auto_create_bucket: Create bucket if it doesn't exist
        """
        self._source_path = source_path
        self._entrypoint = entrypoint
        self._s3_bucket = s3_bucket
        self._auto_create_bucket = auto_create_bucket
        self._package_uri: Optional[str] = None

    @property
    def strategy_name(self) -> str:
        """Return the strategy name."""
        return "direct_code_deploy"

    @property
    def source_path(self) -> str:
        """Return the source path."""
        return self._source_path

    @property
    def entrypoint(self) -> str:
        """Return the entrypoint."""
        return self._entrypoint

    @property
    def image_uri(self) -> Optional[str]:
        """Return the image URI (always None for direct code deploy)."""
        return None

    @property
    def package_uri(self) -> Optional[str]:
        """Return the S3 package URI after build."""
        return self._package_uri

    def validate_prerequisites(self) -> None:
        """Validate that zip utility is available."""
        if not shutil.which("zip"):
            raise RuntimeError("zip utility not found. Install zip to use direct code deploy.")

    def build(
        self,
        agent_name: str,
        region_name: Optional[str] = None,
        tag: str = "latest",
        max_wait: int = 600,
    ) -> Dict[str, Any]:
        """Package Python code and upload to S3.

        Args:
            agent_name: Name of the agent
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

        source_path = os.path.abspath(self._source_path)
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

        self._package_uri = f"s3://{bucket_name}/{s3_key}"
        logger.info("Direct code deploy complete. Package URI: %s", self._package_uri)

        return {
            "packageUri": self._package_uri,
            "s3Bucket": bucket_name,
            "s3Key": s3_key,
            "status": "SUCCEEDED",
            "entrypoint": self._entrypoint,
        }

    def _create_code_package(self, source_path: str, output_path: str) -> str:
        """Create a zip package of the source code."""
        exclude_dirs = {
            ".git", "__pycache__", ".venv", "venv", "node_modules",
            ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build",
            "*.egg-info",
        }

        exclude_patterns = {
            "*.pyc", "*.pyo", "*.pyd", ".DS_Store", "*.so",
            ".env", ".env.*", "*.log",
        }

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_path):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]

                for file in files:
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


# Backwards compatibility aliases
CodeBuildStrategy = CodeBuild
LocalBuildStrategy = LocalBuild
DirectCodeDeployStrategy = DirectCodeDeploy


# Convenience factory functions
def prebuilt(image_uri: str) -> PrebuiltImage:
    """Create a pre-built image strategy.

    Args:
        image_uri: ECR image URI

    Returns:
        PrebuiltImage instance
    """
    return PrebuiltImage(image_uri=image_uri)


def codebuild(source_path: str, entrypoint: str) -> CodeBuild:
    """Create a CodeBuild strategy.

    Args:
        source_path: Path to agent source code
        entrypoint: Entry point (e.g., "main.py:app")

    Returns:
        CodeBuild instance
    """
    return CodeBuild(source_path=source_path, entrypoint=entrypoint)


def local(
    source_path: str,
    entrypoint: str,
    runtime: Optional[str] = None,
) -> LocalBuild:
    """Create a local build strategy.

    Args:
        source_path: Path to agent source code
        entrypoint: Entry point (e.g., "main.py:app")
        runtime: Container runtime ("docker", "finch", "podman") or None to auto-detect

    Returns:
        LocalBuild instance
    """
    return LocalBuild(source_path=source_path, entrypoint=entrypoint, runtime=runtime)


def direct_code_deploy(
    source_path: str,
    entrypoint: str,
    s3_bucket: Optional[str] = None,
    auto_create_bucket: bool = True,
) -> DirectCodeDeploy:
    """Create a direct code deploy strategy.

    Args:
        source_path: Path to agent source code
        entrypoint: Entry point (e.g., "main.py:app")
        s3_bucket: S3 bucket for code packages
        auto_create_bucket: Create bucket if it doesn't exist

    Returns:
        DirectCodeDeploy instance
    """
    return DirectCodeDeploy(
        source_path=source_path,
        entrypoint=entrypoint,
        s3_bucket=s3_bucket,
        auto_create_bucket=auto_create_bucket,
    )
