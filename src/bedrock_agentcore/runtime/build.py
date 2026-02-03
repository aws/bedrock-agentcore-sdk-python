"""Build strategies for Bedrock AgentCore agent deployments.

This module provides an abstract Build class and concrete implementations
for different build/deployment strategies:

- ECR: Deploy container images to ECR (via CodeBuild or pre-built image)
- DirectCodeDeploy: Package Python code as zip for direct deployment to S3

Example:
    from bedrock_agentcore.runtime import Agent
    from bedrock_agentcore.runtime.build import ECR, DirectCodeDeploy

    # Build from source with CodeBuild and push to ECR
    agent = Agent(
        name="my-agent",
        build=ECR(source_path="./agent-src", entrypoint="main.py:app"),
    )

    # Use pre-built docker image
    agent = Agent(
        name="my-agent",
        build=ECR(image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-agent:latest"),
    )

    # Direct code deploy (no container)
    agent = Agent(
        name="my-agent",
        build=DirectCodeDeploy(source_path="./agent-src", entrypoint="main.py:app"),
    )
"""

import logging
import os
import shutil
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
        """Build the agent code locally (if applicable).

        Args:
            agent_name: Name of the agent
            region_name: AWS region name
            tag: Image/version tag
            max_wait: Maximum seconds to wait for build

        Returns:
            Dictionary with build results
        """
        pass

    @abstractmethod
    def deploy(
        self,
        agent_name: str,
        region_name: Optional[str] = None,
        tag: str = "latest",
        max_wait: int = 600,
    ) -> Dict[str, Any]:
        """Build and push the agent code to target repository.

        This builds the code (if needed) and pushes to the target
        (ECR for container strategies, S3 for direct code deploy).

        Args:
            agent_name: Name of the agent
            region_name: AWS region name
            tag: Image/version tag
            max_wait: Maximum seconds to wait for deployment

        Returns:
            Dictionary with deployment results including:
                - imageUri or packageUri depending on strategy
                - status: Deployment status
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
        """Return the image URI if available (after deploy or for pre-built)."""
        pass

    def validate_prerequisites(self) -> None:  # noqa: B027
        """Validate that prerequisites for this build strategy are met.

        This is a hook that subclasses can optionally override. The default
        implementation does nothing (no prerequisites required).

        Raises:
            RuntimeError: If prerequisites are not met
        """
        pass


class ECR(Build):
    """Deploy container images to ECR.

    This strategy supports two modes:
    1. Build from source using AWS CodeBuild (provide source_path and entrypoint)
    2. Use a pre-built docker image (provide image_uri)

    Example:
        # Build from source with CodeBuild
        build = ECR(source_path="./my-agent", entrypoint="agent.py:app")

        # Use pre-built image
        build = ECR(image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-agent:latest")
    """

    def __init__(
        self,
        source_path: Optional[str] = None,
        entrypoint: Optional[str] = None,
        image_uri: Optional[str] = None,
    ):
        """Initialize ECR build strategy.

        Provide either (source_path + entrypoint) for CodeBuild, or image_uri for pre-built.

        Args:
            source_path: Path to agent source code (for CodeBuild)
            entrypoint: Entry point e.g. "main.py:app" (for CodeBuild)
            image_uri: Pre-built ECR image URI

        Raises:
            ValueError: If neither source_path nor image_uri is provided
        """
        if image_uri:
            self._mode = "prebuilt"
            self._image_uri = image_uri
            self._source_path = None
            self._entrypoint = None
        elif source_path and entrypoint:
            self._mode = "codebuild"
            self._image_uri: Optional[str] = None
            self._source_path = source_path
            self._entrypoint = entrypoint
        else:
            raise ValueError(
                "Must provide either image_uri (pre-built) or both source_path and entrypoint (CodeBuild)"
            )

    @property
    def strategy_name(self) -> str:
        """Return the strategy name."""
        return "ecr"

    @property
    def mode(self) -> str:
        """Return the build mode ('prebuilt' or 'codebuild')."""
        return self._mode

    @property
    def source_path(self) -> Optional[str]:
        """Return the source path (None for pre-built)."""
        return self._source_path

    @property
    def entrypoint(self) -> Optional[str]:
        """Return the entrypoint (None for pre-built)."""
        return self._entrypoint

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
        """Build the container image (CodeBuild mode only).

        For pre-built images, this is a no-op.

        Args:
            agent_name: Name of the agent
            region_name: AWS region name
            tag: Image tag
            max_wait: Maximum seconds to wait for build

        Returns:
            Dictionary with build results
        """
        if self._mode == "prebuilt":
            logger.info("Using pre-built image: %s", self._image_uri)
            return {
                "imageUri": self._image_uri,
                "status": "READY",
            }

        # CodeBuild mode - build but don't push yet
        logger.info("Building image with CodeBuild...")
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

        self._image_uri = result.get("imageUri")
        return result

    def deploy(
        self,
        agent_name: str,
        region_name: Optional[str] = None,
        tag: str = "latest",
        max_wait: int = 600,
    ) -> Dict[str, Any]:
        """Build and push the container image to ECR.

        For pre-built images, this validates the image exists.
        For CodeBuild mode, this builds and pushes to ECR.

        Args:
            agent_name: Name of the agent
            region_name: AWS region name
            tag: Image tag
            max_wait: Maximum seconds to wait for deployment

        Returns:
            Dictionary with:
                - imageUri: ECR image URI
                - status: "SUCCEEDED" or "READY"
        """
        if self._mode == "prebuilt":
            logger.info("Using pre-built image: %s", self._image_uri)
            return {
                "imageUri": self._image_uri,
                "status": "READY",
            }

        # CodeBuild mode - build and push
        logger.info("Building and pushing image with CodeBuild...")
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

        self._image_uri = result.get("imageUri")
        logger.info("Deploy complete. Image URI: %s", self._image_uri)
        return result


class DirectCodeDeploy(Build):
    """Package Python code as zip for direct deployment to S3.

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
        """Return the S3 package URI after deploy."""
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
        """Create the zip package locally.

        Args:
            agent_name: Name of the agent
            region_name: AWS region name (unused)
            tag: Version tag
            max_wait: Maximum seconds to wait (unused)

        Returns:
            Dictionary with:
                - localPath: Path to the created zip file
                - status: "BUILT"
        """
        source_path = os.path.abspath(self._source_path)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path not found: {source_path}")

        # Create zip package in temp directory
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, f"{agent_name}-{tag}.zip")
        self._create_code_package(source_path, zip_path)

        logger.info("Built code package: %s", zip_path)
        return {
            "localPath": zip_path,
            "status": "BUILT",
            "entrypoint": self._entrypoint,
        }

    def deploy(
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
        logger.info("Deploy complete. Package URI: %s", self._package_uri)

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
