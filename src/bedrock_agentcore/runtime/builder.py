"""Docker build operations for Bedrock AgentCore Runtime.

This module provides functions for building Docker images
for container-based agent deployments using AWS CodeBuild.
"""

import json
import logging
import os
import tempfile
import time
import uuid
import zipfile
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from bedrock_agentcore._utils.user_agent import build_user_agent_suffix

from .ecr import ensure_ecr_repository
from .iam import get_or_create_codebuild_execution_role

logger = logging.getLogger(__name__)

# Default Dockerfile template for Python agents
DEFAULT_DOCKERFILE_TEMPLATE = '''FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt* pyproject.toml* ./

# Install Python dependencies
RUN if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi
RUN if [ -f "pyproject.toml" ]; then pip install --no-cache-dir .; fi

# Copy application code
COPY . .

# Install bedrock-agentcore SDK
RUN pip install --no-cache-dir bedrock-agentcore

# Expose port
EXPOSE 8080

# Set entrypoint
CMD ["python", "-m", "{entrypoint_module}"]
'''


def get_codebuild_client(region_name: Optional[str] = None) -> Any:
    """Get a CodeBuild client with proper user agent.

    Args:
        region_name: AWS region name

    Returns:
        boto3 CodeBuild client
    """
    user_agent_extra = build_user_agent_suffix()
    client_config = Config(user_agent_extra=user_agent_extra)
    return boto3.client("codebuild", region_name=region_name, config=client_config)


def get_s3_client(region_name: Optional[str] = None) -> Any:
    """Get an S3 client with proper user agent.

    Args:
        region_name: AWS region name

    Returns:
        boto3 S3 client
    """
    user_agent_extra = build_user_agent_suffix()
    client_config = Config(user_agent_extra=user_agent_extra)
    return boto3.client("s3", region_name=region_name, config=client_config)


def generate_dockerfile(
    source_path: str,
    entrypoint: str,
    output_path: Optional[str] = None,
) -> str:
    """Generate a Dockerfile for the agent.

    Args:
        source_path: Path to agent source code
        entrypoint: Entry point (e.g., "agent.py:app" or "agent")
        output_path: Optional path to write Dockerfile (default: source_path/Dockerfile)

    Returns:
        Path to generated Dockerfile
    """
    # Parse entrypoint to get module name
    if ":" in entrypoint:
        module_part = entrypoint.split(":")[0]
    else:
        module_part = entrypoint

    # Remove .py extension if present
    if module_part.endswith(".py"):
        module_part = module_part[:-3]

    # Generate Dockerfile content
    dockerfile_content = DEFAULT_DOCKERFILE_TEMPLATE.format(entrypoint_module=module_part)

    # Determine output path
    if output_path is None:
        output_path = os.path.join(source_path, "Dockerfile")

    # Write Dockerfile
    with open(output_path, "w") as f:
        f.write(dockerfile_content)

    logger.info("Generated Dockerfile at: %s", output_path)
    return output_path


def _create_source_zip(source_path: str, output_path: str) -> str:
    """Create a zip file of the source code for CodeBuild.

    Args:
        source_path: Path to agent source code
        output_path: Path to write zip file

    Returns:
        Path to created zip file
    """
    source_path = os.path.abspath(source_path)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_path):
            # Skip common directories
            dirs[:] = [d for d in dirs if d not in [".git", "__pycache__", ".venv", "venv", "node_modules"]]

            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_path)
                zipf.write(file_path, arcname)

    logger.debug("Created source zip: %s", output_path)
    return output_path


def _ensure_source_bucket(
    agent_name: str,
    region_name: Optional[str] = None,
) -> str:
    """Ensure an S3 bucket exists for CodeBuild source code.

    Args:
        agent_name: Name of the agent
        region_name: AWS region name

    Returns:
        Bucket name
    """
    s3_client = get_s3_client(region_name)
    sts_client = boto3.client("sts")

    account_id = sts_client.get_caller_identity()["Account"]
    region = region_name or boto3.Session().region_name or "us-west-2"

    bucket_name = f"bedrock-agentcore-codebuild-{account_id}-{region}"

    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.debug("S3 bucket '%s' already exists", bucket_name)
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

    return bucket_name


def build_image_codebuild(
    source_path: str,
    image_name: str,
    ecr_repository_uri: str,
    tag: str = "latest",
    region_name: Optional[str] = None,
    wait: bool = True,
    max_wait: int = 600,
    poll_interval: int = 10,
) -> Dict[str, Any]:
    """Build a Docker image using AWS CodeBuild.

    This method uploads source code to S3, creates a CodeBuild project,
    and runs the build to produce an ARM64 image in ECR.

    Args:
        source_path: Path to agent source code
        image_name: Name for the Docker image
        ecr_repository_uri: ECR repository URI
        tag: Image tag
        region_name: AWS region name
        wait: Wait for build to complete
        max_wait: Maximum seconds to wait
        poll_interval: Seconds between status checks

    Returns:
        Dictionary with build details

    Raises:
        RuntimeError: If build fails
        TimeoutError: If wait times out
    """
    codebuild_client = get_codebuild_client(region_name)
    s3_client = get_s3_client(region_name)
    sts_client = boto3.client("sts")

    account_id = sts_client.get_caller_identity()["Account"]
    region = region_name or boto3.Session().region_name or "us-west-2"

    source_path = os.path.abspath(source_path)

    # Verify Dockerfile exists (should be generated before calling this function)
    dockerfile = os.path.join(source_path, "Dockerfile")
    if not os.path.exists(dockerfile):
        raise FileNotFoundError(f"Dockerfile not found at: {dockerfile}")

    # Ensure S3 bucket for source code
    bucket_name = _ensure_source_bucket(image_name, region_name)

    # Get ECR repository ARN
    ecr_repo_name = ecr_repository_uri.split("/")[-1].split(":")[0]
    ecr_repository_arn = f"arn:aws:ecr:{region}:{account_id}:repository/{ecr_repo_name}"

    # Ensure CodeBuild IAM role
    role_result = get_or_create_codebuild_execution_role(
        agent_name=image_name,
        ecr_repository_arn=ecr_repository_arn,
        region_name=region_name,
        source_bucket_name=bucket_name,
    )
    codebuild_role_arn = role_result["roleArn"]

    # Create and upload source zip
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "source.zip")
        _create_source_zip(source_path, zip_path)

        # Upload to S3
        s3_key = f"{image_name}/{uuid.uuid4().hex}/source.zip"
        logger.info("Uploading source code to s3://%s/%s", bucket_name, s3_key)
        s3_client.upload_file(zip_path, bucket_name, s3_key)

    # Create/update CodeBuild project
    project_name = f"bedrock-agentcore-{image_name}"[:255]
    full_image_uri = f"{ecr_repository_uri}:{tag}"

    ecr_registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    ecr_login_cmd = (
        f"aws ecr get-login-password --region {region} | "
        f"docker login --username AWS --password-stdin {ecr_registry}"
    )

    buildspec = {
        "version": "0.2",
        "phases": {
            "pre_build": {
                "commands": [
                    "echo Logging in to Amazon ECR...",
                    ecr_login_cmd,
                ]
            },
            "build": {
                "commands": [
                    "echo Build started on `date`",
                    f"docker build -t {full_image_uri} .",
                ]
            },
            "post_build": {
                "commands": [
                    "echo Build completed on `date`",
                    "echo Pushing the Docker image...",
                    f"docker push {full_image_uri}",
                ]
            },
        },
    }

    project_config = {
        "name": project_name,
        "description": f"Build project for Bedrock AgentCore agent: {image_name}",
        "source": {
            "type": "S3",
            "location": f"{bucket_name}/{s3_key}",
        },
        "artifacts": {"type": "NO_ARTIFACTS"},
        "environment": {
            "type": "ARM_CONTAINER",
            "computeType": "BUILD_GENERAL1_SMALL",
            "image": "aws/codebuild/amazonlinux2-aarch64-standard:3.0",
            "privilegedMode": True,
            "environmentVariables": [
                {"name": "AWS_DEFAULT_REGION", "value": region},
                {"name": "AWS_ACCOUNT_ID", "value": account_id},
                {"name": "IMAGE_REPO_NAME", "value": ecr_repo_name},
                {"name": "IMAGE_TAG", "value": tag},
            ],
        },
        "serviceRole": codebuild_role_arn,
        "buildSpec": json.dumps(buildspec),
        "tags": [
            {"key": "CreatedBy", "value": "bedrock-agentcore-sdk"},
            {"key": "AgentName", "value": image_name},
        ],
    }

    # Create or update project
    try:
        codebuild_client.create_project(**project_config)
        logger.info("Created CodeBuild project: %s", project_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
            codebuild_client.update_project(**project_config)
            logger.info("Updated CodeBuild project: %s", project_name)
        else:
            raise

    # Start build
    logger.info("Starting CodeBuild build for '%s'...", image_name)
    response = codebuild_client.start_build(projectName=project_name)
    build_id = response["build"]["id"]

    if not wait:
        return {
            "buildId": build_id,
            "projectName": project_name,
            "imageUri": full_image_uri,
            "status": "IN_PROGRESS",
        }

    # Wait for build to complete
    return _wait_for_codebuild(
        codebuild_client=codebuild_client,
        build_id=build_id,
        image_uri=full_image_uri,
        max_wait=max_wait,
        poll_interval=poll_interval,
    )


def _wait_for_codebuild(
    codebuild_client: Any,
    build_id: str,
    image_uri: str,
    max_wait: int,
    poll_interval: int,
) -> Dict[str, Any]:
    """Wait for a CodeBuild build to complete.

    Args:
        codebuild_client: boto3 CodeBuild client
        build_id: CodeBuild build ID
        image_uri: Expected image URI
        max_wait: Maximum seconds to wait
        poll_interval: Seconds between status checks

    Returns:
        Dictionary with build result

    Raises:
        RuntimeError: If build fails
        TimeoutError: If wait times out
    """
    start_time = time.time()
    logger.info("Waiting for CodeBuild build to complete...")

    while time.time() - start_time < max_wait:
        response = codebuild_client.batch_get_builds(ids=[build_id])
        build = response["builds"][0]
        status = build["buildStatus"]

        logger.debug("CodeBuild status: %s", status)

        if status == "SUCCEEDED":
            logger.info("CodeBuild build succeeded")
            return {
                "buildId": build_id,
                "imageUri": image_uri,
                "status": "SUCCEEDED",
                "buildOutput": build.get("logs", {}),
            }

        if status in ["FAILED", "FAULT", "STOPPED", "TIMED_OUT"]:
            raise RuntimeError(f"CodeBuild build failed with status: {status}")

        time.sleep(poll_interval)

    raise TimeoutError(f"Timeout waiting for CodeBuild build after {max_wait}s")


def build_and_push(
    source_path: str,
    agent_name: str,
    entrypoint: str,
    region_name: Optional[str] = None,
    tag: str = "latest",
    wait: bool = True,
    max_wait: int = 600,
) -> Dict[str, Any]:
    """Build a Docker image using CodeBuild and push to ECR.

    This is the main entry point for container builds. It handles:
    1. Creating ECR repository (auto-generated name)
    2. Generating Dockerfile if not present
    3. Building ARM64 image via CodeBuild
    4. Pushing to ECR

    Args:
        source_path: Path to agent source code
        agent_name: Name of the agent
        entrypoint: Entry point (e.g., "agent.py:app")
        region_name: AWS region name
        tag: Image tag
        wait: Wait for build to complete
        max_wait: Maximum seconds to wait

    Returns:
        Dictionary with build result including imageUri

    Raises:
        FileNotFoundError: If source path doesn't exist
        RuntimeError: If build fails
    """
    source_path = os.path.abspath(source_path)

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source path not found: {source_path}")

    # Auto-generate ECR repository name
    ecr_repository = f"bedrock-agentcore/{agent_name}"

    # Ensure ECR repository exists
    ecr_result = ensure_ecr_repository(ecr_repository, region_name)
    ecr_repository_uri = ecr_result["repositoryUri"]

    # Generate Dockerfile if not present
    dockerfile_path = os.path.join(source_path, "Dockerfile")
    if not os.path.exists(dockerfile_path):
        logger.info("No Dockerfile found, generating one...")
        generate_dockerfile(source_path, entrypoint)

    # Build image using CodeBuild
    logger.info("Building image using CodeBuild...")
    result = build_image_codebuild(
        source_path=source_path,
        image_name=agent_name,
        ecr_repository_uri=ecr_repository_uri,
        tag=tag,
        region_name=region_name,
        wait=wait,
        max_wait=max_wait,
    )

    logger.info("Build complete. Image URI: %s", result.get("imageUri"))
    return result
