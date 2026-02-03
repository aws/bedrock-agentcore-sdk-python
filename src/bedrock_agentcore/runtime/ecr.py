"""ECR operations for Bedrock AgentCore Runtime.

This module provides functions for managing ECR repositories
for container-based agent deployments.
"""

import base64
import logging
import subprocess
from typing import Any, Dict, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from bedrock_agentcore._utils.user_agent import build_user_agent_suffix

logger = logging.getLogger(__name__)


def get_ecr_client(region_name: Optional[str] = None) -> Any:
    """Get an ECR client with proper user agent.

    Args:
        region_name: AWS region name

    Returns:
        boto3 ECR client
    """
    user_agent_extra = build_user_agent_suffix()
    client_config = Config(user_agent_extra=user_agent_extra)
    return boto3.client("ecr", region_name=region_name, config=client_config)


def ensure_ecr_repository(
    repository_name: str,
    region_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Ensure an ECR repository exists, creating it if needed.

    This is an idempotent operation - if the repository already exists,
    it returns the existing repository details.

    Args:
        repository_name: Name of the ECR repository
        region_name: AWS region name

    Returns:
        Dictionary with repository details including repositoryUri

    Raises:
        ClientError: If repository creation fails (other than already exists)
    """
    ecr_client = get_ecr_client(region_name)

    # Try to describe existing repository
    try:
        response = ecr_client.describe_repositories(repositoryNames=[repository_name])
        repository = response["repositories"][0]
        logger.info("ECR repository '%s' already exists", repository_name)
        return {
            "repositoryName": repository["repositoryName"],
            "repositoryUri": repository["repositoryUri"],
            "repositoryArn": repository["repositoryArn"],
            "created": False,
        }
    except ClientError as e:
        if e.response["Error"]["Code"] != "RepositoryNotFoundException":
            raise

    # Create repository
    logger.info("Creating ECR repository '%s'...", repository_name)
    try:
        response = ecr_client.create_repository(
            repositoryName=repository_name,
            imageScanningConfiguration={"scanOnPush": True},
            imageTagMutability="MUTABLE",
        )
        repository = response["repository"]
        logger.info("Created ECR repository: %s", repository["repositoryUri"])
        return {
            "repositoryName": repository["repositoryName"],
            "repositoryUri": repository["repositoryUri"],
            "repositoryArn": repository["repositoryArn"],
            "created": True,
        }
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryAlreadyExistsException":
            # Race condition - repository was created between describe and create
            response = ecr_client.describe_repositories(repositoryNames=[repository_name])
            repository = response["repositories"][0]
            return {
                "repositoryName": repository["repositoryName"],
                "repositoryUri": repository["repositoryUri"],
                "repositoryArn": repository["repositoryArn"],
                "created": False,
            }
        raise


def delete_ecr_repository(
    repository_name: str,
    region_name: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Delete an ECR repository.

    Args:
        repository_name: Name of the ECR repository
        region_name: AWS region name
        force: If True, delete even if repository contains images

    Returns:
        Dictionary with deletion status

    Raises:
        ClientError: If deletion fails
    """
    ecr_client = get_ecr_client(region_name)

    try:
        ecr_client.delete_repository(repositoryName=repository_name, force=force)
        logger.info("Deleted ECR repository '%s'", repository_name)
        return {"status": "DELETED", "repositoryName": repository_name}
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            logger.warning("ECR repository '%s' not found", repository_name)
            return {"status": "NOT_FOUND", "repositoryName": repository_name}
        raise


def get_ecr_login_credentials(region_name: Optional[str] = None) -> Tuple[str, str, str]:
    """Get ECR login credentials for Docker authentication.

    Args:
        region_name: AWS region name

    Returns:
        Tuple of (username, password, registry_url)

    Raises:
        ClientError: If unable to get authorization token
    """
    ecr_client = get_ecr_client(region_name)

    response = ecr_client.get_authorization_token()
    auth_data = response["authorizationData"][0]

    # Decode the token (base64 encoded "username:password")
    token = base64.b64decode(auth_data["authorizationToken"]).decode("utf-8")
    username, password = token.split(":")
    registry_url = auth_data["proxyEndpoint"]

    return username, password, registry_url


def docker_login_to_ecr(region_name: Optional[str] = None) -> bool:
    """Perform Docker login to ECR.

    Args:
        region_name: AWS region name

    Returns:
        True if login succeeded

    Raises:
        RuntimeError: If Docker login fails
    """
    username, password, registry_url = get_ecr_login_credentials(region_name)

    logger.info("Logging into ECR registry: %s", registry_url)

    # Use docker login command
    result = subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin", registry_url],
        input=password,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Docker login to ECR failed: {result.stderr}")

    logger.info("Successfully logged into ECR")
    return True


def push_image_to_ecr(
    local_image: str,
    repository_uri: str,
    tag: str = "latest",
    region_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Push a local Docker image to ECR.

    Args:
        local_image: Local image name (e.g., "my-agent:latest")
        repository_uri: ECR repository URI
        tag: Image tag
        region_name: AWS region name

    Returns:
        Dictionary with push details including full image URI

    Raises:
        RuntimeError: If Docker operations fail
    """
    # Login to ECR
    docker_login_to_ecr(region_name)

    full_uri = f"{repository_uri}:{tag}"

    # Tag the image
    logger.info("Tagging image '%s' as '%s'", local_image, full_uri)
    result = subprocess.run(
        ["docker", "tag", local_image, full_uri],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to tag image: {result.stderr}")

    # Push the image
    logger.info("Pushing image to ECR: %s", full_uri)
    result = subprocess.run(
        ["docker", "push", full_uri],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to push image: {result.stderr}")

    logger.info("Successfully pushed image to ECR: %s", full_uri)
    return {
        "imageUri": full_uri,
        "repositoryUri": repository_uri,
        "tag": tag,
    }


def get_account_id() -> str:
    """Get the current AWS account ID.

    Returns:
        AWS account ID string
    """
    sts_client = boto3.client("sts")
    return str(sts_client.get_caller_identity()["Account"])


def build_ecr_uri(repository_name: str, region_name: Optional[str] = None, tag: str = "latest") -> str:
    """Build the full ECR URI for a repository.

    Args:
        repository_name: Name of the ECR repository
        region_name: AWS region name
        tag: Image tag

    Returns:
        Full ECR URI (e.g., "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-repo:latest")
    """
    account_id = get_account_id()
    region = region_name or boto3.Session().region_name or "us-west-2"
    return f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}:{tag}"
