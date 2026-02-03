"""IAM operations for Bedrock AgentCore Runtime.

This module provides functions for managing IAM roles
for container-based agent deployments.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from bedrock_agentcore._utils.user_agent import build_user_agent_suffix

logger = logging.getLogger(__name__)


def get_iam_client(region_name: Optional[str] = None) -> Any:
    """Get an IAM client with proper user agent.

    Args:
        region_name: AWS region name

    Returns:
        boto3 IAM client
    """
    user_agent_extra = build_user_agent_suffix()
    client_config = Config(user_agent_extra=user_agent_extra)
    return boto3.client("iam", region_name=region_name, config=client_config)


def _generate_deterministic_suffix(agent_name: str) -> str:
    """Generate a deterministic suffix from agent name using SHA-256.

    Args:
        agent_name: Name of the agent

    Returns:
        10-character deterministic suffix
    """
    hash_obj = hashlib.sha256(agent_name.encode("utf-8"))
    return hash_obj.hexdigest()[:10]


def _get_runtime_trust_policy(region: str, account_id: str) -> Dict[str, Any]:
    """Get the trust policy for the runtime execution role.

    Args:
        region: AWS region
        account_id: AWS account ID

    Returns:
        Trust policy document
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {"StringEquals": {"aws:SourceAccount": account_id}},
            }
        ],
    }


def _get_runtime_execution_policy(
    region: str,
    account_id: str,
    ecr_repository_arn: Optional[str] = None,
) -> Dict[str, Any]:
    """Get the execution policy for the runtime role.

    Args:
        region: AWS region
        account_id: AWS account ID
        ecr_repository_arn: Optional ECR repository ARN

    Returns:
        Execution policy document
    """
    statements: List[Dict[str, Any]] = [
        {
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            "Resource": f"arn:aws:logs:{region}:{account_id}:log-group:/aws/bedrock-agentcore/*",
        },
        {
            "Sid": "ECRAuth",
            "Effect": "Allow",
            "Action": ["ecr:GetAuthorizationToken"],
            "Resource": "*",
        },
    ]

    # Add ECR pull permissions if repository specified
    if ecr_repository_arn:
        statements.append(
            {
                "Sid": "ECRPull",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                "Resource": ecr_repository_arn,
            }
        )

    return {
        "Version": "2012-10-17",
        "Statement": statements,
    }


def _get_codebuild_trust_policy(account_id: str) -> Dict[str, Any]:
    """Get the trust policy for the CodeBuild execution role.

    Args:
        account_id: AWS account ID

    Returns:
        Trust policy document
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "codebuild.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {"StringEquals": {"aws:SourceAccount": account_id}},
            }
        ],
    }


def _get_codebuild_execution_policy(
    region: str,
    account_id: str,
    ecr_repository_arn: str,
    source_bucket_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Get the execution policy for the CodeBuild role.

    Args:
        region: AWS region
        account_id: AWS account ID
        ecr_repository_arn: ECR repository ARN
        source_bucket_name: Optional S3 bucket for source code

    Returns:
        Execution policy document
    """
    statements: List[Dict[str, Any]] = [
        {
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            "Resource": f"arn:aws:logs:{region}:{account_id}:log-group:/aws/codebuild/*",
        },
        {
            "Sid": "ECRAuth",
            "Effect": "Allow",
            "Action": ["ecr:GetAuthorizationToken"],
            "Resource": "*",
        },
        {
            "Sid": "ECRPush",
            "Effect": "Allow",
            "Action": [
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:PutImage",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
            ],
            "Resource": ecr_repository_arn,
        },
    ]

    # Add S3 permissions if bucket specified
    if source_bucket_name:
        statements.append(
            {
                "Sid": "S3Access",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:GetObjectVersion",
                    "s3:PutObject",
                    "s3:ListBucket",
                ],
                "Resource": [
                    f"arn:aws:s3:::{source_bucket_name}",
                    f"arn:aws:s3:::{source_bucket_name}/*",
                ],
            }
        )

    return {
        "Version": "2012-10-17",
        "Statement": statements,
    }


def get_or_create_runtime_execution_role(
    agent_name: str,
    region_name: Optional[str] = None,
    role_name: Optional[str] = None,
    ecr_repository_arn: Optional[str] = None,
) -> Dict[str, Any]:
    """Get or create the IAM execution role for a runtime.

    This is an idempotent operation - if the role already exists,
    it returns the existing role details.

    Args:
        agent_name: Name of the agent (used for deterministic naming)
        region_name: AWS region name
        role_name: Optional explicit role name (otherwise auto-generated)
        ecr_repository_arn: Optional ECR repository ARN for pull permissions

    Returns:
        Dictionary with role details including roleArn

    Raises:
        ClientError: If role creation fails
    """
    iam_client = get_iam_client(region_name)
    sts_client = boto3.client("sts")

    account_id = sts_client.get_caller_identity()["Account"]
    region = region_name or boto3.Session().region_name or "us-west-2"

    # Generate deterministic role name if not provided
    if not role_name:
        suffix = _generate_deterministic_suffix(agent_name)
        role_name = f"AmazonBedrockAgentCoreSDKRuntime-{region}-{suffix}"

    # Try to get existing role
    try:
        response = iam_client.get_role(RoleName=role_name)
        role = response["Role"]
        logger.info("IAM role '%s' already exists", role_name)
        return {
            "roleName": role["RoleName"],
            "roleArn": role["Arn"],
            "created": False,
        }
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    # Create role
    logger.info("Creating IAM execution role '%s'...", role_name)

    trust_policy = _get_runtime_trust_policy(region, account_id)
    execution_policy = _get_runtime_execution_policy(region, account_id, ecr_repository_arn)

    try:
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description=f"Execution role for Bedrock AgentCore runtime: {agent_name}",
            Tags=[
                {"Key": "CreatedBy", "Value": "bedrock-agentcore-sdk"},
                {"Key": "AgentName", "Value": agent_name},
            ],
        )
        role_arn = response["Role"]["Arn"]

        # Attach inline policy
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName="ExecutionPolicy",
            PolicyDocument=json.dumps(execution_policy),
        )

        logger.info("Created IAM execution role: %s", role_arn)
        return {
            "roleName": role_name,
            "roleArn": role_arn,
            "created": True,
        }

    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            # Race condition - role was created between get and create
            response = iam_client.get_role(RoleName=role_name)
            return {
                "roleName": response["Role"]["RoleName"],
                "roleArn": response["Role"]["Arn"],
                "created": False,
            }
        raise


def get_or_create_codebuild_execution_role(
    agent_name: str,
    ecr_repository_arn: str,
    region_name: Optional[str] = None,
    role_name: Optional[str] = None,
    source_bucket_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Get or create the IAM execution role for CodeBuild.

    This is an idempotent operation - if the role already exists,
    it returns the existing role details.

    Args:
        agent_name: Name of the agent (used for deterministic naming)
        ecr_repository_arn: ECR repository ARN for push permissions
        region_name: AWS region name
        role_name: Optional explicit role name (otherwise auto-generated)
        source_bucket_name: Optional S3 bucket for source code

    Returns:
        Dictionary with role details including roleArn

    Raises:
        ClientError: If role creation fails
    """
    iam_client = get_iam_client(region_name)
    sts_client = boto3.client("sts")

    account_id = sts_client.get_caller_identity()["Account"]
    region = region_name or boto3.Session().region_name or "us-west-2"

    # Generate deterministic role name if not provided
    if not role_name:
        suffix = _generate_deterministic_suffix(agent_name)
        role_name = f"AmazonBedrockAgentCoreSDKCodeBuild-{region}-{suffix}"

    # Try to get existing role
    try:
        response = iam_client.get_role(RoleName=role_name)
        role = response["Role"]
        logger.info("CodeBuild IAM role '%s' already exists", role_name)
        return {
            "roleName": role["RoleName"],
            "roleArn": role["Arn"],
            "created": False,
        }
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    # Create role
    logger.info("Creating CodeBuild IAM role '%s'...", role_name)

    trust_policy = _get_codebuild_trust_policy(account_id)
    execution_policy = _get_codebuild_execution_policy(
        region, account_id, ecr_repository_arn, source_bucket_name
    )

    try:
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description=f"CodeBuild role for Bedrock AgentCore agent: {agent_name}",
            Tags=[
                {"Key": "CreatedBy", "Value": "bedrock-agentcore-sdk"},
                {"Key": "AgentName", "Value": agent_name},
            ],
        )
        role_arn = response["Role"]["Arn"]

        # Attach inline policy
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName="CodeBuildExecutionPolicy",
            PolicyDocument=json.dumps(execution_policy),
        )

        logger.info("Created CodeBuild IAM role: %s", role_arn)
        return {
            "roleName": role_name,
            "roleArn": role_arn,
            "created": True,
        }

    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            # Race condition
            response = iam_client.get_role(RoleName=role_name)
            return {
                "roleName": response["Role"]["RoleName"],
                "roleArn": response["Role"]["Arn"],
                "created": False,
            }
        raise


def delete_role(role_name: str, region_name: Optional[str] = None) -> Dict[str, Any]:
    """Delete an IAM role and its inline policies.

    Args:
        role_name: Name of the IAM role
        region_name: AWS region name

    Returns:
        Dictionary with deletion status

    Raises:
        ClientError: If deletion fails
    """
    iam_client = get_iam_client(region_name)

    try:
        # First, delete inline policies
        policies = iam_client.list_role_policies(RoleName=role_name)
        for policy_name in policies.get("PolicyNames", []):
            iam_client.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
            logger.debug("Deleted inline policy: %s", policy_name)

        # Then delete the role
        iam_client.delete_role(RoleName=role_name)
        logger.info("Deleted IAM role '%s'", role_name)
        return {"status": "DELETED", "roleName": role_name}

    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            logger.warning("IAM role '%s' not found", role_name)
            return {"status": "NOT_FOUND", "roleName": role_name}
        raise
