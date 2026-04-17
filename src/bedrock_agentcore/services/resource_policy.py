"""Client for managing resource-based policies on Bedrock AgentCore resources."""

import json
import logging
from typing import Optional, Union

import boto3

from bedrock_agentcore._utils.endpoints import CP_ENDPOINT_OVERRIDE


class ResourcePolicyClient:
    """Client for managing resource-based policies on Bedrock AgentCore resources.

    Resource-based policies control which principals can invoke and manage
    Agent Runtime, Endpoint, and Gateway resources.
    """

    def __init__(self, region: str):
        """Initialize the client for the specified region."""
        self.region = region
        cp_kwargs: dict = {"region_name": region}
        if CP_ENDPOINT_OVERRIDE:
            cp_kwargs["endpoint_url"] = CP_ENDPOINT_OVERRIDE
        self.client = boto3.client("bedrock-agentcore-control", **cp_kwargs)
        self.logger = logging.getLogger("bedrock_agentcore.resource_policy_client")

    def put_resource_policy(self, resource_arn: str, policy: Union[str, dict]) -> dict:
        """Create or update a resource-based policy.

        Args:
            resource_arn: ARN of the resource to attach the policy to.
            policy: Policy document as a dict (auto-serialized) or JSON string.

        Returns:
            The stored policy as a dict.
        """
        policy_str = json.dumps(policy) if isinstance(policy, dict) else policy
        self.logger.info("Putting resource policy for %s", resource_arn)
        resp = self.client.put_resource_policy(resourceArn=resource_arn, policy=policy_str)
        return json.loads(resp["policy"])

    def get_resource_policy(self, resource_arn: str) -> Optional[dict]:
        """Get the resource-based policy for a resource.

        Args:
            resource_arn: ARN of the resource.

        Returns:
            The policy as a dict, or None if no policy is attached.
        """
        self.logger.info("Getting resource policy for %s", resource_arn)
        resp = self.client.get_resource_policy(resourceArn=resource_arn)
        return json.loads(resp["policy"]) if "policy" in resp else None

    def delete_resource_policy(self, resource_arn: str) -> dict:
        """Delete the resource-based policy from a resource.

        Args:
            resource_arn: ARN of the resource.

        Returns:
            Raw boto3 response.

        Raises:
            ClientError: ResourceNotFoundException if no policy exists.
        """
        self.logger.info("Deleting resource policy for %s", resource_arn)
        return self.client.delete_resource_policy(resourceArn=resource_arn)
