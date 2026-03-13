"""Unit tests for ResourcePolicyClient."""

import json
from unittest.mock import Mock, patch

from bedrock_agentcore.services.resource_policy import ResourcePolicyClient

TEST_REGION = "us-east-1"
TEST_ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-agent"


class TestResourcePolicyClient:
    def test_initialization(self):
        with patch("boto3.client") as mock_boto:
            client = ResourcePolicyClient(region=TEST_REGION)
            assert client.region == TEST_REGION
            mock_boto.assert_called_once_with(
                "bedrock-agentcore-control",
                region_name=TEST_REGION,
                endpoint_url=f"https://bedrock-agentcore-control.{TEST_REGION}.amazonaws.com",
            )

    def test_put_serializes_dict_to_json(self):
        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_boto.return_value = mock_client
            policy = {"Version": "2012-10-17", "Statement": []}
            mock_client.put_resource_policy.return_value = {"policy": json.dumps(policy)}

            client = ResourcePolicyClient(region=TEST_REGION)
            result = client.put_resource_policy(TEST_ARN, policy)

            mock_client.put_resource_policy.assert_called_once_with(resourceArn=TEST_ARN, policy=json.dumps(policy))
            assert result == policy

    def test_get_returns_none_when_no_policy(self):
        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_boto.return_value = mock_client
            mock_client.get_resource_policy.return_value = {}

            client = ResourcePolicyClient(region=TEST_REGION)
            assert client.get_resource_policy(TEST_ARN) is None

    def test_get_deserializes_policy(self):
        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_boto.return_value = mock_client
            policy = {"Version": "2012-10-17"}
            mock_client.get_resource_policy.return_value = {"policy": json.dumps(policy)}

            client = ResourcePolicyClient(region=TEST_REGION)
            assert client.get_resource_policy(TEST_ARN) == policy

    def test_delete_passes_through(self):
        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_boto.return_value = mock_client
            mock_client.delete_resource_policy.return_value = {}

            client = ResourcePolicyClient(region=TEST_REGION)
            client.delete_resource_policy(TEST_ARN)
            mock_client.delete_resource_policy.assert_called_once_with(resourceArn=TEST_ARN)
