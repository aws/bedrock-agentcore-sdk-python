"""Integration tests for ResourcePolicyClient.

E2e tests against real AWS resources. No mocking.

Requires:
    RESOURCE_POLICY_TEST_ARN: ARN of an Agent Runtime to attach policies to
    RESOURCE_POLICY_TEST_PRINCIPAL: IAM principal ARN to use in test policies
    BEDROCK_TEST_REGION: AWS region (default: us-west-2)

Run: pytest -xvs tests_integ/services/test_resource_policy.py
"""

import json
import os

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.services.resource_policy import ResourcePolicyClient


def _make_policy(
    resource_arn: str,
    principal_arn: str,
    action: str = "bedrock-agentcore:InvokeAgentRuntime",
) -> dict:
    """Build a minimal valid policy dict for testing."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": principal_arn},
                "Action": action,
                "Resource": resource_arn,
            }
        ],
    }


@pytest.mark.integration
class TestResourcePolicyClient:
    @classmethod
    def setup_class(cls):
        cls.resource_arn = os.environ.get("RESOURCE_POLICY_TEST_ARN")
        cls.principal_arn = os.environ.get("RESOURCE_POLICY_TEST_PRINCIPAL")
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")

        if not cls.resource_arn:
            pytest.fail("RESOURCE_POLICY_TEST_ARN env var is required")
        if not cls.principal_arn:
            pytest.fail("RESOURCE_POLICY_TEST_PRINCIPAL env var is required")

        cls.client = ResourcePolicyClient(region=cls.region)

    def setup_method(self):
        """Runs before each test — ensures no policy is attached."""
        try:
            self.client.delete_resource_policy(self.resource_arn)
        except Exception:
            pass

    @classmethod
    def teardown_class(cls):
        """Remove any policy left by the last test so we don't leave side effects on the resource."""
        try:
            cls.client.delete_resource_policy(cls.resource_arn)
        except Exception:
            pass

    @pytest.mark.order(1)
    def test_get_returns_none_when_no_policy(self):
        """get on a resource with no policy returns None."""
        result = self.client.get_resource_policy(self.resource_arn)
        assert result is None

    @pytest.mark.order(2)
    def test_put_get_round_trip(self):
        """put(policy) then get() returns matching policy as a dict."""
        policy = _make_policy(self.resource_arn, self.principal_arn)

        put_result = self.client.put_resource_policy(self.resource_arn, policy)
        assert isinstance(put_result, dict)
        assert put_result["Version"] == policy["Version"]

        result = self.client.get_resource_policy(self.resource_arn)
        assert isinstance(result, dict)
        assert result["Version"] == policy["Version"]
        assert result["Statement"][0]["Effect"] == "Allow"
        assert result["Statement"][0]["Resource"] == self.resource_arn

    @pytest.mark.order(3)
    def test_put_overwrites(self):
        """put(A) then put(B) then get() returns B."""
        policy_a = _make_policy(self.resource_arn, self.principal_arn, action="bedrock-agentcore:InvokeAgentRuntime")
        policy_b = _make_policy(self.resource_arn, self.principal_arn, action="bedrock-agentcore:GetAgentCard")

        self.client.put_resource_policy(self.resource_arn, policy_a)
        self.client.put_resource_policy(self.resource_arn, policy_b)
        result = self.client.get_resource_policy(self.resource_arn)

        assert result["Statement"][0]["Action"] == "bedrock-agentcore:GetAgentCard"

    @pytest.mark.order(4)
    def test_delete_removes_policy(self):
        """put(policy) then delete() then get() returns None."""
        policy = _make_policy(self.resource_arn, self.principal_arn)
        self.client.put_resource_policy(self.resource_arn, policy)
        self.client.delete_resource_policy(self.resource_arn)

        result = self.client.get_resource_policy(self.resource_arn)
        assert result is None

    @pytest.mark.order(5)
    def test_delete_on_no_policy_raises(self):
        """delete on a resource with no policy raises ResourceNotFoundException."""
        with pytest.raises(ClientError) as exc_info:
            self.client.delete_resource_policy(self.resource_arn)
        assert exc_info.value.response["Error"]["Code"] == "ResourceNotFoundException"

    @pytest.mark.order(6)
    def test_dict_and_string_equivalence(self):
        """put(dict) and put(json.dumps(dict)) produce the same get() result."""
        policy = _make_policy(self.resource_arn, self.principal_arn)

        self.client.put_resource_policy(self.resource_arn, policy)
        result_from_dict = self.client.get_resource_policy(self.resource_arn)

        self.client.put_resource_policy(self.resource_arn, json.dumps(policy))
        result_from_str = self.client.get_resource_policy(self.resource_arn)

        assert result_from_dict == result_from_str
