"""Integration tests for GatewayClient Web Search target helper methods.

Requires environment variables:
    BEDROCK_TEST_REGION: AWS region (default: us-east-1). The web-search connector is
        only offered in us-east-1, eu-west-1 and ap-northeast-1.
    GATEWAY_ROLE_ARN: IAM role ARN with AgentCore gateway trust policy
"""

import os
import time

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.gateway.client import GatewayClient


@pytest.mark.integration
class TestGatewayWebSearchTarget:
    """Integration tests for create_web_search_target."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-east-1")
        cls.gateway_role_arn = os.environ.get("GATEWAY_ROLE_ARN")
        if not cls.gateway_role_arn:
            pytest.fail("GATEWAY_ROLE_ARN must be set")

        cls.gateway_client = GatewayClient(region_name=cls.region)
        cls.test_prefix = f"sdk-integ-ws-tgt-{int(time.time())}"
        cls.gateway_id = None
        cls.target_ids = []

        gw = cls.gateway_client.create_gateway_and_wait(
            name=f"{cls.test_prefix}-gw",
            roleArn=cls.gateway_role_arn,
            authorizerType="NONE",
            protocolType="MCP",
        )
        cls.gateway_id = gw["gatewayId"]

    @classmethod
    def teardown_class(cls):
        for target_id in cls.target_ids:
            try:
                cls.gateway_client.delete_gateway_target_and_wait(
                    gatewayIdentifier=cls.gateway_id,
                    targetId=target_id,
                )
            except Exception as e:
                print(f"Failed to delete target {target_id}: {e}")

        if cls.gateway_id:
            try:
                cls.gateway_client.delete_gateway_and_wait(gatewayIdentifier=cls.gateway_id)
            except Exception as e:
                print(f"Failed to delete gateway {cls.gateway_id}: {e}")

    def _create_target(self, **kwargs):
        """Create a web search target, skipping the test if the account is not entitled.

        The web-search connector is enabled per account. When it is not, CreateGatewayTarget
        rejects the request with "Connector integration web-search is not available for this
        account." Any other error still fails the test.
        """
        try:
            return self.gateway_client.create_web_search_target(gateway_identifier=self.gateway_id, **kwargs)
        except ClientError as e:
            error = e.response.get("Error", {})
            if error.get("Code") == "ValidationException" and "not available for this account" in error.get(
                "Message", ""
            ):
                pytest.skip(f"web-search connector not enabled for this account: {error.get('Message')}")
            raise

    @pytest.mark.order(1)
    def test_create_web_search_target_minimal(self):
        target = self._create_target()
        self.__class__.target_ids.append(target["targetId"])
        assert target["status"] == "READY"
        assert target["name"] == "amazon-web-search"

    @pytest.mark.order(2)
    def test_create_web_search_target_with_options(self):
        target = self._create_target(
            name=f"{self.test_prefix}-custom",
            description="Search the public web",
            exclude_domains=["example.com"],
            include_domains=["docs.aws.amazon.com"],
            connector_version="1.2.0",
            parameter_overrides=[{"path": "$.maxResults", "visible": True, "description": "How many results"}],
        )
        self.__class__.target_ids.append(target["targetId"])
        assert target["status"] == "READY"
        assert target["name"] == f"{self.test_prefix}-custom"

    @pytest.mark.order(3)
    def test_create_web_search_target_with_credential_config(self):
        target = self._create_target(
            name=f"{self.test_prefix}-cred",
            credentialProviderConfigurations=[
                {"credentialProviderType": "GATEWAY_IAM_ROLE"},
            ],
        )
        self.__class__.target_ids.append(target["targetId"])
        assert target["status"] == "READY"
