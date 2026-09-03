"""Integration tests for GatewayClient Web Search target helper methods.

These tests skip for two separate reasons, and both are checked before anything is
created. The connector is only offered in three regions, so the class skips outright
in any other one. And it is enabled per account, so a target creation in a supported
region can still come back saying it is not available, which skips as well.

Requires environment variables:
    BEDROCK_TEST_REGION: AWS region. Defaults to us-east-1 rather than the us-west-2
        the other gateway tests use, because web search is not offered in us-west-2
        and CI sets AWS_REGION to it, so taking the ambient region would skip every
        run.
    GATEWAY_ROLE_ARN: IAM role ARN with AgentCore gateway trust policy
"""

import os
import time

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.gateway.client import GatewayClient

#: Regions where the web-search connector is offered.
WEB_SEARCH_REGIONS = ("us-east-1", "eu-west-1", "ap-northeast-1")


@pytest.mark.integration
class TestGatewayWebSearchTarget:
    """Integration tests for create_web_search_target."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-east-1")
        # Checked before the gateway is created, so an unsupported region does not
        # create and delete a real gateway just to find out the target cannot be made.
        if cls.region not in WEB_SEARCH_REGIONS:
            pytest.skip(f"web search is not offered in {cls.region}, only in {', '.join(WEB_SEARCH_REGIONS)}")
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

        This is the account half of the two reasons in the module docstring, and it can
        only be found out by asking: there is no API that reports whether a connector is
        available to an account. When it is not, CreateGatewayTarget rejects the request
        with "Connector integration web-search is not available for this account." Any
        other error still fails the test.
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
