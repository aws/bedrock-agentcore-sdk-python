"""Integration tests for GatewayClient.

Requires environment variables:
    BEDROCK_TEST_REGION: AWS region (default: us-west-2)
    GATEWAY_ROLE_ARN: IAM role ARN with AgentCore gateway trust policy
"""

import os
import time

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.gateway.client import GatewayClient


@pytest.mark.integration
class TestGatewayClient:
    """Integration tests for GatewayClient CRUD and wait methods."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.role_arn = os.environ.get("GATEWAY_ROLE_ARN")
        if not cls.role_arn:
            pytest.skip("GATEWAY_ROLE_ARN not set")
        cls.client = GatewayClient(region_name=cls.region)
        cls.test_prefix = f"sdk-integ-{int(time.time())}"
        cls.gateway_ids = []

    @classmethod
    def teardown_class(cls):
        for gw_id in cls.gateway_ids:
            try:
                cls.client.delete_gateway(gatewayIdentifier=gw_id)
            except Exception as e:
                print(f"Failed to delete gateway {gw_id}: {e}")

    @pytest.mark.order(1)
    def test_create_gateway_and_wait(self):
        gw = self.client.create_gateway_and_wait(
            name=f"{self.test_prefix}-gw",
            roleArn=self.role_arn,
            authorizerType="NONE",
            protocolType="MCP",
        )
        self.__class__.gateway_ids.append(gw["gatewayId"])
        assert gw["status"] == "READY"
        assert gw["name"] == f"{self.test_prefix}-gw"

    @pytest.mark.order(2)
    def test_get_gateway_passthrough(self):
        if not self.gateway_ids:
            pytest.skip("prerequisite test did not create gateway")
        gw = self.client.get_gateway(
            gatewayIdentifier=self.gateway_ids[0],
        )
        assert gw["status"] == "READY"

    @pytest.mark.order(3)
    def test_get_gateway_snake_case(self):
        if not self.gateway_ids:
            pytest.skip("prerequisite test did not create gateway")
        gw = self.client.get_gateway(
            gateway_identifier=self.gateway_ids[0],
        )
        assert gw["status"] == "READY"

    @pytest.mark.order(4)
    def test_get_gateway_by_name(self):
        if not self.gateway_ids:
            pytest.skip("prerequisite test did not create gateway")
        gw = self.client.get_gateway_by_name(
            name=f"{self.test_prefix}-gw",
        )
        assert gw is not None
        assert gw["gatewayId"] == self.gateway_ids[0]

    @pytest.mark.order(5)
    def test_get_gateway_by_name_not_found(self):
        result = self.client.get_gateway_by_name(
            name="nonexistent-gateway-name",
        )
        assert result is None

    @pytest.mark.order(6)
    def test_list_gateways_passthrough(self):
        gateways = self.client.list_gateways()
        assert "items" in gateways

    @pytest.mark.order(7)
    def test_update_gateway_and_wait(self):
        if not self.gateway_ids:
            pytest.skip("prerequisite test did not create gateway")
        updated = self.client.update_gateway_and_wait(
            gatewayIdentifier=self.gateway_ids[0],
            name=f"{self.test_prefix}-gw",
            roleArn=self.role_arn,
            authorizerType="NONE",
            description="updated by integ test",
        )
        assert updated["status"] == "READY"

    @pytest.mark.order(8)
    def test_delete_gateway_and_wait(self):
        if not self.gateway_ids:
            pytest.skip("prerequisite test did not create gateway")
        gw_id = self.gateway_ids.pop(0)
        self.client.delete_gateway_and_wait(
            gatewayIdentifier=gw_id,
        )
        with pytest.raises(ClientError):
            self.client.get_gateway(gatewayIdentifier=gw_id)


@pytest.mark.integration
class TestGatewayTargetClient:
    """Integration tests for gateway target CRUD.

    Requires GATEWAY_LAMBDA_ARN in addition to GATEWAY_ROLE_ARN.
    """

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.role_arn = os.environ.get("GATEWAY_ROLE_ARN")
        cls.lambda_arn = os.environ.get("GATEWAY_LAMBDA_ARN")
        if not cls.role_arn:
            pytest.skip("GATEWAY_ROLE_ARN not set")
        if not cls.lambda_arn:
            pytest.skip("GATEWAY_LAMBDA_ARN not set")
        cls.client = GatewayClient(region_name=cls.region)
        cls.test_prefix = f"sdk-integ-tgt-{int(time.time())}"
        cls.gateway_id = None
        cls.target_ids = []

        # Create a gateway for target tests
        gw = cls.client.create_gateway_and_wait(
            name=f"{cls.test_prefix}-gw",
            roleArn=cls.role_arn,
            authorizerType="NONE",
            protocolType="MCP",
        )
        cls.gateway_id = gw["gatewayId"]

    @classmethod
    def teardown_class(cls):
        if cls.gateway_id:
            for target_id in cls.target_ids:
                try:
                    cls.client.delete_gateway_target(
                        gatewayIdentifier=cls.gateway_id,
                        targetId=target_id,
                    )
                except Exception as e:
                    print(f"Failed to delete target {target_id}: {e}")
            try:
                cls.client.delete_gateway_and_wait(
                    gatewayIdentifier=cls.gateway_id,
                )
            except Exception as e:
                print(f"Failed to delete gateway {cls.gateway_id}: {e}")

    @pytest.mark.order(1)
    def test_create_gateway_target_and_wait(self):
        target = self.client.create_gateway_target_and_wait(
            gatewayIdentifier=self.gateway_id,
            name=f"{self.test_prefix}-target",
            targetConfiguration={
                "mcp": {
                    "lambda": {
                        "lambdaArn": self.lambda_arn,
                        "toolSchema": {
                            "inlinePayload": [
                                {
                                    "name": "test_tool",
                                    "description": "A test tool",
                                    "inputSchema": {"type": "object"},
                                }
                            ]
                        },
                    }
                },
            },
            credentialProviderConfigurations=[
                {"credentialProviderType": "GATEWAY_IAM_ROLE"},
            ],
        )
        self.__class__.target_ids.append(target["targetId"])
        assert target["status"] == "READY"

    @pytest.mark.order(2)
    def test_get_gateway_target_passthrough(self):
        if not self.target_ids:
            pytest.skip("prerequisite test did not create target")
        target = self.client.get_gateway_target(
            gatewayIdentifier=self.gateway_id,
            targetId=self.target_ids[0],
        )
        assert target["status"] == "READY"

    @pytest.mark.order(3)
    def test_get_gateway_target_by_name(self):
        if not self.target_ids:
            pytest.skip("prerequisite test did not create target")
        target = self.client.get_gateway_target_by_name(
            gateway_identifier=self.gateway_id,
            name=f"{self.test_prefix}-target",
        )
        assert target is not None
        assert target["targetId"] == self.target_ids[0]

    @pytest.mark.order(4)
    def test_list_gateway_targets_passthrough(self):
        targets = self.client.list_gateway_targets(
            gatewayIdentifier=self.gateway_id,
        )
        assert "items" in targets

    @pytest.mark.order(5)
    def test_update_gateway_target_and_wait(self):
        if not self.target_ids:
            pytest.skip("prerequisite test did not create target")
        updated = self.client.update_gateway_target_and_wait(
            gatewayIdentifier=self.gateway_id,
            targetId=self.target_ids[0],
            name=f"{self.test_prefix}-target",
            targetConfiguration={
                "mcp": {
                    "lambda": {
                        "lambdaArn": self.lambda_arn,
                        "toolSchema": {
                            "inlinePayload": [
                                {
                                    "name": "test_tool",
                                    "description": "An updated test tool",
                                    "inputSchema": {"type": "object"},
                                }
                            ]
                        },
                    }
                },
            },
            credentialProviderConfigurations=[
                {"credentialProviderType": "GATEWAY_IAM_ROLE"},
            ],
            description="updated by integ test",
        )
        assert updated["status"] == "READY"

    @pytest.mark.order(6)
    def test_get_gateway_target_by_name_not_found(self):
        result = self.client.get_gateway_target_by_name(
            gateway_identifier=self.gateway_id,
            name="nonexistent-target-name",
        )
        assert result is None

    @pytest.mark.order(7)
    def test_delete_gateway_target_and_wait(self):
        if not self.target_ids:
            pytest.skip("prerequisite test did not create target")
        target_id = self.target_ids.pop(0)
        self.client.delete_gateway_target_and_wait(
            gatewayIdentifier=self.gateway_id,
            targetId=target_id,
        )
        with pytest.raises(ClientError):
            self.client.get_gateway_target(
                gatewayIdentifier=self.gateway_id,
                targetId=target_id,
            )
