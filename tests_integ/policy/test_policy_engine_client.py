"""Integration tests for PolicyEngineClient."""

import os
import time

import boto3
import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.policy.client import PolicyEngineClient


@pytest.mark.integration
class TestPolicyEngineClient:
    """Integration tests for PolicyEngineClient CRUD and wait methods."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.client = PolicyEngineClient(region_name=cls.region)
        cls.test_prefix = f"sdk_integ_{int(time.time())}"
        cls.engine_ids = []
        cls.policy_ids = []
        sts = boto3.client("sts", region_name=cls.region)
        account_id = sts.get_caller_identity()["Account"]
        cls.gateway_resource_arn = f"arn:aws:bedrock-agentcore:{cls.region}:{account_id}:gateway/*"

    @classmethod
    def teardown_class(cls):
        for engine_id, policy_id in cls.policy_ids:
            try:
                cls.client.delete_policy(
                    policyEngineId=engine_id,
                    policyId=policy_id,
                )
            except Exception as e:
                print(f"Failed to delete policy {policy_id}: {e}")
        # Wait for policy deletes to complete before deleting engines
        if cls.policy_ids:
            time.sleep(15)
        for engine_id in cls.engine_ids:
            try:
                cls.client.delete_policy_engine(
                    policyEngineId=engine_id,
                )
            except Exception as e:
                print(f"Failed to delete engine {engine_id}: {e}")

    @pytest.mark.order(1)
    def test_create_policy_engine_and_wait(self):
        engine = self.client.create_policy_engine_and_wait(
            name=f"{self.test_prefix}_engine",
        )
        self.__class__.engine_ids.append(engine["policyEngineId"])
        assert engine["status"] == "ACTIVE"
        assert engine["name"] == f"{self.test_prefix}_engine"

    @pytest.mark.order(2)
    def test_get_policy_engine_passthrough(self):
        if not self.engine_ids:
            pytest.skip("prerequisite test did not create engine")
        engine = self.client.get_policy_engine(
            policyEngineId=self.engine_ids[0],
        )
        assert engine["status"] == "ACTIVE"

    @pytest.mark.order(3)
    def test_get_policy_engine_snake_case(self):
        if not self.engine_ids:
            pytest.skip("prerequisite test did not create engine")
        engine = self.client.get_policy_engine(
            policy_engine_id=self.engine_ids[0],
        )
        assert engine["status"] == "ACTIVE"

    @pytest.mark.order(4)
    def test_update_policy_engine_and_wait(self):
        if not self.engine_ids:
            pytest.skip("prerequisite test did not create engine")
        updated = self.client.update_policy_engine_and_wait(
            policyEngineId=self.engine_ids[0],
            description={"optionalValue": "updated by integ test"},
        )
        assert updated["status"] == "ACTIVE"

    @pytest.mark.order(5)
    def test_create_policy_and_wait(self):
        if not self.engine_ids:
            pytest.skip("prerequisite test did not create engine")
        policy = self.client.create_policy_and_wait(
            policyEngineId=self.engine_ids[0],
            name=f"{self.test_prefix}_policy",
            definition={"cedar": {"statement": ("permit(principal, action, resource is AgentCore::Gateway);")}},
            validationMode="IGNORE_ALL_FINDINGS",
        )
        self.__class__.policy_ids.append(
            (self.engine_ids[0], policy["policyId"]),
        )
        assert policy["status"] == "ACTIVE"

    @pytest.mark.order(6)
    def test_get_policy_passthrough(self):
        if not self.policy_ids:
            pytest.skip("prerequisite test did not create policy")
        engine_id, policy_id = self.policy_ids[0]
        policy = self.client.get_policy(
            policyEngineId=engine_id,
            policyId=policy_id,
        )
        assert policy["status"] == "ACTIVE"

    @pytest.mark.order(7)
    def test_list_policy_engines_passthrough(self):
        engines = self.client.list_policy_engines()
        assert "policyEngines" in engines

    @pytest.mark.order(8)
    def test_list_policies_passthrough(self):
        if not self.engine_ids:
            pytest.skip("prerequisite test did not create engine")
        policies = self.client.list_policies(
            policyEngineId=self.engine_ids[0],
        )
        assert "policies" in policies

    @pytest.mark.order(9)
    def test_delete_policy_and_wait(self):
        if not self.policy_ids:
            pytest.skip("prerequisite test did not create policy")
        engine_id, policy_id = self.policy_ids.pop(0)
        self.client.delete_policy_and_wait(
            policyEngineId=engine_id,
            policyId=policy_id,
        )
        # Verify it's gone
        with pytest.raises(ClientError):
            self.client.get_policy(
                policyEngineId=engine_id,
                policyId=policy_id,
            )

    @pytest.mark.order(10)
    def test_delete_policy_engine_and_wait(self):
        if not self.engine_ids:
            pytest.skip("prerequisite test did not create engine")
        engine_id = self.engine_ids.pop(0)
        self.client.delete_policy_engine_and_wait(
            policyEngineId=engine_id,
        )
        # Verify it's gone
        with pytest.raises(ClientError):
            self.client.get_policy_engine(policyEngineId=engine_id)
