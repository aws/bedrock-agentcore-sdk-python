"""Integration tests for AgentCoreRuntimeClient passthrough and *_and_wait methods.

Requires environment variables:
    BEDROCK_TEST_REGION: AWS region (default: us-west-2)
    RUNTIME_ROLE_ARN: IAM role ARN with AgentCore runtime trust policy
    RUNTIME_S3_CODE_URI: S3 URI for agent code artifact (e.g. s3://bucket/agent.zip)
"""

import os
import time

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.runtime.agent_core_runtime_client import AgentCoreRuntimeClient


@pytest.mark.integration
class TestRuntimeClientPassthrough:
    """Read-only passthrough tests. No resources needed."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.client = AgentCoreRuntimeClient(region=cls.region)

    @pytest.mark.order(1)
    def test_list_agent_runtimes_passthrough(self):
        response = self.client.list_agent_runtimes()
        assert "agentRuntimes" in response

    @pytest.mark.order(2)
    def test_list_agent_runtimes_snake_case(self):
        response = self.client.list_agent_runtimes(max_results=5)
        assert "agentRuntimes" in response

    @pytest.mark.order(3)
    def test_non_allowlisted_method_raises(self):
        with pytest.raises(AttributeError):
            self.client.not_a_real_method()


@pytest.mark.integration
class TestRuntimeCrud:
    """CRUD tests for agent runtimes.

    Requires RUNTIME_ROLE_ARN and RUNTIME_S3_CODE_URI.
    """

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.role_arn = os.environ.get("RUNTIME_ROLE_ARN")
        cls.s3_code_uri = os.environ.get("RUNTIME_S3_CODE_URI")
        if not cls.role_arn or not cls.s3_code_uri:
            pytest.skip("RUNTIME_ROLE_ARN and RUNTIME_S3_CODE_URI must be set")
        cls.client = AgentCoreRuntimeClient(region=cls.region)
        cls.test_prefix = f"sdk_integ_{int(time.time())}"
        cls.runtime_ids = []

        # Parse S3 URI into bucket and key
        s3_parts = cls.s3_code_uri.replace("s3://", "").split("/", 1)
        cls.s3_bucket = s3_parts[0]
        cls.s3_key = s3_parts[1]

    @classmethod
    def teardown_class(cls):
        for rid in cls.runtime_ids:
            try:
                cls.client.teardown_endpoint_and_runtime(
                    rid,
                    endpoint_name="sdk_integ_ep",
                )
            except Exception as e:
                print(f"Failed to teardown runtime {rid}: {e}")

    @pytest.mark.order(4)
    def test_create_agent_runtime_and_wait(self):
        runtime = self.client.create_agent_runtime_and_wait(
            agentRuntimeName=f"{self.test_prefix}_rt",
            roleArn=self.role_arn,
            networkConfiguration={"networkMode": "PUBLIC"},
            agentRuntimeArtifact={
                "codeConfiguration": {
                    "code": {
                        "s3": {
                            "bucket": self.s3_bucket,
                            "prefix": self.s3_key,
                        }
                    },
                    "runtime": "PYTHON_3_12",
                    "entryPoint": ["agent.py"],
                }
            },
        )
        self.__class__.runtime_ids.append(runtime["agentRuntimeId"])
        assert runtime["status"] == "READY"

    @pytest.mark.order(5)
    def test_get_agent_runtime_passthrough(self):
        if not self.runtime_ids:
            pytest.skip("prerequisite test did not create runtime")
        runtime = self.client.get_agent_runtime(
            agentRuntimeId=self.runtime_ids[0],
        )
        assert runtime["status"] == "READY"

    @pytest.mark.order(6)
    def test_get_agent_runtime_snake_case(self):
        if not self.runtime_ids:
            pytest.skip("prerequisite test did not create runtime")
        runtime = self.client.get_agent_runtime(
            agent_runtime_id=self.runtime_ids[0],
        )
        assert runtime["status"] == "READY"

    @pytest.mark.order(7)
    def test_update_agent_runtime_and_wait(self):
        if not self.runtime_ids:
            pytest.skip("prerequisite test did not create runtime")
        updated = self.client.update_agent_runtime_and_wait(
            agentRuntimeId=self.runtime_ids[0],
            roleArn=self.role_arn,
            networkConfiguration={"networkMode": "PUBLIC"},
            agentRuntimeArtifact={
                "codeConfiguration": {
                    "code": {
                        "s3": {
                            "bucket": self.s3_bucket,
                            "prefix": self.s3_key,
                        }
                    },
                    "runtime": "PYTHON_3_12",
                    "entryPoint": ["agent.py"],
                }
            },
            description="updated by integ test",
        )
        assert updated["status"] == "READY"

    @pytest.mark.order(8)
    def test_create_agent_runtime_endpoint_and_wait(self):
        if not self.runtime_ids:
            pytest.skip("prerequisite test did not create runtime")
        endpoint = self.client.create_agent_runtime_endpoint_and_wait(
            agentRuntimeId=self.runtime_ids[0],
            name="sdk_integ_ep",
        )
        assert endpoint["status"] == "READY"

    @pytest.mark.order(9)
    def test_get_aggregated_status(self):
        if not self.runtime_ids:
            pytest.skip("prerequisite test did not create runtime")
        result = self.client.get_aggregated_status(
            self.runtime_ids[0],
            endpoint_name="sdk_integ_ep",
        )
        assert result["runtime"] is not None
        assert result["runtime"]["status"] == "READY"
        assert result["endpoint"] is not None
        assert result["endpoint"]["status"] == "READY"

    @pytest.mark.order(10)
    def test_teardown_endpoint_and_runtime(self):
        if not self.runtime_ids:
            pytest.skip("prerequisite test did not create runtime")
        rid = self.runtime_ids.pop(0)
        self.client.teardown_endpoint_and_runtime(
            rid,
            endpoint_name="sdk_integ_ep",
        )
        with pytest.raises(ClientError):
            self.client.get_agent_runtime(agentRuntimeId=rid)
