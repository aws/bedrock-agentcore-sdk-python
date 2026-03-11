"""Integration tests for MemoryClient.

Requires environment variables:
    MEMORY_KINESIS_ARN: ARN of a pre-existing Kinesis stream
    MEMORY_ROLE_ARN: ARN of an IAM role the memory service can assume

Run with:
    pytest tests_integ/memory/test_memory_client.py -xvs --log-cli-level=INFO
"""

import os
import time
import uuid

import pytest

from bedrock_agentcore.memory import MemoryClient


@pytest.mark.integration
class TestMemoryClient:
    """Integration tests for MemoryClient."""

    @classmethod
    def setup_class(cls):
        cls.kinesis_stream_arn = os.environ.get("MEMORY_KINESIS_ARN")
        cls.execution_role_arn = os.environ.get("MEMORY_ROLE_ARN")

        if not cls.kinesis_stream_arn or not cls.execution_role_arn:
            pytest.fail("MEMORY_KINESIS_ARN and MEMORY_ROLE_ARN must be set")

        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.client = MemoryClient(region_name=cls.region)
        cls.test_prefix = f"test_stream_{int(time.time())}"
        cls.memory_ids = []

    @classmethod
    def teardown_class(cls):
        for memory_id in cls.memory_ids:
            try:
                cls.client.delete_memory(memory_id)
            except Exception as e:
                print(f"Failed to delete memory {memory_id}: {e}")

    def _make_delivery_config(self, level="FULL_CONTENT"):
        return {
            "resources": [
                {
                    "kinesis": {
                        "dataStreamArn": self.kinesis_stream_arn,
                        "contentConfigurations": [{"type": "MEMORY_RECORDS", "level": level}],
                    }
                }
            ]
        }

    def test_stream_delivery_create_and_update(self):
        """Create memory with FULL_CONTENT delivery, update to METADATA_ONLY via passthrough."""
        delivery_config = self._make_delivery_config("FULL_CONTENT")

        memory = self.client.create_memory_and_wait(
            name=f"{self.test_prefix}_stream",
            strategies=[],
            memory_execution_role_arn=self.execution_role_arn,
            stream_delivery_resources=delivery_config,
        )

        memory_id = memory["id"]
        self.__class__.memory_ids.append(memory_id)

        assert memory["streamDeliveryResources"] == delivery_config

        # Test update via MemoryClient.__getattr__ passthrough to boto3 client.
        # Uses camelCase params because the passthrough forwards directly to boto3
        # without the snake_case translation that explicit SDK methods provide.
        updated_config = self._make_delivery_config("METADATA_ONLY")
        response = self.client.update_memory(
            memoryId=memory_id,
            clientToken=str(uuid.uuid4()),
            streamDeliveryResources=updated_config,
        )
        assert response["memory"]["streamDeliveryResources"] == updated_config
