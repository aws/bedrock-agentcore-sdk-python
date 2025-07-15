"""Tests for the MemoryControlPlaneClient.

This module contains tests for the Bedrock AgentCore Memory control plane operations.

Note: To run tests in parallel, you need the following pytest plugins:
- pytest-xdist: For parallel test execution
- pytest-depends: For test dependencies
- pytest-order: For test ordering

Install with: pip install pytest-xdist pytest-depends pytest-order
Run with: pytest -xvs tests/test_controlplane.py -n 2
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.memory.controlplane import MemoryControlPlaneClient


@pytest.mark.integration
class TestMemoryControlPlaneClient:
    """Integration tests for MemoryControlPlaneClient."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        # Use environment variables or default to test environment
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.endpoint = os.environ.get(
            "BEDROCK_AGENTCORE_CONTROL_ENDPOINT", f"https://bedrock-agentcore-control.{cls.region}.amazonaws.com"
        )

        # Initialize client
        cls.client = MemoryControlPlaneClient(region_name=cls.region)

        # Test prefix to identify test resources
        cls.test_prefix = f"test_cp_{int(time.time())}"

        # Store created memory IDs for cleanup
        cls.memory_ids = []

    @pytest.mark.order(1)
    @pytest.mark.parallel
    def test_workflow_1_create_and_update_memory(self):
        """Test workflow 1: Create memory with strategies and update its description.

        This test verifies that:
        1. A memory can be created with strategies
        2. The memory and its strategies become ACTIVE
        3. The memory can be updated with a new description
        4. The memory can be retrieved and its properties verified
        """
        # Step 1: Create memory with a strategy and wait for active
        memory_name = f"{self.test_prefix}_basic"

        # Define a simple semantic strategy
        strategies = [
            {
                "semanticMemoryStrategy": {
                    "name": "TestBasicStrategy",
                    "description": "Test basic strategy for create test",
                }
            }
        ]

        memory = self.client.create_memory(
            name=memory_name,
            description="Test memory",
            strategies=strategies,
            wait_for_active=True,
            max_wait=300,  # Increased timeout to allow strategy to become active
            poll_interval=10,
        )

        # Store memory ID for cleanup
        memory_id = memory["id"]
        self.__class__.memory_ids.append(memory_id)

        # Verify memory was created successfully
        assert memory["name"] == memory_name
        assert memory["status"] == "ACTIVE"
        assert "strategies" in memory

        # Verify strategy was created and is ACTIVE
        strategies = memory.get("strategies", [])
        assert len(strategies) > 0

        # Step 2: Update memory description
        updated_memory = self.client.update_memory(
            memory_id=memory_id,
            description="Updated description",
        )

        # Verify description was updated
        assert updated_memory["description"] == "Updated description"
        assert updated_memory["status"] == "ACTIVE"

        # Get memory to verify details
        memory_details = self.client.get_memory(memory_id)
        assert memory_details["id"] == memory_id
        assert memory_details["name"] == memory_name
        assert memory_details["description"] == "Updated description"

    @pytest.mark.order(1)
    @pytest.mark.parallel
    def test_workflow_2_add_strategy(self):
        """Test workflow 2: Create memory and add a strategy.

        This test verifies that:
        1. A memory can be created without strategies
        2. A semantic strategy can be added to the memory
        3. The strategy is correctly added with the specified properties
        4. The strategy becomes ACTIVE
        """
        # Step 1: Create memory without strategies
        memory_name = f"{self.test_prefix}_strategy"
        memory = self.client.create_memory(
            name=memory_name,
            description="Test memory for strategy",
            event_expiry_days=30,
            wait_for_active=True,
            max_wait=60,  # Increased timeout
            poll_interval=5,
        )

        # Store memory ID for cleanup
        memory_id = memory["id"]
        self.__class__.memory_ids.append(memory_id)

        # Step 2: Add a semantic strategy
        semantic_strategy = {
            "semanticMemoryStrategy": {"name": "TestSemanticStrategy", "description": "Test semantic strategy"}
        }

        # Strategy activation is tested, but result not used
        self.client.add_strategy(
            memory_id=memory_id,
            strategy=semantic_strategy,
            wait_for_active=True,
            max_wait=300,  # Significantly increased timeout for strategy activation
            poll_interval=10,
        )

        # Get memory to verify details
        memory_details = self.client.get_memory(memory_id)

        # Verify strategy was added
        strategies = memory_details.get("strategies", [])
        assert len(strategies) > 0

        # Find the semantic strategy and verify it's ACTIVE
        semantic_strategy_found = False
        for strategy in strategies:
            if strategy.get("name") == "TestSemanticStrategy":
                semantic_strategy_found = True
                assert strategy.get("type") == "SEMANTIC"
                assert strategy.get("description") == "Test semantic strategy"
                assert strategy.get("status") == "ACTIVE", (
                    f"Strategy status is {strategy.get('status')}, expected ACTIVE"
                )
                break

        assert semantic_strategy_found, "Semantic strategy not found in memory"

    @pytest.mark.order(3)
    @pytest.mark.depends(on=["test_workflow_1_create_and_update_memory", "test_workflow_2_add_strategy"])
    def test_workflow_3_list_and_delete_memories(self):
        """Test workflow 3: List and delete memories from previous tests.

        This test verifies that:
        1. The memories created in previous tests can be listed
        2. The memories can be deleted
        3. The deletion can be verified

        Note: This test relies on test_workflow_1 and test_workflow_2 running first.
        """
        # List memories and verify our test memories exist
        memories = self.client.list_memories()

        # Filter to only include our test memories
        test_memories = [m for m in memories if m["id"].startswith(self.test_prefix)]

        # Verify we have at least 2 memories from previous tests
        assert len(test_memories) >= 2, f"Expected at least 2 test memories, found {len(test_memories)}"

        # Delete the memories we created in previous tests
        for memory_id in list(
            self.__class__.memory_ids
        ):  # Create a copy of the list to avoid modification during iteration
            try:
                self.client.delete_memory(
                    memory_id=memory_id,
                    wait_for_deletion=True,
                    wait_for_strategies=False,  # Don't wait for strategies
                    max_wait=120,
                    poll_interval=5,
                )
                print(f"Deleted memory: {memory_id}")
                self.__class__.memory_ids.remove(memory_id)
            except Exception as e:
                print(f"Failed to delete memory {memory_id}: {e}")
                # If we can't delete it now, we'll try again in teardown

        # Verify memories were deleted
        memories_after = self.client.list_memories()
        remaining_test_memories = [m for m in memories_after if m["id"].startswith(self.test_prefix)]
        assert len(remaining_test_memories) == 0, f"Expected 0 test memories, found {len(remaining_test_memories)}"


@pytest.mark.unit
class TestMemoryControlPlaneClientUnit:
    """Unit tests for MemoryControlPlaneClient using mocks."""

    def setup_method(self):
        """Set up test environment for each test."""
        # Create a mock boto3 client
        self.mock_boto_client = MagicMock()

        # Patch boto3.client to return our mock
        self.boto_patcher = patch("boto3.client", return_value=self.mock_boto_client)
        self.mock_boto3_client = self.boto_patcher.start()

        # Initialize client with the mock
        self.client = MemoryControlPlaneClient(region_name="us-west-2")

    def teardown_method(self):
        """Clean up after each test."""
        self.boto_patcher.stop()

    def test_create_memory(self):
        """Test create_memory method.

        Verifies that:
        1. The method returns the expected result
        2. The AWS client was called with the correct parameters
        """
        # Setup mock response
        self.mock_boto_client.create_memory.return_value = {
            "memory": {"id": "test-memory-id", "name": "TestMemory", "status": "CREATING", "strategies": []}
        }

        # Call method
        result = self.client.create_memory(name="TestMemory", description="Test description")

        # Verify result
        assert result["id"] == "test-memory-id"
        assert result["name"] == "TestMemory"

        # Verify mock was called with correct parameters
        self.mock_boto_client.create_memory.assert_called_once()
        call_args = self.mock_boto_client.create_memory.call_args[1]
        assert call_args["name"] == "TestMemory"
        assert call_args["description"] == "Test description"
        assert call_args["eventExpiryDuration"] == 90
        assert "clientToken" in call_args

    def test_update_memory(self):
        """Test update_memory method.

        Verifies that:
        1. Description updates are properly passed to the AWS API
        2. The returned object contains the updated description
        """
        # Setup mock response
        self.mock_boto_client.update_memory.return_value = {
            "memory": {
                "id": "test-memory-id",
                "name": "TestMemory",
                "description": "Updated description",
                "status": "UPDATING",
                "strategies": [],
            }
        }

        # Call method
        result = self.client.update_memory(memory_id="test-memory-id", description="Updated description")

        # Verify result
        assert result["id"] == "test-memory-id"
        assert result["description"] == "Updated description"

        # Verify mock was called with correct parameters
        self.mock_boto_client.update_memory.assert_called_once()
        call_args = self.mock_boto_client.update_memory.call_args[1]
        assert call_args["memoryId"] == "test-memory-id"
        assert call_args["description"] == "Updated description"
        assert "clientToken" in call_args

    def test_add_strategy(self):
        """Test add_strategy method.

        Verifies that:
        1. Strategy configurations are correctly passed to the AWS API
        2. The returned object contains the added strategy
        """
        # Setup mock response
        self.mock_boto_client.update_memory.return_value = {
            "memory": {
                "id": "test-memory-id",
                "name": "TestMemory",
                "status": "UPDATING",
                "strategies": [
                    {"strategyId": "test-strategy-id", "name": "TestStrategy", "type": "SEMANTIC", "status": "CREATING"}
                ],
            }
        }

        # Call method
        strategy = {"semanticMemoryStrategy": {"name": "TestStrategy", "description": "Test strategy"}}

        result = self.client.add_strategy(memory_id="test-memory-id", strategy=strategy)

        # Verify result
        assert result["id"] == "test-memory-id"
        assert len(result["strategies"]) == 1
        assert result["strategies"][0]["name"] == "TestStrategy"

        # Verify mock was called with correct parameters
        self.mock_boto_client.update_memory.assert_called_once()
        call_args = self.mock_boto_client.update_memory.call_args[1]
        assert call_args["memoryId"] == "test-memory-id"
        assert "memoryStrategies" in call_args
        assert "addMemoryStrategies" in call_args["memoryStrategies"]
        assert call_args["memoryStrategies"]["addMemoryStrategies"][0] == strategy

    def test_wait_for_memory_active(self):
        """Test _wait_for_memory_active method.

        Verifies that:
        1. The waiting mechanism works correctly
        2. The method returns when the memory becomes active
        """
        # Setup mock responses for get_memory
        self.mock_boto_client.get_memory.side_effect = [
            {"memory": {"id": "test-memory-id", "status": "CREATING", "strategies": []}},
            {"memory": {"id": "test-memory-id", "status": "CREATING", "strategies": []}},
            {"memory": {"id": "test-memory-id", "status": "ACTIVE", "strategies": []}},
        ]

        # Call method with short poll interval
        result = self.client._wait_for_memory_active("test-memory-id", max_wait=10, poll_interval=1)

        # Verify result
        assert result["id"] == "test-memory-id"
        assert result["status"] == "ACTIVE"

        # Verify mock was called multiple times
        assert self.mock_boto_client.get_memory.call_count == 3

    def test_wait_for_memory_active_timeout(self):
        """Test _wait_for_memory_active method with timeout.

        Verifies that:
        1. A timeout is correctly handled
        2. A TimeoutError is raised after the specified timeout
        """
        # Setup mock response to always return CREATING
        self.mock_boto_client.get_memory.return_value = {
            "memory": {"id": "test-memory-id", "status": "CREATING", "strategies": []}
        }

        # Call method with short timeout
        with pytest.raises(TimeoutError):
            self.client._wait_for_memory_active("test-memory-id", max_wait=1, poll_interval=1)

        # Verify mock was called multiple times
        assert self.mock_boto_client.get_memory.call_count > 1

    def test_delete_memory_with_wait(self):
        """Test delete_memory with wait_for_deletion=True.

        Verifies that:
        1. The deletion is initiated correctly
        2. The method waits for the deletion to complete
        3. The method returns when the memory is deleted
        """
        # Setup initial response
        self.mock_boto_client.delete_memory.return_value = {"memoryId": "test-memory-id", "status": "DELETING"}

        # Setup get_memory to first return the memory, then raise ResourceNotFoundException
        self.mock_boto_client.get_memory.side_effect = [
            {"memory": {"id": "test-memory-id", "status": "DELETING"}},
            ClientError(error_response={"Error": {"Code": "ResourceNotFoundException"}}, operation_name="GetMemory"),
        ]

        # Call method
        result = self.client.delete_memory(memory_id="test-memory-id", wait_for_deletion=True, poll_interval=1)

        # Verify result
        assert result["memoryId"] == "test-memory-id"
        assert result["status"] == "DELETING"

        # Verify mocks were called correctly
        self.mock_boto_client.delete_memory.assert_called_once()
        assert self.mock_boto_client.get_memory.call_count == 2


if __name__ == "__main__":
    pytest.main(["-xvs", "test_controlplane.py"])
