"""Unit tests for Memory Control Plane Client - no external connections."""

import uuid
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from bedrock_agentcore.memory.constants import MemoryStatus
from bedrock_agentcore.memory.controlplane import MemoryControlPlaneClient


def test_create_memory():
    """Test create_memory functionality."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock successful response
        mock_client.create_memory.return_value = {
            "memory": {"id": "mem-123", "name": "Test Memory", "status": "CREATING"}
        }

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test basic memory creation
            result = client.create_memory(name="Test Memory", description="Test description")

            assert result["id"] == "mem-123"
            assert result["name"] == "Test Memory"
            assert mock_client.create_memory.called

            # Verify correct parameters were passed
            args, kwargs = mock_client.create_memory.call_args
            assert kwargs["name"] == "Test Memory"
            assert kwargs["description"] == "Test description"
            assert kwargs["clientToken"] == "12345678-1234-5678-1234-567812345678"


def test_get_memory():
    """Test get_memory functionality."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock response with strategies
        mock_client.get_memory.return_value = {
            "memory": {
                "id": "mem-123",
                "name": "Test Memory",
                "status": "ACTIVE",
                "strategies": [
                    {"strategyId": "strat-1", "type": "SEMANTIC"},
                    {"strategyId": "strat-2", "type": "SUMMARY"},
                ],
            }
        }

        # Test get memory with strategies
        result = client.get_memory("mem-123")

        assert result["id"] == "mem-123"
        assert result["strategyCount"] == 2
        assert "strategies" in result

        # Verify API call
        mock_client.get_memory.assert_called_with(memoryId="mem-123")


def test_list_memories():
    """Test list_memories functionality."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock response
        mock_memories = [
            {"id": "mem-1", "name": "Memory 1", "status": "ACTIVE"},
            {"id": "mem-2", "name": "Memory 2", "status": "ACTIVE"},
        ]
        mock_client.list_memories.return_value = {"memories": mock_memories, "nextToken": None}

        # Test list memories
        result = client.list_memories(max_results=50)

        assert len(result) == 2
        assert result[0]["id"] == "mem-1"
        assert result[0]["strategyCount"] == 0  # List doesn't include strategies

        # Verify API call
        args, kwargs = mock_client.list_memories.call_args
        assert kwargs["maxResults"] == 50


def test_update_memory():
    """Test update_memory functionality."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock response
        mock_client.update_memory.return_value = {
            "memory": {"id": "mem-123", "name": "Updated Memory", "status": "CREATING"}
        }

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test memory update
            result = client.update_memory(memory_id="mem-123", description="Updated description", event_expiry_days=120)

            assert result["id"] == "mem-123"
            assert mock_client.update_memory.called

            # Verify correct parameters
            args, kwargs = mock_client.update_memory.call_args
            assert kwargs["memoryId"] == "mem-123"
            assert kwargs["description"] == "Updated description"
            assert kwargs["eventExpiryDuration"] == 120
            assert kwargs["clientToken"] == "12345678-1234-5678-1234-567812345678"


def test_delete_memory():
    """Test delete_memory functionality."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock response
        mock_client.delete_memory.return_value = {"status": "DELETING"}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test memory deletion
            result = client.delete_memory("mem-123")

            assert result["status"] == "DELETING"
            assert mock_client.delete_memory.called

            # Verify correct parameters
            args, kwargs = mock_client.delete_memory.call_args
            assert kwargs["memoryId"] == "mem-123"
            assert kwargs["clientToken"] == "12345678-1234-5678-1234-567812345678"


def test_delete_memory_wait_for_strategies():
    """Test delete_memory with wait_for_strategies=True."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock get_memory response with strategies in transitional state
        mock_client.get_memory.return_value = {
            "memory": {
                "id": "mem-123",
                "strategies": [
                    {"strategyId": "strat-1", "status": "CREATING"},  # Transitional state
                    {"strategyId": "strat-2", "status": "ACTIVE"},  # Already active
                ],
            }
        }

        # Mock delete_memory response
        mock_client.delete_memory.return_value = {"status": "DELETING"}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            with patch("time.time", return_value=0):
                with patch("time.sleep"):
                    # Mock the _wait_for_status method to avoid actual waiting
                    with patch.object(client, "_wait_for_status") as mock_wait:
                        mock_wait.return_value = {"id": "mem-123", "status": "ACTIVE"}

                        # Test memory deletion with wait_for_strategies=True
                        result = client.delete_memory("mem-123", wait_for_strategies=True)

                        assert result["status"] == "DELETING"

                        # Verify get_memory was called to check strategy status
                        assert mock_client.get_memory.called

                        # Verify _wait_for_status was called due to transitional strategy
                        mock_wait.assert_called_once_with(
                            memory_id="mem-123",
                            target_status=MemoryStatus.ACTIVE.value,
                            max_wait=300,
                            poll_interval=10,
                            check_strategies=True,
                        )

                        # Verify delete_memory was called
                        assert mock_client.delete_memory.called
                        args, kwargs = mock_client.delete_memory.call_args
                        assert kwargs["memoryId"] == "mem-123"


def test_delete_memory_wait_for_deletion():
    """Test delete_memory with wait_for_deletion=True."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock delete_memory response
        mock_client.delete_memory.return_value = {"status": "DELETING"}

        # Mock get_memory to first return the memory, then raise ResourceNotFoundException
        error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Memory not found"}}
        mock_client.get_memory.side_effect = ClientError(error_response, "GetMemory")

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            with patch("time.time", return_value=0):
                with patch("time.sleep"):
                    # Test memory deletion with wait_for_deletion=True
                    result = client.delete_memory("mem-123", wait_for_deletion=True, max_wait=120, poll_interval=5)

                    assert result["status"] == "DELETING"

                    # Verify delete_memory was called
                    assert mock_client.delete_memory.called
                    delete_args, delete_kwargs = mock_client.delete_memory.call_args
                    assert delete_kwargs["memoryId"] == "mem-123"
                    assert delete_kwargs["clientToken"] == "12345678-1234-5678-1234-567812345678"

                    # Verify get_memory was called (to check if memory is gone)
                    assert mock_client.get_memory.called
                    get_args, get_kwargs = mock_client.get_memory.call_args
                    assert get_kwargs["memoryId"] == "mem-123"


def test_add_strategy():
    """Test add_strategy functionality."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock update_memory response (add_strategy uses update_memory internally)
        mock_client.update_memory.return_value = {"memory": {"id": "mem-123", "status": "CREATING"}}

        # Test strategy addition
        strategy = {"semanticMemoryStrategy": {"name": "Test Strategy"}}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            result = client.add_strategy("mem-123", strategy)

            assert result["id"] == "mem-123"
            assert mock_client.update_memory.called

            # Verify strategy was passed correctly
            args, kwargs = mock_client.update_memory.call_args
            assert "memoryStrategies" in kwargs
            assert "addMemoryStrategies" in kwargs["memoryStrategies"]
            assert kwargs["memoryStrategies"]["addMemoryStrategies"][0] == strategy


def test_add_strategy_wait_for_active():
    """Test add_strategy with wait_for_active=True."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock update_memory response (add_strategy uses update_memory internally)
        mock_client.update_memory.return_value = {"memory": {"id": "mem-123", "status": "CREATING"}}

        # Mock get_memory response to find the newly added strategy
        mock_client.get_memory.return_value = {
            "memory": {
                "id": "mem-123",
                "strategies": [{"strategyId": "strat-new-123", "name": "Test Active Strategy", "status": "CREATING"}],
            }
        }

        # Test strategy addition with wait_for_active=True
        strategy = {"semanticMemoryStrategy": {"name": "Test Active Strategy"}}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Mock the _wait_for_strategy_active method to avoid actual waiting
            with patch.object(client, "_wait_for_strategy_active") as mock_wait:
                mock_wait.return_value = {"id": "mem-123", "status": "ACTIVE"}

                result = client.add_strategy("mem-123", strategy, wait_for_active=True, max_wait=120, poll_interval=5)

                assert result["id"] == "mem-123"
                assert mock_client.update_memory.called

                # Verify strategy was passed correctly to update_memory
                args, kwargs = mock_client.update_memory.call_args
                assert "memoryStrategies" in kwargs
                assert "addMemoryStrategies" in kwargs["memoryStrategies"]
                assert kwargs["memoryStrategies"]["addMemoryStrategies"][0] == strategy

                # Verify get_memory was called to find the newly added strategy
                assert mock_client.get_memory.called
                get_args, get_kwargs = mock_client.get_memory.call_args
                assert get_kwargs["memoryId"] == "mem-123"

                # Verify _wait_for_strategy_active was called with correct parameters
                mock_wait.assert_called_once_with("mem-123", "strat-new-123", 120, 5)


def test_get_strategy():
    """Test get_strategy functionality."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock get_memory response with strategies
        mock_client.get_memory.return_value = {
            "memory": {
                "id": "mem-123",
                "strategies": [
                    {"strategyId": "strat-1", "name": "Strategy 1", "type": "SEMANTIC"},
                    {"strategyId": "strat-2", "name": "Strategy 2", "type": "SUMMARY"},
                ],
            }
        }

        # Test getting specific strategy
        result = client.get_strategy("mem-123", "strat-1")

        assert result["strategyId"] == "strat-1"
        assert result["name"] == "Strategy 1"
        assert result["type"] == "SEMANTIC"


def test_update_strategy():
    """Test update_strategy functionality."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock update_memory response (update_strategy uses update_memory internally)
        mock_client.update_memory.return_value = {"memory": {"id": "mem-123", "status": "CREATING"}}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test strategy update
            result = client.update_strategy(
                memory_id="mem-123",
                strategy_id="strat-456",
                description="Updated strategy description",
                namespaces=["custom/namespace1", "custom/namespace2"],
                configuration={"modelId": "test-model"},
            )

            assert result["id"] == "mem-123"
            assert mock_client.update_memory.called

            # Verify correct parameters were passed
            args, kwargs = mock_client.update_memory.call_args
            assert kwargs["memoryId"] == "mem-123"
            assert "memoryStrategies" in kwargs
            assert "modifyMemoryStrategies" in kwargs["memoryStrategies"]

            # Verify the strategy modification details
            modify_strategy = kwargs["memoryStrategies"]["modifyMemoryStrategies"][0]
            assert modify_strategy["memoryStrategyId"] == "strat-456"
            assert modify_strategy["description"] == "Updated strategy description"
            assert modify_strategy["namespaces"] == ["custom/namespace1", "custom/namespace2"]
            assert modify_strategy["configuration"] == {"modelId": "test-model"}


def test_error_handling():
    """Test error handling."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the client to raise an error
        mock_client = MagicMock()
        client.client = mock_client

        error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid parameter"}}
        mock_client.create_memory.side_effect = ClientError(error_response, "CreateMemory")

        try:
            client.create_memory(name="Test Memory")
            raise AssertionError("Error was not raised as expected")
        except ClientError as e:
            assert "ValidationException" in str(e)


def test_wait_for_strategy_active():
    """Test _wait_for_strategy_active helper method."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock get_memory response - strategy becomes ACTIVE
        mock_client.get_memory.return_value = {
            "memory": {
                "id": "mem-123",
                "strategies": [{"strategyId": "strat-456", "status": "ACTIVE", "name": "Test Strategy"}],
            }
        }

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                # Test _wait_for_strategy_active
                result = client._wait_for_strategy_active("mem-123", "strat-456", max_wait=60, poll_interval=5)

                assert result["id"] == "mem-123"
                assert mock_client.get_memory.called

                # Verify correct parameters
                args, kwargs = mock_client.get_memory.call_args
                assert kwargs["memoryId"] == "mem-123"


def test_create_memory_with_strategies():
    """Test create_memory with memory strategies."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock successful response
        mock_client.create_memory.return_value = {
            "memory": {"id": "mem-456", "name": "Memory with Strategies", "status": "CREATING"}
        }

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test memory creation with strategies
            strategies = [{"semanticMemoryStrategy": {"name": "Strategy 1"}}]
            result = client.create_memory(
                name="Memory with Strategies",
                description="Test with strategies",
                strategies=strategies,
                event_expiry_days=120,
                memory_execution_role_arn="arn:aws:iam::123456789012:role/MemoryRole",
            )

            assert result["id"] == "mem-456"
            assert mock_client.create_memory.called

            # Verify all parameters were passed
            args, kwargs = mock_client.create_memory.call_args
            assert kwargs["name"] == "Memory with Strategies"
            assert kwargs["description"] == "Test with strategies"
            assert kwargs["memoryStrategies"] == strategies
            assert kwargs["eventExpiryDuration"] == 120
            assert kwargs["memoryExecutionRoleArn"] == "arn:aws:iam::123456789012:role/MemoryRole"


def test_list_memories_with_pagination():
    """Test list_memories with pagination."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock paginated responses
        first_batch = [{"id": f"mem-{i}", "name": f"Memory {i}", "status": "ACTIVE"} for i in range(1, 101)]
        second_batch = [{"id": f"mem-{i}", "name": f"Memory {i}", "status": "ACTIVE"} for i in range(101, 151)]

        mock_client.list_memories.side_effect = [
            {"memories": first_batch, "nextToken": "token-123"},
            {"memories": second_batch, "nextToken": None},
        ]

        # Test with max_results requiring pagination
        result = client.list_memories(max_results=150)

        assert len(result) == 150
        assert result[0]["id"] == "mem-1"
        assert result[149]["id"] == "mem-150"

        # Verify two API calls were made
        assert mock_client.list_memories.call_count == 2


def test_update_memory_minimal():
    """Test update_memory with minimal parameters."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock response
        mock_client.update_memory.return_value = {"memory": {"id": "mem-123", "status": "ACTIVE"}}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test minimal update (only memory_id)
            result = client.update_memory(memory_id="mem-123")

            assert result["id"] == "mem-123"
            assert mock_client.update_memory.called

            # Verify minimal parameters
            args, kwargs = mock_client.update_memory.call_args
            assert kwargs["memoryId"] == "mem-123"
            assert kwargs["clientToken"] == "12345678-1234-5678-1234-567812345678"


def test_wait_for_status_timeout():
    """Test _wait_for_status with timeout."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock get_memory to always return CREATING (never becomes ACTIVE)
        mock_client.get_memory.return_value = {"memory": {"id": "mem-timeout", "status": "CREATING", "strategies": []}}

        # Mock time to simulate timeout - provide enough values for all calls
        time_values = [0] + [i * 10 for i in range(1, 35)] + [301]  # Enough values for multiple checks
        with patch("time.time", side_effect=time_values):
            with patch("time.sleep"):
                try:
                    client._wait_for_status(
                        memory_id="mem-timeout", target_status="ACTIVE", max_wait=300, poll_interval=10
                    )
                    raise AssertionError("TimeoutError was not raised")
                except TimeoutError as e:
                    assert "did not reach status ACTIVE within 300 seconds" in str(e)


def test_wait_for_status_failure():
    """Test _wait_for_status with FAILED status."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock get_memory to return FAILED status
        mock_client.get_memory.return_value = {
            "memory": {"id": "mem-failed", "status": "FAILED", "failureReason": "Configuration error", "strategies": []}
        }

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                try:
                    client._wait_for_status(
                        memory_id="mem-failed", target_status="ACTIVE", max_wait=300, poll_interval=10
                    )
                    raise AssertionError("RuntimeError was not raised")
                except RuntimeError as e:
                    assert "Memory operation failed: Configuration error" in str(e)


def test_wait_for_strategy_active_timeout():
    """Test _wait_for_strategy_active with timeout."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock get_memory response - strategy never becomes ACTIVE
        mock_client.get_memory.return_value = {
            "memory": {"id": "mem-123", "strategies": [{"strategyId": "strat-timeout", "status": "CREATING"}]}
        }

        # Mock time to simulate timeout - provide enough values for multiple calls
        time_values = [0] + [i * 10 for i in range(1, 35)] + [301]
        with patch("time.time", side_effect=time_values):
            with patch("time.sleep"):
                try:
                    client._wait_for_strategy_active("mem-123", "strat-timeout", max_wait=300, poll_interval=10)
                    raise AssertionError("TimeoutError was not raised")
                except TimeoutError as e:
                    assert "Strategy strat-timeout did not become ACTIVE within 300 seconds" in str(e)


def test_wait_for_strategy_active_not_found():
    """Test _wait_for_strategy_active when strategy is not found."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock get_memory response - strategy doesn't exist
        mock_client.get_memory.return_value = {
            "memory": {"id": "mem-123", "strategies": [{"strategyId": "strat-other", "status": "ACTIVE"}]}
        }

        # Mock time to simulate timeout - provide enough values for multiple calls
        time_values = [0] + [i * 5 for i in range(1, 15)] + [61]
        with patch("time.time", side_effect=time_values):
            with patch("time.sleep"):
                try:
                    client._wait_for_strategy_active("mem-123", "strat-nonexistent", max_wait=60, poll_interval=5)
                    raise AssertionError("TimeoutError was not raised")
                except TimeoutError as e:
                    assert "Strategy strat-nonexistent did not become ACTIVE within 60 seconds" in str(e)


def test_get_strategy_not_found():
    """Test get_strategy when strategy doesn't exist."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock get_memory response without the requested strategy
        mock_client.get_memory.return_value = {
            "memory": {
                "id": "mem-123",
                "strategies": [{"strategyId": "strat-other", "name": "Other Strategy", "type": "SEMANTIC"}],
            }
        }

        try:
            client.get_strategy("mem-123", "strat-nonexistent")
            raise AssertionError("ValueError was not raised")
        except ValueError as e:
            assert "Strategy strat-nonexistent not found in memory mem-123" in str(e)


def test_delete_memory_wait_for_deletion_timeout():
    """Test delete_memory with wait_for_deletion timeout."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock delete_memory response
        mock_client.delete_memory.return_value = {"status": "DELETING"}

        # Mock get_memory to always succeed (memory never gets deleted)
        mock_client.get_memory.return_value = {"memory": {"id": "mem-persistent", "status": "DELETING"}}

        # Mock time to simulate timeout
        # Provide enough values for multiple time.time() calls in the loop
        with patch("time.time", side_effect=[0, 0, 0, 301, 301, 301]):
            with patch("time.sleep"):
                with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                    try:
                        client.delete_memory("mem-persistent", wait_for_deletion=True, max_wait=300, poll_interval=10)
                        raise AssertionError("TimeoutError was not raised")
                    except TimeoutError as e:
                        assert "Memory mem-persistent was not deleted within 300 seconds" in str(e)


def test_wait_for_status_with_strategy_check():
    """Test _wait_for_status with check_strategies=True and transitional strategies."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock get_memory responses - first with transitional strategy, then all active
        mock_client.get_memory.side_effect = [
            {
                "memory": {
                    "id": "mem-123",
                    "status": "ACTIVE",
                    "strategies": [
                        {"strategyId": "strat-1", "status": "CREATING"},  # Transitional
                        {"strategyId": "strat-2", "status": "ACTIVE"},  # Already active
                    ],
                }
            },
            {
                "memory": {
                    "id": "mem-123",
                    "status": "ACTIVE",
                    "strategies": [
                        {"strategyId": "strat-1", "status": "ACTIVE"},  # Now active
                        {"strategyId": "strat-2", "status": "ACTIVE"},
                    ],
                }
            },
        ]

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                # Test _wait_for_status with check_strategies=True
                result = client._wait_for_status(
                    memory_id="mem-123", target_status="ACTIVE", max_wait=120, poll_interval=10, check_strategies=True
                )

                assert result["id"] == "mem-123"
                assert result["status"] == "ACTIVE"

                # Should have made two calls - one found transitional strategy, second found all active
                assert mock_client.get_memory.call_count == 2


def test_add_strategy_strategy_not_found():
    """Test add_strategy when newly added strategy cannot be found."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock update_memory response
        mock_client.update_memory.return_value = {"memory": {"id": "mem-123", "status": "CREATING"}}

        # Mock get_memory response without the newly added strategy
        mock_client.get_memory.return_value = {
            "memory": {
                "id": "mem-123",
                "status": "ACTIVE",
                "strategies": [],  # No strategies found
            }
        }

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            strategy = {"semanticMemoryStrategy": {"name": "Missing Strategy"}}

            # The actual implementation just logs a warning and returns the memory
            # It doesn't raise an exception
            result = client.add_strategy("mem-123", strategy, wait_for_active=True)

            # Should return the memory object from get_memory (since wait_for_active=True)
            assert result["id"] == "mem-123"
            assert result["status"] == "ACTIVE"


def test_initialization_with_env_vars():
    """Test initialization with environment variables."""
    with patch("boto3.client") as mock_boto_client:
        with patch("os.getenv") as mock_getenv:
            # Mock environment variables - use the correct names from controlplane.py
            env_vars = {
                "BEDROCK_AGENTCORE_CONTROL_ENDPOINT": "https://custom-control.amazonaws.com",
                "BEDROCK_AGENTCORE_CONTROL_SERVICE": "custom-control-service",
            }
            mock_getenv.side_effect = lambda key, default=None: env_vars.get(key, default)

            # Test initialization with custom environment
            MemoryControlPlaneClient()

            # Verify boto3.client was called with custom endpoint
            mock_boto_client.assert_called_with(
                "custom-control-service", region_name="us-west-2", endpoint_url="https://custom-control.amazonaws.com"
            )


def test_wait_for_status():
    """Test _wait_for_status helper method."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock get_memory response - memory becomes ACTIVE
        mock_client.get_memory.return_value = {
            "memory": {
                "id": "mem-123",
                "status": "ACTIVE",
                "strategies": [
                    {"strategyId": "strat-1", "status": "ACTIVE"},
                    {"strategyId": "strat-2", "status": "ACTIVE"},
                ],
            }
        }

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                # Test _wait_for_status with check_strategies=True
                result = client._wait_for_status(
                    memory_id="mem-123", target_status="ACTIVE", max_wait=120, poll_interval=10, check_strategies=True
                )

                assert result["id"] == "mem-123"
                assert result["status"] == "ACTIVE"
                assert mock_client.get_memory.called

                # Verify correct parameters
                args, kwargs = mock_client.get_memory.call_args
                assert kwargs["memoryId"] == "mem-123"


def test_get_memory_client_error():
    """Test get_memory with ClientError."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock ClientError
        error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Memory not found"}}
        mock_client.get_memory.side_effect = ClientError(error_response, "GetMemory")

        try:
            client.get_memory("nonexistent-mem-123")
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ResourceNotFoundException" in str(e)


def test_list_memories_client_error():
    """Test list_memories with ClientError."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock ClientError
        error_response = {"Error": {"Code": "AccessDeniedException", "Message": "Insufficient permissions"}}
        mock_client.list_memories.side_effect = ClientError(error_response, "ListMemories")

        try:
            client.list_memories(max_results=50)
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "AccessDeniedException" in str(e)


def test_update_memory_client_error():
    """Test update_memory with ClientError."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock ClientError
        error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid memory parameters"}}
        mock_client.update_memory.side_effect = ClientError(error_response, "UpdateMemory")

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            try:
                client.update_memory(memory_id="mem-123", description="Updated description")
                raise AssertionError("ClientError was not raised")
            except ClientError as e:
                assert "ValidationException" in str(e)


def test_delete_memory_client_error():
    """Test delete_memory with ClientError."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock ClientError
        error_response = {"Error": {"Code": "ConflictException", "Message": "Memory is in use"}}
        mock_client.delete_memory.side_effect = ClientError(error_response, "DeleteMemory")

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            try:
                client.delete_memory("mem-in-use")
                raise AssertionError("ClientError was not raised")
            except ClientError as e:
                assert "ConflictException" in str(e)


def test_get_strategy_client_error():
    """Test get_strategy with ClientError from get_memory."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock ClientError from get_memory call
        error_response = {"Error": {"Code": "ThrottlingException", "Message": "Request throttled"}}
        mock_client.get_memory.side_effect = ClientError(error_response, "GetMemory")

        try:
            client.get_strategy("mem-123", "strat-456")
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ThrottlingException" in str(e)


def test_wait_for_strategy_active_client_error():
    """Test _wait_for_strategy_active with ClientError."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock ClientError
        error_response = {"Error": {"Code": "ServiceException", "Message": "Internal service error"}}
        mock_client.get_memory.side_effect = ClientError(error_response, "GetMemory")

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                try:
                    client._wait_for_strategy_active("mem-123", "strat-456", max_wait=60, poll_interval=5)
                    raise AssertionError("ClientError was not raised")
                except ClientError as e:
                    assert "ServiceException" in str(e)


def test_wait_for_status_client_error():
    """Test _wait_for_status with ClientError."""
    with patch("boto3.client"):
        client = MemoryControlPlaneClient()

        # Mock the boto3 client
        mock_client = MagicMock()
        client.client = mock_client

        # Mock ClientError
        error_response = {"Error": {"Code": "InternalServerError", "Message": "Internal server error"}}
        mock_client.get_memory.side_effect = ClientError(error_response, "GetMemory")

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                try:
                    client._wait_for_status(memory_id="mem-123", target_status="ACTIVE", max_wait=120, poll_interval=10)
                    raise AssertionError("ClientError was not raised")
                except ClientError as e:
                    assert "InternalServerError" in str(e)
