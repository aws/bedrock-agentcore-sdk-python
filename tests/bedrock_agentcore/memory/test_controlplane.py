"""Unit tests for Memory Control Plane Client - no external connections."""

import uuid
import warnings
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.memory.constants import MemoryStatus
from bedrock_agentcore.memory.controlplane import MemoryControlPlaneClient

# Suppress the deprecation warnings for all tests except the ones that explicitly test them
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _create_client():
    """Helper to create a MemoryControlPlaneClient with a mocked boto3 client passed via constructor."""
    mock_client = MagicMock()
    client = MemoryControlPlaneClient(client=mock_client)
    return client, mock_client


def test_deprecation_warning():
    """Test that MemoryControlPlaneClient emits a deprecation warning on init."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        MemoryControlPlaneClient(client=MagicMock())
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "MemoryClient" in str(deprecation_warnings[0].message)
        assert "v1.4.0" in str(deprecation_warnings[0].message)


def test_method_deprecation_warnings():
    """Test that each public method emits its own deprecation warning."""
    client, mock_client = _create_client()

    mock_client.create_memory.return_value = {"memory": {"id": "mem-123", "name": "Test", "status": "CREATING"}}
    mock_client.get_memory.return_value = {"memory": {"id": "mem-123", "status": "ACTIVE", "strategies": []}}
    mock_client.list_memories.return_value = {"memories": [], "nextToken": None}
    mock_client.update_memory.return_value = {"memory": {"id": "mem-123", "status": "CREATING"}}
    mock_client.delete_memory.return_value = {"status": "DELETING"}

    methods_and_suggestions = [
        (lambda: client.create_memory(name="Test"), "create_memory"),
        (lambda: client.get_memory("mem-123"), "get_memory"),
        (lambda: client.list_memories(), "list_memories"),
        (lambda: client.update_memory(memory_id="mem-123"), "update_memory"),
        (lambda: client.delete_memory("mem-123"), "delete_memory"),
        (lambda: client.get_strategy("mem-123", "strat-1"), "get_strategy"),
    ]

    for method_call, method_name in methods_and_suggestions:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                method_call()
            except (ValueError, ClientError):
                pass  # Some methods may raise due to mock setup
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            method_warnings = [
                x for x in deprecation_warnings if f"MemoryControlPlaneClient.{method_name}()" in str(x.message)
            ]
            assert len(method_warnings) >= 1, f"No deprecation warning for {method_name}"
            assert "MemoryClient" in str(method_warnings[0].message)


def test_create_memory():
    """Test create_memory functionality."""
    client, mock_client = _create_client()

    # Mock successful response
    mock_client.create_memory.return_value = {"memory": {"id": "mem-123", "name": "Test Memory", "status": "CREATING"}}

    with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
        result = client.create_memory(name="Test Memory", description="Test description")

        assert result["id"] == "mem-123"
        assert result["name"] == "Test Memory"
        assert mock_client.create_memory.called

        # Verify core parameters were passed
        args, kwargs = mock_client.create_memory.call_args
        assert kwargs["name"] == "Test Memory"
        assert kwargs["description"] == "Test description"
        assert kwargs["clientToken"] == "12345678-1234-5678-1234-567812345678"


def test_get_memory():
    """Test get_memory functionality."""
    client, mock_client = _create_client()

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

    result = client.get_memory("mem-123")

    assert result["id"] == "mem-123"
    assert result["strategyCount"] == 2
    assert "strategies" in result

    # Verify API call
    mock_client.get_memory.assert_called_with(memoryId="mem-123")


def test_list_memories():
    """Test list_memories functionality."""
    client, mock_client = _create_client()

    mock_memories = [
        {"id": "mem-1", "name": "Memory 1", "status": "ACTIVE"},
        {"id": "mem-2", "name": "Memory 2", "status": "ACTIVE"},
    ]
    mock_client.list_memories.return_value = {"memories": mock_memories, "nextToken": None}

    result = client.list_memories(max_results=50)

    assert len(result) == 2
    assert result[0]["id"] == "mem-1"
    assert result[0]["strategyCount"] == 0
    assert result[1]["strategyCount"] == 0

    # Verify API call
    args, kwargs = mock_client.list_memories.call_args
    assert kwargs["maxResults"] == 50


def test_update_memory():
    """Test update_memory functionality."""
    client, mock_client = _create_client()

    mock_client.update_memory.return_value = {
        "memory": {"id": "mem-123", "name": "Updated Memory", "status": "CREATING"}
    }

    with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
        result = client.update_memory(memory_id="mem-123", description="Updated description", event_expiry_days=120)

        assert result["id"] == "mem-123"
        assert mock_client.update_memory.called

        args, kwargs = mock_client.update_memory.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert kwargs["description"] == "Updated description"
        assert kwargs["eventExpiryDuration"] == 120
        assert kwargs["clientToken"] == "12345678-1234-5678-1234-567812345678"


def test_delete_memory():
    """Test delete_memory functionality."""
    client, mock_client = _create_client()

    mock_client.delete_memory.return_value = {"status": "DELETING"}

    with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
        result = client.delete_memory("mem-123")

        assert result["status"] == "DELETING"
        assert mock_client.delete_memory.called

        args, kwargs = mock_client.delete_memory.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert kwargs["clientToken"] == "12345678-1234-5678-1234-567812345678"


def test_delete_memory_wait_for_strategies():
    """Test delete_memory with wait_for_strategies=True."""
    client, mock_client = _create_client()

    # First call from get_memory (initial check): one strategy still CREATING
    # Second call from _wait_for_status polling: all active
    mock_client.get_memory.side_effect = [
        {
            "memory": {
                "id": "mem-123",
                "status": "ACTIVE",
                "strategies": [
                    {"strategyId": "strat-1", "status": "CREATING"},
                    {"strategyId": "strat-2", "status": "ACTIVE"},
                ],
            }
        },
        {
            "memory": {
                "id": "mem-123",
                "status": "ACTIVE",
                "strategies": [
                    {"strategyId": "strat-1", "status": "ACTIVE"},
                    {"strategyId": "strat-2", "status": "ACTIVE"},
                ],
            }
        },
    ]

    mock_client.delete_memory.return_value = {"status": "DELETING"}

    with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                result = client.delete_memory("mem-123", wait_for_strategies=True)

                assert result["status"] == "DELETING"
                assert mock_client.get_memory.call_count == 2
                assert mock_client.delete_memory.called


def test_delete_memory_wait_for_deletion():
    """Test delete_memory with wait_for_deletion=True."""
    client, mock_client = _create_client()

    mock_client.delete_memory.return_value = {"status": "DELETING"}

    error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Memory not found"}}
    mock_client.get_memory.side_effect = ClientError(error_response, "GetMemory")

    with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                result = client.delete_memory("mem-123", wait_for_deletion=True, max_wait=120, poll_interval=5)

                assert result["status"] == "DELETING"
                assert mock_client.delete_memory.called
                assert mock_client.get_memory.called


def test_add_strategy():
    """Test add_strategy functionality."""
    client, mock_client = _create_client()

    mock_client.update_memory.return_value = {"memory": {"id": "mem-123", "status": "CREATING"}}

    strategy = {"semanticMemoryStrategy": {"name": "Test Strategy"}}

    with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
        result = client.add_strategy("mem-123", strategy)

        assert result["id"] == "mem-123"
        assert mock_client.update_memory.called

        args, kwargs = mock_client.update_memory.call_args
        assert "memoryStrategies" in kwargs
        assert "addMemoryStrategies" in kwargs["memoryStrategies"]
        added = kwargs["memoryStrategies"]["addMemoryStrategies"][0]
        assert "semanticMemoryStrategy" in added
        assert added["semanticMemoryStrategy"]["name"] == "Test Strategy"


def test_add_strategy_wait_for_active():
    """Test add_strategy with wait_for_active=True."""
    client, mock_client = _create_client()

    # First: update_memory response
    mock_client.update_memory.return_value = {"memory": {"id": "mem-123", "status": "CREATING"}}

    # Subsequent: get_memory calls for strategy lookup and wait
    mock_client.get_memory.side_effect = [
        # get_memory call to find newly added strategy
        {
            "memory": {
                "id": "mem-123",
                "status": "ACTIVE",
                "strategies": [
                    {"strategyId": "strat-new-123", "name": "Test Active Strategy", "status": "CREATING"},
                ],
            }
        },
        # _wait_for_strategy_active polling: strategy now ACTIVE
        {
            "memory": {
                "id": "mem-123",
                "status": "ACTIVE",
                "strategies": [
                    {"strategyId": "strat-new-123", "name": "Test Active Strategy", "status": "ACTIVE"},
                ],
            }
        },
    ]

    strategy = {"semanticMemoryStrategy": {"name": "Test Active Strategy"}}

    with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                result = client.add_strategy("mem-123", strategy, wait_for_active=True, max_wait=120, poll_interval=5)

                assert result["id"] == "mem-123"
                assert mock_client.update_memory.called


def test_get_strategy():
    """Test get_strategy functionality."""
    client, mock_client = _create_client()

    mock_client.get_memory.return_value = {
        "memory": {
            "id": "mem-123",
            "strategies": [
                {"strategyId": "strat-1", "name": "Strategy 1", "type": "SEMANTIC"},
                {"strategyId": "strat-2", "name": "Strategy 2", "type": "SUMMARY"},
            ],
        }
    }

    result = client.get_strategy("mem-123", "strat-1")

    assert result["strategyId"] == "strat-1"
    assert result["name"] == "Strategy 1"


def test_update_strategy():
    """Test update_strategy functionality."""
    client, mock_client = _create_client()

    mock_client.update_memory.return_value = {"memory": {"id": "mem-123", "status": "CREATING"}}

    with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
        result = client.update_strategy(
            memory_id="mem-123",
            strategy_id="strat-456",
            description="Updated strategy description",
            namespaces=["custom/namespace1/", "custom/namespace2/"],
        )

        assert result["id"] == "mem-123"
        assert mock_client.update_memory.called

        args, kwargs = mock_client.update_memory.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert "memoryStrategies" in kwargs
        assert "modifyMemoryStrategies" in kwargs["memoryStrategies"]

        modify_strategy = kwargs["memoryStrategies"]["modifyMemoryStrategies"][0]
        assert modify_strategy["memoryStrategyId"] == "strat-456"
        assert modify_strategy["description"] == "Updated strategy description"
        assert modify_strategy["namespaces"] == ["custom/namespace1/", "custom/namespace2/"]


def test_remove_strategy():
    """Test remove_strategy functionality."""
    client, mock_client = _create_client()

    mock_client.update_memory.return_value = {"memory": {"id": "mem-123", "status": "CREATING"}}

    with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
        result = client.remove_strategy(memory_id="mem-123", strategy_id="strat-456")

        assert result["id"] == "mem-123"
        assert mock_client.update_memory.called

        args, kwargs = mock_client.update_memory.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert "memoryStrategies" in kwargs
        assert "deleteMemoryStrategies" in kwargs["memoryStrategies"]
        deleted = kwargs["memoryStrategies"]["deleteMemoryStrategies"]
        assert deleted == [{"memoryStrategyId": "strat-456"}]


def test_error_handling():
    """Test error handling."""
    client, mock_client = _create_client()

    error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid parameter"}}
    mock_client.create_memory.side_effect = ClientError(error_response, "CreateMemory")

    try:
        client.create_memory(name="Test Memory")
        raise AssertionError("Error was not raised as expected")
    except ClientError as e:
        assert "ValidationException" in str(e)


def test_create_memory_with_strategies():
    """Test create_memory with memory strategies."""
    client, mock_client = _create_client()

    mock_client.create_memory.return_value = {
        "memory": {"id": "mem-456", "name": "Memory with Strategies", "status": "CREATING"}
    }

    with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
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

        args, kwargs = mock_client.create_memory.call_args
        assert kwargs["name"] == "Memory with Strategies"
        assert kwargs["description"] == "Test with strategies"
        assert kwargs["eventExpiryDuration"] == 120
        assert kwargs["memoryExecutionRoleArn"] == "arn:aws:iam::123456789012:role/MemoryRole"
        assert kwargs["memoryStrategies"] == strategies


def test_list_memories_with_pagination():
    """Test list_memories with pagination."""
    client, mock_client = _create_client()

    first_batch = [{"id": f"mem-{i}", "name": f"Memory {i}", "status": "ACTIVE"} for i in range(1, 101)]
    second_batch = [{"id": f"mem-{i}", "name": f"Memory {i}", "status": "ACTIVE"} for i in range(101, 151)]

    mock_client.list_memories.side_effect = [
        {"memories": first_batch, "nextToken": "token-123"},
        {"memories": second_batch, "nextToken": None},
    ]

    result = client.list_memories(max_results=150)

    assert len(result) == 150
    assert result[0]["id"] == "mem-1"
    assert result[149]["id"] == "mem-150"
    assert result[0]["strategyCount"] == 0

    assert mock_client.list_memories.call_count == 2


def test_update_memory_minimal():
    """Test update_memory with minimal parameters."""
    client, mock_client = _create_client()

    mock_client.update_memory.return_value = {"memory": {"id": "mem-123", "status": "ACTIVE"}}

    with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
        result = client.update_memory(memory_id="mem-123")

        assert result["id"] == "mem-123"
        assert mock_client.update_memory.called

        args, kwargs = mock_client.update_memory.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert kwargs["clientToken"] == "12345678-1234-5678-1234-567812345678"


def test_get_strategy_not_found():
    """Test get_strategy when strategy doesn't exist."""
    client, mock_client = _create_client()

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
    client, mock_client = _create_client()

    mock_client.delete_memory.return_value = {"status": "DELETING"}

    # Mock get_memory to always succeed (memory never gets deleted)
    mock_client.get_memory.return_value = {"memory": {"id": "mem-persistent", "status": "DELETING"}}

    with patch("time.time", side_effect=[0, 0, 0, 301, 301, 301]):
        with patch("time.sleep"):
            with patch(
                "uuid.uuid4",
                return_value=uuid.UUID("12345678-1234-5678-1234-567812345678"),
            ):
                try:
                    client.delete_memory(
                        "mem-persistent",
                        wait_for_deletion=True,
                        max_wait=300,
                        poll_interval=10,
                    )
                    raise AssertionError("TimeoutError was not raised")
                except TimeoutError as e:
                    assert "was not deleted within 300 seconds" in str(e)


def test_wait_for_status_timeout():
    """Test _wait_for_status with timeout."""
    client, mock_client = _create_client()

    mock_client.get_memory.return_value = {"memory": {"id": "mem-timeout", "status": "CREATING", "strategies": []}}

    time_values = [0] + [i * 10 for i in range(1, 35)] + [301]
    with patch("time.time", side_effect=time_values):
        with patch("time.sleep"):
            try:
                client._wait_for_status(
                    "mem-timeout",
                    target_status=MemoryStatus.ACTIVE.value,
                    max_wait=300,
                    poll_interval=10,
                )
                raise AssertionError("TimeoutError was not raised")
            except TimeoutError as e:
                assert "did not reach status ACTIVE within 300 seconds" in str(e)


def test_wait_for_status_failure():
    """Test _wait_for_status with FAILED status."""
    client, mock_client = _create_client()

    mock_client.get_memory.return_value = {
        "memory": {
            "id": "mem-failed",
            "status": "FAILED",
            "failureReason": "Configuration error",
            "strategies": [],
        }
    }

    with patch("time.time", return_value=0):
        with patch("time.sleep"):
            try:
                client._wait_for_status(
                    "mem-failed",
                    target_status=MemoryStatus.ACTIVE.value,
                    max_wait=300,
                    poll_interval=10,
                )
                raise AssertionError("RuntimeError was not raised")
            except RuntimeError as e:
                assert "Memory operation failed: Configuration error" in str(e)


def test_wait_for_status_with_strategy_check():
    """Test _wait_for_status with transitional strategies."""
    client, mock_client = _create_client()

    mock_client.get_memory.side_effect = [
        {
            "memory": {
                "id": "mem-123",
                "status": "ACTIVE",
                "strategies": [
                    {"strategyId": "strat-1", "status": "CREATING"},
                    {"strategyId": "strat-2", "status": "ACTIVE"},
                ],
            }
        },
        {
            "memory": {
                "id": "mem-123",
                "status": "ACTIVE",
                "strategies": [
                    {"strategyId": "strat-1", "status": "ACTIVE"},
                    {"strategyId": "strat-2", "status": "ACTIVE"},
                ],
            }
        },
    ]

    with patch("time.time", return_value=0):
        with patch("time.sleep"):
            result = client._wait_for_status(
                "mem-123",
                target_status=MemoryStatus.ACTIVE.value,
                max_wait=120,
                poll_interval=10,
            )

            assert result["id"] == "mem-123"
            assert result["status"] == "ACTIVE"
            assert mock_client.get_memory.call_count == 2


def test_get_memory_client_error():
    """Test get_memory with ClientError."""
    client, mock_client = _create_client()

    error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Memory not found"}}
    mock_client.get_memory.side_effect = ClientError(error_response, "GetMemory")

    try:
        client.get_memory("nonexistent-mem-123")
        raise AssertionError("ClientError was not raised")
    except ClientError as e:
        assert "ResourceNotFoundException" in str(e)


def test_list_memories_client_error():
    """Test list_memories with ClientError."""
    client, mock_client = _create_client()

    error_response = {"Error": {"Code": "AccessDeniedException", "Message": "Insufficient permissions"}}
    mock_client.list_memories.side_effect = ClientError(error_response, "ListMemories")

    try:
        client.list_memories(max_results=50)
        raise AssertionError("ClientError was not raised")
    except ClientError as e:
        assert "AccessDeniedException" in str(e)


def test_update_memory_client_error():
    """Test update_memory with ClientError."""
    client, mock_client = _create_client()

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
    client, mock_client = _create_client()

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
    client, mock_client = _create_client()

    error_response = {"Error": {"Code": "ThrottlingException", "Message": "Request throttled"}}
    mock_client.get_memory.side_effect = ClientError(error_response, "GetMemory")

    try:
        client.get_strategy("mem-123", "strat-456")
        raise AssertionError("ClientError was not raised")
    except ClientError as e:
        assert "ThrottlingException" in str(e)


def test_wait_for_status_client_error():
    """Test _wait_for_status with ClientError."""
    client, mock_client = _create_client()

    error_response = {"Error": {"Code": "InternalServerError", "Message": "Internal server error"}}
    mock_client.get_memory.side_effect = ClientError(error_response, "GetMemory")

    with patch("time.time", return_value=0):
        with patch("time.sleep"):
            try:
                client._wait_for_status(
                    "mem-123",
                    target_status=MemoryStatus.ACTIVE.value,
                    max_wait=120,
                    poll_interval=10,
                )
                raise AssertionError("ClientError was not raised")
            except ClientError as e:
                assert "InternalServerError" in str(e)


def test_constructor_client_param():
    """Test that passing client= in constructor sets self.client directly."""
    mock_client = MagicMock()
    client = MemoryControlPlaneClient(client=mock_client)

    assert client.client is mock_client


def test_constructor_default_client():
    """Test that omitting client= in constructor creates a default boto3 client."""
    with patch("boto3.client") as mock_boto:
        mock_boto.return_value = MagicMock()
        client = MemoryControlPlaneClient()

    assert client.client is mock_boto.return_value
