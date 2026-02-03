"""Tests for Memory class."""

from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.memory.config import StrategyType
from bedrock_agentcore.memory.memory import Memory


class TestMemoryInit:
    """Tests for Memory initialization."""

    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_minimal_init(self, mock_client_class: MagicMock) -> None:
        """Test minimal memory initialization."""
        mock_client = MagicMock()
        mock_client.region_name = "us-west-2"
        mock_client_class.return_value = mock_client

        memory = Memory(name="test-memory")

        assert memory.name == "test-memory"
        assert memory.config.name == "test-memory"
        assert memory.config.description is None
        assert memory.config.strategies is None
        assert memory.memory_id is None

    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_full_init(self, mock_client_class: MagicMock) -> None:
        """Test full memory initialization with all parameters."""
        mock_client = MagicMock()
        mock_client.region_name = "us-west-2"
        mock_client_class.return_value = mock_client

        memory = Memory(
            name="test-memory",
            description="Test memory description",
            strategies=[
                {"type": "SEMANTIC", "namespace": "facts/{sessionId}/"},
                {"type": "SUMMARY", "namespace": "summaries/{sessionId}/"},
            ],
            encryption_key_arn="arn:aws:kms:us-west-2:123456789012:key/abc123",
            tags={"Environment": "test"},
            region="us-east-1",
        )

        assert memory.name == "test-memory"
        assert memory.config.description == "Test memory description"
        assert memory.config.strategies is not None
        assert len(memory.config.strategies) == 2
        assert memory.config.strategies[0].strategy_type == StrategyType.SEMANTIC
        assert memory.config.strategies[0].namespace == "facts/{sessionId}/"
        assert memory.config.strategies[1].strategy_type == StrategyType.SUMMARY
        assert memory.config.encryption_key_arn == "arn:aws:kms:us-west-2:123456789012:key/abc123"
        assert memory.config.tags == {"Environment": "test"}

    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_init_with_custom_prompt(self, mock_client_class: MagicMock) -> None:
        """Test memory initialization with custom prompt strategy."""
        mock_client = MagicMock()
        mock_client.region_name = "us-west-2"
        mock_client_class.return_value = mock_client

        memory = Memory(
            name="test-memory",
            strategies=[
                {
                    "type": "CUSTOM_SEMANTIC",
                    "namespace": "custom/",
                    "customPrompt": "Extract key facts from conversation.",
                },
            ],
        )

        assert memory.config.strategies is not None
        assert len(memory.config.strategies) == 1
        assert memory.config.strategies[0].strategy_type == StrategyType.CUSTOM_SEMANTIC
        assert memory.config.strategies[0].custom_prompt == "Extract key facts from conversation."


class TestMemoryIsActive:
    """Tests for Memory is_active property."""

    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_is_active_without_memory_id(self, mock_client_class: MagicMock) -> None:
        """Test is_active returns False when memory_id is not set."""
        mock_client = MagicMock()
        mock_client.region_name = "us-west-2"
        mock_client_class.return_value = mock_client

        memory = Memory(name="test-memory")

        assert memory.is_active is False

    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_is_active_with_active_memory(self, mock_client_class: MagicMock) -> None:
        """Test is_active returns True when memory is active."""
        mock_client = MagicMock()
        mock_client.region_name = "us-west-2"
        mock_client.get_memory_status.return_value = "ACTIVE"
        mock_client_class.return_value = mock_client

        memory = Memory(name="test-memory")
        memory._memory_id = "memory-123"

        assert memory.is_active is True


class TestMemoryOperations:
    """Tests for Memory create/delete operations."""

    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_delete_without_memory_id(self, mock_client_class: MagicMock) -> None:
        """Test delete when memory is not created."""
        mock_client = MagicMock()
        mock_client.region_name = "us-west-2"
        mock_client_class.return_value = mock_client

        memory = Memory(name="test-memory")

        result = memory.delete()

        assert result["status"] == "NOT_CREATED"

    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_get_session_without_memory_id_raises(self, mock_client_class: MagicMock) -> None:
        """Test that get_session raises ValueError when memory is not launched."""
        mock_client = MagicMock()
        mock_client.region_name = "us-west-2"
        mock_client_class.return_value = mock_client

        memory = Memory(name="test-memory")

        with pytest.raises(ValueError, match="Memory is not launched"):
            memory.get_session(actor_id="user-123", session_id="session-456")

    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_list_events_without_memory_id_raises(self, mock_client_class: MagicMock) -> None:
        """Test that list_events raises ValueError when memory is not launched."""
        mock_client = MagicMock()
        mock_client.region_name = "us-west-2"
        mock_client_class.return_value = mock_client

        memory = Memory(name="test-memory")

        with pytest.raises(ValueError, match="Memory is not launched"):
            memory.list_events(actor_id="user-123", session_id="session-456")

    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_search_records_without_memory_id_raises(self, mock_client_class: MagicMock) -> None:
        """Test that search_records raises ValueError when memory is not launched."""
        mock_client = MagicMock()
        mock_client.region_name = "us-west-2"
        mock_client_class.return_value = mock_client

        memory = Memory(name="test-memory")

        with pytest.raises(ValueError, match="Memory is not launched"):
            memory.search_records(query="test", namespace="facts/")

    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_add_strategy_without_memory_id_raises(self, mock_client_class: MagicMock) -> None:
        """Test that add_strategy raises ValueError when memory is not launched."""
        mock_client = MagicMock()
        mock_client.region_name = "us-west-2"
        mock_client_class.return_value = mock_client

        memory = Memory(name="test-memory")

        with pytest.raises(ValueError, match="Memory is not launched"):
            memory.add_strategy(strategy_type="SEMANTIC", namespace="facts/")
