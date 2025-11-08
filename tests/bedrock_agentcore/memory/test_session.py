"""Unit tests for Session Manager and MemorySession classes - no external connections."""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import pytest
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

from bedrock_agentcore.memory.constants import BlobMessage, ConversationalMessage, MessageRole, RetrievalConfig
from bedrock_agentcore.memory.models import (
    ActorSummary,
    Branch,
    Event,
    EventMessage,
    MemoryRecord,
    SessionSummary,
)
from bedrock_agentcore.memory.session import Actor, MemorySession, MemorySessionManager


class TestBotocoreConfigSupport:
    """Test cases for botocore.config support in MemorySessionManager."""

    def test_session_manager_initialization_with_boto_client_config(self):
        """Test MemorySessionManager initialization with boto_client_config."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Create custom botocore config
            custom_config = BotocoreConfig(
                retries={"max_attempts": 5, "mode": "adaptive"},
                max_pool_connections=50,
                connect_timeout=10,
                read_timeout=30,
            )

            manager = MemorySessionManager(
                memory_id="testMemory-1234567890", region_name="us-west-2", boto_client_config=custom_config
            )

            assert manager._memory_id == "testMemory-1234567890"
            assert manager.region_name == "us-west-2"
            assert manager._data_plane_client == mock_client_instance

            # Verify client was called with merged config
            mock_session.client.assert_called_once()
            call_args = mock_session.client.call_args
            assert call_args[0] == ("bedrock-agentcore",)
            assert call_args[1]["region_name"] == "us-west-2"
            assert "config" in call_args[1]

            # Verify the config was merged with user agent
            passed_config = call_args[1]["config"]
            assert passed_config.user_agent_extra == "bedrock-agentcore-sdk"
            assert passed_config.retries == {"max_attempts": 5, "mode": "adaptive"}
            assert passed_config.max_pool_connections == 50

    def test_session_manager_initialization_with_existing_user_agent(self):
        """Test MemorySessionManager initialization preserves existing user agent."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Create custom botocore config with existing user agent
            custom_config = BotocoreConfig(user_agent_extra="my-custom-app", retries={"max_attempts": 3})

            MemorySessionManager(
                memory_id="testMemory-1234567890", region_name="us-west-2", boto_client_config=custom_config
            )

            # Verify client was called with merged config
            mock_session.client.assert_called_once()
            call_args = mock_session.client.call_args
            passed_config = call_args[1]["config"]

            # Verify existing user agent was preserved and SDK user agent was appended
            assert passed_config.user_agent_extra == "my-custom-app bedrock-agentcore-sdk"
            assert passed_config.retries == {"max_attempts": 3}

    def test_session_manager_initialization_without_boto_client_config(self):
        """Test MemorySessionManager initialization without boto_client_config uses default."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            MemorySessionManager(
                memory_id="testMemory-1234567890",
                region_name="us-west-2",
                # No boto_client_config provided
            )

            # Verify client was called with default config
            mock_session.client.assert_called_once()
            call_args = mock_session.client.call_args
            passed_config = call_args[1]["config"]

            # Verify default user agent was set
            assert passed_config.user_agent_extra == "bedrock-agentcore-sdk"

    def test_session_manager_with_custom_retry_config(self):
        """Test MemorySessionManager with custom retry configuration."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-east-1"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Create config with custom retry settings
            retry_config = BotocoreConfig(
                retries={"max_attempts": 10, "mode": "standard"}, connect_timeout=60, read_timeout=120
            )

            MemorySessionManager(
                memory_id="testMemory-1234567890", region_name="us-east-1", boto_client_config=retry_config
            )

            # Verify the retry configuration was applied
            call_args = mock_session.client.call_args
            passed_config = call_args[1]["config"]
            assert passed_config.retries["max_attempts"] == 10
            assert passed_config.retries["mode"] == "standard"
            assert passed_config.connect_timeout == 60
            assert passed_config.read_timeout == 120

    def test_session_manager_with_connection_pool_config(self):
        """Test MemorySessionManager with custom connection pool configuration."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-east-1"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Create config with connection pool settings
            pool_config = BotocoreConfig(max_pool_connections=100, retries={"max_attempts": 2})

            MemorySessionManager(
                memory_id="testMemory-1234567890", region_name="us-east-1", boto_client_config=pool_config
            )

            # Verify the connection pool configuration was applied
            call_args = mock_session.client.call_args
            passed_config = call_args[1]["config"]
            assert passed_config.max_pool_connections == 100
            assert passed_config.retries["max_attempts"] == 2

    def test_session_manager_config_with_boto3_session(self):
        """Test MemorySessionManager with both boto3_session and boto_client_config."""
        custom_session = MagicMock()
        custom_session.region_name = "us-west-2"
        mock_client_instance = MagicMock()
        custom_session.client.return_value = mock_client_instance

        # Create custom botocore config
        custom_config = BotocoreConfig(user_agent_extra="test-app", retries={"max_attempts": 7})

        MemorySessionManager(
            memory_id="testMemory-1234567890",
            region_name="us-west-2",
            boto3_session=custom_session,
            boto_client_config=custom_config,
        )

        # Verify the custom session was used with the merged config
        custom_session.client.assert_called_once()
        call_args = custom_session.client.call_args
        passed_config = call_args[1]["config"]

        # Verify user agent was merged correctly
        assert passed_config.user_agent_extra == "test-app bedrock-agentcore-sdk"
        assert passed_config.retries["max_attempts"] == 7

    def test_session_manager_config_merge_behavior(self):
        """Test that botocore config merge behavior works correctly."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-east-1"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Create config with multiple settings
            original_config = BotocoreConfig(
                retries={"max_attempts": 5, "mode": "adaptive"},
                max_pool_connections=25,
                connect_timeout=30,
                user_agent_extra="original-app",
            )

            MemorySessionManager(
                memory_id="testMemory-1234567890", region_name="us-east-1", boto_client_config=original_config
            )

            # Verify all original settings were preserved and user agent was merged
            call_args = mock_session.client.call_args
            passed_config = call_args[1]["config"]

            assert passed_config.retries["max_attempts"] == 5
            assert passed_config.retries["mode"] == "adaptive"
            assert passed_config.max_pool_connections == 25
            assert passed_config.connect_timeout == 30
            assert passed_config.user_agent_extra == "original-app bedrock-agentcore-sdk"

    def test_functional_test_with_custom_config(self):
        """Test that MemorySessionManager functions correctly with custom config."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-east-1"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Create config optimized for high throughput
            high_throughput_config = BotocoreConfig(
                retries={"max_attempts": 3, "mode": "adaptive"},
                max_pool_connections=50,
                connect_timeout=5,
                read_timeout=60,
            )

            manager = MemorySessionManager(
                memory_id="testMemory-1234567890", region_name="us-east-1", boto_client_config=high_throughput_config
            )

            # Mock a successful add_turns operation
            mock_response = {"event": {"eventId": "test-event-123"}}
            mock_client_instance.create_event.return_value = mock_response

            # Test that the manager works normally with custom config
            result = manager.add_turns(
                actor_id="user-123",
                session_id="session-456",
                messages=[ConversationalMessage("Hello", MessageRole.USER)],
            )

            # Verify the operation succeeded
            assert isinstance(result, Event)
            assert result["eventId"] == "test-event-123"

            # Verify the client was created with our custom config
            call_args = mock_session.client.call_args
            passed_config = call_args[1]["config"]
            assert passed_config.max_pool_connections == 50
            assert passed_config.connect_timeout == 5
            assert passed_config.read_timeout == 60

    def test_config_parameter_validation(self):
        """Test that invalid config parameters are handled appropriately."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-east-1"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Test with None config (should work)
            MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-east-1", boto_client_config=None)

            # Verify default config was applied
            call_args = mock_session.client.call_args
            passed_config = call_args[1]["config"]
            assert passed_config.user_agent_extra == "bedrock-agentcore-sdk"

    def test_config_with_all_parameters(self):
        """Test MemorySessionManager with all initialization parameters including config."""
        custom_session = MagicMock()
        custom_session.region_name = "eu-west-1"
        mock_client_instance = MagicMock()
        custom_session.client.return_value = mock_client_instance

        # Create comprehensive config
        comprehensive_config = BotocoreConfig(
            retries={"max_attempts": 4, "mode": "standard"},
            max_pool_connections=75,
            connect_timeout=15,
            read_timeout=90,
            user_agent_extra="comprehensive-test",
        )

        manager = MemorySessionManager(
            memory_id="testMemory-comprehensive",
            region_name="eu-west-1",
            boto3_session=custom_session,
            boto_client_config=comprehensive_config,
        )

        # Verify all parameters were handled correctly
        assert manager._memory_id == "testMemory-comprehensive"
        assert manager.region_name == "eu-west-1"

        # Verify client creation with all parameters
        custom_session.client.assert_called_once()
        call_args = custom_session.client.call_args

        assert call_args[0] == ("bedrock-agentcore",)
        assert call_args[1]["region_name"] == "eu-west-1"

        passed_config = call_args[1]["config"]
        assert passed_config.retries["max_attempts"] == 4
        assert passed_config.max_pool_connections == 75
        assert passed_config.connect_timeout == 15
        assert passed_config.read_timeout == 90
        assert passed_config.user_agent_extra == "comprehensive-test bedrock-agentcore-sdk"


class TestSessionManager:
    """Test cases for MemorySessionManager class."""

    def test_session_manager_initialization(self):
        """Test MemorySessionManager initialization."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            assert manager._memory_id == "testMemory-1234567890"
            assert manager.region_name == "us-west-2"
            assert manager._data_plane_client == mock_client_instance

            # Verify client was called with config parameter (default user agent)
            mock_session.client.assert_called_once()
            call_args = mock_session.client.call_args
            assert call_args[0] == ("bedrock-agentcore",)
            assert call_args[1]["region_name"] == "us-west-2"
            assert "config" in call_args[1]

            # Verify the default config has the correct user agent
            passed_config = call_args[1]["config"]
            assert passed_config.user_agent_extra == "bedrock-agentcore-sdk"

    def test_getattr_allowed_method(self):
        """Test __getattr__ forwards allowed data plane methods."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Test accessing an allowed method
            mock_method = MagicMock()
            mock_client_instance.retrieve_memory_records = mock_method

            result = manager.retrieve_memory_records
            assert result == mock_method

    def test_getattr_disallowed_method(self):
        """Test __getattr__ raises AttributeError for disallowed methods."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Test accessing a disallowed method
            with pytest.raises(AttributeError) as exc_info:
                _ = manager.some_disallowed_method

            assert "'MemorySessionManager' object has no attribute 'some_disallowed_method'" in str(exc_info.value)
            assert "Method not found on _data_plane_client" in str(exc_info.value)

    def test_process_turn_with_llm_success(self):
        """Test process_turn_with_llm successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock search_long_term_memories
            mock_memories = [{"content": {"text": "Previous context"}, "memoryRecordId": "rec-123"}]
            with patch.object(manager, "search_long_term_memories", return_value=mock_memories):
                # Mock add_turns
                mock_event = {"eventId": "event-123", "memoryId": "testMemory-1234567890"}
                with patch.object(manager, "add_turns", return_value=Event(mock_event)):
                    # Define LLM callback
                    def mock_llm_callback(user_input: str, memories: List[Dict[str, Any]]) -> str:
                        return f"Response to: {user_input} with {len(memories)} memories"

                    # Test process_turn_with_llm with new RetrievalConfig API
                    retrieval_config = {"test/namespace": RetrievalConfig(top_k=5)}
                    memories, response, event = manager.process_turn_with_llm(
                        actor_id="user-123",
                        session_id="session-456",
                        user_input="Hello",
                        llm_callback=mock_llm_callback,
                        retrieval_config=retrieval_config,
                    )

                    assert len(memories) == 1
                    assert "Response to: Hello with 1 memories" in response
                    assert event["eventId"] == "event-123"

    def test_process_turn_with_llm_no_retrieval(self):
        """Test process_turn_with_llm without memory retrieval."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock add_turns
            mock_event = {"eventId": "event-123", "memoryId": "testMemory-1234567890"}
            with patch.object(manager, "add_turns", return_value=Event(mock_event)):
                # Define LLM callback
                def mock_llm_callback(user_input: str, memories: List[Dict[str, Any]]) -> str:
                    return f"Response to: {user_input}"

                # Test process_turn_with_llm without retrieval (None retrieval_config)
                memories, response, event = manager.process_turn_with_llm(
                    actor_id="user-123",
                    session_id="session-456",
                    user_input="Hello",
                    llm_callback=mock_llm_callback,
                    retrieval_config=None,
                )

                assert len(memories) == 0
                assert response == "Response to: Hello"
                assert event["eventId"] == "event-123"

    def test_process_turn_with_llm_async_method(self):
        """Test process_turn_with_llm_async method."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock add_turns
            mock_event = {"eventId": "event-123", "memoryId": "testMemory-1234567890"}
            with patch.object(manager, "add_turns", return_value=Event(mock_event)):
                # Define async LLM callback
                async def mock_async_llm_callback(user_input: str, memories: List[Dict[str, Any]]) -> str:
                    return f"Async method response to: {user_input}"

                # Test process_turn_with_llm_async
                async def run_test():
                    memories, response, event = await manager.process_turn_with_llm_async(
                        actor_id="user-123",
                        session_id="session-456",
                        user_input="Hello async method",
                        llm_callback=mock_async_llm_callback,
                        retrieval_config=None,
                    )
                    return memories, response, event

                memories, response, event = asyncio.run(run_test())
                assert len(memories) == 0
                assert response == "Async method response to: Hello async method"
                assert event["eventId"] == "event-123"

    def test_process_turn_with_llm_callback_error(self):
        """Test process_turn_with_llm with LLM callback error."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Define failing LLM callback
            def failing_llm_callback(user_input: str, memories: List[Dict[str, Any]]) -> str:
                raise Exception("LLM service error")

            # Test process_turn_with_llm with callback error
            with pytest.raises(Exception) as exc_info:
                manager.process_turn_with_llm(
                    actor_id="user-123",
                    session_id="session-456",
                    user_input="Hello",
                    llm_callback=failing_llm_callback,
                    retrieval_config=None,
                )

            assert "LLM service error" in str(exc_info.value)

    def test_process_turn_with_llm_invalid_callback_return(self):
        """Test process_turn_with_llm with invalid callback return type."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Define callback that returns non-string
            def invalid_llm_callback(user_input: str, memories: List[Dict[str, Any]]) -> int:
                return 123  # type: ignore

            # Test process_turn_with_llm with invalid return type
            with pytest.raises(ValueError) as exc_info:
                manager.process_turn_with_llm(
                    actor_id="user-123",
                    session_id="session-456",
                    user_input="Hello",
                    llm_callback=invalid_llm_callback,  # type: ignore
                    retrieval_config=None,
                )

            assert "LLM callback must return a string response" in str(exc_info.value)

    def test_add_turns_success(self):
        """Test add_turns successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock create_event response
            mock_response = {"event": {"eventId": "turn-event-123", "memoryId": "testMemory-1234567890"}}
            mock_client_instance.create_event.return_value = mock_response

            messages = [
                ConversationalMessage("Hello", MessageRole.USER),
                ConversationalMessage("Hi there", MessageRole.ASSISTANT),
            ]
            result = manager.add_turns(actor_id="user-123", session_id="session-456", messages=messages)

            assert isinstance(result, Event)
            assert result["eventId"] == "turn-event-123"

            # Verify call parameters
            call_args = mock_client_instance.create_event.call_args[1]
            assert len(call_args["payload"]) == 2
            assert call_args["payload"][0]["conversational"]["role"] == MessageRole.USER.value
            assert call_args["payload"][1]["conversational"]["role"] == MessageRole.ASSISTANT.value

    def test_add_turns_empty_messages(self):
        """Test add_turns with empty messages raises ValueError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            with pytest.raises(ValueError) as exc_info:
                manager.add_turns(actor_id="user-123", session_id="session-456", messages=[])

            assert "At least one message is required" in str(exc_info.value)

    def test_add_turns_single_element_tuple_as_blob(self):
        """Test add_turns treats single-element tuple as blob object."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock create_event response
            mock_response = {"event": {"eventId": "blob-event-123", "memoryId": "testMemory-1234567890"}}
            mock_client_instance.create_event.return_value = mock_response

            # Single-element tuple should be treated as blob using BlobMessage
            result = manager.add_turns(
                actor_id="user-123",
                session_id="session-456",
                messages=[BlobMessage(("Hello",))],  # Single-element tuple treated as blob
            )

            assert isinstance(result, Event)
            assert result["eventId"] == "blob-event-123"

            # Verify it was treated as blob
            call_args = mock_client_instance.create_event.call_args[1]
            payload = call_args["payload"]
            assert len(payload) == 1
            assert "blob" in payload[0]
            assert payload[0]["blob"] == ("Hello",)

    def test_add_turns_invalid_role(self):
        """Test add_turns with invalid role."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            with pytest.raises(ValueError) as exc_info:
                manager.add_turns(
                    actor_id="user-123",
                    session_id="session-456",
                    messages=[ConversationalMessage("Hello", "INVALID_ROLE")],
                )

            assert "ConversationalMessage.role must be a MessageRole" in str(exc_info.value)

    def test_add_turns_with_branch(self):
        """Test add_turns with branch parameter."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock create_event response
            mock_response = {"event": {"eventId": "branch-turn-event-123", "memoryId": "testMemory-1234567890"}}
            mock_client_instance.create_event.return_value = mock_response

            branch = {"name": "test-branch", "rootEventId": "event-root-123"}
            messages = [ConversationalMessage("Branch message", MessageRole.USER)]
            result = manager.add_turns(actor_id="user-123", session_id="session-456", messages=messages, branch=branch)

            assert result["eventId"] == "branch-turn-event-123"

            # Verify branch was passed
            call_args = mock_client_instance.create_event.call_args[1]
            assert call_args["branch"] == branch

    def test_add_turns_client_error(self):
        """Test add_turns with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock ClientError
            error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid parameters"}}
            mock_client_instance.create_event.side_effect = ClientError(error_response, "CreateEvent")

            with pytest.raises(ClientError):
                manager.add_turns(
                    actor_id="user-123",
                    session_id="session-456",
                    messages=[ConversationalMessage("Hello", MessageRole.USER)],
                )

    def test_fork_conversation_success(self):
        """Test fork_conversation successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock add_turns
            mock_event = {"eventId": "fork-event-123", "memoryId": "testMemory-1234567890"}
            with patch.object(manager, "add_turns", return_value=Event(mock_event)) as mock_add_turns:
                result = manager.fork_conversation(
                    actor_id="user-123",
                    session_id="session-456",
                    root_event_id="event-root-123",
                    branch_name="test-branch",
                    messages=[ConversationalMessage("Fork message", MessageRole.USER)],
                )

                assert result["eventId"] == "fork-event-123"

                # Verify add_turns was called with correct branch
                mock_add_turns.assert_called_once()
                call_args = mock_add_turns.call_args[1]
                assert call_args["branch"]["rootEventId"] == "event-root-123"
                assert call_args["branch"]["name"] == "test-branch"

    def test_fork_conversation_client_error(self):
        """Test fork_conversation with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock add_turns to raise ClientError
            with patch.object(
                manager,
                "add_turns",
                side_effect=ClientError(
                    {"Error": {"Code": "ValidationException", "Message": "Invalid root event"}}, "CreateEvent"
                ),
            ):
                with pytest.raises(ClientError):
                    manager.fork_conversation(
                        actor_id="user-123",
                        session_id="session-456",
                        root_event_id="invalid-event",
                        branch_name="test-branch",
                        messages=[ConversationalMessage("Fork message", MessageRole.USER)],
                    )

    def test_list_events_success(self):
        """Test list_events successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock list_events response
            mock_events = [
                {"eventId": "event-1", "eventTimestamp": datetime.now()},
                {"eventId": "event-2", "eventTimestamp": datetime.now()},
            ]
            mock_client_instance.list_events.return_value = {"events": mock_events, "nextToken": None}

            result = manager.list_events(actor_id="user-123", session_id="session-456")

            assert len(result) == 2
            assert all(isinstance(event, Event) for event in result)
            assert result[0]["eventId"] == "event-1"
            assert result[1]["eventId"] == "event-2"

    def test_list_events_with_pagination(self):
        """Test list_events with pagination."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock paginated responses
            first_batch = [{"eventId": f"event-{i}", "eventTimestamp": datetime.now()} for i in range(100)]
            second_batch = [{"eventId": f"event-{i}", "eventTimestamp": datetime.now()} for i in range(100, 150)]

            mock_client_instance.list_events.side_effect = [
                {"events": first_batch, "nextToken": "token-123"},
                {"events": second_batch, "nextToken": None},
            ]

            result = manager.list_events(actor_id="user-123", session_id="session-456", max_results=150)

            assert len(result) == 150
            assert mock_client_instance.list_events.call_count == 2

    def test_list_events_with_branch_filter(self):
        """Test list_events with branch filtering."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock response
            mock_events = [{"eventId": "branch-event-1", "eventTimestamp": datetime.now()}]
            mock_client_instance.list_events.return_value = {"events": mock_events, "nextToken": None}

            result = manager.list_events(
                actor_id="user-123", session_id="session-456", branch_name="test-branch", include_parent_branches=True
            )

            assert len(result) == 1

            # Verify filter was applied
            call_args = mock_client_instance.list_events.call_args[1]
            assert "filter" in call_args
            assert call_args["filter"]["branch"]["name"] == "test-branch"
            assert call_args["filter"]["branch"]["includeParentBranches"] is True

    def test_list_events_main_branch_no_filter(self):
        """Test list_events with main branch doesn't apply filter."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock response
            mock_events = [{"eventId": "main-event-1", "eventTimestamp": datetime.now()}]
            mock_client_instance.list_events.return_value = {"events": mock_events, "nextToken": None}

            result = manager.list_events(actor_id="user-123", session_id="session-456", branch_name="main")

            assert len(result) == 1

            # Verify no filter was applied for main branch
            call_args = mock_client_instance.list_events.call_args[1]
            assert "filter" not in call_args

    def test_list_events_client_error(self):
        """Test list_events with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock ClientError
            error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid parameters"}}
            mock_client_instance.list_events.side_effect = ClientError(error_response, "ListEvents")

            with pytest.raises(ClientError):
                manager.list_events(actor_id="user-123", session_id="session-456")

    def test_list_branches_success(self):
        """Test list_branches successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock events with branches
            mock_events = [
                {"eventId": "event-1", "eventTimestamp": datetime(2023, 1, 1, 10, 0, 0)},
                {
                    "eventId": "event-2",
                    "eventTimestamp": datetime(2023, 1, 1, 10, 5, 0),
                    "branch": {"name": "branch-1", "rootEventId": "event-1"},
                },
            ]
            mock_client_instance.list_events.return_value = {"events": mock_events, "nextToken": None}

            result = manager.list_branches(actor_id="user-123", session_id="session-456")

            assert len(result) == 2  # main + branch-1
            assert all(isinstance(branch, Branch) for branch in result)

            # Check main branch
            main_branch = next(b for b in result if b["name"] == "main")
            assert main_branch["rootEventId"] is None
            assert main_branch["eventCount"] == 1

            # Check custom branch
            custom_branch = next(b for b in result if b["name"] == "branch-1")
            assert custom_branch["rootEventId"] == "event-1"
            assert custom_branch["eventCount"] == 1

    def test_list_branches_no_main_events(self):
        """Test list_branches with no main branch events."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock events with only branch events
            mock_events = [
                {
                    "eventId": "event-1",
                    "eventTimestamp": datetime(2023, 1, 1, 10, 0, 0),
                    "branch": {"name": "branch-1", "rootEventId": "event-root"},
                }
            ]
            mock_client_instance.list_events.return_value = {"events": mock_events, "nextToken": None}

            result = manager.list_branches(actor_id="user-123", session_id="session-456")

            assert len(result) == 1  # Only branch-1, no main
            assert result[0]["name"] == "branch-1"

    def test_list_branches_client_error(self):
        """Test list_branches with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock ClientError
            error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid parameters"}}
            mock_client_instance.list_events.side_effect = ClientError(error_response, "ListEvents")

            with pytest.raises(ClientError):
                manager.list_branches(actor_id="user-123", session_id="session-456")

    def test_get_last_k_turns_success(self):
        """Test get_last_k_turns successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock list_events
            mock_events = [
                Event(
                    {
                        "eventId": "event-1",
                        "eventTimestamp": datetime(2023, 1, 1, 10, 0, 0),
                        "payload": [
                            {"conversational": {"role": "USER", "content": {"text": "Hello"}}},
                            {"conversational": {"role": "ASSISTANT", "content": {"text": "Hi there"}}},
                        ],
                    }
                ),
                Event(
                    {
                        "eventId": "event-2",
                        "eventTimestamp": datetime(2023, 1, 1, 10, 5, 0),
                        "payload": [
                            {"conversational": {"role": "USER", "content": {"text": "How are you?"}}},
                            {"conversational": {"role": "ASSISTANT", "content": {"text": "I'm doing well"}}},
                        ],
                    }
                ),
            ]
            with patch.object(manager, "list_events", return_value=mock_events):
                result = manager.get_last_k_turns(
                    actor_id="user-123",
                    session_id="session-456",
                    k=2,
                )

                assert len(result) == 2
                assert len(result[0]) == 2  # First turn has 2 messages
                assert all(isinstance(msg, EventMessage) for msg in result[0])

    def test_get_last_k_turns_empty_events(self):
        """Test get_last_k_turns with no events."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock empty list_events
            with patch.object(manager, "list_events", return_value=[]):
                result = manager.get_last_k_turns(actor_id="user-123", session_id="session-456", k=5)

                assert result == []

    def test_get_last_k_turns_client_error(self):
        """Test get_last_k_turns with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock list_events to raise ClientError
            with patch.object(
                manager,
                "list_events",
                side_effect=ClientError(
                    {"Error": {"Code": "ValidationException", "Message": "Invalid parameters"}}, "ListEvents"
                ),
            ):
                with pytest.raises(ClientError):
                    manager.get_last_k_turns(actor_id="user-123", session_id="session-456", k=5)

    def test_get_last_k_turns_with_include_parent_branches_parameter(self):
        """Test get_last_k_turns with include_parent_branches parameter to cover new functionality."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock list_events
            mock_events = [
                Event(
                    {
                        "eventId": "event-1",
                        "eventTimestamp": datetime(2023, 1, 1, 10, 0, 0),
                        "payload": [
                            {"conversational": {"role": "USER", "content": {"text": "Hello from branch"}}},
                            {"conversational": {"role": "ASSISTANT", "content": {"text": "Hi from branch"}}},
                        ],
                    }
                ),
                Event(
                    {
                        "eventId": "event-2",
                        "eventTimestamp": datetime(2023, 1, 1, 10, 5, 0),
                        "payload": [
                            {"conversational": {"role": "USER", "content": {"text": "Another message"}}},
                            {"conversational": {"role": "ASSISTANT", "content": {"text": "Another response"}}},
                        ],
                    }
                ),
            ]
            with patch.object(manager, "list_events", return_value=mock_events) as mock_list_events:
                # Test with include_parent_branches=True
                result = manager.get_last_k_turns(
                    actor_id="user-123",
                    session_id="session-456",
                    k=3,
                    branch_name="test-branch",
                    include_parent_branches=True,
                    max_results=50,
                )

                assert len(result) == 2
                assert len(result[0]) == 2  # First turn has 2 messages
                assert len(result[1]) == 2  # Second turn has 2 messages
                assert all(isinstance(msg, EventMessage) for msg in result[0])
                assert all(isinstance(msg, EventMessage) for msg in result[1])

                # Verify list_events was called with include_parent_branches=True when include_parent_branches=True
                mock_list_events.assert_called_once_with(
                    actor_id="user-123",
                    session_id="session-456",
                    branch_name="test-branch",
                    include_parent_branches=True,  # This should be True when include_parent_branches=True
                    max_results=50,
                )

                # Test with include_parent_branches=False (default behavior)
                mock_list_events.reset_mock()
                manager.get_last_k_turns(
                    actor_id="user-123",
                    session_id="session-456",
                    k=2,
                    branch_name="test-branch",
                    include_parent_branches=False,
                )

                # Verify list_events was called with include_parent_branches=False when include_parent_branches=False
                mock_list_events.assert_called_once_with(
                    actor_id="user-123",
                    session_id="session-456",
                    branch_name="test-branch",
                    include_parent_branches=False,  # This should be False when include_parent_branches=False
                    max_results=100,  # Default max_results
                )

    def test_get_event_success(self):
        """Test get_event successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock get_event response
            mock_response = {
                "event": {
                    "eventId": "event-123",
                    "memoryId": "testMemory-1234567890",
                    "actorId": "user-123",
                    "sessionId": "session-456",
                }
            }
            mock_client_instance.get_event.return_value = mock_response

            result = manager.get_event(actor_id="user-123", session_id="session-456", event_id="event-123")

            assert isinstance(result, Event)
            assert result["eventId"] == "event-123"
            assert result["memoryId"] == "testMemory-1234567890"

    def test_get_event_client_error(self):
        """Test get_event with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock ClientError
            error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Event not found"}}
            mock_client_instance.get_event.side_effect = ClientError(error_response, "GetEvent")

            with pytest.raises(ClientError):
                manager.get_event(actor_id="user-123", session_id="session-456", event_id="invalid-event")

    def test_delete_event_success(self):
        """Test delete_event successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Test delete_event (no return value expected)
            manager.delete_event(actor_id="user-123", session_id="session-456", event_id="event-123")

            # Verify API call
            mock_client_instance.delete_event.assert_called_once_with(
                memoryId="testMemory-1234567890", actorId="user-123", sessionId="session-456", eventId="event-123"
            )

    def test_delete_event_client_error(self):
        """Test delete_event with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock ClientError
            error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Event not found"}}
            mock_client_instance.delete_event.side_effect = ClientError(error_response, "DeleteEvent")

            with pytest.raises(ClientError):
                manager.delete_event(actor_id="user-123", session_id="session-456", event_id="invalid-event")

    def test_search_long_term_memories_success(self):
        """Test search_long_term_memories successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock retrieve_memory_records response
            mock_response = {
                "memoryRecordSummaries": [
                    {"content": {"text": "Memory 1"}, "memoryRecordId": "rec-1"},
                    {"content": {"text": "Memory 2"}, "memoryRecordId": "rec-2"},
                ]
            }
            mock_client_instance.retrieve_memory_records.return_value = mock_response

            result = manager.search_long_term_memories(query="test query", namespace_prefix="test/namespace", top_k=5)

            assert len(result) == 2
            assert all(isinstance(record, MemoryRecord) for record in result)
            assert result[0]["memoryRecordId"] == "rec-1"
            assert result[1]["memoryRecordId"] == "rec-2"

            # Verify API call
            call_args = mock_client_instance.retrieve_memory_records.call_args[1]
            assert call_args["memoryId"] == "testMemory-1234567890"
            assert call_args["searchCriteria"]["searchQuery"] == "test query"
            assert call_args["searchCriteria"]["topK"] == 5
            assert call_args["namespace"] == "test/namespace"

    def test_search_long_term_memories_with_strategy(self):
        """Test search_long_term_memories with strategy_id."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock retrieve_memory_records response
            mock_response = {"memoryRecordSummaries": []}
            mock_client_instance.retrieve_memory_records.return_value = mock_response

            result = manager.search_long_term_memories(
                query="test query", namespace_prefix="test/namespace", strategy_id="strategy-123"
            )

            assert result == []

            # Verify strategy_id was passed
            call_args = mock_client_instance.retrieve_memory_records.call_args[1]
            assert call_args["searchCriteria"]["strategyId"] == "strategy-123"

    def test_search_long_term_memories_client_error(self):
        """Test search_long_term_memories with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock ClientError
            error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid query"}}
            mock_client_instance.retrieve_memory_records.side_effect = ClientError(
                error_response, "RetrieveMemoryRecords"
            )

            with pytest.raises(ClientError):
                manager.search_long_term_memories(query="invalid query", namespace_prefix="test/namespace")

    def test_list_long_term_memory_records_success(self):
        """Test list_long_term_memory_records successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock paginator
            mock_paginator = MagicMock()
            mock_client_instance.get_paginator.return_value = mock_paginator
            mock_paginator.paginate.return_value = [
                {"memoryRecords": [{"memoryRecordId": "rec-1"}, {"memoryRecordId": "rec-2"}]}
            ]

            result = manager.list_long_term_memory_records(namespace_prefix="test/namespace")

            assert len(result) == 2
            assert all(isinstance(record, MemoryRecord) for record in result)

    def test_list_long_term_memory_records_with_strategy(self):
        """Test list_long_term_memory_records with strategy_id."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock paginator
            mock_paginator = MagicMock()
            mock_client_instance.get_paginator.return_value = mock_paginator
            mock_paginator.paginate.return_value = [{"memoryRecords": []}]

            result = manager.list_long_term_memory_records(
                namespace_prefix="test/namespace", strategy_id="strategy-123"
            )

            assert result == []

            # Verify strategy_id was passed
            call_args = mock_paginator.paginate.call_args[1]
            assert call_args["memoryStrategyId"] == "strategy-123"

    def test_list_long_term_memory_records_client_error(self):
        """Test list_long_term_memory_records with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock paginator to raise ClientError
            mock_paginator = MagicMock()
            mock_client_instance.get_paginator.return_value = mock_paginator
            mock_paginator.paginate.side_effect = ClientError(
                {"Error": {"Code": "ValidationException", "Message": "Invalid namespace"}}, "ListMemoryRecords"
            )

            with pytest.raises(ClientError):
                manager.list_long_term_memory_records(namespace_prefix="invalid/namespace")

    def test_list_actors_success(self):
        """Test list_actors successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock paginator
            mock_paginator = MagicMock()
            mock_client_instance.get_paginator.return_value = mock_paginator
            mock_paginator.paginate.return_value = [{"actorSummaries": [{"actorId": "user-1"}, {"actorId": "user-2"}]}]

            result = manager.list_actors()

            assert len(result) == 2
            assert all(isinstance(actor, ActorSummary) for actor in result)

    def test_list_actors_client_error(self):
        """Test list_actors with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock paginator to raise ClientError
            mock_paginator = MagicMock()
            mock_client_instance.get_paginator.return_value = mock_paginator
            mock_paginator.paginate.side_effect = ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}}, "ListActors"
            )

            with pytest.raises(ClientError):
                manager.list_actors()

    def test_get_memory_record_success(self):
        """Test get_memory_record successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock get_memory_record response
            mock_response = {"memoryRecord": {"memoryRecordId": "rec-123", "content": {"text": "Memory content"}}}
            mock_client_instance.get_memory_record.return_value = mock_response

            result = manager.get_memory_record(record_id="rec-123")

            assert isinstance(result, MemoryRecord)
            assert result["memoryRecordId"] == "rec-123"

    def test_get_memory_record_client_error(self):
        """Test get_memory_record with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock ClientError
            error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Record not found"}}
            mock_client_instance.get_memory_record.side_effect = ClientError(error_response, "GetMemoryRecord")

            with pytest.raises(ClientError):
                manager.get_memory_record(record_id="invalid-record")

    def test_delete_memory_record_success(self):
        """Test delete_memory_record successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Test delete_memory_record (no return value expected)
            manager.delete_memory_record(record_id="rec-123")

            # Verify API call
            mock_client_instance.delete_memory_record.assert_called_once_with(
                memoryId="testMemory-1234567890", memoryRecordId="rec-123"
            )

    def test_delete_memory_record_client_error(self):
        """Test delete_memory_record with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock ClientError
            error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Record not found"}}
            mock_client_instance.delete_memory_record.side_effect = ClientError(error_response, "DeleteMemoryRecord")

            with pytest.raises(ClientError):
                manager.delete_memory_record(record_id="invalid-record")

    def test_delete_all_long_term_memories_in_namespace_success(self):
        """Test delete_all_long_term_memories_in_namespace successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock list_long_term_memory_records
            mock_records = [
                {"memoryRecordId": "rec-1", "content": {"text": "Memory 1"}},
                {"memoryRecordId": "rec-2", "content": {"text": "Memory 2"}},
            ]
            with patch.object(manager, "list_long_term_memory_records", return_value=mock_records):
                # Mock batch_delete_memory_records response
                mock_response = {
                    "successfulRecords": [
                        {"memoryRecordId": "rec-1", "status": "SUCCEEDED"},
                        {"memoryRecordId": "rec-2", "status": "SUCCEEDED"},
                    ],
                    "failedRecords": [],
                }
                mock_client_instance.batch_delete_memory_records.return_value = mock_response

                result = manager.delete_all_long_term_memories_in_namespace("test/namespace")

                assert len(result["successfulRecords"]) == 2
                assert len(result["failedRecords"]) == 0

                # Verify API call
                mock_client_instance.batch_delete_memory_records.assert_called_once_with(
                    memoryId="testMemory-1234567890",
                    records=[{"memoryRecordId": "rec-1"}, {"memoryRecordId": "rec-2"}],
                )

    def test_delete_all_long_term_memories_in_namespace_empty(self):
        """Test delete_all_long_term_memories_in_namespace with no records."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock empty list_long_term_memory_records
            with patch.object(manager, "list_long_term_memory_records", return_value=[]):
                result = manager.delete_all_long_term_memories_in_namespace("empty/namespace")

                assert result == {"successfulRecords": [], "failedRecords": []}
                # Should not call batch_delete_memory_records
                mock_client_instance.batch_delete_memory_records.assert_not_called()

    def test_delete_all_long_term_memories_in_namespace_client_error(self):
        """Test delete_all_long_term_memories_in_namespace with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock list_long_term_memory_records
            mock_records = [{"memoryRecordId": "rec-1", "content": {"text": "Memory 1"}}]
            with patch.object(manager, "list_long_term_memory_records", return_value=mock_records):
                # Mock ClientError
                error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid request"}}
                mock_client_instance.batch_delete_memory_records.side_effect = ClientError(
                    error_response, "BatchDeleteMemoryRecords"
                )

                with pytest.raises(ClientError):
                    manager.delete_all_long_term_memories_in_namespace("test/namespace")

    def test_delete_all_long_term_memories_in_namespace_over_100_records(self):
        """Test deleting more than 100 records in namespace."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock 150 memory records
            mock_records = [{"memoryRecordId": f"rec-{i}", "content": {"text": f"Memory {i}"}} for i in range(150)]
            with patch.object(manager, "list_long_term_memory_records", return_value=mock_records):
                # Mock batch_delete_memory_records responses for each chunk
                mock_client_instance.batch_delete_memory_records.side_effect = [
                    {"successfulRecords": [{"memoryRecordId": f"rec-{i}"} for i in range(100)], "failedRecords": []},
                    {
                        "successfulRecords": [{"memoryRecordId": f"rec-{i}"} for i in range(100, 150)],
                        "failedRecords": [],
                    },
                ]

                result = manager.delete_all_long_term_memories_in_namespace("test/namespace")

                # Verify two batch calls were made
                assert mock_client_instance.batch_delete_memory_records.call_count == 2
                assert len(result["successfulRecords"]) == 150
                assert len(result["failedRecords"]) == 0

                # Verify first batch had 100 records
                first_batch = mock_client_instance.batch_delete_memory_records.call_args_list[0][1]["records"]
                assert len(first_batch) == 100

                # Verify second batch had 50 records
                second_batch = mock_client_instance.batch_delete_memory_records.call_args_list[1][1]["records"]
                assert len(second_batch) == 50

    def test_delete_all_long_term_memories_in_namespace_partial_failure(self):
        """Test delete_all_long_term_memories_in_namespace with some failed records."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock list_long_term_memory_records
            mock_records = [
                {"memoryRecordId": "rec-1", "content": {"text": "Memory 1"}},
                {"memoryRecordId": "rec-2", "content": {"text": "Memory 2"}},
                {"memoryRecordId": "rec-3", "content": {"text": "Memory 3"}},
            ]
            with patch.object(manager, "list_long_term_memory_records", return_value=mock_records):
                # Mock batch_delete_memory_records response with partial failure
                mock_response = {
                    "successfulRecords": [
                        {"memoryRecordId": "rec-1", "status": "SUCCEEDED"},
                        {"memoryRecordId": "rec-3", "status": "SUCCEEDED"},
                    ],
                    "failedRecords": [{"memoryRecordId": "rec-2", "status": "FAILED", "errorMessage": "Access denied"}],
                }
                mock_client_instance.batch_delete_memory_records.return_value = mock_response

                result = manager.delete_all_long_term_memories_in_namespace("test/namespace")

                assert len(result["successfulRecords"]) == 2
                assert len(result["failedRecords"]) == 1
                assert result["failedRecords"][0]["memoryRecordId"] == "rec-2"
                assert result["failedRecords"][0]["errorMessage"] == "Access denied"

    def test_list_actor_sessions_success(self):
        """Test list_actor_sessions successful execution."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock paginator
            mock_paginator = MagicMock()
            mock_client_instance.get_paginator.return_value = mock_paginator
            mock_paginator.paginate.return_value = [
                {"sessionSummaries": [{"sessionId": "session-1"}, {"sessionId": "session-2"}]}
            ]

            result = manager.list_actor_sessions(actor_id="user-123")

            assert len(result) == 2
            assert all(isinstance(session, SessionSummary) for session in result)

    def test_list_actor_sessions_client_error(self):
        """Test list_actor_sessions with ClientError."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock paginator to raise ClientError
            mock_paginator = MagicMock()
            mock_client_instance.get_paginator.return_value = mock_paginator
            mock_paginator.paginate.side_effect = ClientError(
                {"Error": {"Code": "ValidationException", "Message": "Invalid actor"}}, "ListSessions"
            )

            with pytest.raises(ClientError):
                manager.list_actor_sessions(actor_id="invalid-actor")

    def test_create_session_success(self):
        """Test create_memory_session successful execution."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Test with provided session_id
            session = manager.create_memory_session(actor_id="user-123", session_id="session-456")

            assert isinstance(session, MemorySession)
            assert session._actor_id == "user-123"
            assert session._session_id == "session-456"
            assert session._memory_id == "testMemory-1234567890"

    def test_create_session_auto_generate_id(self):
        """Test create_memory_session with auto-generated session_id."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Test without session_id (should auto-generate)
            with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                session = manager.create_memory_session(actor_id="user-123")

                assert isinstance(session, MemorySession)
                assert session._actor_id == "user-123"
                assert session._session_id == "12345678-1234-5678-1234-567812345678"


class TestSession:
    """Test cases for MemorySession class."""

    def test_session_initialization(self):
        """Test MemorySession initialization."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            assert session._memory_id == "testMemory-1234567890"
            assert session._actor_id == "user-123"
            assert session._session_id == "session-456"
            assert session._manager == manager

            # Test dictionary representation
            assert session["memoryId"] == "testMemory-1234567890"
            assert session["actorId"] == "user-123"
            assert session["sessionId"] == "session-456"

    def test_session_add_turns_delegation(self):
        """Test MemorySession.add_turns delegates to manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_event = Event({"eventId": "event-123"})
            with patch.object(manager, "add_turns", return_value=mock_event) as mock_add_turns:
                result = session.add_turns(messages=[ConversationalMessage("Hello", MessageRole.USER)])

                assert result == mock_event
            mock_add_turns.assert_called_once_with(
                "user-123", "session-456", [ConversationalMessage("Hello", MessageRole.USER)], None, None, None
            )

    def test_session_fork_conversation_delegation(self):
        """Test MemorySession.fork_conversation delegates to manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_event = Event({"eventId": "fork-event-123"})
            with patch.object(manager, "fork_conversation", return_value=mock_event) as mock_fork:
                result = session.fork_conversation(
                    messages=[ConversationalMessage("Fork message", MessageRole.USER)],
                    root_event_id="event-root-123",
                    branch_name="test-branch",
                )

                assert result == mock_event
                mock_fork.assert_called_once_with(
                    "user-123",
                    "session-456",
                    "event-root-123",
                    "test-branch",
                    [ConversationalMessage("Fork message", MessageRole.USER)],
                    None,
                    None,
                )

    def test_session_create_blob_event_delegation(self):
        """Test MemorySession can create blob events using add_turns."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_event = Event({"eventId": "blob-event-123"})
            with patch.object(manager, "add_turns", return_value=mock_event) as mock_add_turns:
                blob_data = {"data": "test"}
                result = session.add_turns(messages=[BlobMessage(blob_data)])

                assert result == mock_event
                mock_add_turns.assert_called_once_with(
                    "user-123", "session-456", [BlobMessage(blob_data)], None, None, None
                )

    def test_session_process_turn_with_llm_delegation(self):
        """Test MemorySession.process_turn_with_llm delegates to manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_memories = [{"content": {"text": "Memory"}}]
            mock_response = "LLM response"
            mock_event = {"eventId": "event-123"}
            with patch.object(
                manager, "process_turn_with_llm", return_value=(mock_memories, mock_response, mock_event)
            ) as mock_process:

                def mock_llm(user_input: str, memories: List[Dict[str, Any]]) -> str:
                    return "Response"

                memories, response, event = session.process_turn_with_llm(
                    user_input="Hello", llm_callback=mock_llm, retrieval_config=None
                )

                assert memories == mock_memories
                assert response == mock_response
                assert event == mock_event
                mock_process.assert_called_once_with("user-123", "session-456", "Hello", mock_llm, None, None, None)

    def test_session_get_last_k_turns_delegation(self):
        """Test MemorySession.get_last_k_turns delegates to manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_turns = [[EventMessage({"role": "USER", "content": {"text": "Hello"}})]]
            with patch.object(manager, "get_last_k_turns", return_value=mock_turns) as mock_get_turns:
                result = session.get_last_k_turns(k=3)

                assert result == mock_turns
                # Updated to match the new method signature with include_parent_branches parameter
                mock_get_turns.assert_called_once_with("user-123", "session-456", 3, None, None, 100)

    def test_session_get_event_delegation(self):
        """Test MemorySession.get_event delegates to manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_event = Event({"eventId": "event-123"})
            with patch.object(manager, "get_event", return_value=mock_event) as mock_get_event:
                result = session.get_event(event_id="event-123")

                assert result == mock_event
                mock_get_event.assert_called_once_with("user-123", "session-456", "event-123")

    def test_session_delete_event_delegation(self):
        """Test MemorySession.delete_event delegates to manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            with patch.object(manager, "delete_event") as mock_delete_event:
                session.delete_event(event_id="event-123")

                mock_delete_event.assert_called_once_with("user-123", "session-456", "event-123")

    def test_session_get_memory_record_delegation(self):
        """Test MemorySession.get_memory_record delegates to manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_record = MemoryRecord({"memoryRecordId": "rec-123"})
            with patch.object(manager, "get_memory_record", return_value=mock_record) as mock_get_record:
                result = session.get_memory_record(record_id="rec-123")

                assert result == mock_record
                mock_get_record.assert_called_once_with("rec-123")

    def test_session_delete_memory_record_delegation(self):
        """Test MemorySession.delete_memory_record delegates to manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            with patch.object(manager, "delete_memory_record") as mock_delete_record:
                session.delete_memory_record(record_id="rec-123")

                mock_delete_record.assert_called_once_with("rec-123")

    def test_session_search_long_term_memories_delegation(self):
        """Test MemorySession.search_long_term_memories delegates to manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_records = [MemoryRecord({"memoryRecordId": "rec-123"})]
            with patch.object(manager, "search_long_term_memories", return_value=mock_records) as mock_search:
                result = session.search_long_term_memories(query="test query", namespace_prefix="test/namespace")

                assert result == mock_records
                mock_search.assert_called_once_with("test query", "test/namespace", 3, None, 20)

    def test_session_list_long_term_memory_records_delegation(self):
        """Test MemorySession.list_long_term_memory_records delegates to manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_records = [MemoryRecord({"memoryRecordId": "rec-123"})]
            with patch.object(manager, "list_long_term_memory_records", return_value=mock_records) as mock_list:
                result = session.list_long_term_memory_records(namespace_prefix="test/namespace")

                assert result == mock_records
                mock_list.assert_called_once_with("test/namespace", None, 10)

    def test_session_list_actors_delegation(self):
        """Test MemorySession.list_actors delegates to manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_actors = [ActorSummary({"actorId": "user-1"})]
            with patch.object(manager, "list_actors", return_value=mock_actors) as mock_list_actors:
                result = session.list_actors()

                assert result == mock_actors
                mock_list_actors.assert_called_once()

    def test_session_list_events_delegation(self):
        """Test MemorySession.list_events delegates to manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_events = [Event({"eventId": "event-1"})]
            with patch.object(manager, "list_events", return_value=mock_events) as mock_list_events:
                result = session.list_events(branch_name="test-branch")

                assert result == mock_events
                mock_list_events.assert_called_once_with(
                    actor_id="user-123",
                    session_id="session-456",
                    branch_name="test-branch",
                    include_parent_branches=False,
                    eventMetadata=None,
                    include_payload=True,
                    max_results=100,
                )

    def test_session_list_branches_delegation(self):
        """Test MemorySession.list_branches delegates to manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_branches = [Branch({"name": "main"})]
            with patch.object(manager, "list_branches", return_value=mock_branches) as mock_list_branches:
                result = session.list_branches()

                assert result == mock_branches
                mock_list_branches.assert_called_once_with("user-123", "session-456")

    def test_session_get_actor(self):
        """Test MemorySession.get_actor returns Actor instance."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            actor = session.get_actor()

            assert isinstance(actor, Actor)
            assert actor._id == "user-123"
            assert actor._session_manager == manager


class TestActor:
    """Test cases for Actor class."""

    def test_actor_initialization(self):
        """Test Actor initialization."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            actor = Actor(actor_id="user-123", session_manager=manager)

            assert actor._id == "user-123"
            assert actor._session_manager == manager

            # Test dictionary representation
            assert actor["actorId"] == "user-123"

    def test_actor_list_sessions_delegation(self):
        """Test Actor.list_sessions delegates to session manager."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            actor = Actor(actor_id="user-123", session_manager=manager)

            # Mock manager method
            mock_sessions = [SessionSummary({"sessionId": "session-1"})]
            with patch.object(manager, "list_actor_sessions", return_value=mock_sessions) as mock_list_sessions:
                result = actor.list_sessions()

                assert result == mock_sessions
                mock_list_sessions.assert_called_once_with("user-123")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_add_turns_custom_timestamp(self):
        """Test add_turns with custom timestamp."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock create_event response
            mock_response = {"event": {"eventId": "event-123"}}
            mock_client_instance.create_event.return_value = mock_response

            custom_timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            manager.add_turns(
                actor_id="user-123",
                session_id="session-456",
                messages=[ConversationalMessage("Hello", MessageRole.USER)],
                event_timestamp=custom_timestamp,
            )

            # Verify custom timestamp was passed
            call_args = mock_client_instance.create_event.call_args[1]
            assert call_args["eventTimestamp"] == custom_timestamp

    def test_process_turn_with_llm_custom_retrieval_config(self):
        """Test process_turn_with_llm with custom retrieval config."""
        with patch("boto3.Session") as mock_boto_client:
            mock_client_instance = MagicMock()
            mock_boto_client.return_value = mock_client_instance

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock search_long_term_memories
            mock_memories = []
            with patch.object(manager, "search_long_term_memories", return_value=mock_memories) as mock_search:
                # Mock add_turns
                mock_event = {"eventId": "event-123"}
                with patch.object(manager, "add_turns", return_value=Event(mock_event)):

                    def mock_llm_callback(user_input: str, memories: List[Dict[str, Any]]) -> str:
                        return "Response"

                    # Test with custom retrieval config
                    retrieval_config = {"test/namespace": RetrievalConfig(top_k=5, retrieval_query="custom query")}
                    manager.process_turn_with_llm(
                        actor_id="user-123",
                        session_id="session-456",
                        user_input="Hello",
                        llm_callback=mock_llm_callback,
                        retrieval_config=retrieval_config,
                    )

                    # Verify custom query was used
                    mock_search.assert_called_once_with(
                        query="custom query Hello", namespace_prefix="test/namespace", top_k=5
                    )

    def test_list_events_max_results_respected(self):
        """Test list_events respects max_results parameter."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock response with more events than max_results
            large_batch = [{"eventId": f"event-{i}", "eventTimestamp": datetime.now()} for i in range(200)]
            mock_client_instance.list_events.return_value = {"events": large_batch, "nextToken": "has-more"}

            result = manager.list_events(actor_id="user-123", session_id="session-456", max_results=50)

            # Should only return 50 events
            assert len(result) == 50

    def test_get_last_k_turns_turn_grouping(self):
        """Test get_last_k_turns properly groups messages into turns."""
        with patch("boto3.Session") as mock_boto_client:
            mock_client_instance = MagicMock()
            mock_boto_client.return_value = mock_client_instance

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock events with multiple turns
            mock_events = [
                Event(
                    {
                        "eventId": "event-1",
                        "payload": [
                            {"conversational": {"role": "USER", "content": {"text": "Hello"}}},
                            {"conversational": {"role": "ASSISTANT", "content": {"text": "Hi"}}},
                            {
                                "conversational": {"role": "USER", "content": {"text": "How are you?"}}
                            },  # New turn starts
                            {"conversational": {"role": "ASSISTANT", "content": {"text": "Good"}}},
                        ],
                    }
                )
            ]
            with patch.object(manager, "list_events", return_value=mock_events):
                result = manager.get_last_k_turns(actor_id="user-123", session_id="session-456", k=5)

                # Should group into 2 turns
                assert len(result) == 2
                assert len(result[0]) == 2  # First turn: USER + ASSISTANT
                assert len(result[1]) == 2  # Second turn: USER + ASSISTANT

    def test_session_delegation_with_optional_parameters(self):
        """Test MemorySession methods properly pass optional parameters."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Test add_turns with all optional parameters
            mock_event = Event({"eventId": "event-123"})
            with patch.object(manager, "add_turns", return_value=mock_event) as mock_add_turns:
                custom_timestamp = datetime.now(timezone.utc)
                branch = {"name": "test-branch", "rootEventId": "root-123"}

                session.add_turns(
                    messages=[ConversationalMessage("Hello", MessageRole.USER)],
                    branch=branch,
                    event_timestamp=custom_timestamp,
                )

                mock_add_turns.assert_called_once_with(
                    "user-123",
                    "session-456",
                    [ConversationalMessage("Hello", MessageRole.USER)],
                    branch,
                    None,
                    custom_timestamp,
                )

    def test_comprehensive_error_coverage(self):
        """Test comprehensive error handling across different methods."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Test various ClientError scenarios
            error_scenarios = [
                ("ValidationException", "Invalid parameters"),
                ("ResourceNotFoundException", "Resource not found"),
                ("AccessDeniedException", "Access denied"),
                ("ThrottlingException", "Request throttled"),
                ("ServiceException", "Internal service error"),
            ]

            for error_code, error_message in error_scenarios:
                error_response = {"Error": {"Code": error_code, "Message": error_message}}
                mock_client_instance.create_event.side_effect = ClientError(error_response, "CreateEvent")

                with pytest.raises(ClientError) as exc_info:
                    manager.add_turns(
                        actor_id="user-123",
                        session_id="session-456",
                        messages=[ConversationalMessage("Hello", MessageRole.USER)],
                    )

                assert error_code in str(exc_info.value)
                assert error_message in str(exc_info.value)

    def test_getattr_method_not_in_allowed_but_exists_on_client(self):
        """Test __getattr__ when method exists on client but not in allowed methods."""
        with patch("boto3.Session") as mock_boto_client:
            mock_client_instance = MagicMock()
            mock_boto_client.return_value = mock_client_instance

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock hasattr to return True for a method not in allowed list
            with patch("builtins.hasattr", return_value=True):
                with pytest.raises(AttributeError) as exc_info:
                    _ = manager.some_disallowed_method

                assert "'MemorySessionManager' object has no attribute 'some_disallowed_method'" in str(exc_info.value)

    def test_search_long_term_memories_without_strategy_id(self):
        """Test search_long_term_memories without strategy_id to cover missing branch."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock retrieve_memory_records response
            mock_response = {"memoryRecordSummaries": []}
            mock_client_instance.retrieve_memory_records.return_value = mock_response

            result = manager.search_long_term_memories(query="test query", namespace_prefix="test/namespace")

            assert result == []

            # Verify strategy_id was not passed
            call_args = mock_client_instance.retrieve_memory_records.call_args[1]
            assert "strategyId" not in call_args["searchCriteria"]

    def test_list_long_term_memory_records_without_strategy_id(self):
        """Test list_long_term_memory_records without strategy_id to cover missing branch."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock paginator
            mock_paginator = MagicMock()
            mock_client_instance.get_paginator.return_value = mock_paginator
            mock_paginator.paginate.return_value = [{"memoryRecords": []}]

            result = manager.list_long_term_memory_records(namespace_prefix="test/namespace")

            assert result == []

            # Verify strategy_id was not passed
            call_args = mock_paginator.paginate.call_args[1]
            assert "strategyId" not in call_args

    def test_list_events_no_next_token_break(self):
        """Test list_events when no next_token is returned to cover break condition."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock response without nextToken
            mock_events = [{"eventId": "event-1", "eventTimestamp": datetime.now()}]
            mock_client_instance.list_events.return_value = {"events": mock_events}  # No nextToken

            result = manager.list_events(actor_id="user-123", session_id="session-456")

            assert len(result) == 1
            assert mock_client_instance.list_events.call_count == 1

    def test_list_events_max_results_break(self):
        """Test list_events when max_results is reached to cover break condition."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock response with exactly max_results events
            mock_events = [{"eventId": f"event-{i}", "eventTimestamp": datetime.now()} for i in range(5)]
            mock_client_instance.list_events.return_value = {"events": mock_events, "nextToken": "has-more"}

            result = manager.list_events(actor_id="user-123", session_id="session-456", max_results=5)

            # Should break when max_results is reached
            assert len(result) == 5
            assert mock_client_instance.list_events.call_count == 1

    def test_get_last_k_turns_no_conversational_payload(self):
        """Test get_last_k_turns with payload that has no conversational items."""
        with patch("boto3.Session") as mock_boto_client:
            mock_client_instance = MagicMock()
            mock_boto_client.return_value = mock_client_instance

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock events with non-conversational payload
            mock_events = [
                Event(
                    {
                        "eventId": "event-1",
                        "eventTimestamp": datetime(2023, 1, 1, 10, 0, 0),
                        "payload": [{"blob": {"data": "some_data"}}],  # Non-conversational payload
                    }
                )
            ]
            with patch.object(manager, "list_events", return_value=mock_events):
                result = manager.get_last_k_turns(actor_id="user-123", session_id="session-456", k=5)

                assert len(result) == 0  # No turns due to no conversational messages

    def test_get_last_k_turns_break_on_k_limit(self):
        """Test get_last_k_turns breaks when k limit is reached."""
        with patch("boto3.Session") as mock_boto_client:
            mock_client_instance = MagicMock()
            mock_boto_client.return_value = mock_client_instance

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock many events to test k limit
            mock_events = []
            for i in range(10):  # Create 10 events, each with a USER message (new turn)
                mock_events.append(
                    Event(
                        {
                            "eventId": f"event-{i}",
                            "eventTimestamp": datetime(2023, 1, 1, 10, i, 0),
                            "payload": [{"conversational": {"role": "USER", "content": {"text": f"Message {i}"}}}],
                        }
                    )
                )

            with patch.object(manager, "list_events", return_value=mock_events):
                result = manager.get_last_k_turns(actor_id="user-123", session_id="session-456", k=3)

                # Should only return 3 turns even though there are 10 events
                assert len(result) == 3

    def test_actor_list_sessions_return_type(self):
        """Test Actor.list_sessions returns correct type."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            actor = Actor(actor_id="user-123", session_manager=manager)

            # Mock manager method to return SessionSummary objects
            mock_sessions = [SessionSummary({"sessionId": "session-1"})]
            with patch.object(manager, "list_actor_sessions", return_value=mock_sessions) as mock_list_sessions:
                result = actor.list_sessions()

                # The method should return the SessionSummary objects, not Session objects
                assert result == mock_sessions
                assert all(isinstance(session, SessionSummary) for session in result)
                mock_list_sessions.assert_called_once_with("user-123")

    def test_getattr_method_exists_but_not_allowed(self):
        """Test __getattr__ when method exists on client but not in allowed methods."""
        with patch("boto3.Session") as mock_boto_client:
            mock_client_instance = MagicMock()
            mock_boto_client.return_value = mock_client_instance

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock a method that exists on client but is not in allowed methods
            mock_client_instance.some_other_method = MagicMock()

            with pytest.raises(AttributeError) as exc_info:
                _ = manager.some_other_method

            assert "'MemorySessionManager' object has no attribute 'some_other_method'" in str(exc_info.value)

    def test_session_add_turns_parameter_order(self):
        """Test MemorySession.add_turns passes parameters in correct order."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method to verify parameter order
            mock_event = Event({"eventId": "event-123"})
            with patch.object(manager, "add_turns", return_value=mock_event) as mock_add_turns:
                custom_timestamp = datetime.now(timezone.utc)
                branch = {"name": "test-branch"}

                session.add_turns(
                    messages=[ConversationalMessage("Hello", MessageRole.USER)],
                    branch=branch,
                    event_timestamp=custom_timestamp,
                )

                # Verify the exact parameter order: actor_id, session_id, messages, branch, event_timestamp
                mock_add_turns.assert_called_once_with(
                    "user-123",
                    "session-456",
                    [ConversationalMessage("Hello", MessageRole.USER)],
                    branch,
                    None,
                    custom_timestamp,
                )


class TestEventMetadataFlow:
    """Test cases for metadata support for STM in MemorySessionManager."""

    def test_fork_conversation_with_metadata_parameter(self):
        """Test fork_conversation with new metadata parameter."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock add_turns
            mock_event = {"eventId": "fork-event-123", "memoryId": "testMemory-1234567890"}
            with patch.object(manager, "add_turns", return_value=Event(mock_event)) as mock_add_turns:
                metadata = {"location": {"stringValue": "NYC"}}

                result = manager.fork_conversation(
                    actor_id="user-123",
                    session_id="session-456",
                    root_event_id="event-root-123",
                    branch_name="test-branch",
                    messages=[ConversationalMessage("Fork message", MessageRole.USER)],
                    metadata=metadata,
                )

                assert result["eventId"] == "fork-event-123"

                # Verify add_turns was called with metadata
                mock_add_turns.assert_called_once()
                call_args = mock_add_turns.call_args[1]
                assert call_args["metadata"] == metadata
                assert call_args["branch"]["rootEventId"] == "event-root-123"
                assert call_args["branch"]["name"] == "test-branch"

    def test_list_events_with_event_metadata_filter(self):
        """Test list_events with eventMetadata filter parameter."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock response
            mock_events = [{"eventId": "filtered-event-1", "eventTimestamp": datetime.now()}]
            mock_client_instance.list_events.return_value = {"events": mock_events, "nextToken": None}

            # Test with eventMetadata filter
            event_metadata_filter = [
                {
                    "left": {"metadataKey": "location"},
                    "operator": "EQUALS_TO",
                    "right": {"metadataValue": {"stringValue": "NYC"}},
                }
            ]

            result = manager.list_events(
                actor_id="user-123", session_id="session-456", eventMetadata=event_metadata_filter
            )

            assert len(result) == 1
            assert result[0]["eventId"] == "filtered-event-1"

            # Verify filter was applied
            call_args = mock_client_instance.list_events.call_args[1]
            assert "filter" in call_args
            assert call_args["filter"]["eventMetadata"] == event_metadata_filter

    def test_list_events_with_both_branch_and_metadata_filters(self):
        """Test list_events with both branch and eventMetadata filters."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock response
            mock_events = [{"eventId": "filtered-event-1", "eventTimestamp": datetime.now()}]
            mock_client_instance.list_events.return_value = {"events": mock_events, "nextToken": None}

            # Test with both branch and eventMetadata filters
            event_metadata_filter = [
                {
                    "left": {"metadataKey": "location"},
                    "operator": "EQUALS_TO",
                    "right": {"metadataValue": {"stringValue": "NYC"}},
                }
            ]

            result = manager.list_events(
                actor_id="user-123",
                session_id="session-456",
                branch_name="test-branch",
                include_parent_branches=True,
                eventMetadata=event_metadata_filter,
            )

            assert len(result) == 1

            # Verify both filters were applied - eventMetadata should override branch filter
            call_args = mock_client_instance.list_events.call_args[1]
            assert "filter" in call_args
            assert call_args["filter"]["eventMetadata"] == event_metadata_filter
            # Branch filter should be present when eventMetadata is specified
            assert "branch" in call_args["filter"]

    def test_memory_session_list_events_with_event_metadata(self):
        """Test MemorySession.list_events with eventMetadata parameter."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_events = [Event({"eventId": "event-1"})]
            event_metadata_filter = [
                {
                    "left": {"metadataKey": "location"},
                    "operator": "EQUALS_TO",
                    "right": {"metadataValue": {"stringValue": "NYC"}},
                }
            ]

            with patch.object(manager, "list_events", return_value=mock_events) as mock_list_events:
                result = session.list_events(branch_name="test-branch", eventMetadata=event_metadata_filter)

                assert result == mock_events
                mock_list_events.assert_called_once_with(
                    actor_id="user-123",
                    session_id="session-456",
                    branch_name="test-branch",
                    include_parent_branches=False,
                    eventMetadata=event_metadata_filter,
                    include_payload=True,
                    max_results=100,
                )

    def test_memory_session_fork_conversation_with_metadata(self):
        """Test MemorySession.fork_conversation with metadata parameter."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_event = Event({"eventId": "fork-event-123"})
            metadata = {"location": {"stringValue": "NYC"}}

            with patch.object(manager, "fork_conversation", return_value=mock_event) as mock_fork:
                result = session.fork_conversation(
                    messages=[ConversationalMessage("Fork message", MessageRole.USER)],
                    root_event_id="event-root-123",
                    branch_name="test-branch",
                    metadata=metadata,
                )

                assert result == mock_event
                mock_fork.assert_called_once_with(
                    "user-123",
                    "session-456",
                    "event-root-123",
                    "test-branch",
                    [ConversationalMessage("Fork message", MessageRole.USER)],
                    metadata,
                    None,
                )

    def test_memory_session_process_turn_with_llm_with_metadata(self):
        """Test MemorySession.process_turn_with_llm with metadata parameter."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_memories = [{"content": {"text": "Memory"}}]
            mock_response = "LLM response"
            mock_event = {"eventId": "event-123"}
            metadata = {"location": {"stringValue": "NYC"}}

            with patch.object(
                manager, "process_turn_with_llm", return_value=(mock_memories, mock_response, mock_event)
            ) as mock_process:

                def mock_llm(user_input: str, memories: List[Dict[str, Any]]) -> str:
                    return "Response"

                memories, response, event = session.process_turn_with_llm(
                    user_input="Hello", llm_callback=mock_llm, retrieval_config=None, metadata=metadata
                )

                assert memories == mock_memories
                assert response == mock_response
                assert event == mock_event
                mock_process.assert_called_once_with("user-123", "session-456", "Hello", mock_llm, None, metadata, None)

    def test_process_turn_with_llm_with_metadata_parameter(self):
        """Test process_turn_with_llm with metadata parameter."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock search_long_term_memories
            mock_memories = [{"content": {"text": "Previous context"}, "memoryRecordId": "rec-123"}]
            with patch.object(manager, "search_long_term_memories", return_value=mock_memories):
                # Mock add_turns
                mock_event = {"eventId": "event-123", "memoryId": "testMemory-1234567890"}
                with patch.object(manager, "add_turns", return_value=Event(mock_event)) as mock_add_turns:
                    # Define LLM callback
                    def mock_llm_callback(user_input: str, memories: List[Dict[str, Any]]) -> str:
                        return f"Response to: {user_input} with {len(memories)} memories"

                    # Test process_turn_with_llm with metadata
                    retrieval_config = {"test/namespace": RetrievalConfig(top_k=5)}
                    metadata = {"location": {"stringValue": "NYC"}}

                    memories, response, event = manager.process_turn_with_llm(
                        actor_id="user-123",
                        session_id="session-456",
                        user_input="Hello",
                        llm_callback=mock_llm_callback,
                        retrieval_config=retrieval_config,
                        metadata=metadata,
                    )

                    assert len(memories) == 1
                    assert "Response to: Hello with 1 memories" in response
                    assert event["eventId"] == "event-123"

                    # Verify add_turns was called with metadata
                    mock_add_turns.assert_called_once()
                    call_args = mock_add_turns.call_args[1]
                    assert call_args["metadata"] == metadata

    def test_add_turns_with_metadata_parameter(self):
        """Test add_turns with metadata parameter."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock create_event response
            mock_response = {"event": {"eventId": "turn-event-123", "memoryId": "testMemory-1234567890"}}
            mock_client_instance.create_event.return_value = mock_response

            messages = [
                ConversationalMessage("Hello", MessageRole.USER),
                ConversationalMessage("Hi there", MessageRole.ASSISTANT),
            ]
            metadata = {"location": {"stringValue": "NYC"}}

            result = manager.add_turns(
                actor_id="user-123", session_id="session-456", messages=messages, metadata=metadata
            )

            assert isinstance(result, Event)
            assert result["eventId"] == "turn-event-123"

            # Verify metadata was passed to create_event
            call_args = mock_client_instance.create_event.call_args[1]
            assert call_args["metadata"] == metadata
            assert len(call_args["payload"]) == 2

    def test_memory_session_add_turns_with_metadata(self):
        """Test MemorySession.add_turns with metadata parameter."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")
            session = MemorySession(
                memory_id="testMemory-1234567890", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method
            mock_event = Event({"eventId": "event-123"})
            metadata = {"location": {"stringValue": "NYC"}}

            with patch.object(manager, "add_turns", return_value=mock_event) as mock_add_turns:
                result = session.add_turns(
                    messages=[ConversationalMessage("Hello", MessageRole.USER)], metadata=metadata
                )

                assert result == mock_event
                mock_add_turns.assert_called_once_with(
                    "user-123", "session-456", [ConversationalMessage("Hello", MessageRole.USER)], None, metadata, None
                )


class TestAdditionalCoverage:
    """Additional tests to reach 99% coverage."""

    def test_getattr_method_not_in_allowed_and_not_on_client(self):
        """Test __getattr__ when method doesn't exist on client and not in allowed methods."""
        with patch("boto3.Session") as mock_boto_client:
            mock_client_instance = MagicMock()
            mock_boto_client.return_value = mock_client_instance

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock hasattr to return False (method doesn't exist on client)
            with patch("builtins.hasattr", return_value=False):
                with pytest.raises(AttributeError) as exc_info:
                    _ = manager.nonexistent_method

                assert "'MemorySessionManager' object has no attribute 'nonexistent_method'" in str(exc_info.value)

    def test_process_turn_with_llm_with_retrieval_query_fallback(self):
        """Test process_turn_with_llm uses user_input when retrieval_query is None."""
        with patch("boto3.Session") as mock_boto_client:
            mock_client_instance = MagicMock()
            mock_boto_client.return_value = mock_client_instance

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock search_long_term_memories
            mock_memories = []
            with patch.object(manager, "search_long_term_memories", return_value=mock_memories) as mock_search:
                # Mock add_turns
                mock_event = {"eventId": "event-123"}
                with patch.object(manager, "add_turns", return_value=Event(mock_event)):

                    def mock_llm_callback(user_input: str, memories: List[Dict[str, Any]]) -> str:
                        return "Response"

                    # Test with retrieval_config but no retrieval_query (should use user_input)
                    retrieval_config = {"test/namespace": RetrievalConfig(top_k=3, retrieval_query=None)}
                    manager.process_turn_with_llm(
                        actor_id="user-123",
                        session_id="session-456",
                        user_input="Hello",
                        llm_callback=mock_llm_callback,
                        retrieval_config=retrieval_config,
                    )

                    # Verify user_input was used as query
                    mock_search.assert_called_once_with(query="Hello", namespace_prefix="test/namespace", top_k=3)

    def test_add_turns_with_custom_timestamp(self):
        """Test add_turns with custom timestamp."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock create_event response
            mock_response = {"event": {"eventId": "turn-event-123", "memoryId": "testMemory-1234567890"}}
            mock_client_instance.create_event.return_value = mock_response

            custom_timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            messages = [ConversationalMessage("Hello", MessageRole.USER)]

            result = manager.add_turns(
                actor_id="user-123", session_id="session-456", messages=messages, event_timestamp=custom_timestamp
            )

            assert isinstance(result, Event)
            assert result["eventId"] == "turn-event-123"

            # Verify custom timestamp was passed
            call_args = mock_client_instance.create_event.call_args[1]
            assert call_args["eventTimestamp"] == custom_timestamp

    def test_list_events_with_next_token(self):
        """Test list_events with next_token parameter."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock paginated responses
            first_batch = [{"eventId": "event-1", "eventTimestamp": datetime.now()}]
            second_batch = [{"eventId": "event-2", "eventTimestamp": datetime.now()}]

            mock_client_instance.list_events.side_effect = [
                {"events": first_batch, "nextToken": "token-123"},
                {"events": second_batch, "nextToken": None},
            ]

            result = manager.list_events(actor_id="user-123", session_id="session-456", max_results=2)

            assert len(result) == 2
            assert mock_client_instance.list_events.call_count == 2

            # Verify nextToken was passed in second call
            second_call_args = mock_client_instance.list_events.call_args_list[1][1]
            assert second_call_args["nextToken"] == "token-123"

    def test_validate_and_resolve_region_no_session_region(self):
        """Test _validate_and_resolve_region when session has no region - covers line 154."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = None  # No region in session
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Should use provided region_name when session has no region
            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-1")
            assert manager.region_name == "us-west-1"

    def test_build_client_config_no_existing_user_agent(self):
        """Test _build_client_config when boto_client_config has no user_agent_extra - covers lines 197-200."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-east-1"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Create config without user_agent_extra
            custom_config = BotocoreConfig(retries={"max_attempts": 3})
            # Ensure user_agent_extra is None
            custom_config.user_agent_extra = None

            MemorySessionManager(memory_id="test-memory", region_name="us-east-1", boto_client_config=custom_config)

            # Verify client was called with merged config
            call_args = mock_session.client.call_args[1]
            passed_config = call_args["config"]

            # Should set user agent to just the SDK user agent (no existing agent to merge)
            assert passed_config.user_agent_extra == "bedrock-agentcore-sdk"

    def test_process_turn_with_llm_with_relevance_score_filtering(self):
        """Test process_turn_with_llm with relevance score filtering - covers line 316."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-2")

            # Mock search_long_term_memories to return records with different relevance scores
            mock_memories = [
                {"content": {"text": "High relevance"}, "memoryRecordId": "rec-1", "relevanceScore": 0.8},
                {"content": {"text": "Low relevance"}, "memoryRecordId": "rec-2", "relevanceScore": 0.2},
                {"content": {"text": "Medium relevance"}, "memoryRecordId": "rec-3", "relevanceScore": 0.5},
            ]
            with patch.object(manager, "search_long_term_memories", return_value=mock_memories):
                # Mock add_turns
                mock_event = {"eventId": "event-123"}
                with patch.object(manager, "add_turns", return_value=Event(mock_event)):

                    def mock_llm_callback(user_input: str, memories: List[Dict[str, Any]]) -> str:
                        return f"Response with {len(memories)} memories"

                    # Test with relevance_score filtering (should filter out low relevance)
                    retrieval_config = {"test/namespace": RetrievalConfig(top_k=5, relevance_score=0.4)}
                    memories, response, event = manager.process_turn_with_llm(
                        actor_id="user-123",
                        session_id="session-456",
                        user_input="Hello",
                        llm_callback=mock_llm_callback,
                        retrieval_config=retrieval_config,
                    )

                    # Should have filtered out the record with relevance_score 0.2
                    assert len(memories) == 2  # Only records with score >= 0.4
                    assert "Response with 2 memories" in response

    def test_list_events_empty_events_break(self):
        """Test list_events when empty events are returned - covers lines 493->526."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-2")

            # Mock response with empty events on second call
            mock_client_instance.list_events.side_effect = [
                {"events": [{"eventId": "event-1", "eventTimestamp": datetime.now()}], "nextToken": "token-123"},
                {"events": [], "nextToken": "token-456"},  # Empty events should break the loop
            ]

            result = manager.list_events(actor_id="user-123", session_id="session-456")

            assert len(result) == 1
            assert mock_client_instance.list_events.call_count == 2

    def test_list_events_max_iterations_warning(self):
        """Test list_events max iterations warning - covers lines 517-518."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-2")

            # Mock response that would cause infinite loop (always has nextToken)
            mock_client_instance.list_events.return_value = {
                "events": [{"eventId": "event-1", "eventTimestamp": datetime.now()}],
                "nextToken": "always-has-token",
            }

            with patch("bedrock_agentcore.memory.session.logger") as mock_logger:
                # Set max_results high enough that we hit max_iterations first
                result = manager.list_events(actor_id="user-123", session_id="session-456", max_results=10000)

                # Should have hit max iterations and logged warning
                mock_logger.warning.assert_called_with(
                    "Reached maximum iteration limit (%d) in list_events pagination", 1000
                )
                assert len(result) > 0

    def test_list_events_debug_logging_no_events(self):
        """Test list_events debug logging when no events returned - covers line 527."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-2")

            # Mock response with no events
            mock_client_instance.list_events.return_value = {"events": []}

            with patch("bedrock_agentcore.memory.session.logger") as mock_logger:
                result = manager.list_events(actor_id="user-123", session_id="session-456")

                # Should have logged debug message about no events
                mock_logger.debug.assert_called_with("No more events returned, ending pagination")
                assert len(result) == 0

    def test_list_branches_empty_events_break(self):
        """Test list_branches when empty events are returned - covers lines 553->575."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-2")

            # Mock response with empty events on second call
            mock_client_instance.list_events.side_effect = [
                {"events": [{"eventId": "event-1", "eventTimestamp": datetime.now()}], "nextToken": "token-123"},
                {"events": [], "nextToken": "token-456"},  # Empty events should break the loop
            ]

            result = manager.list_branches(actor_id="user-123", session_id="session-456")

            assert len(result) == 1  # Should have main branch
            assert mock_client_instance.list_events.call_count == 2

    def test_list_branches_max_iterations_warning(self):
        """Test list_branches max iterations warning - covers lines 566-567."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-2")

            # Mock response that would cause infinite loop
            mock_client_instance.list_events.return_value = {
                "events": [{"eventId": "event-1", "eventTimestamp": datetime.now()}],
                "nextToken": "always-has-token",
            }

            with patch("bedrock_agentcore.memory.session.logger") as mock_logger:
                result = manager.list_branches(actor_id="user-123", session_id="session-456")

                # Should have hit max iterations and logged warning
                mock_logger.warning.assert_called_with(
                    "Reached maximum iteration limit (%d) in list_branches pagination", 1000
                )
                assert len(result) > 0

    def test_list_branches_debug_logging_no_events(self):
        """Test list_branches debug logging when no events returned - covers line 576."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-2")

            # Mock response with no events
            mock_client_instance.list_events.return_value = {"events": []}

            with patch("bedrock_agentcore.memory.session.logger") as mock_logger:
                result = manager.list_branches(actor_id="user-123", session_id="session-456")

                # Should have logged debug message about no events
                mock_logger.debug.assert_called_with("No more events returned, ending pagination in list_branches")
                assert len(result) == 0

    def test_list_branches_multiple_events_same_branch(self):
        """Test list_branches with multiple events in same branch - covers line 594."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-2")

            # Mock events with multiple events in same branch
            mock_events = [
                {
                    "eventId": "event-1",
                    "eventTimestamp": datetime(2023, 1, 1, 10, 0, 0),
                    "branch": {"name": "test-branch", "rootEventId": "root-1"},
                },
                {
                    "eventId": "event-2",
                    "eventTimestamp": datetime(2023, 1, 1, 10, 5, 0),
                    "branch": {"name": "test-branch", "rootEventId": "root-1"},  # Same branch
                },
            ]
            mock_client_instance.list_events.return_value = {"events": mock_events, "nextToken": None}

            result = manager.list_branches(actor_id="user-123", session_id="session-456")

            assert len(result) == 1  # Only one branch
            assert result[0]["name"] == "test-branch"
            assert result[0]["eventCount"] == 2  # Should increment count for second event

    def test_memory_session_add_turns_parameter_order(self):
        """Test MemorySession.add_turns parameter order - covers line 779."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-2")
            session = MemorySession(
                memory_id="test-memory", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method to verify exact parameter order
            mock_event = Event({"eventId": "event-123"})
            with patch.object(manager, "add_turns", return_value=mock_event) as mock_add_turns:
                messages = [ConversationalMessage("Hello", MessageRole.USER)]
                custom_timestamp = datetime.now(timezone.utc)
                branch = {"name": "test-branch"}

                # Call with all parameters to test the exact order
                session.add_turns(messages=messages, branch=branch, event_timestamp=custom_timestamp)

                # Verify the exact parameter order: actor_id, session_id, messages, branch, event_timestamp
                mock_add_turns.assert_called_once_with(
                    "user-123", "session-456", messages, branch, None, custom_timestamp
                )

    def test_process_turn_with_llm_no_relevance_score_config(self):
        """Test process_turn_with_llm when RetrievalConfig has no relevance_score."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-2")

            # Mock search_long_term_memories
            mock_memories = [{"content": {"text": "Memory"}, "memoryRecordId": "rec-1"}]
            with patch.object(manager, "search_long_term_memories", return_value=mock_memories):
                # Mock add_turns
                mock_event = {"eventId": "event-123"}
                with patch.object(manager, "add_turns", return_value=Event(mock_event)):

                    def mock_llm_callback(user_input: str, memories: List[Dict[str, Any]]) -> str:
                        return "Response"

                    # Test with RetrievalConfig that has a very low relevance_score (effectively no filtering)
                    retrieval_config = {"test/namespace": RetrievalConfig(top_k=3, relevance_score=0.0)}
                    memories, response, event = manager.process_turn_with_llm(
                        actor_id="user-123",
                        session_id="session-456",
                        user_input="Hello",
                        llm_callback=mock_llm_callback,
                        retrieval_config=retrieval_config,
                    )

                    # Should not filter any memories when relevance_score is very low
                    assert len(memories) == 1
                    assert response == "Response"

    def test_validate_and_resolve_region_edge_case(self):
        """Test _validate_and_resolve_region edge case - covers line 154."""
        with patch("boto3.Session") as mock_session_class, patch.dict("os.environ", {}, clear=True):
            mock_session = MagicMock()
            mock_session.region_name = None  # No region in session
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Test when both region_name and session region are None, should fallback to us-west-2
            manager = MemorySessionManager(memory_id="test-memory", region_name=None)
            assert manager.region_name == "us-west-2"

    def test_region_resolution_priority_order(self):
        """Test region resolution follows documented priority order:
        1. region_name parameter
        2. boto3_session region
        3. AWS_REGION env var
        4. boto3.Session().region_name (checks AWS_DEFAULT_REGION and AWS config)
        5. us-west-2 fallback
        """
        # Test 1: region_name parameter takes highest priority
        with (
            patch("boto3.Session") as mock_session_class,
            patch.dict("os.environ", {"AWS_REGION": "eu-west-1"}, clear=True),
        ):
            mock_session = MagicMock()
            mock_session.region_name = None
            mock_session.client.return_value = MagicMock()
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory", region_name="us-east-1")
            assert manager.region_name == "us-east-1"

        # Test 2: boto3_session region takes priority over AWS_REGION
        with patch.dict("os.environ", {"AWS_REGION": "eu-west-1"}, clear=True):
            mock_session = MagicMock()
            mock_session.region_name = "ap-south-1"
            mock_session.client.return_value = MagicMock()

            manager = MemorySessionManager(memory_id="test-memory", boto3_session=mock_session)
            assert manager.region_name == "ap-south-1"

        # Test 3: AWS_REGION env var takes priority over boto3.Session().region_name
        with (
            patch("boto3.Session") as mock_session_class,
            patch.dict("os.environ", {"AWS_REGION": "eu-central-1"}, clear=True),
        ):
            mock_session = MagicMock()
            mock_session.region_name = "ca-central-1"  # This would be from AWS config
            mock_session.client.return_value = MagicMock()
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory")
            assert manager.region_name == "eu-central-1"

        # Test 4: boto3.Session().region_name used when AWS_REGION not set
        with patch("boto3.Session") as mock_session_class, patch.dict("os.environ", {}, clear=True):
            mock_session = MagicMock()
            mock_session.region_name = "sa-east-1"  # From AWS_DEFAULT_REGION or AWS config
            mock_session.client.return_value = MagicMock()
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory")
            assert manager.region_name == "sa-east-1"

        # Test 5: us-west-2 fallback when nothing is set
        with patch("boto3.Session") as mock_session_class, patch.dict("os.environ", {}, clear=True):
            mock_session = MagicMock()
            mock_session.region_name = None
            mock_session.client.return_value = MagicMock()
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory")
            assert manager.region_name == "us-west-2"

    def test_memory_session_add_turns_branch_parameter_order(self):
        """Test MemorySession.add_turns with branch parameter order - covers line 779."""
        with patch("boto3.Session"):
            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-2")
            session = MemorySession(
                memory_id="test-memory", actor_id="user-123", session_id="session-456", manager=manager
            )

            # Mock manager method to verify exact parameter order
            mock_event = Event({"eventId": "event-123"})
            with patch.object(manager, "add_turns", return_value=mock_event) as mock_add_turns:
                messages = [ConversationalMessage("Hello", MessageRole.USER)]
                branch = {"name": "test-branch"}

                # Call with branch parameter only (no timestamp)
                session.add_turns(messages=messages, branch=branch)

                # Verify the exact parameter order: actor_id, session_id, messages, branch, event_timestamp
                mock_add_turns.assert_called_once_with("user-123", "session-456", messages, branch, None, None)

    def test_list_long_term_memory_records_memoryRecordSummaries_fallback(self):
        """Test list_long_term_memory_records fallback to memoryRecordSummaries."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-2")

            # Mock paginator that returns memoryRecordSummaries instead of memoryRecords
            mock_paginator = MagicMock()
            mock_client_instance.get_paginator.return_value = mock_paginator
            mock_paginator.paginate.return_value = [
                {
                    "memoryRecords": [],  # Empty memoryRecords
                    "memoryRecordSummaries": [  # Should fall back to this
                        {"memoryRecordId": "rec-1"},
                        {"memoryRecordId": "rec-2"},
                    ],
                }
            ]

            result = manager.list_long_term_memory_records(namespace_prefix="test/namespace")

            assert len(result) == 2
            assert all(isinstance(record, MemoryRecord) for record in result)
            assert result[0]["memoryRecordId"] == "rec-1"
            assert result[1]["memoryRecordId"] == "rec-2"

    def test_region_validation_with_non_string_session_region(self):
        """Test region validation when session region is not a string."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = 123  # Non-string region (shouldn't cause conflict)
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Should not raise ValueError when session region is not a string
            manager = MemorySessionManager(memory_id="test-memory", region_name="us-west-1")
            assert manager.region_name == "us-west-1"

    def test_region_validation_order_change(self):
        """Test that region validation happens before session creation - covers recent commit changes."""
        # Test case: Conflicting regions should raise ValueError
        custom_session = MagicMock()
        custom_session.region_name = "us-east-1"
        mock_client_instance = MagicMock()
        custom_session.client.return_value = mock_client_instance

        with pytest.raises(ValueError) as exc_info:
            MemorySessionManager(
                memory_id="test-memory",
                region_name="us-west-1",  # Different from session region
                boto3_session=custom_session,
            )

        assert "Region mismatch" in str(exc_info.value)
        assert "us-west-1" in str(exc_info.value)
        assert "us-east-1" in str(exc_info.value)

    def test_region_validation_with_none_session(self):
        """Test region validation when boto3_session is None - covers recent commit changes."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-east-1"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Test validation when boto3_session parameter is None
            manager = MemorySessionManager(
                memory_id="test-memory",
                region_name="us-west-1",
                boto3_session=None,  # Explicitly None
            )

            # Should use the provided region_name
            assert manager.region_name == "us-west-1"

    def test_region_validation_simplified_logic(self):
        """Test the simplified region validation logic - covers recent commit changes."""
        # Test case 1: Conflicting regions should raise ValueError
        custom_session = MagicMock()
        custom_session.region_name = "us-east-1"
        mock_client_instance = MagicMock()
        custom_session.client.return_value = mock_client_instance

        with pytest.raises(ValueError) as exc_info:
            MemorySessionManager(
                memory_id="test-memory",
                region_name="us-west-1",  # Different from session region
                boto3_session=custom_session,
            )

        assert "Region mismatch" in str(exc_info.value)
        assert "us-west-1" in str(exc_info.value)
        assert "us-east-1" in str(exc_info.value)

        # Test case 2: Matching regions should work
        custom_session2 = MagicMock()
        custom_session2.region_name = "us-west-1"
        custom_session2.client.return_value = mock_client_instance

        manager = MemorySessionManager(
            memory_id="test-memory",
            region_name="us-west-1",  # Same as session region
            boto3_session=custom_session2,
        )

        assert manager.region_name == "us-west-1"

    def test_configure_timestamp_serialization_non_datetime_value(self):
        """Test timestamp serialization with non-datetime value."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Mock the original serializer method
            original_serialize = MagicMock()
            mock_client_instance._serializer._serializer._serialize_type_timestamp = original_serialize

            MemorySessionManager(memory_id="test-memory", region_name="us-west-2")

            # Get the overridden serialization function
            overridden_func = mock_client_instance._serializer._serializer._serialize_type_timestamp

            # Test with non-datetime value (should call original function)
            serialized = {}
            shape = MagicMock()
            overridden_func(serialized, "not-a-datetime", shape, "test_field")

            # Should have called the original function
            original_serialize.assert_called_once_with(serialized, "not-a-datetime", shape, "test_field")

    def test_configure_timestamp_serialization_datetime_value(self):
        """Test timestamp serialization with datetime value."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            # Mock the original serializer method
            original_serialize = MagicMock()
            mock_client_instance._serializer._serializer._serialize_type_timestamp = original_serialize

            MemorySessionManager(memory_id="test-memory", region_name="us-west-2")

            # Get the overridden serialization function
            overridden_func = mock_client_instance._serializer._serializer._serialize_type_timestamp

            # Test with datetime value
            serialized = {}
            test_datetime = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            shape = MagicMock()
            overridden_func(serialized, test_datetime, shape, "test_field")

            # Should have set the timestamp as float
            assert serialized["test_field"] == test_datetime.timestamp()
            # Should NOT have called the original function
            original_serialize.assert_not_called()


class TestAddTurnsWithDataClasses:
    """Test add_turns function with new ConversationalMessage and BlobMessage data classes."""

    @pytest.fixture
    def session_manager(self):
        """Create a MemorySessionManager instance for testing."""
        with patch("boto3.Session"):
            manager = MemorySessionManager("test-memory-id", "us-east-1")
            manager._data_plane_client = Mock()
            return manager

    def test_add_turns_with_conversational_messages(self, session_manager):
        """Test add_turns with ConversationalMessage objects."""
        # Mock the client response
        mock_response = {
            "event": {"eventId": "test-event-id", "eventTimestamp": datetime.now(timezone.utc), "payload": []}
        }
        session_manager._data_plane_client.create_event.return_value = mock_response

        # Create messages using new data classes
        messages = [
            ConversationalMessage("Hello, how are you?", MessageRole.USER),
            ConversationalMessage("I'm doing well, thank you!", MessageRole.ASSISTANT),
        ]

        # Call add_turns
        session_manager.add_turns(actor_id="test-actor", session_id="test-session", messages=messages)

        # Verify the client was called correctly
        session_manager._data_plane_client.create_event.assert_called_once()
        call_args = session_manager._data_plane_client.create_event.call_args[1]

        assert call_args["memoryId"] == "test-memory-id"
        assert call_args["actorId"] == "test-actor"
        assert call_args["sessionId"] == "test-session"

        # Check payload structure
        payload = call_args["payload"]
        assert len(payload) == 2

        # First message
        assert "conversational" in payload[0]
        assert payload[0]["conversational"]["content"]["text"] == "Hello, how are you?"
        assert payload[0]["conversational"]["role"] == MessageRole.USER.value

        # Second message
        assert "conversational" in payload[1]
        assert payload[1]["conversational"]["content"]["text"] == "I'm doing well, thank you!"
        assert payload[1]["conversational"]["role"] == MessageRole.ASSISTANT.value

    def test_add_turns_with_blob_messages(self, session_manager):
        """Test add_turns with BlobMessage objects."""
        # Mock the client response
        mock_response = {
            "event": {"eventId": "test-event-id", "eventTimestamp": datetime.now(timezone.utc), "payload": []}
        }
        session_manager._data_plane_client.create_event.return_value = mock_response

        # Create messages using new data classes
        messages = [BlobMessage({"type": "image", "data": "base64_encoded_image"}), BlobMessage([1, 2, 3, 4, 5])]

        # Call add_turns
        session_manager.add_turns(actor_id="test-actor", session_id="test-session", messages=messages)

        # Verify the client was called correctly
        session_manager._data_plane_client.create_event.assert_called_once()
        call_args = session_manager._data_plane_client.create_event.call_args[1]

        # Check payload structure
        payload = call_args["payload"]
        assert len(payload) == 2

        # First blob message
        assert "blob" in payload[0]
        assert payload[0]["blob"] == {"type": "image", "data": "base64_encoded_image"}

        # Second blob message
        assert "blob" in payload[1]
        assert payload[1]["blob"] == [1, 2, 3, 4, 5]

    def test_add_turns_mixed_data_classes(self, session_manager):
        """Test add_turns with mixed ConversationalMessage and BlobMessage objects."""
        # Mock the client response
        mock_response = {
            "event": {"eventId": "test-event-id", "eventTimestamp": datetime.now(timezone.utc), "payload": []}
        }
        session_manager._data_plane_client.create_event.return_value = mock_response

        # Create mixed messages using new data classes
        messages = [
            ConversationalMessage("What's in this image?", MessageRole.USER),
            BlobMessage({"type": "image", "filename": "photo.jpg", "data": "base64_data"}),
            ConversationalMessage("I can see a beautiful landscape with mountains.", MessageRole.ASSISTANT),
            BlobMessage({"analysis": {"objects": ["mountain", "tree", "sky"], "confidence": 0.95}}),
        ]

        # Call add_turns
        session_manager.add_turns(actor_id="test-actor", session_id="test-session", messages=messages)

        # Verify the client was called correctly
        session_manager._data_plane_client.create_event.assert_called_once()
        call_args = session_manager._data_plane_client.create_event.call_args[1]

        # Check payload structure
        payload = call_args["payload"]
        assert len(payload) == 4

        # First message - conversational
        assert "conversational" in payload[0]
        assert payload[0]["conversational"]["content"]["text"] == "What's in this image?"
        assert payload[0]["conversational"]["role"] == MessageRole.USER.value

        # Second message - blob
        assert "blob" in payload[1]
        assert payload[1]["blob"]["type"] == "image"
        assert payload[1]["blob"]["filename"] == "photo.jpg"

        # Third message - conversational
        assert "conversational" in payload[2]
        assert payload[2]["conversational"]["content"]["text"] == "I can see a beautiful landscape with mountains."
        assert payload[2]["conversational"]["role"] == MessageRole.ASSISTANT.value

        # Fourth message - blob
        assert "blob" in payload[3]
        assert payload[3]["blob"]["analysis"]["objects"] == ["mountain", "tree", "sky"]
        assert payload[3]["blob"]["analysis"]["confidence"] == 0.95

    def test_add_turns_conversational_message_validation_error(self, session_manager):
        """Test that ConversationalMessage validation errors are properly handled."""
        # Test with invalid role type (not MessageRole enum)
        with pytest.raises(ValueError, match="ConversationalMessage.role must be a MessageRole"):
            ConversationalMessage("Hello", "INVALID_ROLE")

    def test_add_turns_empty_data_class_list(self, session_manager):
        """Test add_turns with empty message list using data classes."""
        messages = []

        # Should raise ValueError for empty messages
        with pytest.raises(ValueError, match="At least one message is required"):
            session_manager.add_turns(actor_id="test-actor", session_id="test-session", messages=messages)

    def test_add_turns_data_classes_with_custom_timestamp(self, session_manager):
        """Test add_turns with data classes and custom timestamp."""
        # Mock the client response
        mock_response = {
            "event": {"eventId": "test-event-id", "eventTimestamp": datetime.now(timezone.utc), "payload": []}
        }
        session_manager._data_plane_client.create_event.return_value = mock_response

        # Create messages and custom timestamp
        messages = [ConversationalMessage("Test message", MessageRole.USER)]
        custom_timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Call add_turns with custom timestamp
        session_manager.add_turns(
            actor_id="test-actor", session_id="test-session", messages=messages, event_timestamp=custom_timestamp
        )

        # Verify the custom timestamp was used
        call_args = session_manager._data_plane_client.create_event.call_args[1]
        assert call_args["eventTimestamp"] == custom_timestamp

    def test_add_turns_data_classes_with_branch(self, session_manager):
        """Test add_turns with data classes and branch information."""
        # Mock the client response
        mock_response = {
            "event": {"eventId": "test-event-id", "eventTimestamp": datetime.now(timezone.utc), "payload": []}
        }
        session_manager._data_plane_client.create_event.return_value = mock_response

        # Create messages and branch info
        messages = [ConversationalMessage("Branch message", MessageRole.USER)]
        branch_info = {"rootEventId": "root-123", "name": "test-branch"}

        # Call add_turns with branch
        session_manager.add_turns(
            actor_id="test-actor", session_id="test-session", messages=messages, branch=branch_info
        )

        # Verify the branch info was included
        call_args = session_manager._data_plane_client.create_event.call_args[1]
        assert call_args["branch"] == branch_info

    def test_add_turns_invalid_message_type(self, session_manager):
        """Test add_turns with invalid message type raises ValueError."""
        # Test with invalid message type (not ConversationalMessage or BlobMessage)
        messages = ["invalid_message_type"]

        with pytest.raises(ValueError, match="Invalid message format. Must be ConversationalMessage or BlobMessage"):
            session_manager.add_turns(actor_id="test-actor", session_id="test-session", messages=messages)

    def test_getattr_method_exists_on_client_but_not_allowed(self):
        """Test __getattr__ when method exists on client but not in allowed methods - covers lines 112-116."""
        with patch("boto3.Session") as mock_boto_client:
            mock_client_instance = MagicMock()
            mock_boto_client.return_value = mock_client_instance

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock a method that exists on client but is not in allowed methods
            mock_client_instance.some_control_plane_method = MagicMock()

            # This should trigger the condition where method is not in allowed but exists on client
            with pytest.raises(AttributeError) as exc_info:
                _ = manager.some_control_plane_method

            assert "'MemorySessionManager' object has no attribute 'some_control_plane_method'" in str(exc_info.value)
            assert "Method not found on _data_plane_client" in str(exc_info.value)

    def test_search_long_term_memories_info_logging_on_client_error(self):
        """Test search_long_term_memories logs info on ClientError - covers line 481."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock ClientError
            error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid query"}}
            mock_client_instance.retrieve_memory_records.side_effect = ClientError(
                error_response, "RetrieveMemoryRecords"
            )

            with patch("bedrock_agentcore.memory.session.logger") as mock_logger:
                with pytest.raises(ClientError):
                    manager.search_long_term_memories(query="invalid query", namespace_prefix="test/namespace")

                # Verify info logging was called (not error logging)
                mock_logger.info.assert_called_with("     ❌ Error querying long-term memory", mock.ANY)

    def test_list_long_term_memory_records_multiple_pages(self):
        """Test list_long_term_memory_records with multiple pages - covers line 533."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock paginator with multiple pages
            mock_paginator = MagicMock()
            mock_client_instance.get_paginator.return_value = mock_paginator

            # Multiple pages of results
            mock_paginator.paginate.return_value = [
                {"memoryRecords": [{"memoryRecordId": "rec-1"}, {"memoryRecordId": "rec-2"}]},
                {"memoryRecords": [{"memoryRecordId": "rec-3"}, {"memoryRecordId": "rec-4"}]},
            ]

            result = manager.list_long_term_memory_records(namespace_prefix="test/namespace")

            assert len(result) == 4
            assert all(isinstance(record, MemoryRecord) for record in result)
            # Verify all records from both pages are included
            record_ids = [record["memoryRecordId"] for record in result]
            assert "rec-1" in record_ids
            assert "rec-4" in record_ids

    def test_get_last_k_turns_with_include_parent_branches_true(self):
        """Test get_last_k_turns with include_parent_branches=True - covers line 539->529."""
        with patch("boto3.Session") as mock_boto_client:
            mock_client_instance = MagicMock()
            mock_boto_client.return_value = mock_client_instance

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock list_events
            mock_events = [
                Event(
                    {
                        "eventId": "event-1",
                        "eventTimestamp": datetime(2023, 1, 1, 10, 0, 0),
                        "payload": [{"conversational": {"role": "USER", "content": {"text": "Hello"}}}],
                    }
                )
            ]
            with patch.object(manager, "list_events", return_value=mock_events) as mock_list_events:
                result = manager.get_last_k_turns(
                    actor_id="user-123",
                    session_id="session-456",
                    k=2,
                    branch_name="test-branch",
                    include_parent_branches=True,  # This should trigger include_parent_branches=True
                )

                assert len(result) == 1

                # Verify list_events was called with include_parent_branches=True
                mock_list_events.assert_called_once_with(
                    actor_id="user-123",
                    session_id="session-456",
                    branch_name="test-branch",
                    include_parent_branches=True,  # This is the key parameter
                    max_results=100,
                )

    def test_getattr_debug_logging(self):
        """Test __getattr__ debug logging when method is found."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock an allowed method
            mock_method = MagicMock()
            mock_client_instance.retrieve_memory_records = mock_method

            with patch("bedrock_agentcore.memory.session.logger") as mock_logger:
                result = manager.retrieve_memory_records

                # Verify debug logging was called
                mock_logger.debug.assert_called_once_with(
                    "Forwarding method '%s' to _data_plane_client", "retrieve_memory_records"
                )
                assert result == mock_method

    def test_process_turn_with_llm_no_retrieval_namespace(self):
        """Test process_turn_with_llm without retrieval_config (no memory retrieval)."""
        with patch("boto3.Session") as mock_boto_client:
            mock_client_instance = MagicMock()
            mock_boto_client.return_value = mock_client_instance

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock add_turns
            mock_event = {"eventId": "event-123"}
            with patch.object(manager, "add_turns", return_value=Event(mock_event)):
                # Mock search_long_term_memories to ensure it's not called
                with patch.object(manager, "search_long_term_memories") as mock_search:

                    def mock_llm_callback(user_input: str, memories: List[Dict[str, Any]]) -> str:
                        return "Response"

                    # Test without retrieval_config (should not call search)
                    memories, response, event = manager.process_turn_with_llm(
                        actor_id="user-123",
                        session_id="session-456",
                        user_input="Hello",
                        llm_callback=mock_llm_callback,
                        retrieval_config=None,  # No retrieval
                    )

                    # Verify search was not called
                    mock_search.assert_not_called()
                    assert len(memories) == 0
                    assert response == "Response"

    def test_add_turns_default_timestamp(self):
        """Test add_turns uses default timestamp when none provided."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock create_event response
            mock_response = {"event": {"eventId": "turn-event-123", "memoryId": "testMemory-1234567890"}}
            mock_client_instance.create_event.return_value = mock_response

            messages = [ConversationalMessage("Hello", MessageRole.USER)]

            # Mock datetime.now to control the timestamp
            with patch("bedrock_agentcore.memory.session.datetime") as mock_datetime:
                mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
                mock_datetime.now.return_value = mock_now
                mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

                result = manager.add_turns(
                    actor_id="user-123",
                    session_id="session-456",
                    messages=messages,
                    # No event_timestamp provided - should use default
                )

                assert isinstance(result, Event)
                assert result["eventId"] == "turn-event-123"

                # Verify default timestamp was used
                call_args = mock_client_instance.create_event.call_args[1]
                assert call_args["eventTimestamp"] == mock_now

    def test_list_events_no_branch_filter(self):
        """Test list_events without branch filtering."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock response
            mock_events = [{"eventId": "event-1", "eventTimestamp": datetime.now()}]
            mock_client_instance.list_events.return_value = {"events": mock_events, "nextToken": None}

            result = manager.list_events(actor_id="user-123", session_id="session-456", branch_name=None)

            assert len(result) == 1

            # Verify no filter was applied
            call_args = mock_client_instance.list_events.call_args[1]
            assert "filter" not in call_args

    def test_list_branches_no_pagination(self):
        """Test list_branches without pagination (single response)."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock single response (no pagination)
            mock_events = [
                {"eventId": "event-1", "eventTimestamp": datetime(2023, 1, 1, 10, 0, 0)},
                {
                    "eventId": "event-2",
                    "eventTimestamp": datetime(2023, 1, 1, 10, 5, 0),
                    "branch": {"name": "branch-1", "rootEventId": "event-1"},
                },
            ]
            mock_client_instance.list_events.return_value = {"events": mock_events}  # No nextToken

            result = manager.list_branches(actor_id="user-123", session_id="session-456")

            assert len(result) == 2  # main + branch-1
            assert mock_client_instance.list_events.call_count == 1  # Only one call

    def test_list_branches_with_pagination(self):
        """Test list_branches with pagination."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client_instance = MagicMock()
            mock_session.client.return_value = mock_client_instance
            mock_session_class.return_value = mock_session

            manager = MemorySessionManager(memory_id="testMemory-1234567890", region_name="us-west-2")

            # Mock paginated responses
            first_batch = [{"eventId": "event-1", "eventTimestamp": datetime(2023, 1, 1, 10, 0, 0)}]
            second_batch = [
                {
                    "eventId": "event-2",
                    "eventTimestamp": datetime(2023, 1, 1, 10, 5, 0),
                    "branch": {"name": "branch-1", "rootEventId": "event-1"},
                }
            ]

            mock_client_instance.list_events.side_effect = [
                {"events": first_batch, "nextToken": "token-123"},
                {"events": second_batch, "nextToken": None},
            ]

            result = manager.list_branches(actor_id="user-123", session_id="session-456")

            assert len(result) == 2  # main + branch-1
            assert mock_client_instance.list_events.call_count == 2

            # Verify nextToken was passed in second call
            second_call_args = mock_client_instance.list_events.call_args_list[1][1]
            assert second_call_args["nextToken"] == "token-123"
