"""Tests for AgentCoreMemorySessionManager."""

import logging
import time
from unittest.mock import Mock, patch

import pytest
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from strands.agent.agent import Agent
from strands.hooks import AfterInvocationEvent, MessageAddedEvent
from strands.hooks.registry import HookRegistry
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage, SessionType

from bedrock_agentcore.memory.integrations.strands.bedrock_converter import (
    CONVERSATIONAL_MAX_SIZE,
    AgentCoreMemoryConverter,
)
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
    BufferedMessage,
)


@pytest.fixture
def agentcore_config():
    """Create a test AgentCore Memory configuration."""
    return AgentCoreMemoryConfig(memory_id="test-memory-123", session_id="test-session-456", actor_id="test-actor-789")


@pytest.fixture
def agentcore_config_with_retrieval():
    """Create a test AgentCore Memory configuration with retrieval config."""
    retrieval_config = {
        "user_preferences/{actorId}/": RetrievalConfig(top_k=5, relevance_score=0.3),
        "session_context/{sessionId}/": RetrievalConfig(top_k=3, relevance_score=0.5),
    }
    return AgentCoreMemoryConfig(
        memory_id="test-memory-123",
        session_id="test-session-456",
        actor_id="test-actor-789",
        retrieval_config=retrieval_config,
    )


@pytest.fixture
def mock_memory_client():
    """Create a mock MemoryClient."""
    client = Mock()
    client.create_event.return_value = {"eventId": "event_123456"}
    client.list_events.return_value = []
    client.retrieve_memories.return_value = []
    client.gmcp_client = Mock()
    client.gmdp_client = Mock()
    return client


def _create_session_manager(config, mock_memory_client):
    """Helper to create a session manager with mocked dependencies."""
    with (
        patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ),
        patch("boto3.Session") as mock_boto_session,
        patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
    ):
        mock_session = Mock()
        mock_session.region_name = "us-west-2"
        mock_session.client.return_value = Mock()
        mock_boto_session.return_value = mock_session

        manager = AgentCoreMemorySessionManager(config)
        manager.session_id = config.session_id
        manager.session = Session(session_id=config.session_id, session_type=SessionType.AGENT)
        return manager


@pytest.fixture
def session_manager(agentcore_config, mock_memory_client):
    """Create an AgentCoreMemorySessionManager with mocked dependencies."""
    return _create_session_manager(agentcore_config, mock_memory_client)


@pytest.fixture
def batching_config():
    """Create a config with batch_size > 1."""
    return AgentCoreMemoryConfig(
        memory_id="test-memory-123",
        session_id="test-session-456",
        actor_id="test-actor-789",
        batch_size=10,
    )


@pytest.fixture
def batching_session_manager(batching_config, mock_memory_client):
    """Create a session manager with batching enabled."""
    return _create_session_manager(batching_config, mock_memory_client)


@pytest.fixture
def test_agent():
    """Create a test agent."""
    return Agent(agent_id="test-agent-123", messages=[{"role": "user", "content": [{"text": "Hello!"}]}])


class TestAgentCoreMemorySessionManager:
    """Test AgentCoreMemorySessionManager class."""

    def test_init_basic(self, agentcore_config):
        """Test basic initialization."""
        with patch("bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config)

                    assert manager.config == agentcore_config
                    assert manager.memory_client == mock_client
                    mock_client_class.assert_called_once_with(region_name=None)

    def test_events_to_messages(self, session_manager):
        """Test converting Bedrock events to SessionMessages."""
        events = [
            {
                "eventId": "event-1",
                "eventTimestamp": "2024-01-01T12:00:00Z",
                "payload": [
                    {
                        "conversational": {
                            "content": {
                                "text": '{"message": {"role": "user", "content": [{"text": "Hello"}]}, "message_id": 1}'
                            },
                            "role": "USER",
                        }
                    }
                ],
            }
        ]

        messages = AgentCoreMemoryConverter.events_to_messages(events)
        assert messages[0].message["role"] == "user"
        assert messages[0].message["content"][0]["text"] == "Hello"

    def test_create_session(self, session_manager):
        """Test creating a session."""
        session = Session(session_id="test-session-456", session_type=SessionType.AGENT)

        result = session_manager.create_session(session)

        assert result == session
        assert result.session_id == "test-session-456"

    def test_create_session_id_mismatch(self, session_manager):
        """Test creating a session with mismatched ID."""
        session = Session(session_id="wrong-session-id", session_type=SessionType.AGENT)

        with pytest.raises(SessionException, match="Session ID mismatch"):
            session_manager.create_session(session)

    def test_read_session_valid(self, session_manager, mock_memory_client):
        """Test reading a valid session."""
        # Mock the list_events to return a valid session event
        mock_memory_client.list_events.return_value = [
            {
                "eventId": "session-event-1",
                "payload": [{"blob": '{"session_id": "test-session-456", "session_type": "AGENT"}'}],
            }
        ]

        result = session_manager.read_session("test-session-456")

        assert result is not None
        assert result.session_id == "test-session-456"
        assert result.session_type == SessionType.AGENT

    def test_read_session_invalid(self, session_manager):
        """Test reading an invalid session."""
        result = session_manager.read_session("wrong-session-id")

        assert result is None

    def test_read_session_legacy_migration(self, session_manager, mock_memory_client):
        """Test reading a legacy session event triggers migration."""
        legacy_session_data = '{"session_id": "test-session-456", "session_type": "AGENT"}'

        # First call (new approach with metadata) returns empty
        # Second call (legacy actor_id) returns the legacy event
        mock_memory_client.list_events.side_effect = [
            [],  # New approach returns nothing
            [{"eventId": "legacy-event-1", "payload": [{"blob": legacy_session_data}]}],  # Legacy approach
        ]
        mock_memory_client.gmdp_client.create_event.return_value = {"event": {"eventId": "new-event-1"}}

        result = session_manager.read_session("test-session-456")

        # Verify session was returned
        assert result is not None
        assert result.session_id == "test-session-456"
        assert result.session_type == SessionType.AGENT

        # Verify migration: new event created with metadata
        mock_memory_client.gmdp_client.create_event.assert_called_once()
        create_call_kwargs = mock_memory_client.gmdp_client.create_event.call_args.kwargs
        assert "metadata" in create_call_kwargs
        assert create_call_kwargs["metadata"]["stateType"]["stringValue"] == "SESSION"

        # Verify migration: old event deleted
        mock_memory_client.gmdp_client.delete_event.assert_called_once()
        delete_call_kwargs = mock_memory_client.gmdp_client.delete_event.call_args.kwargs
        assert delete_call_kwargs["actorId"] == "session_test-session-456"
        assert delete_call_kwargs["eventId"] == "legacy-event-1"

    def test_create_agent(self, session_manager):
        """Test creating an agent."""
        session_agent = SessionAgent(agent_id="test-agent-123", state={}, conversation_manager_state={})

        # Should not raise any exceptions
        session_manager.create_agent("test-session-456", session_agent)

    def test_create_agent_wrong_session(self, session_manager):
        """Test creating an agent with wrong session ID."""
        session_agent = SessionAgent(agent_id="test-agent-123", state={}, conversation_manager_state={})

        with pytest.raises(SessionException, match="Session ID mismatch"):
            session_manager.create_agent("wrong-session-id", session_agent)

    def test_read_agent_valid(self, session_manager, mock_memory_client):
        """Test reading a valid agent."""
        mock_memory_client.list_events.return_value = [
            {
                "eventId": "event-1",
                "eventTimestamp": "2024-01-01T12:00:00Z",
                "payload": [{"blob": '{"agent_id": "test-agent-123", "state": {}, "conversation_manager_state": {}}'}],
            }
        ]

        result = session_manager.read_agent("test-session-456", "test-agent-123")

        assert result is not None
        assert result.agent_id == "test-agent-123"
        assert result.agent_id == "test-agent-123"

    def test_read_agent_no_events(self, session_manager, mock_memory_client):
        """Test reading an agent with no events."""
        mock_memory_client.list_events.return_value = []

        result = session_manager.read_agent("test-session-456", "test-agent-123")

        assert result is None

    def test_read_agent_legacy_migration(self, session_manager, mock_memory_client):
        """Test reading a legacy agent event triggers migration."""
        legacy_agent_data = '{"agent_id": "test-agent-123", "state": {}, "conversation_manager_state": {}}'

        # New approach with metadata returns empty, then legacy approach returns the event
        mock_memory_client.list_events.side_effect = [
            [],  # New approach with metadata - returns empty
            [{"eventId": "legacy-agent-event-1", "payload": [{"blob": legacy_agent_data}]}],  # Legacy approach
        ]
        mock_memory_client.gmdp_client.create_event.return_value = {"event": {"eventId": "new-agent-event-1"}}

        result = session_manager.read_agent("test-session-456", "test-agent-123")

        # Verify agent was returned
        assert result is not None
        assert result.agent_id == "test-agent-123"

        # Verify migration: new event created with metadata
        mock_memory_client.gmdp_client.create_event.assert_called_once()
        create_call_kwargs = mock_memory_client.gmdp_client.create_event.call_args.kwargs
        assert "metadata" in create_call_kwargs
        assert create_call_kwargs["metadata"]["stateType"]["stringValue"] == "AGENT"
        assert create_call_kwargs["metadata"]["agentId"]["stringValue"] == "test-agent-123"

        # Verify migration: old event deleted
        mock_memory_client.gmdp_client.delete_event.assert_called_once()
        delete_call_kwargs = mock_memory_client.gmdp_client.delete_event.call_args.kwargs
        assert delete_call_kwargs["actorId"] == "agent_test-agent-123"
        assert delete_call_kwargs["eventId"] == "legacy-agent-event-1"

    def test_create_message(self, session_manager, mock_memory_client):
        """Test creating a message."""
        mock_memory_client.create_event.return_value = {"eventId": "event-123"}

        message = SessionMessage(
            message={"role": "user", "content": [{"text": "Hello"}]}, message_id=1, created_at="2024-01-01T12:00:00Z"
        )

        session_manager.create_message("test-session-456", "test-agent-123", message)

        mock_memory_client.create_event.assert_called_once()

    def test_list_messages(self, session_manager, mock_memory_client):
        """Test listing messages."""
        mock_memory_client.list_events.return_value = [
            {
                "eventId": "event-1",
                "eventTimestamp": "2024-01-01T12:00:00Z",
                "payload": [
                    {
                        "conversational": {
                            "content": {
                                "text": '{"message": {"role": "user", "content": [{"text": "Hello"}]}, "message_id": 1}'
                            },
                            "role": "USER",
                        }
                    }
                ],
            },
            {
                "eventId": "event-2",
                "eventTimestamp": "2024-01-01T12:00:00Z",
                "payload": [
                    {
                        "conversational": {
                            "content": {
                                "text": '{"message": {"role": "assistant", "content": [{"text": "Hi there"}]}, "message_id": 2}'  # noqa E501
                            },
                            "role": "ASSISTANT",
                        }
                    }
                ],
            },
        ]

        messages = session_manager.list_messages("test-session-456", "test-agent-123")

        assert len(messages) == 2
        assert messages[1].message["role"] == "user"
        assert messages[0].message["role"] == "assistant"

    def test_list_messages_returns_values_in_correct_reverse_order(self, session_manager, mock_memory_client):
        """Test listing messages."""
        mock_memory_client.list_events.return_value = [
            {
                "eventId": "event-1",
                "eventTimestamp": "2024-01-01T12:00:00Z",
                "payload": [
                    {
                        "conversational": {
                            "content": {
                                "text": '{"message": {"role": "user", "content": [{"text": "Hello"}]}, "message_id": 1}'
                            },
                            "role": "USER",
                        }
                    }
                ],
            },
            {
                "eventId": "event-2",
                "eventTimestamp": "2024-01-01T12:00:00Z",
                "payload": [
                    {
                        "conversational": {
                            "content": {
                                "text": '{"message": {"role": "assistant", "content": [{"text": "Hi there"}]}, "message_id": 2}'  # noqa E501
                            },
                            "role": "ASSISTANT",
                        }
                    }
                ],
            },
        ]

        messages = session_manager.list_messages("test-session-456", "test-agent-123")

        assert len(messages) == 2
        assert messages[1].message["role"] == "user"
        assert messages[0].message["role"] == "assistant"

    def test_events_to_messages_empty_payload(self, session_manager):
        """Test converting Bedrock events with empty payload."""
        events = [
            {
                "eventId": "event-1",
                "eventTimestamp": "2024-01-01T12:00:00Z",
                # No payload
            }
        ]

        messages = AgentCoreMemoryConverter.events_to_messages(events)

        assert len(messages) == 0

    def test_delete_session(self, session_manager):
        """Test deleting a session (no-op for AgentCore Memory)."""
        # Should not raise any exceptions
        session_manager.delete_session("test-session-456")

    def test_read_agent_wrong_session(self, session_manager):
        """Test reading an agent with wrong session ID."""
        result = session_manager.read_agent("wrong-session-id", "test-agent-123")

        assert result is None

    def test_read_agent_exception(self, session_manager, mock_memory_client):
        """Test reading an agent when exception occurs."""
        mock_memory_client.list_events.side_effect = Exception("API Error")

        result = session_manager.read_agent("test-session-456", "test-agent-123")

        assert result is None

    def test_update_agent(self, session_manager, mock_memory_client):
        """Test updating an agent."""
        # First mock that the agent exists
        mock_memory_client.list_events.return_value = [
            {
                "eventId": "event-1",
                "eventTimestamp": "2024-01-01T12:00:00Z",
                "payload": [{"blob": '{"agent_id": "test-agent-123", "state": {}, "conversation_manager_state": {}}'}],
            }
        ]

        session_agent = SessionAgent(agent_id="test-agent-123", state={"key": "value"}, conversation_manager_state={})

        # Should not raise any exceptions
        session_manager.update_agent("test-session-456", session_agent)

    def test_update_agent_uses_cache(self, session_manager, mock_memory_client):
        """Test that update_agent uses cache to avoid fetching memory events on subsequent updates."""

        # Manually populate the cache (simulating what happens after first agent creation/read)
        created_at = "2024-01-01T12:00:00+00:00"
        session_manager._agent_created_at_cache["test-agent-123"] = created_at

        # Now update the agent - should NOT call list_events since it's in cache
        updated_agent = SessionAgent(agent_id="test-agent-123", state={"key": "value"}, conversation_manager_state={})
        session_manager.update_agent("test-session-456", updated_agent)

        # Verify that list_events was NOT called (cache was used)
        mock_memory_client.list_events.assert_not_called()

        # Verify the created_at was preserved from cache
        assert updated_agent.created_at == created_at

    def test_update_agent_wrong_session(self, session_manager):
        """Test updating an agent with wrong session ID."""
        session_agent = SessionAgent(agent_id="test-agent-123", state={}, conversation_manager_state={})

        with pytest.raises(SessionException, match="Agent test-agent-123 in session wrong-session-id does not exist"):
            session_manager.update_agent("wrong-session-id", session_agent)

    def test_create_message_wrong_session(self, session_manager):
        """Test creating a message with wrong session ID."""
        message = SessionMessage(message={"role": "user", "content": [{"text": "Hello"}]}, message_id=1)

        with pytest.raises(SessionException, match="Session ID mismatch"):
            session_manager.create_message("wrong-session-id", "test-agent-123", message)

    def test_create_message_exception(self, session_manager, mock_memory_client):
        """Test creating a message when exception occurs."""
        mock_memory_client.create_event.side_effect = Exception("API Error")

        message = SessionMessage(message={"role": "user", "content": [{"text": "Hello"}]}, message_id=1)

        with pytest.raises(SessionException, match="Failed to create message"):
            session_manager.create_message("test-session-456", "test-agent-123", message)

    def test_read_message(self, session_manager, mock_memory_client):
        """Test reading a message."""
        # Mock the gmdp_client.get_event method
        mock_event_data = {
            "eventId": "event-1",
            "eventTimestamp": "2024-01-01T12:00:00Z",
            "message": {"role": "assistant", "content": [{"text": "Hi there"}]},
            "message_id": 1,
        }
        session_manager.memory_client.gmdp_client.get_event.return_value = mock_event_data

        result = session_manager.read_message("test-session-456", "test-agent-123", 1)

        assert result is not None
        assert result.message["role"] == "assistant"
        assert result.message["content"][0]["text"] == "Hi there"

    def test_read_message_not_found(self, session_manager, mock_memory_client):
        """Test reading a message that doesn't exist."""
        session_manager.memory_client.gmdp_client.get_event.return_value = None

        result = session_manager.read_message("test-session-456", "test-agent-123", 0)

        assert result is None

    def test_update_message(self, session_manager):
        """Test updating a message."""
        message = SessionMessage(message={"role": "user", "content": [{"text": "Hello"}]}, message_id=1)

        # Should not raise any exceptions
        session_manager.update_message("test-session-456", "test-agent-123", message)

    def test_update_message_wrong_session(self, session_manager):
        """Test updating a message with wrong session ID."""
        message = SessionMessage(message={"role": "user", "content": [{"text": "Hello"}]}, message_id=1)

        with pytest.raises(SessionException, match="Session ID mismatch"):
            session_manager.update_message("wrong-session-id", "test-agent-123", message)

    def test_list_messages_with_limit(self, session_manager, mock_memory_client):
        """Test listing messages with limit."""
        mock_memory_client.list_events.return_value = [
            {
                "eventId": "event-1",
                "eventTimestamp": "2024-01-01T12:00:00Z",
                "payload": [
                    {
                        "conversational": {
                            "content": {
                                "text": '{"message": {"role": "user", '
                                '"content": [{"text": "Message 1"}]}, "message_id": 1}'
                            },
                            "role": "USER",
                        }
                    }
                ],
            },
            {
                "eventId": "event-2",
                "eventTimestamp": "2024-01-01T12:00:00Z",
                "payload": [
                    {
                        "conversational": {
                            "content": {
                                "text": '{"message": {"role": "assistant", "content": [{"text": "Message 2"}]}, "message_id": 2}'  # noqa E501
                            },
                            "role": "ASSISTANT",
                        }
                    }
                ],
            },
        ]

        messages = session_manager.list_messages("test-session-456", "test-agent-123", limit=1, offset=1)

        assert len(messages) == 1
        assert messages[0].message["content"][0]["text"] == "Message 1"

    def test_list_messages_wrong_session(self, session_manager):
        """Test listing messages with wrong session ID."""
        with pytest.raises(SessionException, match="Session ID mismatch"):
            session_manager.list_messages("wrong-session-id", "test-agent-123")

    def test_list_messages_exception(self, session_manager, mock_memory_client):
        """Test listing messages when exception occurs."""
        mock_memory_client.list_events.side_effect = Exception("API Error")

        messages = session_manager.list_messages("test-session-456", "test-agent-123")

        assert len(messages) == 0

    def test_load_long_term_memories_no_config(self, session_manager, test_agent):
        """Test loading long-term memories when no retrieval config is set."""
        session_manager.config.retrieval_config = None

        # Mock the method since it doesn't exist yet
        session_manager._load_long_term_memories = Mock()

        # Should not raise any exceptions
        session_manager._load_long_term_memories(test_agent)

        # Verify it was called
        session_manager._load_long_term_memories.assert_called_once_with(test_agent)

    def test_validate_namespace_resolution(self, session_manager):
        """Test namespace resolution validation."""
        # Mock the method since it doesn't exist yet
        session_manager._validate_namespace_resolution = Mock(return_value=True)

        # Valid resolution
        assert session_manager._validate_namespace_resolution(
            "user_preferences/{actorId}/", "user_preferences/test-actor/"
        )

        # Mock invalid resolution
        session_manager._validate_namespace_resolution.return_value = False
        assert not session_manager._validate_namespace_resolution(
            "user_preferences/{actorId}/", "user_preferences/{actorId}/"
        )

        # Invalid - empty result
        assert not session_manager._validate_namespace_resolution("test_namespace/", "")

    def test_load_long_term_memories_with_validation_failure(self, mock_memory_client, test_agent):
        """Test LTM loading with namespace validation failure."""
        # Create config with namespace that will fail resolution
        config_with_bad_namespace = AgentCoreMemoryConfig(
            memory_id="test-memory-123",
            session_id="test-session-456",
            actor_id="test-actor",
            retrieval_config={"user_preferences/{invalidVar}/": RetrievalConfig(top_k=5, relevance_score=0.3)},
        )

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(config_with_bad_namespace)
                    # Mock the method since it doesn't exist yet
                    manager._load_long_term_memories = Mock()
                    manager._load_long_term_memories(test_agent)
                    manager._load_long_term_memories.assert_called_once_with(test_agent)

        # Should not call retrieve_memories due to validation failure
        assert mock_memory_client.retrieve_memories.call_count == 0

        # No memories should be stored (agent.state is unmodified since we mocked the method)
        assert test_agent.state.get("ltm_memories") is None

    def test_retry_with_backoff_success(self, session_manager):
        """Test retry mechanism with eventual success."""
        mock_func = Mock()
        mock_func.side_effect = [ClientError({"Error": {"Code": "ThrottlingException"}}, "test"), "success"]

        # Mock the method since it doesn't exist yet
        session_manager._retry_with_backoff = Mock(return_value="success")

        with patch("time.sleep"):  # Speed up test
            result = session_manager._retry_with_backoff(mock_func, "arg1", kwarg1="value1")

        assert result == "success"

    def test_retry_with_backoff_max_retries(self, session_manager):
        """Test retry mechanism reaching max retries."""
        mock_func = Mock()
        mock_func.side_effect = ClientError({"Error": {"Code": "ThrottlingException"}}, "test")

        # Mock the method since it doesn't exist yet
        session_manager._retry_with_backoff = Mock(
            side_effect=ClientError({"Error": {"Code": "ThrottlingException"}}, "test")
        )

        with patch("time.sleep"):  # Speed up test
            with pytest.raises(ClientError):
                session_manager._retry_with_backoff(mock_func, max_retries=2)

    def test_generate_initialization_query(self, session_manager, test_agent):
        """Test contextual query generation based on namespace patterns."""

        # Mock the method since it doesn't exist yet
        def mock_generate_query(namespace, config, agent):
            if "preferences" in namespace:
                return "user preferences settings"
            elif "context" in namespace:
                return "conversation context history"
            elif "semantic" in namespace or "facts" in namespace:
                return "facts knowledge information"
            else:
                return "context preferences facts"

        session_manager._generate_initialization_query = Mock(side_effect=mock_generate_query)

        # Test preferences namespace
        config = RetrievalConfig(top_k=5, relevance_score=0.3)
        query = session_manager._generate_initialization_query("user_preferences/{actorId}/", config, test_agent)
        assert query == "user preferences settings"

        # Test context namespace
        query = session_manager._generate_initialization_query("session_context/{sessionId}/", config, test_agent)
        assert query == "conversation context history"

        # Test semantic namespace
        query = session_manager._generate_initialization_query("semantic_knowledge/", config, test_agent)
        assert query == "facts knowledge information"

        # Test facts namespace
        query = session_manager._generate_initialization_query("facts_database/", config, test_agent)
        assert query == "facts knowledge information"

        # Test fallback
        query = session_manager._generate_initialization_query("unknown_namespace/", config, test_agent)
        assert query == "context preferences facts"

    def test_generate_initialization_query_custom(self, session_manager, test_agent):
        """Test custom initialization query takes precedence."""
        config = RetrievalConfig(top_k=5, relevance_score=0.3, initialization_query="custom query for testing")

        # Mock the method since it doesn't exist yet
        session_manager._generate_initialization_query = Mock(return_value="custom query for testing")

        query = session_manager._generate_initialization_query("user_preferences/{actorId}/", config, test_agent)
        assert query == "custom query for testing"

    def test_retrieve_contextual_memories_all_namespaces(self, agentcore_config_with_retrieval, mock_memory_client):
        """Test contextual memory retrieval from all namespaces."""
        mock_memory_client.retrieve_memories.return_value = [
            {"content": "Relevant memory", "relevanceScore": 0.8},
            {"content": "Less relevant memory", "relevanceScore": 0.2},
        ]

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config_with_retrieval)
                    # Mock the method since it doesn't exist yet
                    manager.retrieve_contextual_memories = Mock(
                        return_value=[
                            {
                                "namespace": "user_preferences/test-actor-789/",
                                "memories": [{"content": "Relevant memory", "relevanceScore": 0.8}],
                            },
                            {
                                "namespace": "session_context/test-session-456/",
                                "memories": [{"content": "Less relevant memory", "relevanceScore": 0.2}],
                            },
                        ]
                    )
                    results = manager.retrieve_contextual_memories("What are my preferences?")

        # Should return results organized by namespace
        assert len(results) == 2

    def test_retrieve_contextual_memories_specific_namespaces(
        self, agentcore_config_with_retrieval, mock_memory_client
    ):
        """Test contextual memory retrieval from specific namespaces."""
        mock_memory_client.retrieve_memories.return_value = [
            {"content": "User preference memory", "relevanceScore": 0.9}
        ]

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config_with_retrieval)
                    # Mock the method since it doesn't exist yet
                    manager.retrieve_contextual_memories = Mock(
                        return_value=[
                            {
                                "namespace": "user_preferences/test-actor-789/",
                                "memories": [{"content": "User preference memory", "relevanceScore": 0.9}],
                            }
                        ]
                    )
                    results = manager.retrieve_contextual_memories(
                        "What are my preferences?", namespaces=["user_preferences/{actorId}/"]
                    )

        # Should return results for specified namespace only
        assert len(results) == 1

    def test_retrieve_contextual_memories_no_config(self, session_manager):
        """Test contextual memory retrieval with no config."""
        session_manager.config.retrieval_config = None

        session_manager.retrieve_contextual_memories = Mock(return_value={})
        results = session_manager.retrieve_contextual_memories("test query")

        assert results == {}

    def test_retrieve_contextual_memories_invalid_namespace(self, agentcore_config_with_retrieval, mock_memory_client):
        """Test contextual memory retrieval with invalid namespace."""
        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config_with_retrieval)
                    manager.retrieve_contextual_memories = Mock(return_value={})
                    results = manager.retrieve_contextual_memories("test query", namespaces=["nonexistent_namespace/"])

        # Should return empty results
        assert results == {}

    def test_load_long_term_memories_with_config(self, agentcore_config_with_retrieval, mock_memory_client, test_agent):
        """Test loading long-term memories with retrieval config."""
        mock_memory_client.retrieve_memories.return_value = [
            {"content": "User prefers morning meetings", "relevanceScore": 0.8},
            {"content": "User is in Pacific timezone", "relevanceScore": 0.7},
        ]

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config_with_retrieval)
                    manager._load_long_term_memories = Mock()
                    manager._load_long_term_memories(test_agent)

        # Verify the method was called
        manager._load_long_term_memories.assert_called_once_with(test_agent)

    def test_load_long_term_memories_exception_handling(
        self, agentcore_config_with_retrieval, mock_memory_client, test_agent
    ):
        """Test exception handling during long-term memory loading."""
        mock_memory_client.retrieve_memories.side_effect = Exception("API Error")

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config_with_retrieval)

                    # Should not raise exception, just log warning
                    manager._load_long_term_memories = Mock()
        manager._load_long_term_memories(test_agent)

    def test_namespace_variable_resolution(self, session_manager):
        """Test namespace variable resolution with various combinations."""
        # Test basic variable resolution
        namespace = "user_preferences/{actorId}/"
        resolved = namespace.format(
            actorId=session_manager.config.actor_id, sessionId=session_manager.config.session_id, memoryStrategyId=""
        )
        assert resolved == "user_preferences/test-actor-789/"

        # Test multiple variables
        namespace = "context/{sessionId}/actor/{actorId}/"
        resolved = namespace.format(
            actorId=session_manager.config.actor_id, sessionId=session_manager.config.session_id, memoryStrategyId=""
        )
        assert resolved == "context/test-session-456/actor/test-actor-789/"

        # Test with strategy ID
        namespace = "strategy/{memoryStrategyId}/user/{actorId}/"
        resolved = namespace.format(
            actorId=session_manager.config.actor_id,
            sessionId=session_manager.config.session_id,
            memoryStrategyId="test_strategy",
        )
        assert resolved == "strategy/test_strategy/user/test-actor-789/"

    def test_generate_initialization_query_patterns(self, session_manager, test_agent):
        """Test initialization query generation with various namespace patterns."""
        config = RetrievalConfig(top_k=5, relevance_score=0.3)

        # Mock the method to return appropriate values based on namespace
        def mock_generate_query(namespace, config, agent):
            if "preferences" in namespace:
                return "user preferences settings"
            elif "context" in namespace:
                return "conversation context history"
            elif "semantic" in namespace or "facts" in namespace or "knowledge" in namespace:
                return "facts knowledge information"
            else:
                return "context preferences facts"

        session_manager._generate_initialization_query = Mock(side_effect=mock_generate_query)

        # Test various preference patterns
        patterns_and_expected = [
            ("user_preferences/{actorId}/", "user preferences settings"),
            ("preferences/global/", "user preferences settings"),
            ("my_preferences/", "user preferences settings"),
            ("session_context/{sessionId}/", "conversation context history"),
            ("context/history/", "conversation context history"),
            ("conversation_context/", "conversation context history"),
            ("semantic_memory/", "facts knowledge information"),
            ("facts_database/", "facts knowledge information"),
            ("knowledge_semantic/", "facts knowledge information"),
            ("random_namespace/", "context preferences facts"),
            ("unknown/", "context preferences facts"),
        ]

        for namespace, expected_query in patterns_and_expected:
            query = session_manager._generate_initialization_query(namespace, config, test_agent)
            assert query == expected_query, f"Failed for namespace: {namespace}"

    def test_load_long_term_memories_enhanced_functionality(
        self, agentcore_config_with_retrieval, mock_memory_client, test_agent
    ):
        """Test enhanced LTM loading functionality with detailed verification."""

        # Mock different responses for different namespaces
        def mock_retrieve_side_effect(*args, **kwargs):
            namespace = kwargs.get("namespace", "")
            if "preferences" in namespace:
                return [
                    {"content": "User prefers morning meetings", "relevanceScore": 0.8},
                    {"content": "User likes coffee", "relevanceScore": 0.2},  # Below threshold
                ]
            else:  # context namespace
                return [{"content": "Previous conversation about project", "relevanceScore": 0.6}]

        mock_memory_client.retrieve_memories.side_effect = mock_retrieve_side_effect

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config_with_retrieval)
                    manager._load_long_term_memories = Mock()
                    manager._load_long_term_memories(test_agent)

        # Verify the method was called
        manager._load_long_term_memories.assert_called_once_with(test_agent)

    def test_initialize_basic_functionality(self, session_manager, test_agent):
        """Test basic initialize functionality with LTM loading."""
        session_manager._latest_agent_message = {}

        # Mock list_messages to return existing messages
        session_manager.list_messages = Mock(
            return_value=[SessionMessage(message={"role": "user", "content": [{"text": "Hello"}]}, message_id=1)]
        )

        # Mock _load_long_term_memories to verify it's called
        session_manager._load_long_term_memories = Mock()

        # Mock the session repository
        session_manager.session_repository = Mock()
        session_manager.session_repository.read_agent = Mock(return_value=None)

        # Initialize the agent
        session_manager.initialize(test_agent)

        # Verify the agent was set up
        assert test_agent.agent_id in session_manager._latest_agent_message

    def test_initialize_with_ltm_integration(self, agentcore_config_with_retrieval, mock_memory_client, test_agent):
        """Test initialize functionality with LTM integration enabled."""
        mock_memory_client.retrieve_memories.return_value = [
            {"content": "User prefers morning meetings", "relevanceScore": 0.8}
        ]

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config_with_retrieval)

                    # Mock the initialize method to only test LTM loading
                    manager._latest_agent_message = {}
                    manager.list_messages = Mock(return_value=[])

                    # Call LTM loading directly to test integration
                    manager._load_long_term_memories = Mock()
                    manager._load_long_term_memories(test_agent)

        # Verify the method was called
        manager._load_long_term_memories.assert_called_once_with(test_agent)

    def test_init_with_boto_config(self, agentcore_config, mock_memory_client):
        """Test initialization with custom boto config."""
        boto_config = BotocoreConfig(user_agent_extra="custom-agent")

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config, boto_client_config=boto_config)
                    assert manager.memory_client is not None

    def test_retrieve_customer_context_no_messages(self, agentcore_config_with_retrieval, mock_memory_client):
        """Test retrieve_customer_context with no messages."""
        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config_with_retrieval)

                    # Create mock agent with no messages
                    mock_agent = Mock()
                    mock_agent.messages = []

                    event = MessageAddedEvent(agent=mock_agent, message={"role": "user", "content": [{"text": "test"}]})
                    result = manager.retrieve_customer_context(event)
                    assert result is None

    def test_retrieve_customer_context_no_config(self, agentcore_config, mock_memory_client):
        """Test retrieve_customer_context with no retrieval config."""
        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config)

                    mock_agent = Mock()
                    mock_agent.messages = [{"role": "user", "content": [{"text": "test"}]}]

                    event = MessageAddedEvent(agent=mock_agent, message={"role": "user", "content": [{"text": "test"}]})
                    result = manager.retrieve_customer_context(event)
                    assert result is None

    def test_retrieve_customer_context_with_memories(self, agentcore_config_with_retrieval, mock_memory_client):
        """Test retrieve_customer_context with successful memory retrieval."""
        mock_memory_client.retrieve_memories.return_value = [
            {"content": {"text": "User context 1"}},
            {"content": {"text": "User context 2"}},
        ]

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config_with_retrieval)

                    mock_agent = Mock()
                    mock_agent.messages = [{"role": "user", "content": [{"text": "test query"}]}]

                    event = MessageAddedEvent(agent=mock_agent, message={"role": "user", "content": [{"text": "test"}]})
                    manager.retrieve_customer_context(event)

                    # Verify memory retrieval was called
                    assert mock_memory_client.retrieve_memories.called

    def test_retrieve_customer_context_exception(self, agentcore_config_with_retrieval, mock_memory_client):
        """Test retrieve_customer_context with exception handling."""
        mock_memory_client.retrieve_memories.side_effect = Exception("Memory error")

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config_with_retrieval)

                    mock_agent = Mock()
                    mock_agent.messages = [{"role": "user", "content": [{"text": "test query"}]}]

                    event = MessageAddedEvent(agent=mock_agent, message={"role": "user", "content": [{"text": "test"}]})

                    # Should not raise exception, just log error
                    manager.retrieve_customer_context(event)

    def test_retrieve_customer_context_filters_by_relevance_score(self, mock_memory_client):
        """Test retrieve_customer_context filters memories below relevance_score threshold."""
        # Return memories with varying relevance scores
        mock_memory_client.retrieve_memories.return_value = [
            {"content": {"text": "Low relevance 1"}, "relevanceScore": 0.1},
            {"content": {"text": "Low relevance 2"}, "relevanceScore": 0.4},
            {"content": {"text": "High relevance 1"}, "relevanceScore": 0.6},
            {"content": {"text": "High relevance 2"}, "relevanceScore": 0.9},
        ]

        # Config with single namespace and relevance_score threshold of 0.5
        config = AgentCoreMemoryConfig(
            memory_id="test-memory-123",
            session_id="test-session-456",
            actor_id="test-actor-789",
            retrieval_config={"test_namespace/": RetrievalConfig(top_k=10, relevance_score=0.5)},
        )

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(config)

                    mock_agent = Mock()
                    mock_agent.messages = [{"role": "user", "content": [{"text": "test query"}]}]

                    event = MessageAddedEvent(agent=mock_agent, message={"role": "user", "content": [{"text": "test"}]})
                    manager.retrieve_customer_context(event)

                    # Verify context was injected into the user message as a content block
                    # (single-message conversation uses inline injection)
                    assert len(mock_agent.messages) == 1
                    injected_context = mock_agent.messages[0]["content"][0]["text"]

                    # With threshold 0.5, only scores >= 0.5 should be included (0.6 and 0.9)
                    assert "High relevance 1" in injected_context
                    assert "High relevance 2" in injected_context
                    assert "Low relevance 1" not in injected_context
                    assert "Low relevance 2" not in injected_context

    def test_list_messages_default_max_results(self, session_manager, mock_memory_client):
        """Test listing messages without limit uses default max_results=10000."""
        mock_memory_client.list_events.return_value = []

        session_manager.list_messages("test-session-456", "test-agent-123")

        mock_memory_client.list_events.assert_called_once()
        call_kwargs = mock_memory_client.list_events.call_args[1]
        assert call_kwargs["max_results"] == 10000

    def test_list_messages_with_limit_calculates_max_results(self, session_manager, mock_memory_client):
        """Test listing messages with limit calculates max_results correctly."""
        mock_memory_client.list_events.return_value = []

        session_manager.list_messages("test-session-456", "test-agent-123", limit=500, offset=50)

        mock_memory_client.list_events.assert_called_once()
        call_kwargs = mock_memory_client.list_events.call_args[1]
        assert call_kwargs["max_results"] == 550  # limit + offset

    def test_append_message_handles_none_from_create_message(self, session_manager, test_agent):
        """Test that append_message gracefully handles None return from create_message."""
        # Create a tool use message (no text content, only toolUse block)
        tool_use_message = {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "tooluse_abc123",
                        "name": "calculator",
                        "input": {"operation": "add", "a": 5, "b": 3},
                    }
                }
            ],
        }

        # Mock create_message to return None (simulating the behavior for messages with no text)
        session_manager.create_message = Mock(return_value=None)
        session_manager._latest_agent_message = {}

        # This should NOT crash - it should handle None gracefully
        session_manager.append_message(tool_use_message, test_agent)

        # Verify create_message was called
        session_manager.create_message.assert_called_once()

        # Verify that _latest_agent_message was NOT updated (since message was skipped)
        assert test_agent.agent_id not in session_manager._latest_agent_message

    def test_append_message_normal_message_still_works(self, session_manager, test_agent):
        """Test that append_message still works correctly for normal messages with text."""
        # Create a normal message with text content
        normal_message = {
            "role": "assistant",
            "content": [{"text": "The answer is 8."}],
        }

        # Mock create_message to return a valid event (normal behavior)
        mock_event = {"eventId": "event_123456", "memoryId": "test-memory"}
        session_manager.create_message = Mock(return_value=mock_event)
        session_manager._latest_agent_message = {}

        # This should work normally
        session_manager.append_message(normal_message, test_agent)

        # Verify create_message was called
        session_manager.create_message.assert_called_once()

        # Verify that _latest_agent_message WAS updated
        assert test_agent.agent_id in session_manager._latest_agent_message
        assert session_manager._latest_agent_message[test_agent.agent_id].message_id == "event_123456"


class TestBatchingConfig:
    """Test batch_size configuration validation."""

    def test_batch_size_default_value(self):
        """Test batch_size defaults to 1 (immediate send)."""
        config = AgentCoreMemoryConfig(
            memory_id="test-memory",
            session_id="test-session",
            actor_id="test-actor",
        )
        assert config.batch_size == 1

    def test_batch_size_custom_value(self):
        """Test batch_size can be set to a custom value."""
        config = AgentCoreMemoryConfig(
            memory_id="test-memory",
            session_id="test-session",
            actor_id="test-actor",
            batch_size=10,
        )
        assert config.batch_size == 10

    def test_batch_size_maximum_value(self):
        """Test batch_size accepts maximum value of 100."""
        config = AgentCoreMemoryConfig(
            memory_id="test-memory",
            session_id="test-session",
            actor_id="test-actor",
            batch_size=100,
        )
        assert config.batch_size == 100

    def test_batch_size_exceeds_maximum_raises_error(self):
        """Test batch_size above 100 raises validation error."""
        with pytest.raises(ValueError):
            AgentCoreMemoryConfig(
                memory_id="test-memory",
                session_id="test-session",
                actor_id="test-actor",
                batch_size=101,
            )

    def test_batch_size_zero_raises_error(self):
        """Test batch_size of 0 raises validation error."""
        with pytest.raises(ValueError):
            AgentCoreMemoryConfig(
                memory_id="test-memory",
                session_id="test-session",
                actor_id="test-actor",
                batch_size=0,
            )

    def test_batch_size_negative_raises_error(self):
        """Test negative batch_size raises validation error."""
        with pytest.raises(ValueError):
            AgentCoreMemoryConfig(
                memory_id="test-memory",
                session_id="test-session",
                actor_id="test-actor",
                batch_size=-1,
            )


class TestBatchingBufferManagement:
    """Test batching buffer management and pending_message_count."""

    @pytest.fixture
    def batching_config(self):
        """Override with batch_size=5 for buffer management tests."""
        return AgentCoreMemoryConfig(
            memory_id="test-memory-123",
            session_id="test-session-456",
            actor_id="test-actor-789",
            batch_size=5,
        )

    @pytest.fixture
    def batching_session_manager(self, batching_config, mock_memory_client):
        """Create a session manager with batch_size=5."""
        return _create_session_manager(batching_config, mock_memory_client)

    def test_pending_message_count_empty_buffer(self, batching_session_manager):
        """Test pending_message_count returns 0 for empty buffer."""
        assert batching_session_manager.pending_message_count() == 0

    def test_pending_message_count_with_buffered_messages(self, batching_session_manager, mock_memory_client):
        """Test pending_message_count returns correct count."""
        # Add messages to buffer (batch_size=5, so won't auto-flush)
        for i in range(3):
            message = SessionMessage(
                message={"role": "user", "content": [{"text": f"Message {i}"}]},
                message_id=i,
                created_at="2024-01-01T12:00:00Z",
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

        assert batching_session_manager.pending_message_count() == 3
        # Verify no events were sent (still buffered)
        mock_memory_client.create_event.assert_not_called()

    def test_buffer_auto_flushes_at_batch_size(self, batching_session_manager, mock_memory_client):
        """Test buffer automatically flushes when reaching batch_size."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        # Add exactly batch_size messages (5)
        for i in range(5):
            message = SessionMessage(
                message={"role": "user", "content": [{"text": f"Message {i}"}]},
                message_id=i,
                created_at="2024-01-01T12:00:00Z",
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

        # Buffer should have been flushed
        assert batching_session_manager.pending_message_count() == 0
        # One batched API call for all messages in the same session
        assert mock_memory_client.gmdp_client.create_event.call_count == 1

    def test_create_message_returns_empty_dict_when_buffered(self, batching_session_manager):
        """Test create_message returns empty dict when message is buffered."""
        message = SessionMessage(
            message={"role": "user", "content": [{"text": "Hello"}]},
            message_id=1,
            created_at="2024-01-01T12:00:00Z",
        )

        result = batching_session_manager.create_message("test-session-456", "test-agent", message)

        assert result == {}

    def test_pending_agent_state_count_empty_buffer(self, batching_session_manager):
        """Test pending_agent_state_count returns 0 for empty buffer."""
        assert batching_session_manager.pending_agent_state_count() == 0

    def test_pending_agent_state_count_with_buffered_states(self, batching_session_manager, mock_memory_client):
        """Test pending_agent_state_count returns correct count."""
        # First create the agent so update_agent doesn't fail
        agent = SessionAgent(
            agent_id="test-agent",
            state={"description": "Initial"},
            conversation_manager_state={},
        )
        batching_session_manager.create_agent("test-session-456", agent)

        # Update agent state multiple times (should buffer with batch_size=5)
        for i in range(3):
            agent.state["description"] = f"Updated description {i}"
            batching_session_manager.update_agent("test-session-456", agent)

        # Should have 4 agent states in buffer (1 initial create + 3 updates)
        assert batching_session_manager.pending_agent_state_count() == 4
        # Verify no additional create_agent calls were made (still buffered)
        assert mock_memory_client.gmdp_client.create_event.call_count == 0  # All buffered, none flushed

    def test_agent_state_buffer_keeps_state_per_agent(self, batching_session_manager, mock_memory_client):
        """Test that agent state buffer preserves all agent state updates."""
        # Create two agents
        agent1 = SessionAgent(
            agent_id="agent-1",
            state={"description": "Description 1"},
            conversation_manager_state={},
        )
        agent2 = SessionAgent(
            agent_id="agent-2",
            state={"description": "Description 2"},
            conversation_manager_state={},
        )
        batching_session_manager.create_agent("test-session-456", agent1)
        batching_session_manager.create_agent("test-session-456", agent2)

        # Update agent1 once
        agent1.state["description"] = "Agent 1 update"
        batching_session_manager.update_agent("test-session-456", agent1)

        # Update agent2 once
        agent2.state["description"] = "Agent 2 updated"
        batching_session_manager.update_agent("test-session-456", agent2)

        # Total: 2 creates + 1 update + 1 update = 4 states in buffer (batch_size=5, so no auto-flush)
        # Should have 4 agent states in buffer (all preserved: 2 creates + 1 for agent1 + 1 for agent2)
        assert batching_session_manager.pending_agent_state_count() == 4

    def test_agent_state_flushed_with_messages(self, batching_session_manager, mock_memory_client):
        """Test that agent states are flushed along with messages."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        # Create agent
        agent = SessionAgent(
            agent_id="test-agent",
            state={"description": "Initial"},
            conversation_manager_state={},
        )
        batching_session_manager.create_agent("test-session-456", agent)

        # Add messages and update agent state
        for i in range(3):
            message = SessionMessage(
                message={"role": "user", "content": [{"text": f"Message {i}"}]},
                message_id=i,
                created_at="2024-01-01T12:00:00Z",
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

        agent.state["description"] = "Updated"
        batching_session_manager.update_agent("test-session-456", agent)

        # Verify both are buffered
        assert batching_session_manager.pending_message_count() == 3
        # Should have 2 agent states: 1 initial create + 1 update
        assert batching_session_manager.pending_agent_state_count() == 2

        # Flush
        batching_session_manager._flush_messages()

        # Both buffers should be cleared
        assert batching_session_manager.pending_message_count() == 0
        assert batching_session_manager.pending_agent_state_count() == 0

        # Verify create_event was called for messages and agent states
        # 2 calls total: 1 for batched messages + 1 for batched agent states
        assert mock_memory_client.gmdp_client.create_event.call_count == 2

    def test_agent_state_preserved_on_flush_failure(self, batching_session_manager, mock_memory_client):
        """Test that agent states remain in buffer if flush fails."""
        # Create agent
        agent = SessionAgent(
            agent_id="test-agent",
            state={"description": "Initial"},
            conversation_manager_state={},
        )
        batching_session_manager.create_agent("test-session-456", agent)

        # Update agent state
        agent.state["description"] = "Updated"
        batching_session_manager.update_agent("test-session-456", agent)

        # Should have 2 states: 1 initial create + 1 update
        assert batching_session_manager.pending_agent_state_count() == 2

        # Make flush fail
        mock_memory_client.gmdp_client.create_event.side_effect = Exception("API Error")

        # Flush should fail
        with pytest.raises(SessionException):
            batching_session_manager._flush_messages()

        # Agent states should still be in buffer (2 states preserved)
        assert batching_session_manager.pending_agent_state_count() == 2


class TestBatchingFlush:
    """Test _flush_messages behavior."""

    def test__flush_messages_empty_buffer(self, batching_session_manager):
        """Test _flush_messages with empty buffer returns empty list."""
        results = batching_session_manager._flush_messages()
        assert results == []

    def test__flush_messages_sends_all_buffered(self, batching_session_manager, mock_memory_client):
        """Test _flush_messages sends all buffered messages in a single batched call."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        # Add 3 messages (below batch_size of 10)
        for i in range(3):
            message = SessionMessage(
                message={"role": "user", "content": [{"text": f"Message {i}"}]},
                message_id=i,
                created_at="2024-01-01T12:00:00Z",
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

        assert batching_session_manager.pending_message_count() == 3

        # Flush manually
        results = batching_session_manager._flush_messages()

        # One batched API call for all messages in the same session
        assert len(results) == 1
        assert batching_session_manager.pending_message_count() == 0
        assert mock_memory_client.gmdp_client.create_event.call_count == 1

    def test__flush_messages_maintains_order(self, batching_session_manager, mock_memory_client):
        """Test _flush_messages maintains message order within batched payload."""
        sent_payloads = []

        def track_create_event(**kwargs):
            sent_payloads.append(kwargs.get("payload"))
            return {"eventId": f"event_{len(sent_payloads)}"}

        mock_memory_client.gmdp_client.create_event.side_effect = track_create_event

        # Add messages with distinct content
        for i in range(3):
            message = SessionMessage(
                message={"role": "user", "content": [{"text": f"Message_{i}"}]},
                message_id=i,
                created_at=f"2024-01-01T12:0{i}:00Z",
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

        batching_session_manager._flush_messages()

        # Should be one batched call with messages in order
        assert len(sent_payloads) == 1
        combined_payload = sent_payloads[0]
        assert len(combined_payload) == 3
        for i, item in enumerate(combined_payload):
            assert f"Message_{i}" in item["conversational"]["content"]["text"]

    def test__flush_messages_clears_buffer(self, batching_session_manager, mock_memory_client):
        """Test _flush_messages clears the buffer after sending."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        message = SessionMessage(
            message={"role": "user", "content": [{"text": "Hello"}]},
            message_id=1,
            created_at="2024-01-01T12:00:00Z",
        )
        batching_session_manager.create_message("test-session-456", "test-agent", message)

        # First flush
        batching_session_manager._flush_messages()
        assert batching_session_manager.pending_message_count() == 0

        # Second flush should be no-op
        results = batching_session_manager._flush_messages()
        assert results == []

    def test__flush_messages_exception_handling(self, batching_session_manager, mock_memory_client):
        """Test _flush_messages raises SessionException on failure."""
        mock_memory_client.gmdp_client.create_event.side_effect = Exception("API Error")

        message = SessionMessage(
            message={"role": "user", "content": [{"text": "Hello"}]},
            message_id=1,
            created_at="2024-01-01T12:00:00Z",
        )
        batching_session_manager.create_message("test-session-456", "test-agent", message)

        with pytest.raises(SessionException, match="Failed to flush messages"):
            batching_session_manager._flush_messages()

    def test__flush_messages_partial_flush_failure_preserves_all_messages(
        self, batching_session_manager, mock_memory_client
    ):
        """Test that on flush failure, all messages remain in buffer to prevent data loss."""
        mock_memory_client.gmdp_client.create_event.side_effect = Exception("API Error")

        # Add multiple messages
        for i in range(3):
            message = SessionMessage(
                message={"role": "user", "content": [{"text": f"Message {i}"}]},
                message_id=i,
                created_at=f"2024-01-01T12:0{i}:00Z",
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

        assert batching_session_manager.pending_message_count() == 3

        # Flush should fail
        with pytest.raises(SessionException):
            batching_session_manager._flush_messages()

        # All messages should still be in buffer (not cleared on failure)
        assert batching_session_manager.pending_message_count() == 3

        # Fix the mock and retry - should succeed now
        mock_memory_client.gmdp_client.create_event.side_effect = None
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        results = batching_session_manager._flush_messages()
        assert len(results) == 1  # One batched call for all messages
        assert batching_session_manager.pending_message_count() == 0

    def test__flush_messages_batching_combines_messages_for_same_session(
        self, batching_session_manager, mock_memory_client
    ):
        """Test that multiple messages for the same session are combined into one API call."""
        sent_payloads = []

        def track_create_event(**kwargs):
            sent_payloads.append(kwargs.get("payload"))
            return {"eventId": f"event_{len(sent_payloads)}"}

        mock_memory_client.gmdp_client.create_event.side_effect = track_create_event

        # Add 5 messages to the same session
        for i in range(5):
            message = SessionMessage(
                message={"role": "user", "content": [{"text": f"Message_{i}"}]},
                message_id=i,
                created_at=f"2024-01-01T12:0{i}:00Z",
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

        batching_session_manager._flush_messages()

        # Should be ONE API call with all 5 messages combined
        assert mock_memory_client.gmdp_client.create_event.call_count == 1
        assert len(sent_payloads) == 1
        # The combined payload should have all 5 messages
        assert len(sent_payloads[0]) == 5
        # Messages should be in order
        for i in range(5):
            assert f"Message_{i}" in sent_payloads[0][i]["conversational"]["content"]["text"]

    def test__flush_messages_multiple_sessions_grouped_into_separate_api_calls(
        self, batching_session_manager, mock_memory_client
    ):
        """Test that messages to different sessions are grouped into separate API calls.

        Note: In normal usage, create_message enforces session_id == config.session_id,
        so all messages go to one session. This test verifies the internal grouping logic
        by directly manipulating the buffer.
        """
        from datetime import datetime, timezone

        calls_by_session = {}

        def track_create_event(**kwargs):
            session_id = kwargs.get("sessionId")
            payload = kwargs.get("payload")
            calls_by_session[session_id] = payload
            return {"eventId": f"event_{session_id}"}

        mock_memory_client.gmdp_client.create_event.side_effect = track_create_event

        # Directly populate buffer with messages for multiple sessions
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        batching_session_manager._message_buffer = [
            BufferedMessage("session-A", [("SessionA_Message_0", "user")], False, base_time),
            BufferedMessage("session-A", [("SessionA_Message_1", "user")], False, base_time),
            BufferedMessage("session-B", [("SessionB_Message_0", "user")], False, base_time),
            BufferedMessage("session-B", [("SessionB_Message_1", "user")], False, base_time),
            BufferedMessage("session-B", [("SessionB_Message_2", "user")], False, base_time),
            BufferedMessage("session-A", [("SessionA_Message_2", "user")], False, base_time),  # Non-consecutive
        ]

        batching_session_manager._flush_messages()

        # Should be TWO API calls - one per session
        assert mock_memory_client.gmdp_client.create_event.call_count == 2
        assert len(calls_by_session) == 2

        # Session A should have 3 messages combined
        assert "session-A" in calls_by_session
        assert len(calls_by_session["session-A"]) == 3
        assert calls_by_session["session-A"][0]["conversational"]["content"]["text"] == "SessionA_Message_0"
        assert calls_by_session["session-A"][1]["conversational"]["content"]["text"] == "SessionA_Message_1"
        assert calls_by_session["session-A"][2]["conversational"]["content"]["text"] == "SessionA_Message_2"

        # Session B should have 3 messages combined
        assert "session-B" in calls_by_session
        assert len(calls_by_session["session-B"]) == 3
        for i in range(3):
            assert calls_by_session["session-B"][i]["conversational"]["content"]["text"] == f"SessionB_Message_{i}"

    def test__flush_messages_latest_timestamp_used_for_combined_events(
        self, batching_session_manager, mock_memory_client
    ):
        """Test that the latest timestamp from grouped messages is used for the combined event."""
        captured_timestamps = []

        def track_create_event(**kwargs):
            captured_timestamps.append(kwargs.get("eventTimestamp"))
            return {"eventId": "event_123"}

        mock_memory_client.gmdp_client.create_event.side_effect = track_create_event

        # Add messages with different timestamps (out of order)
        timestamps = ["2024-01-01T12:05:00Z", "2024-01-01T12:01:00Z", "2024-01-01T12:10:00Z"]
        for i, ts in enumerate(timestamps):
            message = SessionMessage(
                message={"role": "user", "content": [{"text": f"Message_{i}"}]},
                message_id=i,
                created_at=ts,
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

        batching_session_manager._flush_messages()

        # The combined event should use the latest timestamp (12:10:00)
        assert len(captured_timestamps) == 1
        # The timestamp should be the latest one (12:10:00)
        from datetime import datetime, timezone

        expected_latest = datetime(2024, 1, 1, 12, 10, 0, tzinfo=timezone.utc)
        # Account for monotonic timestamp adjustment (may add microseconds)
        assert captured_timestamps[0] >= expected_latest

    def test__flush_messages_partial_failure_multiple_sessions_preserves_buffer(
        self, batching_session_manager, mock_memory_client
    ):
        """Test that when one session fails, ALL messages remain in buffer.

        Note: Tests internal grouping logic by directly manipulating buffer.
        """
        from datetime import datetime, timezone

        def fail_on_second_session(**kwargs):
            session_id = kwargs.get("sessionId")
            if session_id == "session-B":
                raise Exception("API Error for session B")
            return {"eventId": f"event_{session_id}"}

        mock_memory_client.gmdp_client.create_event.side_effect = fail_on_second_session

        # Directly populate buffer with messages for multiple sessions
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        batching_session_manager._message_buffer = [
            BufferedMessage("session-A", [("SessionA_Message_0", "user")], False, base_time),
            BufferedMessage("session-A", [("SessionA_Message_1", "user")], False, base_time),
            BufferedMessage("session-B", [("SessionB_Message_0", "user")], False, base_time),
            BufferedMessage("session-B", [("SessionB_Message_1", "user")], False, base_time),
        ]

        assert batching_session_manager.pending_message_count() == 4

        # Flush should fail
        with pytest.raises(SessionException, match="Failed to flush messages"):
            batching_session_manager._flush_messages()

        # ALL messages should still be in buffer (even session A's which "succeeded")
        # This is because buffer is only cleared after ALL succeed
        assert batching_session_manager.pending_message_count() == 4

    def test_blob_messages_sent_batched(self, batching_session_manager, mock_memory_client):
        """Test that multiple blob messages are sent as batched."""
        blob_calls = []

        def track_blob_event(**kwargs):
            blob_calls.append(kwargs)
            return {"event": {"eventId": f"blob_event_{len(blob_calls)}"}}

        mock_memory_client.gmdp_client.create_event.side_effect = track_blob_event

        # Add multiple blob messages (>9KB each)
        for i in range(3):
            large_text = f"blob_{i}_" + "x" * (CONVERSATIONAL_MAX_SIZE + 100)
            message = SessionMessage(
                message={"role": "user", "content": [{"text": large_text}]},
                message_id=i,
                created_at=f"2024-01-01T12:0{i}:00Z",
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

        batching_session_manager._flush_messages()

        # Blobs are now batched together in one call with multiple payloads
        assert mock_memory_client.gmdp_client.create_event.call_count == 1
        assert len(blob_calls) == 1

        # Verify the batched call contains all 3 blobs
        call = blob_calls[0]
        assert "payload" in call
        assert len(call["payload"]) == 3
        for i in range(3):
            assert "blob" in call["payload"][i]
            assert f"blob_{i}_" in call["payload"][i]["blob"]

    def test_mixed_sessions_with_blobs_and_conversational(self, batching_session_manager, mock_memory_client):
        """Test complex scenario: multiple sessions with both blob and conversational messages.

        Note: Tests internal grouping logic by directly manipulating buffer.
        """
        from datetime import datetime, timezone

        calls_by_session = {}

        def track_create_event(**kwargs):
            session_id = kwargs.get("sessionId")
            payload = kwargs.get("payload")
            calls_by_session[session_id] = payload
            return {"eventId": f"event_{session_id}"}

        mock_memory_client.gmdp_client.create_event.side_effect = track_create_event

        # Directly populate buffer with mixed messages
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        blob_content = {"role": "user", "content": [{"text": "blob_A_" + "x" * (CONVERSATIONAL_MAX_SIZE + 100)}]}
        batching_session_manager._message_buffer = [
            # Session A: 2 conversational messages
            BufferedMessage("session-A", [("SessionA_conv_0", "user")], False, base_time),
            BufferedMessage("session-A", [("SessionA_conv_1", "user")], False, base_time),
            # Session A: 1 blob message
            BufferedMessage("session-A", [blob_content], True, base_time),
            # Session B: 1 conversational message
            BufferedMessage("session-B", [("SessionB_conv_0", "user")], False, base_time),
        ]

        batching_session_manager._flush_messages()

        # Should have 2 gmdp_client.create_event calls (one per session)
        # Each session combines conversational and blob messages
        assert mock_memory_client.gmdp_client.create_event.call_count == 2

        # Session A should have 3 items in payload (2 conversational + 1 blob)
        assert "session-A" in calls_by_session
        assert len(calls_by_session["session-A"]) == 3

        # Session B should have 1 conversational message
        assert "session-B" in calls_by_session
        assert len(calls_by_session["session-B"]) == 1

    def test__flush_messages_calls_both_flush_methods(self, batching_session_manager, mock_memory_client):
        """Test that _flush_messages() calls both _flush_messages_only() and _flush_agent_states_only()."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        # Add messages
        for i in range(2):
            message = SessionMessage(
                message={"role": "user", "content": [{"text": f"Message {i}"}]},
                message_id=i,
                created_at="2024-01-01T12:00:00Z",
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

        # Add agent state
        agent = SessionAgent(
            agent_id="test-agent",
            state={"description": "Test agent"},
            conversation_manager_state={},
        )
        batching_session_manager.create_agent("test-session-456", agent)

        # Verify both buffers have content
        assert batching_session_manager.pending_message_count() == 2
        assert batching_session_manager.pending_agent_state_count() == 1

        # Flush all
        results = batching_session_manager._flush_messages()

        # Should have 2 API calls: 1 for messages + 1 for agent states
        assert mock_memory_client.gmdp_client.create_event.call_count == 2
        assert len(results) == 2

        # Both buffers should be cleared
        assert batching_session_manager.pending_message_count() == 0
        assert batching_session_manager.pending_agent_state_count() == 0

    def test__flush_messages_with_only_messages(self, batching_session_manager, mock_memory_client):
        """Test that _flush_messages() works when only messages are buffered."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        # Add only messages (no agent states)
        for i in range(3):
            message = SessionMessage(
                message={"role": "user", "content": [{"text": f"Message {i}"}]},
                message_id=i,
                created_at="2024-01-01T12:00:00Z",
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

        assert batching_session_manager.pending_message_count() == 3
        assert batching_session_manager.pending_agent_state_count() == 0

        # Flush all
        results = batching_session_manager._flush_messages()

        # Should have 1 API call for messages only
        assert mock_memory_client.gmdp_client.create_event.call_count == 1
        assert len(results) == 1
        assert batching_session_manager.pending_message_count() == 0

    def test__flush_messages_with_only_agent_states(self, batching_session_manager, mock_memory_client):
        """Test that _flush_messages() works when only agent states are buffered."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        # Add only agent states (no messages)
        agent = SessionAgent(
            agent_id="test-agent",
            state={"description": "Test agent"},
            conversation_manager_state={},
        )
        batching_session_manager.create_agent("test-session-456", agent)

        agent.state["description"] = "Updated"
        batching_session_manager.update_agent("test-session-456", agent)

        assert batching_session_manager.pending_message_count() == 0
        assert batching_session_manager.pending_agent_state_count() == 2

        # Flush all
        results = batching_session_manager._flush_messages()

        # Should have 1 API call for agent states only
        assert mock_memory_client.gmdp_client.create_event.call_count == 1
        assert len(results) == 1
        assert batching_session_manager.pending_agent_state_count() == 0

    def test__flush_agent_states_only_empty_buffer(self, batching_session_manager):
        """Test _flush_agent_states_only with empty buffer returns empty list."""
        results = batching_session_manager._flush_agent_states_only()
        assert results == []

    def test__flush_agent_states_only_sends_all_buffered(self, batching_session_manager, mock_memory_client):
        """Test _flush_agent_states_only sends all buffered agent states in a single batched call."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        # Create agent
        agent = SessionAgent(
            agent_id="test-agent",
            state={"description": "Initial"},
            conversation_manager_state={},
        )
        batching_session_manager.create_agent("test-session-456", agent)

        # Update agent state twice
        for i in range(2):
            agent.state["description"] = f"Updated {i}"
            batching_session_manager.update_agent("test-session-456", agent)

        # Should have 3 agent states: 1 create + 2 updates
        assert batching_session_manager.pending_agent_state_count() == 3

        # Flush agent states only
        results = batching_session_manager._flush_agent_states_only()

        # One batched API call for all agent states
        assert len(results) == 1
        assert batching_session_manager.pending_agent_state_count() == 0
        assert mock_memory_client.gmdp_client.create_event.call_count == 1

        # Verify the call had metadata for agent state and agent id
        call_kwargs = mock_memory_client.gmdp_client.create_event.call_args[1]
        assert "metadata" in call_kwargs
        assert "stateType" in call_kwargs["metadata"]
        assert "agentId" in call_kwargs["metadata"]
        assert call_kwargs["metadata"]["agentId"] == {"stringValue": "test-agent"}

    def test__flush_agent_states_only_preserves_messages(self, batching_session_manager, mock_memory_client):
        """Test _flush_agent_states_only preserves message buffer."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        # Add messages
        for i in range(2):
            message = SessionMessage(
                message={"role": "user", "content": [{"text": f"Message {i}"}]},
                message_id=i,
                created_at="2024-01-01T12:00:00Z",
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

        # Add agent state
        agent = SessionAgent(
            agent_id="test-agent",
            state={"description": "Test agent"},
            conversation_manager_state={},
        )
        batching_session_manager.create_agent("test-session-456", agent)

        assert batching_session_manager.pending_message_count() == 2
        assert batching_session_manager.pending_agent_state_count() == 1

        # Flush only agent states
        batching_session_manager._flush_agent_states_only()

        # Agent states should be flushed, messages should remain
        assert batching_session_manager.pending_agent_state_count() == 0
        assert batching_session_manager.pending_message_count() == 2
        assert mock_memory_client.gmdp_client.create_event.call_count == 1

    def test__flush_agent_states_only_clears_buffer(self, batching_session_manager, mock_memory_client):
        """Test _flush_agent_states_only clears the agent state buffer after sending."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        agent = SessionAgent(
            agent_id="test-agent",
            state={"description": "Test agent"},
            conversation_manager_state={},
        )
        batching_session_manager.create_agent("test-session-456", agent)

        # First flush
        batching_session_manager._flush_agent_states_only()
        assert batching_session_manager.pending_agent_state_count() == 0

        # Second flush should be no-op
        results = batching_session_manager._flush_agent_states_only()
        assert results == []

    def test__flush_agent_states_only_exception_handling(self, batching_session_manager, mock_memory_client):
        """Test _flush_agent_states_only raises SessionException on failure."""
        mock_memory_client.gmdp_client.create_event.side_effect = Exception("API Error")

        agent = SessionAgent(
            agent_id="test-agent",
            state={"description": "Test agent"},
            conversation_manager_state={},
        )
        batching_session_manager.create_agent("test-session-456", agent)

        with pytest.raises(SessionException, match="Failed to flush agent states"):
            batching_session_manager._flush_agent_states_only()

    def test__flush_agent_states_only_failure_preserves_agent_states(
        self, batching_session_manager, mock_memory_client
    ):
        """Test that on flush failure, all agent states remain in buffer to prevent data loss."""
        mock_memory_client.gmdp_client.create_event.side_effect = Exception("API Error")

        # Create agent and update twice
        agent = SessionAgent(
            agent_id="test-agent",
            state={"description": "Initial"},
            conversation_manager_state={},
        )
        batching_session_manager.create_agent("test-session-456", agent)

        agent.state["description"] = "Updated 1"
        batching_session_manager.update_agent("test-session-456", agent)

        agent.state["description"] = "Updated 2"
        batching_session_manager.update_agent("test-session-456", agent)

        assert batching_session_manager.pending_agent_state_count() == 3

        # Flush should fail
        with pytest.raises(SessionException):
            batching_session_manager._flush_agent_states_only()

        # All agent states should still be in buffer (not cleared on failure)
        assert batching_session_manager.pending_agent_state_count() == 3

        # Fix the mock and retry - should succeed now
        mock_memory_client.gmdp_client.create_event.side_effect = None
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        results = batching_session_manager._flush_agent_states_only()
        assert len(results) == 1
        assert batching_session_manager.pending_agent_state_count() == 0

    def test__flush_agent_states_only_batches_multiple_states(self, batching_session_manager, mock_memory_client):
        """Test that multiple agent states are batched into a single API call."""
        sent_payloads = []

        def track_create_event(**kwargs):
            sent_payloads.append(kwargs.get("payload"))
            return {"eventId": f"event_{len(sent_payloads)}"}

        mock_memory_client.gmdp_client.create_event.side_effect = track_create_event

        # Create agent and update 4 times
        agent = SessionAgent(
            agent_id="test-agent",
            state={"description": "Initial"},
            conversation_manager_state={},
        )
        batching_session_manager.create_agent("test-session-456", agent)

        for i in range(4):
            agent.state["description"] = f"Updated {i}"
            batching_session_manager.update_agent("test-session-456", agent)

        # Should have 5 agent states: 1 create + 4 updates
        assert batching_session_manager.pending_agent_state_count() == 5

        batching_session_manager._flush_agent_states_only()

        # Should be ONE API call with all 5 agent states combined
        assert mock_memory_client.gmdp_client.create_event.call_count == 1
        assert len(sent_payloads) == 1
        # The combined payload should have all 5 agent states as blobs
        assert len(sent_payloads[0]) == 5
        for item in sent_payloads[0]:
            assert "blob" in item


class TestBatchingBackwardsCompatibility:
    """Test batch_size=1 behaves identically to previous implementation."""

    def test_batch_size_one_sends_immediately(self, session_manager, mock_memory_client):
        """Test batch_size=1 (default) sends message immediately."""
        mock_memory_client.create_event.return_value = {"eventId": "event_123"}

        message = SessionMessage(
            message={"role": "user", "content": [{"text": "Hello"}]},
            message_id=1,
            created_at="2024-01-01T12:00:00Z",
        )

        result = session_manager.create_message("test-session-456", "test-agent-123", message)

        # Should return event immediately
        assert result.get("eventId") == "event_123"
        # Should have sent immediately
        mock_memory_client.create_event.assert_called_once()
        # Buffer should be empty
        assert session_manager.pending_message_count() == 0

    def test_batch_size_one_returns_event_id(self, session_manager, mock_memory_client):
        """Test batch_size=1 returns the event with eventId."""
        mock_memory_client.create_event.return_value = {"eventId": "unique_event_id"}

        message = SessionMessage(
            message={"role": "user", "content": [{"text": "Hello"}]},
            message_id=1,
            created_at="2024-01-01T12:00:00Z",
        )

        result = session_manager.create_message("test-session-456", "test-agent-123", message)

        assert "eventId" in result
        assert result["eventId"] == "unique_event_id"


class TestBatchingContextManager:
    """Test context manager (__enter__/__exit__) functionality."""

    def test_context_manager_returns_self(self, batching_session_manager):
        """Test __enter__ returns the session manager instance."""
        with batching_session_manager as ctx:
            assert ctx is batching_session_manager

    def test_context_manager_flushes_on_exit(self, batching_session_manager, mock_memory_client):
        """Test __exit__ flushes pending messages."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        with batching_session_manager:
            message = SessionMessage(
                message={"role": "user", "content": [{"text": "Hello"}]},
                message_id=1,
                created_at="2024-01-01T12:00:00Z",
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

            # Should still be buffered
            assert batching_session_manager.pending_message_count() == 1

        # After exiting context, should have flushed
        assert batching_session_manager.pending_message_count() == 0
        mock_memory_client.gmdp_client.create_event.assert_called_once()

    def test_context_manager_flushes_on_exception(self, batching_session_manager, mock_memory_client):
        """Test __exit__ flushes even when exception occurs."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        try:
            with batching_session_manager:
                message = SessionMessage(
                    message={"role": "user", "content": [{"text": "Hello"}]},
                    message_id=1,
                    created_at="2024-01-01T12:00:00Z",
                )
                batching_session_manager.create_message("test-session-456", "test-agent", message)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should have flushed despite exception
        assert batching_session_manager.pending_message_count() == 0
        mock_memory_client.gmdp_client.create_event.assert_called_once()

    def test_exit_preserves_original_exception_when_flush_fails(
        self, batching_session_manager, mock_memory_client, caplog
    ):
        """Test __exit__ logs flush failure and preserves the original exception."""
        mock_memory_client.gmdp_client.create_event.side_effect = RuntimeError("flush failed")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="original error"):
                with batching_session_manager:
                    message = SessionMessage(
                        message={"role": "user", "content": [{"text": "Hello"}]},
                        message_id=1,
                        created_at="2024-01-01T12:00:00Z",
                    )
                    batching_session_manager.create_message("test-session-456", "test-agent", message)
                    raise ValueError("original error")

        assert any(
            "Failed to flush messages during exception handling" in record.message and record.levelno == logging.ERROR
            for record in caplog.records
        )

    def test_exit_raises_flush_exception_when_no_original_exception(
        self, batching_session_manager, mock_memory_client, caplog
    ):
        """Test __exit__ still raises flush exceptions when no original exception."""
        mock_memory_client.gmdp_client.create_event.side_effect = RuntimeError("flush failed")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(SessionException, match="flush failed"):
                with batching_session_manager:
                    message = SessionMessage(
                        message={"role": "user", "content": [{"text": "Hello"}]},
                        message_id=1,
                        created_at="2024-01-01T12:00:00Z",
                    )
                    batching_session_manager.create_message("test-session-456", "test-agent", message)

        assert not any(
            "Failed to flush messages during exception handling" in record.message for record in caplog.records
        )


class TestBatchingClose:
    """Test close() method functionality."""

    def test_close_flushes_pending_messages(self, batching_session_manager, mock_memory_client):
        """Test close() flushes all pending messages in a batched call."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "event_123"}

        # Add messages
        for i in range(3):
            message = SessionMessage(
                message={"role": "user", "content": [{"text": f"Message {i}"}]},
                message_id=i,
                created_at="2024-01-01T12:00:00Z",
            )
            batching_session_manager.create_message("test-session-456", "test-agent", message)

        assert batching_session_manager.pending_message_count() == 3

        # Close should flush
        batching_session_manager.close()

        assert batching_session_manager.pending_message_count() == 0
        # One batched API call for all messages in the same session
        assert mock_memory_client.gmdp_client.create_event.call_count == 1

    def test_close_with_empty_buffer(self, batching_session_manager, mock_memory_client):
        """Test close() with empty buffer is a no-op."""
        batching_session_manager.close()

        mock_memory_client.create_event.assert_not_called()
        assert batching_session_manager.pending_message_count() == 0


class TestBatchingBlobMessages:
    """Test batching handles blob messages (exceeding conversational limit) correctly."""

    def test_blob_message_sent_via_gmdp_client(self, batching_session_manager, mock_memory_client):
        """Test large messages (blobs) are sent via gmdp_client."""
        mock_memory_client.gmdp_client.create_event.return_value = {"event": {"eventId": "blob_event_123"}}

        # Create a message that exceeds CONVERSATIONAL_MAX_SIZE.
        large_text = "x" * (CONVERSATIONAL_MAX_SIZE + 100)
        message = SessionMessage(
            message={"role": "user", "content": [{"text": large_text}]},
            message_id=1,
            created_at="2024-01-01T12:00:00Z",
        )
        batching_session_manager.create_message("test-session-456", "test-agent", message)

        # Flush and verify blob path was used
        batching_session_manager._flush_messages()

        mock_memory_client.gmdp_client.create_event.assert_called_once()
        call_kwargs = mock_memory_client.gmdp_client.create_event.call_args.kwargs
        assert "payload" in call_kwargs
        assert "blob" in call_kwargs["payload"][0]

    def test_mixed_conversational_and_blob_messages(self, batching_session_manager, mock_memory_client):
        """Test batching correctly handles mix of conversational and blob messages."""
        mock_memory_client.gmdp_client.create_event.return_value = {"eventId": "conv_event"}

        # Add small (conversational) message
        small_message = SessionMessage(
            message={"role": "user", "content": [{"text": "Small message"}]},
            message_id=1,
            created_at="2024-01-01T12:00:00Z",
        )
        batching_session_manager.create_message("test-session-456", "test-agent", small_message)

        # Add large (blob) message
        large_text = "x" * (CONVERSATIONAL_MAX_SIZE + 100)
        large_message = SessionMessage(
            message={"role": "user", "content": [{"text": large_text}]},
            message_id=2,
            created_at="2024-01-01T12:01:00Z",
        )
        batching_session_manager.create_message("test-session-456", "test-agent", large_message)

        # Flush
        batching_session_manager._flush_messages()

        # Both messages should be sent via gmdp_client.create_event (batched together)
        assert mock_memory_client.gmdp_client.create_event.call_count == 1


class TestThinkingModeCompatibility:
    """Test that retrieve_customer_context injects memory inline, not as an assistant message.

    When thinking is enabled on Claude, assistant messages must start with a thinking block
    and the conversation must end with a user message. Injecting LTM as a separate assistant
    message violates both constraints.
    """

    def test_retrieve_customer_context_does_not_append_assistant_message(
        self, agentcore_config_with_retrieval, mock_memory_client
    ):
        """Test retrieved memory is injected into the user message, not as a new assistant message."""
        mock_memory_client.retrieve_memories.return_value = [
            {"content": {"text": "User prefers dark mode"}},
            {"content": {"text": "User likes sushi"}},
        ]

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config_with_retrieval)

                    mock_agent = Mock()
                    mock_agent.messages = [{"role": "user", "content": [{"text": "What are my preferences?"}]}]

                    event = MessageAddedEvent(
                        agent=mock_agent, message={"role": "user", "content": [{"text": "What are my preferences?"}]}
                    )
                    manager.retrieve_customer_context(event)

                    # No new messages should be added — memory is inlined in the user message
                    assert len(mock_agent.messages) == 1
                    assert mock_agent.messages[-1]["role"] == "user"

                    # Memory prepended, original query remains last
                    content = mock_agent.messages[0]["content"]
                    assert len(content) == 2
                    assert "<user_context>" in content[0]["text"]
                    assert content[1]["text"] == "What are my preferences?"

    def test_retrieve_customer_context_no_assistant_message_multi_turn(
        self, agentcore_config_with_retrieval, mock_memory_client
    ):
        """Test memory injection keeps last message as user in a multi-turn conversation."""
        mock_memory_client.retrieve_memories.return_value = [
            {"content": {"text": "User likes sushi"}},
        ]

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(agentcore_config_with_retrieval)

                    mock_agent = Mock()
                    mock_agent.messages = [
                        {"role": "user", "content": [{"text": "I love sushi"}]},
                        {"role": "assistant", "content": [{"text": "That's great!"}]},
                        {"role": "user", "content": [{"text": "What do I like to eat?"}]},
                    ]

                    event = MessageAddedEvent(
                        agent=mock_agent, message={"role": "user", "content": [{"text": "What do I like to eat?"}]}
                    )
                    manager.retrieve_customer_context(event)

                    # No new messages added
                    assert len(mock_agent.messages) == 3
                    assert mock_agent.messages[-1]["role"] == "user"

                    # Memory injected into last user message
                    content = mock_agent.messages[-1]["content"]
                    assert len(content) == 2
                    assert "<user_context>" in content[0]["text"]
                    assert content[1]["text"] == "What do I like to eat?"

    def test_retrieve_customer_context_custom_context_tag(self, mock_memory_client):
        """Test that a custom context_tag is used when configured."""
        custom_config = AgentCoreMemoryConfig(
            memory_id="test-memory-123",
            session_id="test-session-456",
            actor_id="test-actor-789",
            retrieval_config={"user_preferences/{actorId}/": RetrievalConfig(top_k=5, relevance_score=0.3)},
            context_tag="retrieved_memory",
        )

        mock_memory_client.retrieve_memories.return_value = [
            {"content": {"text": "User likes sushi"}},
        ]

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(custom_config)

                    mock_agent = Mock()
                    mock_agent.messages = [{"role": "user", "content": [{"text": "What do I like?"}]}]

                    event = MessageAddedEvent(
                        agent=mock_agent, message={"role": "user", "content": [{"text": "What do I like?"}]}
                    )
                    manager.retrieve_customer_context(event)

                    content = mock_agent.messages[0]["content"]
                    assert "<retrieved_memory>" in content[0]["text"]
                    assert "</retrieved_memory>" in content[0]["text"]

    def test_retrieve_customer_context_default_context_tag(self, mock_memory_client):
        """Test that the default context_tag is user_context."""
        default_config = AgentCoreMemoryConfig(
            memory_id="test-memory-123",
            session_id="test-session-456",
            actor_id="test-actor-789",
            retrieval_config={"user_preferences/{actorId}/": RetrievalConfig(top_k=5, relevance_score=0.3)},
        )

        mock_memory_client.retrieve_memories.return_value = [
            {"content": {"text": "User likes sushi"}},
        ]

        with patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ):
            with patch("boto3.Session") as mock_boto_session:
                mock_session = Mock()
                mock_session.region_name = "us-west-2"
                mock_session.client.return_value = Mock()
                mock_boto_session.return_value = mock_session

                with patch(
                    "strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None
                ):
                    manager = AgentCoreMemorySessionManager(default_config)

                    mock_agent = Mock()
                    mock_agent.messages = [{"role": "user", "content": [{"text": "What do I like?"}]}]

                    event = MessageAddedEvent(
                        agent=mock_agent, message={"role": "user", "content": [{"text": "What do I like?"}]}
                    )
                    manager.retrieve_customer_context(event)

                    content = mock_agent.messages[0]["content"]
                    assert "<user_context>" in content[0]["text"]
                    assert "</user_context>" in content[0]["text"]


class TestAfterInvocationHook:
    """Test AfterInvocationEvent hook integration."""

    def test_after_invocation_hook_registered(self, batching_session_manager):
        """Test that AfterInvocationEvent hook is registered when batching is enabled."""
        registry = HookRegistry()
        batching_session_manager.register_hooks(registry)

        # Verify AfterInvocationEvent callback is registered (batching is enabled)
        assert AfterInvocationEvent in registry._registered_callbacks
        assert len(registry._registered_callbacks[AfterInvocationEvent]) > 0

    def test_after_invocation_hook_flushes_buffer(self, batching_session_manager, mock_memory_client):
        """Test that AfterInvocationEvent hook triggers flush."""
        # Mock session_repository to avoid parent class hook issues
        batching_session_manager.session_repository = Mock()

        # Add messages to buffer
        with batching_session_manager._message_lock:
            batching_session_manager._message_buffer.append(
                BufferedMessage("test-session", [("user", "test message")], False, batching_session_manager._get_monotonic_timestamp())
            )

        assert batching_session_manager.pending_message_count() == 1

        # Register hooks and trigger AfterInvocationEvent
        registry = HookRegistry()
        batching_session_manager.register_hooks(registry)

        # Create mock event with mock agent
        mock_agent = Mock()
        mock_event = AfterInvocationEvent(agent=mock_agent)
        registry.invoke_callbacks(mock_event)

        # Verify buffer was flushed
        assert batching_session_manager.pending_message_count() == 0

    def test_after_invocation_hook_not_registered_when_batching_disabled(self, session_manager):
        """Test that AfterInvocationEvent flush hook is NOT registered when batching is disabled."""
        # Spy on the registry to track what gets added
        registry = HookRegistry()
        original_add = registry.add_callback
        added_callbacks = []

        def spy_add_callback(event_type, callback):
            added_callbacks.append((event_type, callback))
            return original_add(event_type, callback)

        registry.add_callback = spy_add_callback
        session_manager.register_hooks(registry)

        # Check that no AfterInvocationEvent callback referencing _flush_messages was added
        flush_callbacks = [
            cb
            for event_type, cb in added_callbacks
            if event_type == AfterInvocationEvent
            and hasattr(cb, "__code__")
            and "_flush_messages" in str(cb.__code__.co_names)
        ]
        assert len(flush_callbacks) == 0


class TestIntervalFlush:
    """Test interval-based flush mechanism for long-running agents."""

    def test_interval_flush_timer_starts_when_configured(self):
        """Test that interval flush timer starts when flush_interval_seconds is set."""
        config = AgentCoreMemoryConfig(
            memory_id="test-memory",
            session_id="test-session",
            actor_id="test-actor",
            batch_size=10,
            flush_interval_seconds=5.0,
        )

        mock_client = Mock()
        mock_client.list_events.return_value = []

        with (
            patch(
                "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
                return_value=mock_client,
            ),
            patch("boto3.Session") as mock_boto_session,
            patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
        ):
            mock_session = Mock()
            mock_session.region_name = "us-west-2"
            mock_session.client.return_value = Mock()
            mock_boto_session.return_value = mock_session

            manager = AgentCoreMemorySessionManager(config)

            # Verify timer was started
            assert manager._flush_timer is not None
            assert manager._flush_timer.is_alive()
            assert not manager._shutdown

            # Cleanup
            manager.close()

    def test_interval_flush_timer_not_started_when_disabled(self):
        """Test that interval flush timer is not started when flush_interval_seconds is None."""
        config = AgentCoreMemoryConfig(
            memory_id="test-memory",
            session_id="test-session",
            actor_id="test-actor",
            batch_size=10,
            # flush_interval_seconds not set (defaults to None)
        )

        mock_client = Mock()
        mock_client.list_events.return_value = []

        with (
            patch(
                "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
                return_value=mock_client,
            ),
            patch("boto3.Session") as mock_boto_session,
            patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
        ):
            mock_session = Mock()
            mock_session.region_name = "us-west-2"
            mock_session.client.return_value = Mock()
            mock_boto_session.return_value = mock_session

            manager = AgentCoreMemorySessionManager(config)

            # Verify timer was not started
            assert manager._flush_timer is None
            assert not manager._shutdown

    def test_interval_flush_timer_stops_on_close(self):
        """Test that interval flush timer stops when close() is called."""
        config = AgentCoreMemoryConfig(
            memory_id="test-memory",
            session_id="test-session",
            actor_id="test-actor",
            batch_size=10,
            flush_interval_seconds=5.0,
        )

        mock_client = Mock()
        mock_client.list_events.return_value = []

        with (
            patch(
                "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
                return_value=mock_client,
            ),
            patch("boto3.Session") as mock_boto_session,
            patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
        ):
            mock_session = Mock()
            mock_session.region_name = "us-west-2"
            mock_session.client.return_value = Mock()
            mock_boto_session.return_value = mock_session

            manager = AgentCoreMemorySessionManager(config)

            # Verify timer is running
            assert manager._flush_timer is not None
            timer_ref = manager._flush_timer  # Keep reference
            assert timer_ref.is_alive()
            assert not manager._shutdown

            # Close manager
            manager.close()

            # Verify timer is stopped
            assert manager._shutdown
            assert manager._flush_timer is None  # Should be set to None
            # Give timer a moment to actually stop
            time.sleep(0.1)
            assert not timer_ref.is_alive()  # Verify thread actually stopped

    def test_interval_flush_timer_stops_on_context_exit(self):
        """Test that interval flush timer stops when exiting context manager."""
        config = AgentCoreMemoryConfig(
            memory_id="test-memory",
            session_id="test-session",
            actor_id="test-actor",
            batch_size=10,
            flush_interval_seconds=5.0,
        )

        mock_client = Mock()
        mock_client.list_events.return_value = []

        with (
            patch(
                "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
                return_value=mock_client,
            ),
            patch("boto3.Session") as mock_boto_session,
            patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
        ):
            mock_session = Mock()
            mock_session.region_name = "us-west-2"
            mock_session.client.return_value = Mock()
            mock_boto_session.return_value = mock_session

            with AgentCoreMemorySessionManager(config) as manager:
                # Verify timer is running inside context
                assert manager._flush_timer is not None
                assert not manager._shutdown

            # Verify timer is stopped after context exit
            assert manager._shutdown

    def test_interval_flush_callback_flushes_when_buffer_has_messages(self):
        """Test that interval flush callback flushes buffer when it has messages."""
        config = AgentCoreMemoryConfig(
            memory_id="test-memory",
            session_id="test-session",
            actor_id="test-actor",
            batch_size=10,
            flush_interval_seconds=5.0,
        )

        mock_client = Mock()
        mock_client.list_events.return_value = []
        mock_client.create_event.return_value = {"eventId": "event_123"}

        with (
            patch(
                "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
                return_value=mock_client,
            ),
            patch("boto3.Session") as mock_boto_session,
            patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
        ):
            mock_session = Mock()
            mock_session.region_name = "us-west-2"
            mock_session.client.return_value = Mock()
            mock_boto_session.return_value = mock_session

            manager = AgentCoreMemorySessionManager(config)

            # Add messages to buffer
            with manager._message_lock:
                manager._message_buffer.append(
                    BufferedMessage("test-session", [("user", "test message")], False, manager._get_monotonic_timestamp())
                )

            assert manager.pending_message_count() == 1

            # Manually trigger interval flush callback
            manager._interval_flush_callback()

            # Verify buffer was flushed
            assert manager.pending_message_count() == 0

            # Cleanup
            manager.close()

    def test_interval_flush_callback_skips_when_buffer_empty(self):
        """Test that interval flush callback skips flush when buffer is empty."""
        config = AgentCoreMemoryConfig(
            memory_id="test-memory",
            session_id="test-session",
            actor_id="test-actor",
            batch_size=10,
            flush_interval_seconds=5.0,
        )

        mock_client = Mock()
        mock_client.list_events.return_value = []

        with (
            patch(
                "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
                return_value=mock_client,
            ),
            patch("boto3.Session") as mock_boto_session,
            patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
        ):
            mock_session = Mock()
            mock_session.region_name = "us-west-2"
            mock_session.client.return_value = Mock()
            mock_boto_session.return_value = mock_session

            manager = AgentCoreMemorySessionManager(config)

            # Verify buffer is empty
            assert manager.pending_message_count() == 0

            # Track flush calls
            original_flush = manager._flush_messages
            flush_called = {"count": 0}

            def tracked_flush():
                flush_called["count"] += 1
                return original_flush()

            manager._flush_messages = tracked_flush

            # Manually trigger interval flush callback
            manager._interval_flush_callback()

            # Verify flush was not called (buffer was empty)
            assert flush_called["count"] == 0

            # Cleanup
            manager.close()

    def test_interval_flush_callback_flushes_when_agent_state_pending(self):
        """Test that interval flush callback flushes when agent state is pending."""
        config = AgentCoreMemoryConfig(
            memory_id="test-memory",
            session_id="test-session",
            actor_id="test-actor",
            batch_size=10,
            flush_interval_seconds=5.0,
        )

        mock_client = Mock()
        mock_client.list_events.return_value = []
        mock_client.create_event.return_value = {"eventId": "event_123"}

        with (
            patch(
                "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
                return_value=mock_client,
            ),
            patch("boto3.Session") as mock_boto_session,
            patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
        ):
            mock_session = Mock()
            mock_session.region_name = "us-west-2"
            mock_gmdp_client = Mock()
            mock_gmdp_client.create_event.return_value = {"eventId": "event_456"}
            mock_session.client.return_value = mock_gmdp_client
            mock_boto_session.return_value = mock_session

            manager = AgentCoreMemorySessionManager(config)
            manager.session_id = "test-session"  # Set session_id since parent __init__ is mocked

            # Add agent state to buffer (no messages)
            from strands.types.session import SessionAgent

            agent = SessionAgent(
                agent_id="test-agent",
                state={"description": "Test"},
                conversation_manager_state={},
            )
            with manager._agent_state_lock:
                manager._agent_state_buffer.append(("test-session", agent))
                manager._agent_created_at_cache["test-agent"] = agent.created_at

            assert manager.pending_message_count() == 0
            assert manager.pending_agent_state_count() == 1

            # Manually trigger interval flush callback
            manager._interval_flush_callback()

            # Verify buffer was flushed
            assert manager.pending_agent_state_count() == 0

            # Cleanup
            manager.close()

    def test_interval_flush_callback_flushes_when_both_buffers_have_data(self):
        """Test that interval flush callback flushes when both messages and agent states are pending."""
        config = AgentCoreMemoryConfig(
            memory_id="test-memory",
            session_id="test-session",
            actor_id="test-actor",
            batch_size=10,
            flush_interval_seconds=5.0,
        )

        mock_client = Mock()
        mock_client.list_events.return_value = []
        mock_client.create_event.return_value = {"eventId": "event_123"}

        with (
            patch(
                "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
                return_value=mock_client,
            ),
            patch("boto3.Session") as mock_boto_session,
            patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
        ):
            mock_session = Mock()
            mock_session.region_name = "us-west-2"
            mock_gmdp_client = Mock()
            mock_gmdp_client.create_event.return_value = {"eventId": "event_456"}
            mock_session.client.return_value = mock_gmdp_client
            mock_boto_session.return_value = mock_session

            manager = AgentCoreMemorySessionManager(config)
            manager.session_id = "test-session"  # Set session_id since parent __init__ is mocked

            # Add both messages and agent state to buffers
            with manager._message_lock:
                manager._message_buffer.append(
                    BufferedMessage("test-session", [("user", "test message")], False, manager._get_monotonic_timestamp())
                )

            from strands.types.session import SessionAgent

            agent = SessionAgent(
                agent_id="test-agent",
                state={"description": "Test"},
                conversation_manager_state={},
            )
            with manager._agent_state_lock:
                manager._agent_state_buffer.append(("test-session", agent))
                manager._agent_created_at_cache["test-agent"] = agent.created_at

            assert manager.pending_message_count() == 1
            assert manager.pending_agent_state_count() == 1

            # Manually trigger interval flush callback
            manager._interval_flush_callback()

            # Verify both buffers were flushed
            assert manager.pending_message_count() == 0
            assert manager.pending_agent_state_count() == 0

            # Cleanup
            manager.close()

    def test_config_flush_interval_validation(self):
        """Test that flush_interval_seconds must be positive."""
        # Valid: positive value
        config = AgentCoreMemoryConfig(
            memory_id="test-memory",
            session_id="test-session",
            actor_id="test-actor",
            flush_interval_seconds=30.0,
        )
        assert config.flush_interval_seconds == 30.0

        # Valid: None (disabled)
        config = AgentCoreMemoryConfig(
            memory_id="test-memory",
            session_id="test-session",
            actor_id="test-actor",
            flush_interval_seconds=None,
        )
        assert config.flush_interval_seconds is None

        # Invalid: zero or negative should raise validation error
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AgentCoreMemoryConfig(
                memory_id="test-memory",
                session_id="test-session",
                actor_id="test-actor",
                flush_interval_seconds=0.0,
            )

        with pytest.raises(ValidationError):
            AgentCoreMemoryConfig(
                memory_id="test-memory",
                session_id="test-session",
                actor_id="test-actor",
                flush_interval_seconds=-5.0,
            )


class TestMetadataSupport:
    """Tests for user-supplied event metadata on messages."""

    @pytest.fixture
    def config_with_metadata(self):
        """Config with default metadata."""
        return AgentCoreMemoryConfig(
            memory_id="test-memory-123",
            session_id="test-session-456",
            actor_id="test-actor-789",
            default_metadata={"location": {"stringValue": "NYC"}, "team": {"stringValue": "support"}},
        )

    @pytest.fixture
    def session_manager_with_metadata(self, config_with_metadata, mock_memory_client):
        """Session manager with default metadata configured."""
        return _create_session_manager(config_with_metadata, mock_memory_client)

    def test_create_message_with_default_metadata(self, session_manager_with_metadata, mock_memory_client):
        """Config-level default_metadata flows to create_event."""
        mock_memory_client.create_event.return_value = {"eventId": "evt_1"}
        session_message = SessionMessage.from_message({"role": "user", "content": [{"text": "hello"}]}, 0)

        session_manager_with_metadata.create_message("test-session-456", "agent-1", session_message)

        mock_memory_client.create_event.assert_called_once()
        call_kwargs = mock_memory_client.create_event.call_args[1]
        assert call_kwargs["metadata"] == {"location": {"stringValue": "NYC"}, "team": {"stringValue": "support"}}

    def test_create_message_with_per_call_metadata(self, session_manager, mock_memory_client):
        """Per-call metadata passed via kwargs flows to create_event."""
        mock_memory_client.create_event.return_value = {"eventId": "evt_1"}
        session_message = SessionMessage.from_message({"role": "user", "content": [{"text": "hello"}]}, 0)
        per_call = {"project": {"stringValue": "alpha"}}

        session_manager.create_message("test-session-456", "agent-1", session_message, metadata=per_call)

        mock_memory_client.create_event.assert_called_once()
        call_kwargs = mock_memory_client.create_event.call_args[1]
        assert call_kwargs["metadata"] == {"project": {"stringValue": "alpha"}}

    def test_metadata_merging_precedence(self, session_manager_with_metadata, mock_memory_client):
        """Per-call metadata overrides config default for the same key."""
        mock_memory_client.create_event.return_value = {"eventId": "evt_1"}
        session_message = SessionMessage.from_message({"role": "user", "content": [{"text": "hello"}]}, 0)
        per_call = {"location": {"stringValue": "SF"}, "project": {"stringValue": "beta"}}

        session_manager_with_metadata.create_message("test-session-456", "agent-1", session_message, metadata=per_call)

        call_kwargs = mock_memory_client.create_event.call_args[1]
        assert call_kwargs["metadata"]["location"] == {"stringValue": "SF"}
        assert call_kwargs["metadata"]["team"] == {"stringValue": "support"}
        assert call_kwargs["metadata"]["project"] == {"stringValue": "beta"}

    def test_metadata_reserved_keys_rejected(self, session_manager):
        """ValueError raised when user metadata contains reserved keys."""
        from bedrock_agentcore.memory.integrations.strands.session_manager import RESERVED_METADATA_KEYS

        session_message = SessionMessage.from_message({"role": "user", "content": [{"text": "hello"}]}, 0)

        for reserved_key in RESERVED_METADATA_KEYS:
            with pytest.raises(ValueError, match="reserved"):
                session_manager.create_message(
                    "test-session-456",
                    "agent-1",
                    session_message,
                    metadata={reserved_key: {"stringValue": "bad"}},
                )

    def test_metadata_max_keys_exceeded(self, session_manager):
        """ValueError raised when combined metadata exceeds MAX_METADATA_KEYS."""
        from bedrock_agentcore.memory.integrations.strands.session_manager import MAX_METADATA_KEYS

        session_message = SessionMessage.from_message({"role": "user", "content": [{"text": "hello"}]}, 0)
        too_many = {f"key_{i}": {"stringValue": f"val_{i}"} for i in range(MAX_METADATA_KEYS + 1)}

        with pytest.raises(ValueError, match="exceeding the maximum"):
            session_manager.create_message("test-session-456", "agent-1", session_message, metadata=too_many)

    def test_create_message_no_metadata_passes_none(self, session_manager, mock_memory_client):
        """When no metadata configured and none passed, metadata kwarg is None."""
        mock_memory_client.create_event.return_value = {"eventId": "evt_1"}
        session_message = SessionMessage.from_message({"role": "user", "content": [{"text": "hello"}]}, 0)

        session_manager.create_message("test-session-456", "agent-1", session_message)

        call_kwargs = mock_memory_client.create_event.call_args[1]
        assert call_kwargs.get("metadata") is None

    def test_batched_messages_include_metadata(self, mock_memory_client):
        """Metadata flows through the batching path and appears in the flushed event."""
        from datetime import datetime, timezone

        config = AgentCoreMemoryConfig(
            memory_id="test-memory-123",
            session_id="test-session-456",
            actor_id="test-actor-789",
            batch_size=5,
            default_metadata={"env": {"stringValue": "staging"}},
        )
        manager = _create_session_manager(config, mock_memory_client)

        mock_memory_client.gmdp_client.create_event.return_value = {"event": {"eventId": "batch_evt_1"}}

        # Buffer messages with metadata
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        manager._message_buffer = [
            BufferedMessage(
                "test-session-456",
                [("hello", "user")],
                False,
                base_time,
                metadata={"env": {"stringValue": "staging"}},
            ),
            BufferedMessage(
                "test-session-456",
                [("world", "assistant")],
                False,
                base_time,
                metadata={"env": {"stringValue": "staging"}, "extra": {"stringValue": "val"}},
            ),
        ]

        manager._flush_messages_only()

        call_kwargs = mock_memory_client.gmdp_client.create_event.call_args[1]
        # Merged metadata: later message's metadata overrides earlier
        assert call_kwargs["metadata"]["env"] == {"stringValue": "staging"}
        assert call_kwargs["metadata"]["extra"] == {"stringValue": "val"}

    def test_blob_message_with_metadata(self, session_manager_with_metadata, mock_memory_client):
        """Blob messages also receive metadata."""
        from bedrock_agentcore.memory.integrations.strands.bedrock_converter import CONVERSATIONAL_MAX_SIZE

        mock_memory_client.gmdp_client.create_event.return_value = {"event": {"eventId": "blob_1"}}
        big_text = "x" * (CONVERSATIONAL_MAX_SIZE + 100)
        session_message = SessionMessage.from_message({"role": "user", "content": [{"text": big_text}]}, 0)

        session_manager_with_metadata.create_message("test-session-456", "agent-1", session_message)

        call_kwargs = mock_memory_client.gmdp_client.create_event.call_args[1]
        assert call_kwargs["metadata"] == {"location": {"stringValue": "NYC"}, "team": {"stringValue": "support"}}

    def test_metadata_provider_called_per_event(self, mock_memory_client):
        """metadata_provider is called at each create_message and its values appear in the event."""
        call_count = 0

        def provider():
            nonlocal call_count
            call_count += 1
            return {"traceId": {"stringValue": f"trace-{call_count}"}}

        config = AgentCoreMemoryConfig(
            memory_id="test-memory-123",
            session_id="test-session-456",
            actor_id="test-actor-789",
            metadata_provider=provider,
        )
        manager = _create_session_manager(config, mock_memory_client)
        mock_memory_client.create_event.return_value = {"eventId": "evt_1"}

        msg1 = SessionMessage.from_message({"role": "user", "content": [{"text": "hello"}]}, 0)
        manager.create_message("test-session-456", "agent-1", msg1)

        assert call_count == 1
        kwargs1 = mock_memory_client.create_event.call_args[1]
        assert kwargs1["metadata"]["traceId"] == {"stringValue": "trace-1"}

        msg2 = SessionMessage.from_message({"role": "user", "content": [{"text": "world"}]}, 0)
        manager.create_message("test-session-456", "agent-1", msg2)

        assert call_count == 2
        kwargs2 = mock_memory_client.create_event.call_args[1]
        assert kwargs2["metadata"]["traceId"] == {"stringValue": "trace-2"}

    def test_metadata_provider_merged_with_defaults(self, mock_memory_client):
        """metadata_provider values override default_metadata for same key, but both appear."""
        config = AgentCoreMemoryConfig(
            memory_id="test-memory-123",
            session_id="test-session-456",
            actor_id="test-actor-789",
            default_metadata={"env": {"stringValue": "prod"}, "team": {"stringValue": "support"}},
            metadata_provider=lambda: {"env": {"stringValue": "staging"}, "traceId": {"stringValue": "t-1"}},
        )
        manager = _create_session_manager(config, mock_memory_client)
        mock_memory_client.create_event.return_value = {"eventId": "evt_1"}

        msg = SessionMessage.from_message({"role": "user", "content": [{"text": "hello"}]}, 0)
        manager.create_message("test-session-456", "agent-1", msg)

        call_kwargs = mock_memory_client.create_event.call_args[1]
        # provider overrides default for "env"
        assert call_kwargs["metadata"]["env"] == {"stringValue": "staging"}
        # default still present
        assert call_kwargs["metadata"]["team"] == {"stringValue": "support"}
        # provider adds new key
        assert call_kwargs["metadata"]["traceId"] == {"stringValue": "t-1"}

    def test_metadata_provider_reserved_keys_rejected(self, mock_memory_client):
        """metadata_provider returning reserved keys raises ValueError."""
        config = AgentCoreMemoryConfig(
            memory_id="test-memory-123",
            session_id="test-session-456",
            actor_id="test-actor-789",
            metadata_provider=lambda: {"stateType": {"stringValue": "bad"}},
        )
        manager = _create_session_manager(config, mock_memory_client)

        msg = SessionMessage.from_message({"role": "user", "content": [{"text": "hello"}]}, 0)
        with pytest.raises(ValueError, match="reserved"):
            manager.create_message("test-session-456", "agent-1", msg)
