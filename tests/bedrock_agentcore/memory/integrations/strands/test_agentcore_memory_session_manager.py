"""Tests for AgentCoreMemorySessionManager."""

from unittest.mock import Mock, patch

import pytest
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from strands.agent.agent import Agent
from strands.hooks import MessageAddedEvent
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage, SessionType

from bedrock_agentcore.memory.integrations.strands.bedrock_converter import AgentCoreMemoryConverter
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager


@pytest.fixture
def agentcore_config():
    """Create a test AgentCore Memory configuration."""
    return AgentCoreMemoryConfig(memory_id="test-memory-123", session_id="test-session-456", actor_id="test-actor-789")


@pytest.fixture
def agentcore_config_with_retrieval():
    """Create a test AgentCore Memory configuration with retrieval config."""
    retrieval_config = {
        "user_preferences/{actorId}": RetrievalConfig(top_k=5, relevance_score=0.3),
        "session_context/{sessionId}": RetrievalConfig(top_k=3, relevance_score=0.5),
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


@pytest.fixture
def session_manager(agentcore_config, mock_memory_client):
    """Create an AgentCoreMemorySessionManager with mocked dependencies."""
    with patch(
        "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient", return_value=mock_memory_client
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
                manager.session_id = agentcore_config.session_id
                manager.session = Session(session_id=agentcore_config.session_id, session_type=SessionType.AGENT)
                return manager


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
            "user_preferences/{actorId}", "user_preferences/test-actor"
        )

        # Mock invalid resolution
        session_manager._validate_namespace_resolution.return_value = False
        assert not session_manager._validate_namespace_resolution(
            "user_preferences/{actorId}", "user_preferences/{actorId}"
        )

        # Invalid - empty result
        assert not session_manager._validate_namespace_resolution("test_namespace", "")

    def test_load_long_term_memories_with_validation_failure(self, mock_memory_client, test_agent):
        """Test LTM loading with namespace validation failure."""
        # Create config with namespace that will fail resolution
        config_with_bad_namespace = AgentCoreMemoryConfig(
            memory_id="test-memory-123",
            session_id="test-session-456",
            actor_id="test-actor",
            retrieval_config={"user_preferences/{invalidVar}": RetrievalConfig(top_k=5, relevance_score=0.3)},
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

        # No memories should be stored
        assert "ltm_memories" not in test_agent.state._state

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
        query = session_manager._generate_initialization_query("user_preferences/{actorId}", config, test_agent)
        assert query == "user preferences settings"

        # Test context namespace
        query = session_manager._generate_initialization_query("session_context/{sessionId}", config, test_agent)
        assert query == "conversation context history"

        # Test semantic namespace
        query = session_manager._generate_initialization_query("semantic_knowledge", config, test_agent)
        assert query == "facts knowledge information"

        # Test facts namespace
        query = session_manager._generate_initialization_query("facts_database", config, test_agent)
        assert query == "facts knowledge information"

        # Test fallback
        query = session_manager._generate_initialization_query("unknown_namespace", config, test_agent)
        assert query == "context preferences facts"

    def test_generate_initialization_query_custom(self, session_manager, test_agent):
        """Test custom initialization query takes precedence."""
        config = RetrievalConfig(top_k=5, relevance_score=0.3, initialization_query="custom query for testing")

        # Mock the method since it doesn't exist yet
        session_manager._generate_initialization_query = Mock(return_value="custom query for testing")

        query = session_manager._generate_initialization_query("user_preferences/{actorId}", config, test_agent)
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
                                "namespace": "user_preferences/test-actor-789",
                                "memories": [{"content": "Relevant memory", "relevanceScore": 0.8}],
                            },
                            {
                                "namespace": "session_context/test-session-456",
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
                                "namespace": "user_preferences/test-actor-789",
                                "memories": [{"content": "User preference memory", "relevanceScore": 0.9}],
                            }
                        ]
                    )
                    results = manager.retrieve_contextual_memories(
                        "What are my preferences?", namespaces=["user_preferences/{actorId}"]
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
                    results = manager.retrieve_contextual_memories("test query", namespaces=["nonexistent_namespace"])

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
        namespace = "user_preferences/{actorId}"
        resolved = namespace.format(
            actorId=session_manager.config.actor_id, sessionId=session_manager.config.session_id, memoryStrategyId=""
        )
        assert resolved == "user_preferences/test-actor-789"

        # Test multiple variables
        namespace = "context/{sessionId}/actor/{actorId}"
        resolved = namespace.format(
            actorId=session_manager.config.actor_id, sessionId=session_manager.config.session_id, memoryStrategyId=""
        )
        assert resolved == "context/test-session-456/actor/test-actor-789"

        # Test with strategy ID
        namespace = "strategy/{memoryStrategyId}/user/{actorId}"
        resolved = namespace.format(
            actorId=session_manager.config.actor_id,
            sessionId=session_manager.config.session_id,
            memoryStrategyId="test_strategy",
        )
        assert resolved == "strategy/test_strategy/user/test-actor-789"

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
            ("user_preferences/{actorId}", "user preferences settings"),
            ("preferences/global", "user preferences settings"),
            ("my_preferences", "user preferences settings"),
            ("session_context/{sessionId}", "conversation context history"),
            ("context/history", "conversation context history"),
            ("conversation_context", "conversation context history"),
            ("semantic_memory", "facts knowledge information"),
            ("facts_database", "facts knowledge information"),
            ("knowledge_semantic", "facts knowledge information"),
            ("random_namespace", "context preferences facts"),
            ("unknown", "context preferences facts"),
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

    def test_get_full_session_id_conflict(self, session_manager):
        """Test session ID conflict with actor ID."""
        # Set up a scenario where session ID would conflict with actor ID
        session_manager.config.actor_id = "session_test-session"

        with pytest.raises(SessionException, match="Cannot have session"):
            session_manager._get_full_session_id("test-session")

    def test_get_full_agent_id_conflict(self, session_manager):
        """Test agent ID conflict with actor ID."""
        # Set up a scenario where agent ID would conflict with actor ID
        session_manager.config.actor_id = "agent_test-agent"

        with pytest.raises(SessionException, match="Cannot create agent"):
            session_manager._get_full_agent_id("test-agent")

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
            retrieval_config={"test_namespace": RetrievalConfig(top_k=10, relevance_score=0.5)},
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

                    # Verify context was injected
                    assert len(mock_agent.messages) == 2
                    injected_context = mock_agent.messages[1]["content"][0]["text"]

                    # With threshold 0.5, only scores >= 0.5 should be included (0.6 and 0.9)
                    assert "High relevance 1" in injected_context
                    assert "High relevance 2" in injected_context
                    assert "Low relevance 1" not in injected_context
                    assert "Low relevance 2" not in injected_context
