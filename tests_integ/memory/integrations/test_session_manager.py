"""
Integration tests for AgentCore Memory Session Manager.

Run with: python -m pytest tests_integ/memory/integrations/test_session_manager.py -v
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

import pytest
from strands import Agent
from strands.types.session import Session, SessionAgent, SessionType

from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.integrations.strands.bedrock_converter import AgentCoreMemoryConverter
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.memory.models.filters import EventMetadataFilter, LeftExpression, OperatorType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REGION = os.environ.get("BEDROCK_TEST_REGION", "us-east-1")


@pytest.mark.integration
class TestAgentCoreMemorySessionManager:
    """Integration tests for AgentCore Memory Session Manager."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-east-1")
        cls.client = MemoryClient(region_name=cls.region)

    @pytest.fixture(scope="session")
    def memory_client(self):
        """Create a memory client for testing."""
        return MemoryClient(region_name=REGION)

    @pytest.fixture(scope="session")
    def test_memory_stm(self, memory_client):
        """Create a test memory for integration tests."""
        memory_name = f"testmemorySTM{uuid.uuid4().hex[:8]}"
        memory = memory_client.create_memory_and_wait(
            name=memory_name, description="Test STM memory for integration tests", strategies=[]
        )
        yield memory
        # Cleanup
        try:
            memory_client.delete_memory(memory["id"])
        except Exception:
            pass  # Memory might already be deleted

    @pytest.fixture(scope="session")
    def test_memory_ltm(self, memory_client):
        """Create a test memory for integration tests."""
        memory_name = f"testmemoryLTM{uuid.uuid4().hex[:8]}"
        memory = memory_client.create_memory_and_wait(
            name=memory_name,
            description="Full-featured memory with all built-in strategies",
            strategies=[
                {
                    "summaryMemoryStrategy": {
                        "name": "SessionSummarizer",
                        "namespaces": ["/summaries/{actorId}/{sessionId}/"],
                    }
                },
                {
                    "userPreferenceMemoryStrategy": {
                        "name": "PreferenceLearner",
                        "namespaces": ["/preferences/{actorId}/"],
                    }
                },
                {"semanticMemoryStrategy": {"name": "FactExtractor", "namespaces": ["/facts/{actorId}/"]}},
            ],
        )
        yield memory
        try:
            memory_client.delete_memory(memory["id"])
        except Exception:
            pass  # Memory might already be deleted

    def test_session_manager_initialization(self, test_memory_stm):
        """Test session manager initialization."""
        session_config = AgentCoreMemoryConfig(
            memory_id=test_memory_stm["id"],
            session_id=f"test-session-{int(time.time())}",
            actor_id=f"test-actor-{int(time.time())}",
        )
        session_manager = AgentCoreMemorySessionManager(agentcore_memory_config=session_config, region_name=REGION)

        assert session_manager.config == session_config
        assert session_manager.memory_client is not None

    def test_agent_with_session_manager(self, test_memory_stm):
        """Test creating an agent with the session manager."""
        session_config = AgentCoreMemoryConfig(
            memory_id=test_memory_stm["id"],
            session_id=f"test-session-{int(time.time())}",
            actor_id=f"test-actor-{int(time.time())}",
        )
        session_manager = AgentCoreMemorySessionManager(agentcore_memory_config=session_config, region_name=REGION)

        agent = Agent(system_prompt="You are a helpful assistant.", session_manager=session_manager)

        assert agent._session_manager == session_manager

    def test_conversation_persistence(self, test_memory_stm):
        """Test that conversations are persisted to memory."""
        session_config = AgentCoreMemoryConfig(
            memory_id=test_memory_stm["id"],
            session_id=f"test-session-{int(time.time())}",
            actor_id=f"test-actor-{int(time.time())}",
        )
        session_manager = AgentCoreMemorySessionManager(agentcore_memory_config=session_config, region_name=REGION)

        agent = Agent(system_prompt="You are a helpful assistant.", session_manager=session_manager)

        # Have a conversation
        response1 = agent("Hello, my name is John")
        assert response1 is not None

        time.sleep(2)  # throttling
        response2 = agent("What is my name?")
        assert response2 is not None
        assert "John" in response2.message["content"][0]["text"]

    def test_session_manager_with_retrieval_config_adds_context(self, test_memory_ltm):
        """Test session manager with custom retrieval configuration."""
        actor_id = f"test-actor-{int(time.time())}"
        config = AgentCoreMemoryConfig(
            memory_id=test_memory_ltm["id"],
            session_id=f"test-session-{int(time.time())}",
            actor_id=actor_id,
            retrieval_config={"/preferences/{actorId}/": RetrievalConfig(top_k=5, relevance_score=0.7)},
        )

        session_manager = AgentCoreMemorySessionManager(agentcore_memory_config=config, region_name=REGION)

        agent = Agent(system_prompt="You are a helpful assistant.", session_manager=session_manager)

        response1 = agent("I like sushi with tuna")
        assert response1 is not None
        self.client.wait_for_memories(
            memory_id=test_memory_ltm["id"],
            namespace=f"/preferences/{actor_id}/",
            test_query="sushi",
            max_wait=180,
            poll_interval=10,
        )

        response2 = agent("What do I like to eat?")
        assert response2 is not None
        assert "sushi" in str(agent.messages)
        assert "<user_context>" in str(agent.messages)

    def test_multiple_namespace_retrieval_config(self, test_memory_ltm):
        """Test session manager with multiple namespace retrieval configurations."""
        actor_id = f"test-actor-{int(time.time())}"
        config = AgentCoreMemoryConfig(
            memory_id=test_memory_ltm["id"],
            session_id=f"test-session-{int(time.time())}",
            actor_id=actor_id,
            retrieval_config={
                "/preferences/{actorId}/": RetrievalConfig(top_k=5, relevance_score=0.7),
                "/facts/{actorId}/": RetrievalConfig(top_k=10, relevance_score=0.3),
                "/summaries/{actorId}/{sessionId}/": RetrievalConfig(top_k=5, relevance_score=0.5),
            },
        )

        session_manager = AgentCoreMemorySessionManager(agentcore_memory_config=config, region_name=REGION)

        assert len(session_manager.config.retrieval_config) == 3
        agent = Agent(
            system_prompt="You are a helpful assistant that understands user preferences.",
            session_manager=session_manager,
        )

        response1 = agent("I like sushi with tuna")
        assert response1 is not None
        self.client.wait_for_memories(
            memory_id=test_memory_ltm["id"],
            namespace=f"/preferences/{actor_id}/",
            test_query="sushi",
            max_wait=180,
            poll_interval=10,
        )

        response2 = agent("What do I like to eat?")
        assert response2 is not None
        assert "sushi" in str(agent.messages)
        assert "<user_context>" in str(agent.messages)

    def test_session_manager_error_handling(self):
        """Test session manager error handling with invalid configuration."""
        with pytest.raises(Exception):  # noqa: B017
            # Invalid memory ID should raise an error
            config = AgentCoreMemoryConfig(
                memory_id="invalid-memory-id", session_id="test-session", actor_id="test-actor"
            )

            session_manager = AgentCoreMemorySessionManager(agentcore_memory_config=config, region_name=REGION)

            # This should fail when trying to use the session manager
            agent = Agent(system_prompt="Test", session_manager=session_manager)
            agent("Test message")

    def test_legacy_event_migration(self, test_memory_stm, memory_client):
        """Test that legacy events with prefixed actorIds are migrated to metadata format.

        The constructor calls read_session which creates a metadata-path session if none exists.
        To test legacy migration, we create the legacy event BEFORE constructing the session manager,
        so the constructor's read_session finds it via the fallback and migrates it on first access.
        """
        session_id = f"test-legacy-{uuid.uuid4().hex[:8]}"
        actor_id = f"test-actor-{uuid.uuid4().hex[:8]}"

        # --- Session migration ---
        # Create a legacy session event BEFORE constructing the session manager.
        # Legacy events use blob payloads with the session data, so we use gmdp_client directly.
        legacy_session_actor_id = f"session_{session_id}"
        session_data = Session(session_id=session_id, session_type=SessionType.AGENT)
        memory_client.gmdp_client.create_event(
            memoryId=test_memory_stm["id"],
            actorId=legacy_session_actor_id,
            sessionId=session_id,
            payload=[{"blob": json.dumps(session_data.to_dict())}],
            eventTimestamp=datetime.now(timezone.utc),
        )

        # Verify legacy event exists before migration
        legacy_events_before = memory_client.list_events(
            memory_id=test_memory_stm["id"],
            actor_id=legacy_session_actor_id,
            session_id=session_id,
        )
        assert len(legacy_events_before) >= 1

        # Constructing the session manager triggers read_session in __init__,
        # which should find the legacy event, migrate it, and delete the old one
        config = AgentCoreMemoryConfig(
            memory_id=test_memory_stm["id"],
            session_id=session_id,
            actor_id=actor_id,
        )
        session_manager = AgentCoreMemorySessionManager(agentcore_memory_config=config, region_name=REGION)

        # Verify migration: legacy event should be deleted
        legacy_events_after = memory_client.list_events(
            memory_id=test_memory_stm["id"],
            actor_id=legacy_session_actor_id,
            session_id=session_id,
        )
        assert len(legacy_events_after) == 0

        # Verify migration: read_session finds it via the new metadata path
        read_session_result = session_manager.read_session(session_id)
        assert read_session_result is not None
        assert read_session_result.session_id == session_id

        # --- Agent migration ---
        agent_id = f"test-agent-{uuid.uuid4().hex[:8]}"
        legacy_agent_actor_id = f"agent_{agent_id}"
        agent_data = SessionAgent(
            agent_id=agent_id,
            state={"key": "value"},
            conversation_manager_state={},
        )
        memory_client.gmdp_client.create_event(
            memoryId=test_memory_stm["id"],
            actorId=legacy_agent_actor_id,
            sessionId=session_id,
            payload=[{"blob": json.dumps(agent_data.to_dict())}],
            eventTimestamp=datetime.now(timezone.utc),
        )

        # read_agent should find via fallback and migrate
        read_agent_result = session_manager.read_agent(session_id, agent_id)
        assert read_agent_result is not None
        assert read_agent_result.agent_id == agent_id

        # Verify migration: legacy event should be deleted
        legacy_agent_events = memory_client.list_events(
            memory_id=test_memory_stm["id"],
            actor_id=legacy_agent_actor_id,
            session_id=session_id,
        )
        assert len(legacy_agent_events) == 0

    # endregion Event metadata integration tests

    # region End-to-end agent with batching tests

    def test_agent_conversation_with_context_manager(self, test_memory_stm):
        """Test that Agent messages are flushed when the context manager exits, and session resume loads them."""
        session_id = f"test-agent-ctx-{uuid.uuid4().hex[:8]}"
        actor_id = f"test-actor-{uuid.uuid4().hex[:8]}"

        config = AgentCoreMemoryConfig(
            memory_id=test_memory_stm["id"],
            session_id=session_id,
            actor_id=actor_id,
            batch_size=10,
        )

        # Use context manager — __exit__ calls _flush_messages() which is blocking
        with AgentCoreMemorySessionManager(agentcore_memory_config=config, region_name=REGION) as sm:
            agent = Agent(system_prompt="You are a helpful assistant.", session_manager=sm)
            response1 = agent("Hello, my name is Bob")
            assert response1 is not None

        # After __exit__, buffered messages have been flushed (blocking).
        # Resume session with a new session manager to verify persistence.
        config2 = AgentCoreMemoryConfig(
            memory_id=test_memory_stm["id"],
            session_id=session_id,
            actor_id=actor_id,
            batch_size=10,
        )
        sm2 = AgentCoreMemorySessionManager(agentcore_memory_config=config2, region_name=REGION)
        agent2 = Agent(system_prompt="You are a helpful assistant.", session_manager=sm2)

        response2 = agent2("What is my name?")
        assert response2 is not None
        assert "Bob" in response2.message["content"][0]["text"]

        sm2.close()

    def test_agent_multi_turn_with_batching(self, test_memory_stm):
        """Test that a multi-turn conversation within a single Agent works with batching."""
        session_id = f"test-agent-multi-{uuid.uuid4().hex[:8]}"
        actor_id = f"test-actor-{uuid.uuid4().hex[:8]}"

        config = AgentCoreMemoryConfig(
            memory_id=test_memory_stm["id"],
            session_id=session_id,
            actor_id=actor_id,
            batch_size=10,
        )
        session_manager = AgentCoreMemorySessionManager(agentcore_memory_config=config, region_name=REGION)

        agent = Agent(system_prompt="You are a helpful assistant.", session_manager=session_manager)

        agent("Hello, my name is Charlie")
        agent("I live in Seattle")
        response3 = agent("What is my name and where do I live?")
        assert response3 is not None
        response_text = response3.message["content"][0]["text"]
        assert "Charlie" in response_text
        assert "Seattle" in response_text

        # Flush remaining buffered messages (blocking)
        session_manager.close()

        # Verify batched messages are persisted — filter out state events
        message_filter = EventMetadataFilter.build_expression(
            left_operand=LeftExpression.build("stateType"),
            operator=OperatorType.NOT_EXISTS,
        )
        events = session_manager.memory_client.list_events(
            memory_id=test_memory_stm["id"],
            actor_id=actor_id,
            session_id=session_id,
            event_metadata=[message_filter],
        )

        # Convert events back to messages and verify all turns are present
        messages = AgentCoreMemoryConverter.events_to_messages(events)
        # At least 3 user + 3 assistant messages
        assert len(messages) >= 6

    # endregion End-to-end agent with batching tests

    # region Multi-agent tests

    def test_multi_agent_conversation(self, test_memory_stm):
        """Test two agents write to same session, each retrieves only their own messages."""
        session_id = f"multi-agent-session-{int(time.time())}"
        actor_id = f"test-actor-{int(time.time())}"

        # Create session manager for agent A
        config_a = AgentCoreMemoryConfig(
            memory_id=test_memory_stm["id"],
            session_id=session_id,
            actor_id=actor_id,
        )
        sm_a = AgentCoreMemorySessionManager(agentcore_memory_config=config_a, region_name=REGION)

        agent_a = Agent(
            agent_id="agent-A",
            system_prompt="You are Agent A. Always start your responses with 'Agent A here:'.",
            session_manager=sm_a,
        )

        # Agent A has a conversation
        response_a = agent_a("Hello from user to Agent A. Remember the word 'pineapple'.")
        assert response_a is not None

        time.sleep(2)

        # Create session manager for agent B (same session, same actor)
        config_b = AgentCoreMemoryConfig(
            memory_id=test_memory_stm["id"],
            session_id=session_id,
            actor_id=actor_id,
        )
        sm_b = AgentCoreMemorySessionManager(agentcore_memory_config=config_b, region_name=REGION)

        agent_b = Agent(
            agent_id="agent-B",
            system_prompt="You are Agent B. Always start your responses with 'Agent B here:'.",
            session_manager=sm_b,
        )

        # Agent B has a conversation
        response_b = agent_b("Hello from user to Agent B. Remember the word 'strawberry'.")
        assert response_b is not None

        time.sleep(2)

        # Now verify each agent retrieves only its own messages via metadata filter
        agent_a_filter = EventMetadataFilter.build_expression(
            left_operand=LeftExpression.build("agentId"),
            operator=OperatorType.EQUALS_TO,
            right_operand={"metadataValue": {"stringValue": "agent-A"}},
        )
        agent_b_filter = EventMetadataFilter.build_expression(
            left_operand=LeftExpression.build("agentId"),
            operator=OperatorType.EQUALS_TO,
            right_operand={"metadataValue": {"stringValue": "agent-B"}},
        )

        events_a = sm_a.memory_client.list_events(
            memory_id=test_memory_stm["id"],
            actor_id=actor_id,
            session_id=session_id,
            event_metadata=[agent_a_filter],
        )
        events_b = sm_b.memory_client.list_events(
            memory_id=test_memory_stm["id"],
            actor_id=actor_id,
            session_id=session_id,
            event_metadata=[agent_b_filter],
        )

        # Each agent should have its own events (at least user + assistant)
        assert len(events_a) >= 2, f"Expected >=2 events for agent-A, got {len(events_a)}"
        assert len(events_b) >= 2, f"Expected >=2 events for agent-B, got {len(events_b)}"

        # The two sets should be disjoint
        event_ids_a = {e["eventId"] for e in events_a}
        event_ids_b = {e["eventId"] for e in events_b}
        assert event_ids_a.isdisjoint(event_ids_b), "Agent A and B events should be disjoint"

    def test_multi_agent_with_batching(self, test_memory_stm):
        """Test two agents with batching enabled, verify flush groups correctly."""
        session_id = f"multi-agent-batch-{int(time.time())}"
        actor_id = f"test-actor-{int(time.time())}"

        config = AgentCoreMemoryConfig(
            memory_id=test_memory_stm["id"],
            session_id=session_id,
            actor_id=actor_id,
            batch_size=10,
        )

        with AgentCoreMemorySessionManager(agentcore_memory_config=config, region_name=REGION) as sm:
            agent_a = Agent(
                agent_id="agent-A",
                system_prompt="You are Agent A. Keep responses very short (one sentence).",
                session_manager=sm,
            )
            agent_b = Agent(
                agent_id="agent-B",
                system_prompt="You are Agent B. Keep responses very short (one sentence).",
                session_manager=sm,
            )

            agent_a("Hello Agent A, remember the color blue.")
            time.sleep(2)
            agent_b("Hello Agent B, remember the color red.")
        # Context manager flushes remaining buffered messages

        time.sleep(2)

        # Verify events are tagged correctly with agent metadata
        agent_a_filter = EventMetadataFilter.build_expression(
            left_operand=LeftExpression.build("agentId"),
            operator=OperatorType.EQUALS_TO,
            right_operand={"metadataValue": {"stringValue": "agent-A"}},
        )
        agent_b_filter = EventMetadataFilter.build_expression(
            left_operand=LeftExpression.build("agentId"),
            operator=OperatorType.EQUALS_TO,
            right_operand={"metadataValue": {"stringValue": "agent-B"}},
        )

        client = MemoryClient(region_name=REGION)
        events_a = client.list_events(
            memory_id=test_memory_stm["id"],
            actor_id=actor_id,
            session_id=session_id,
            event_metadata=[agent_a_filter],
        )
        events_b = client.list_events(
            memory_id=test_memory_stm["id"],
            actor_id=actor_id,
            session_id=session_id,
            event_metadata=[agent_b_filter],
        )

        # Both agents should have events
        assert len(events_a) >= 1, f"Expected >=1 events for agent-A, got {len(events_a)}"
        assert len(events_b) >= 1, f"Expected >=1 events for agent-B, got {len(events_b)}"

        # Events should be disjoint between agents
        event_ids_a = {e["eventId"] for e in events_a}
        event_ids_b = {e["eventId"] for e in events_b}
        assert event_ids_a.isdisjoint(event_ids_b), "Agent A and B events should be disjoint"

    def test_multi_agent_list_messages_isolation(self, test_memory_stm):
        """Test that list_messages returns only the calling agent's messages."""
        session_id = f"multi-agent-list-{int(time.time())}"
        actor_id = f"test-actor-{int(time.time())}"

        config = AgentCoreMemoryConfig(
            memory_id=test_memory_stm["id"],
            session_id=session_id,
            actor_id=actor_id,
        )
        sm = AgentCoreMemorySessionManager(agentcore_memory_config=config, region_name=REGION)

        agent_a = Agent(
            agent_id="agent-A",
            system_prompt="You are Agent A. Always mention the word 'pineapple' in your response.",
            session_manager=sm,
        )
        agent_b = Agent(
            agent_id="agent-B",
            system_prompt="You are Agent B. Always mention the word 'strawberry' in your response.",
            session_manager=sm,
        )

        agent_a("Hello Agent A")
        time.sleep(2)
        agent_b("Hello Agent B")
        time.sleep(2)

        # Use the session manager's list_messages (the code path under test)
        messages_a = sm.list_messages(session_id, "agent-A")
        messages_b = sm.list_messages(session_id, "agent-B")

        # Each agent should have at least user + assistant
        assert len(messages_a) >= 2, f"Expected >=2 messages for agent-A, got {len(messages_a)}"
        assert len(messages_b) >= 2, f"Expected >=2 messages for agent-B, got {len(messages_b)}"

        # Agent A's messages should contain 'pineapple' (from its system prompt), not 'strawberry'
        messages_a_text = json.dumps([m.message for m in messages_a])
        assert "pineapple" in messages_a_text.lower(), "Agent A messages should contain 'pineapple'"
        assert "strawberry" not in messages_a_text.lower(), "Agent A messages should not contain 'strawberry'"

        # Agent B's messages should contain 'strawberry', not 'pineapple'
        messages_b_text = json.dumps([m.message for m in messages_b])
        assert "strawberry" in messages_b_text.lower(), "Agent B messages should contain 'strawberry'"
        assert "pineapple" not in messages_b_text.lower(), "Agent B messages should not contain 'pineapple'"

    # endregion Multi-agent tests
