"""
Integration tests for AgentCore Memory Session Manager.

Run with: python -m pytest tests_integ/memory/integrations/test_session_manager.py -v
"""

import logging
import os
import time
import uuid

import pytest
from strands import Agent

from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

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
        memory = memory_client.create_memory(name=memory_name, description="Test STM memory for integration tests")
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
                        "namespaces": ["/summaries/{actorId}/{sessionId}"],
                    }
                },
                {
                    "userPreferenceMemoryStrategy": {
                        "name": "PreferenceLearner",
                        "namespaces": ["/preferences/{actorId}"],
                    }
                },
                {"semanticMemoryStrategy": {"name": "FactExtractor", "namespaces": ["/facts/{actorId}"]}},
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

        time.sleep(15)  # throttling
        response2 = agent("What is my name?")
        assert response2 is not None
        assert "John" in response2.message["content"][0]["text"]

    def test_session_manager_with_retrieval_config_adds_context(self, test_memory_ltm):
        """Test session manager with custom retrieval configuration."""
        config = AgentCoreMemoryConfig(
            memory_id=test_memory_ltm["id"],
            session_id=f"test-session-{int(time.time())}",
            actor_id=f"test-actor-{int(time.time())}",
            retrieval_config={"/preferences/{actorId}": RetrievalConfig(top_k=5, relevance_score=0.7)},
        )

        session_manager = AgentCoreMemorySessionManager(agentcore_memory_config=config, region_name=REGION)

        agent = Agent(system_prompt="You are a helpful assistant.", session_manager=session_manager)

        response1 = agent("I like sushi with tuna")
        assert response1 is not None
        logger.info("\nWaiting 90 seconds for memory extraction...")
        time.sleep(90)

        response2 = agent("What do I like to eat?")
        assert response2 is not None
        assert "sushi" in str(agent.messages)
        assert "<user_context>" in str(agent.messages)

    def test_multiple_namespace_retrieval_config(self, test_memory_ltm):
        """Test session manager with multiple namespace retrieval configurations."""
        config = AgentCoreMemoryConfig(
            memory_id=test_memory_ltm["id"],
            session_id=f"test-session-{int(time.time())}",
            actor_id=f"test-actor-{int(time.time())}",
            retrieval_config={
                "/preferences/{actorId}": RetrievalConfig(top_k=5, relevance_score=0.7),
                "/facts/{actorId}": RetrievalConfig(top_k=10, relevance_score=0.3),
                "/summaries/{actorId}/{sessionId}": RetrievalConfig(top_k=5, relevance_score=0.5),
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
        logger.info("\nWaiting 90 seconds for memory extraction...")
        time.sleep(90)

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
