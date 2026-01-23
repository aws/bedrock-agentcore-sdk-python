"""Integration tests for AgentCoreMemorySessionManager.

These tests require AWS credentials and make actual API calls.
Run with: pytest -m integration tests/.../test_session_manager_integration.py -v
Skip with: pytest -m "not integration"

To skip these tests in normal runs, they are marked with @pytest.mark.integration.
"""

import json
import os
import uuid

import pytest

# Skip all tests in this module if --run-integration is not provided
pytestmark = pytest.mark.integration


def get_memory_id():
    """Get Memory ID from environment."""
    return os.environ.get("MEMORY_ID")


def get_region():
    """Get AWS region from environment."""
    return os.environ.get("AWS_REGION", "us-west-2")


@pytest.fixture(scope="module")
def memory_id():
    """Get memory ID, skip if not available."""
    mid = get_memory_id()
    if not mid:
        pytest.skip("MEMORY_ID environment variable not set")
    return mid


@pytest.fixture(scope="module")
def region():
    """Get AWS region."""
    return get_region()


@pytest.fixture
def unique_session_id():
    """Generate unique session ID for each test."""
    return f"test-{uuid.uuid4().hex[:12]}"


class APICallCounter:
    """Count API calls to AgentCore Memory."""

    def __init__(self):
        self.call_count = 0
        self.call_details = []

    def reset(self):
        self.call_count = 0
        self.call_details = []

    def increment(self, method_name: str):
        self.call_count += 1
        self.call_details.append(method_name)


class TestOptimizedStorageIntegration:
    """Integration tests for optimized storage."""

    def test_batched_api_calls_simple_message(self, memory_id, region, unique_session_id):
        """Test that batched storage reduces API calls for simple message."""
        from strands import Agent
        from strands.models import BedrockModel

        from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
        from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

        actor_id = "test-user"

        config = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=unique_session_id,
            actor_id=actor_id,
        )
        session_manager = AgentCoreMemorySessionManager(
            agentcore_memory_config=config,
            region_name=region,
        )

        model = BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", region_name=region)

        agent = Agent(
            model=model,
            session_manager=session_manager,
            system_prompt="You are a helpful assistant. Keep responses very brief.",
        )

        # Count API calls
        counter = APICallCounter()
        original_create_event = session_manager.memory_client.gmdp_client.create_event

        def counted(*args, **kwargs):
            counter.increment("create_event")
            return original_create_event(*args, **kwargs)

        session_manager.memory_client.gmdp_client.create_event = counted
        agent("Say hello briefly.")
        api_calls = counter.call_count

        # Batched storage: 1 (session) + 1 (user msg+state) + 1 (assistant msg+state) = 3
        # AfterInvocation sync skips due to state hash tracking
        assert api_calls <= 4, f"Expected at most 4 API calls, got {api_calls}"

    def test_multi_turn_preserves_context(self, memory_id, region, unique_session_id):
        """Test that context is preserved across turns."""
        from strands import Agent
        from strands.models import BedrockModel

        from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
        from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

        actor_id = "test-user"
        session_id = f"multi-{unique_session_id}"

        config = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=session_id,
            actor_id=actor_id,
        )

        model = BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", region_name=region)

        # Turn 1: Tell the agent something to remember
        session_manager_1 = AgentCoreMemorySessionManager(
            agentcore_memory_config=config,
            region_name=region,
        )

        agent_1 = Agent(
            model=model,
            session_manager=session_manager_1,
            system_prompt="You are a helpful assistant. Remember what the user tells you.",
        )

        agent_1("My favorite color is blue. Remember this.")

        # Turn 2: Create NEW agent instance with SAME session ID
        session_manager_2 = AgentCoreMemorySessionManager(
            agentcore_memory_config=config,
            region_name=region,
        )

        agent_2 = Agent(
            model=model,
            session_manager=session_manager_2,
            system_prompt="You are a helpful assistant. Remember what the user tells you.",
        )

        # Verify messages were loaded
        assert len(agent_2.messages) >= 2, f"Expected at least 2 messages loaded, got {len(agent_2.messages)}"

        # Ask about the remembered information
        response = agent_2("What is my favorite color?")

        # Extract response text
        response_text = ""
        if response.message and response.message.get("content"):
            for block in response.message["content"]:
                if block.get("text"):
                    response_text = block["text"].lower()
                    break

        assert "blue" in response_text, f"Expected 'blue' in response, got: {response_text}"

    def test_messages_correctly_parsed(self, memory_id, region, unique_session_id):
        """Test that messages are correctly saved and loaded."""
        from strands import Agent
        from strands.models import BedrockModel

        from bedrock_agentcore.memory.client import MemoryClient
        from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
        from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

        actor_id = "test-user"
        session_id = f"parse-{unique_session_id}"

        config = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=session_id,
            actor_id=actor_id,
        )

        model = BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", region_name=region)

        session_manager = AgentCoreMemorySessionManager(
            agentcore_memory_config=config,
            region_name=region,
        )

        agent = Agent(model=model, session_manager=session_manager, system_prompt="You are a helpful assistant.")

        agent("Test message for format.")

        # Verify stored format
        memory_client = MemoryClient(region_name=region)
        events = memory_client.list_events(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            max_results=10,
        )

        assert len(events) > 0, "Expected events to be saved"

        # Check that at least one event has type markers
        found_message = False
        found_agent_state = False

        for event in events:
            for payload_item in event.get("payload", []):
                if "blob" in payload_item:
                    blob_data = json.loads(payload_item["blob"])
                    if isinstance(blob_data, dict):
                        if blob_data.get("_type") == "message":
                            found_message = True
                        if blob_data.get("_type") == "agent_state":
                            found_agent_state = True

        assert found_message, "Expected message with _type marker"
        assert found_agent_state, "Expected agent_state with _type marker"

    def test_batched_api_calls_with_tools(self, memory_id, region, unique_session_id):
        """Test that batched storage works with tool calls."""
        from datetime import datetime

        from strands import Agent, tool
        from strands.models import BedrockModel

        from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
        from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

        @tool
        def get_time() -> str:
            """Get the current time."""
            return datetime.now().strftime("%H:%M:%S")

        @tool
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        actor_id = "test-user"

        model = BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", region_name=region)

        config = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=f"tools-{unique_session_id}",
            actor_id=actor_id,
        )
        session_manager = AgentCoreMemorySessionManager(
            agentcore_memory_config=config,
            region_name=region,
        )

        agent = Agent(
            model=model,
            session_manager=session_manager,
            tools=[get_time, add_numbers],
            system_prompt="You are a helpful assistant. Use tools when asked. Keep responses brief.",
        )

        counter = APICallCounter()
        original_create_event = session_manager.memory_client.gmdp_client.create_event

        def counted(*args, **kwargs):
            counter.increment("create_event")
            return original_create_event(*args, **kwargs)

        session_manager.memory_client.gmdp_client.create_event = counted
        agent("What time is it? And what is 10 + 20?")
        api_calls = counter.call_count

        # With tools: more messages but still batched
        # Should be less than 10 calls (unbatched would be ~9+ calls per message)
        assert api_calls <= 7, f"Expected at most 7 API calls with tools, got {api_calls}"
