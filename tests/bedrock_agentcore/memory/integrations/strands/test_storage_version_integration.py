"""Integration tests for storage_version parameter.

These tests require AWS credentials and make actual API calls.
Run with: pytest -m integration tests/.../test_storage_version_integration.py -v
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


class TestStorageVersionIntegration:
    """Integration tests for storage_version parameter."""

    def test_v2_reduces_api_calls_simple_message(self, memory_id, region, unique_session_id):
        """Test that v2 reduces API calls for simple message."""
        from strands import Agent
        from strands.models import BedrockModel

        from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
        from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

        actor_id = "test-user"

        # Test v1
        config_v1 = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=f"v1-{unique_session_id}",
            actor_id=actor_id,
        )
        session_manager_v1 = AgentCoreMemorySessionManager(
            agentcore_memory_config=config_v1,
            region_name=region,
            storage_version="v1",
        )

        model = BedrockModel(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name=region
        )

        agent_v1 = Agent(
            model=model,
            session_manager=session_manager_v1,
            system_prompt="You are a helpful assistant. Keep responses very brief."
        )

        # Count API calls for v1
        counter_v1 = APICallCounter()
        original_create_event_v1 = session_manager_v1.memory_client.gmdp_client.create_event

        def counted_v1(*args, **kwargs):
            counter_v1.increment("create_event")
            return original_create_event_v1(*args, **kwargs)

        session_manager_v1.memory_client.gmdp_client.create_event = counted_v1
        agent_v1("Say hello briefly.")
        v1_calls = counter_v1.call_count

        # Test v2
        config_v2 = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=f"v2-{unique_session_id}",
            actor_id=actor_id,
        )
        session_manager_v2 = AgentCoreMemorySessionManager(
            agentcore_memory_config=config_v2,
            region_name=region,
            storage_version="v2",
        )

        agent_v2 = Agent(
            model=model,
            session_manager=session_manager_v2,
            system_prompt="You are a helpful assistant. Keep responses very brief."
        )

        # Count API calls for v2
        counter_v2 = APICallCounter()
        original_create_event_v2 = session_manager_v2.memory_client.gmdp_client.create_event

        def counted_v2(*args, **kwargs):
            counter_v2.increment("create_event")
            return original_create_event_v2(*args, **kwargs)

        session_manager_v2.memory_client.gmdp_client.create_event = counted_v2
        agent_v2("Say hello briefly.")
        v2_calls = counter_v2.call_count

        # v2 should have fewer API calls than v1
        assert v2_calls < v1_calls, f"v2 ({v2_calls}) should have fewer calls than v1 ({v1_calls})"
        # Expected: v1=5, v2=3
        assert v1_calls == 5, f"Expected v1=5 calls, got {v1_calls}"
        assert v2_calls == 3, f"Expected v2=3 calls, got {v2_calls}"

    def test_v2_multi_turn_preserves_context(self, memory_id, region, unique_session_id):
        """Test that v2 correctly preserves context across turns."""
        from strands import Agent
        from strands.models import BedrockModel

        from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
        from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

        actor_id = "test-user"
        session_id = f"v2-multi-{unique_session_id}"

        config = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=session_id,
            actor_id=actor_id,
        )

        model = BedrockModel(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name=region
        )

        # Turn 1: Tell the agent something to remember
        session_manager_1 = AgentCoreMemorySessionManager(
            agentcore_memory_config=config,
            region_name=region,
            storage_version="v2",
        )

        agent_1 = Agent(
            model=model,
            session_manager=session_manager_1,
            system_prompt="You are a helpful assistant. Remember what the user tells you."
        )

        agent_1("My favorite color is blue. Remember this.")

        # Turn 2: Create NEW agent instance with SAME session ID
        session_manager_2 = AgentCoreMemorySessionManager(
            agentcore_memory_config=config,
            region_name=region,
            storage_version="v2",
        )

        agent_2 = Agent(
            model=model,
            session_manager=session_manager_2,
            system_prompt="You are a helpful assistant. Remember what the user tells you."
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

    def test_v2_parses_messages_correctly(self, memory_id, region, unique_session_id):
        """Test that v2 format messages are correctly saved and loaded."""
        from strands import Agent
        from strands.models import BedrockModel

        from bedrock_agentcore.memory.client import MemoryClient
        from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
        from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

        actor_id = "test-user"
        session_id = f"v2-parse-{unique_session_id}"

        config = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=session_id,
            actor_id=actor_id,
        )

        model = BedrockModel(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name=region
        )

        session_manager = AgentCoreMemorySessionManager(
            agentcore_memory_config=config,
            region_name=region,
            storage_version="v2",
        )

        agent = Agent(
            model=model,
            session_manager=session_manager,
            system_prompt="You are a helpful assistant."
        )

        agent("Test message for v2 format.")

        # Verify stored format
        memory_client = MemoryClient(region_name=region)
        events = memory_client.list_events(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            max_results=10,
        )

        assert len(events) > 0, "Expected events to be saved"

        # Check that at least one event has v2 format
        found_v2_message = False
        found_v2_agent_state = False

        for event in events:
            for payload_item in event.get("payload", []):
                if "blob" in payload_item:
                    blob_data = json.loads(payload_item["blob"])
                    if isinstance(blob_data, dict):
                        if blob_data.get("_type") == "message":
                            found_v2_message = True
                        if blob_data.get("_type") == "agent_state":
                            found_v2_agent_state = True

        assert found_v2_message, "Expected v2 format message with _type marker"
        assert found_v2_agent_state, "Expected v2 format agent_state with _type marker"

    def test_v2_reduces_api_calls_with_tools(self, memory_id, region, unique_session_id):
        """Test that v2 reduces API calls when using tools."""
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

        model = BedrockModel(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name=region
        )

        # Test v1 with tools
        config_v1 = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=f"v1-tools-{unique_session_id}",
            actor_id=actor_id,
        )
        session_manager_v1 = AgentCoreMemorySessionManager(
            agentcore_memory_config=config_v1,
            region_name=region,
            storage_version="v1",
        )

        agent_v1 = Agent(
            model=model,
            session_manager=session_manager_v1,
            tools=[get_time, add_numbers],
            system_prompt="You are a helpful assistant. Use tools when asked. Keep responses brief."
        )

        counter_v1 = APICallCounter()
        original_create_event_v1 = session_manager_v1.memory_client.gmdp_client.create_event

        def counted_v1(*args, **kwargs):
            counter_v1.increment("create_event")
            return original_create_event_v1(*args, **kwargs)

        session_manager_v1.memory_client.gmdp_client.create_event = counted_v1
        agent_v1("What time is it? And what is 10 + 20?")
        v1_calls = counter_v1.call_count

        # Test v2 with tools
        config_v2 = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=f"v2-tools-{unique_session_id}",
            actor_id=actor_id,
        )
        session_manager_v2 = AgentCoreMemorySessionManager(
            agentcore_memory_config=config_v2,
            region_name=region,
            storage_version="v2",
        )

        agent_v2 = Agent(
            model=model,
            session_manager=session_manager_v2,
            tools=[get_time, add_numbers],
            system_prompt="You are a helpful assistant. Use tools when asked. Keep responses brief."
        )

        counter_v2 = APICallCounter()
        original_create_event_v2 = session_manager_v2.memory_client.gmdp_client.create_event

        def counted_v2(*args, **kwargs):
            counter_v2.increment("create_event")
            return original_create_event_v2(*args, **kwargs)

        session_manager_v2.memory_client.gmdp_client.create_event = counted_v2
        agent_v2("What time is it? And what is 10 + 20?")
        v2_calls = counter_v2.call_count

        # v2 should have fewer API calls than v1
        assert v2_calls < v1_calls, f"v2 ({v2_calls}) should have fewer calls than v1 ({v1_calls})"
        # Expected: v1=9, v2=5 (with 2 tool calls)
        assert v1_calls == 9, f"Expected v1=9 calls with tools, got {v1_calls}"
        assert v2_calls == 5, f"Expected v2=5 calls with tools, got {v2_calls}"
