"""Expected-failure mypy fixture for intentionally absent optional methods."""

from typing import cast

from bedrock_agentcore.memory.integrations.strands.memorystore import AgentCoreMemoryStore
from bedrock_agentcore.memory.integrations.strands.memorystore.types import AgentCoreDataPlaneClient

client = cast(AgentCoreDataPlaneClient, object())
store = AgentCoreMemoryStore(
    memory_id="memory",
    actor_id="actor",
    session_id="session",
    namespace="/facts/{actorId}",
    client=client,
)
store.add("content")
store.initialize()
store.get_tools()
