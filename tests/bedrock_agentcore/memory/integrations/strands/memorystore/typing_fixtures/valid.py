"""Expected-success mypy consumer fixture for package-root exports."""

from typing import cast

from strands.memory import MemoryManager, MemoryStore
from typing_extensions import assert_type

from bedrock_agentcore.memory.integrations.strands.memorystore import (
    AgentCoreEventSender,
    AgentCoreEventSenderConfig,
    AgentCoreMemoryStore,
    AgentCoreMemoryStoreConfig,
    create_agentcore_memory_stores,
)
from bedrock_agentcore.memory.integrations.strands.memorystore.types import AgentCoreDataPlaneClient

client = cast(AgentCoreDataPlaneClient, object())
store = AgentCoreMemoryStore(
    memory_id="memory",
    actor_id="actor",
    session_id="session",
    namespace="/facts/{actorId}",
    client=client,
)
protocol_store: MemoryStore = store
MemoryManager(stores=[store])

stores = create_agentcore_memory_stores(
    memory_id="memory",
    actor_id="actor",
    session_id="session",
    namespaces=[{"namespace": "/facts/{actorId}"}],
    client=client,
)
assert_type(stores, list[MemoryStore])
MemoryManager(stores=stores)
assert_type(AgentCoreEventSender, type[AgentCoreEventSender])

sender_config: AgentCoreEventSenderConfig = {
    "client": client,
    "memory_id": "memory",
    "actor_id": "actor",
    "session_id": "session",
}
store_config: AgentCoreMemoryStoreConfig = {
    "memory_id": "memory",
    "actor_id": "actor",
    "session_id": "session",
    "namespace": "/facts/{actorId}",
}
assert protocol_store.name == store.name
assert sender_config["memory_id"] == store_config["memory_id"]
