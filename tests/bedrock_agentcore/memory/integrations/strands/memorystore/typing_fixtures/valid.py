"""Expected-success mypy consumer fixture for package-root exports."""

from typing import cast

from strands.memory import MemoryManager, MemoryStore
from typing_extensions import assert_type

from bedrock_agentcore.memory.client import MemoryClient
from bedrock_agentcore.memory.integrations.strands.memorystore import (
    AgentCoreEventSender,
    AgentCoreEventSenderConfig,
    AgentCoreMemoryStore,
    AgentCoreMemoryStoreConfig,
    create_agentcore_memory_stores,
)
from bedrock_agentcore.memory.integrations.strands.memorystore.types import (
    AgentCoreDataPlaneClient,
    CreateAgentCoreMemoryStoresInput,
)

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
# The SDK's own MemoryClient is accepted wherever a data-plane client is.
memory_client = cast(MemoryClient, object())
memory_client_store = AgentCoreMemoryStore(
    memory_id="memory",
    actor_id="actor",
    session_id="session",
    namespace="/facts/{actorId}",
    client=memory_client,
)
memory_client_stores = create_agentcore_memory_stores(
    memory_id="memory",
    actor_id="actor",
    session_id="session",
    namespaces=[{"namespace": "/facts/{actorId}"}],
    client=memory_client,
)
assert_type(memory_client_stores, list[MemoryStore])
memory_client_store_config: AgentCoreMemoryStoreConfig = {
    "memory_id": "memory",
    "actor_id": "actor",
    "session_id": "session",
    "namespace": "/facts/{actorId}",
    "client": memory_client,
}
memory_client_factory_input: CreateAgentCoreMemoryStoresInput = {
    "memory_id": "memory",
    "actor_id": "actor",
    "session_id": "session",
    "namespaces": [{"namespace": "/facts/{actorId}"}],
    "client": memory_client,
}
memory_client_sender_config: AgentCoreEventSenderConfig = {
    "client": memory_client,
    "memory_id": "memory",
    "actor_id": "actor",
    "session_id": "session",
}

assert protocol_store.name == store.name
assert sender_config["memory_id"] == store_config["memory_id"]
assert memory_client_store.name == memory_client_store_config["namespace"].strip("/").replace("/", "-")
assert memory_client_factory_input["memory_id"] == memory_client_sender_config["memory_id"]
