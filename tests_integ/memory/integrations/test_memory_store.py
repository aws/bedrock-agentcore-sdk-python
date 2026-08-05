"""Live AgentCore Memory tests for the native Strands ``MemoryStore`` integration.

Requires ``MEMORY_PREPOPULATED_ID`` (a pre-provisioned memory with semantic and summary
strategies), ``BEDROCK_TEST_REGION``, and credentials allowed to invoke a Bedrock model.
Optionally set ``STRANDS_TEST_MODEL_ID`` to override the default test model. Long-term
extraction is eventually consistent, so these tests poll and may take several minutes.
"""

import asyncio
import copy
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest
from strands import Agent
from strands.memory import AddMessagesContext, MemoryEntry, MemoryManager
from strands.models import BedrockModel
from strands.types.content import Message

from bedrock_agentcore.memory.client import MemoryClient
from bedrock_agentcore.memory.integrations.strands.memorystore import (
    AgentCoreMemoryStore,
    create_agentcore_memory_stores,
)

REGION = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
MODEL_ID = os.environ.get("STRANDS_TEST_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
FACTS_NAMESPACE = "/facts/{actorId}/"
SUMMARY_NAMESPACE = "/summaries/{actorId}/{sessionId}/"


async def poll_for_records(
    search: Callable[[], Awaitable[list[MemoryEntry]]],
    timeout_seconds: int = 240,
) -> list[MemoryEntry]:
    """Poll eventual long-term extraction until records appear or time expires."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        results = await search()
        if results or time.monotonic() > deadline:
            return results
        await asyncio.sleep(10)


@pytest.fixture(scope="module")
def memory_client() -> Any:
    """Create the live ``MemoryClient`` this integration accepts."""
    return MemoryClient(region_name=REGION)


def wrapping_memory_client(source: Any, data_plane: Any) -> Any:
    """Copy a ``MemoryClient`` but vend a wrapped data-plane client.

    Args:
        source: Live client to copy region/telemetry settings from.
        data_plane: Data-plane object the copy should vend.

    Returns:
        A ``MemoryClient`` whose ``gmdp_client`` is ``data_plane``.
    """
    client = copy.copy(source)
    client.gmdp_client = data_plane
    return client


@pytest.fixture(scope="module")
def semantic_memory() -> dict[str, str]:
    """Use a pre-provisioned memory; this test must never create an untagged resource."""
    memory_id = os.environ.get("MEMORY_PREPOPULATED_ID")
    if not memory_id:
        pytest.skip("MEMORY_PREPOPULATED_ID is required for native Strands memory-store tests")
    return {"id": memory_id}


@pytest.mark.integration
class TestAgentCoreMemoryStore:
    """Store-level tests against the live AgentCore data plane."""

    async def test_write_idempotency_batching_and_recall(self, semantic_memory: Any, memory_client: Any) -> None:
        """Write one batched event, re-fire it idempotently, and recall extracted facts."""
        actor_id = f"batch-actor-{uuid.uuid4().hex}"
        session_id = f"batch-session-{uuid.uuid4().hex}"
        create_event_calls = 0

        class CountingDataPlane:
            """Count create-event calls while delegating to the live data plane."""

            def create_event(self, **kwargs: Any) -> dict[str, Any]:
                nonlocal create_event_calls
                create_event_calls += 1
                return cast(dict[str, Any], memory_client.gmdp_client.create_event(**kwargs))

            def retrieve_memory_records(self, **kwargs: Any) -> dict[str, Any]:
                return cast(dict[str, Any], memory_client.gmdp_client.retrieve_memory_records(**kwargs))

        store = AgentCoreMemoryStore(
            memory_id=semantic_memory["id"],
            actor_id=actor_id,
            session_id=session_id,
            namespace=FACTS_NAMESPACE,
            writable=True,
            extraction=True,
            client=wrapping_memory_client(memory_client, CountingDataPlane()),
        )
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "I am a pilot based in Denver and fly Cessnas."}]},
            {"role": "assistant", "content": [{"text": "Flying Cessnas out of Denver — nice."}]},
            {"role": "user", "content": [{"text": "I also play cello in my spare time."}]},
            {"role": "assistant", "content": [{"text": "A pilot and a cellist!"}]},
        ]
        context = AddMessagesContext(sequence_numbers=[0, 1, 2, 3])
        await store.add_messages(messages, context)
        assert create_event_calls == 1
        await store.add_messages(messages, context)
        assert create_event_calls == 2  # same token; the service accepts/deduplicates the re-fire

        results = await poll_for_records(lambda: store.search("what does the user do and where"))
        assert results, "No records surfaced before the AgentCore extraction timeout"
        joined = " ".join(result.content.lower() for result in results)
        assert any(term in joined for term in ("pilot", "denver", "cello", "cessna"))
        expected_namespace = FACTS_NAMESPACE.replace("{actorId}", actor_id)
        assert any(expected_namespace in (result.metadata or {}).get("_namespaces", []) for result in results), (
            f"Expected recalled _namespaces to contain {expected_namespace}"
        )

    async def test_extraction_mode_wire_passthrough_and_recall_only_guard(
        self, semantic_memory: Any, memory_client: Any
    ) -> None:
        """Prove live ``SKIP`` acceptance and direct recall-only write rejection."""
        captured: dict[str, Any] = {}

        class CapturingDataPlane:
            """Capture create-event parameters while delegating live calls."""

            def create_event(self, **kwargs: Any) -> dict[str, Any]:
                captured.update(kwargs)
                return cast(dict[str, Any], memory_client.gmdp_client.create_event(**kwargs))

            def retrieve_memory_records(self, **kwargs: Any) -> dict[str, Any]:
                return cast(dict[str, Any], memory_client.gmdp_client.retrieve_memory_records(**kwargs))

        identity = {
            "memory_id": semantic_memory["id"],
            "actor_id": f"skip-actor-{uuid.uuid4().hex}",
            "session_id": f"skip-session-{uuid.uuid4().hex}",
            "namespace": FACTS_NAMESPACE,
            "client": wrapping_memory_client(memory_client, CapturingDataPlane()),
        }
        writer = AgentCoreMemoryStore(
            **identity,
            writable=True,
            extraction=True,
            extraction_mode="SKIP",
        )
        await writer.add_messages(
            [{"role": "user", "content": [{"text": "Short-term only temporary data."}]}],
            AddMessagesContext(sequence_numbers=[0]),
        )
        assert captured["extractionMode"] == "SKIP"

        captured.clear()
        default_writer = AgentCoreMemoryStore(**identity, writable=True, extraction=True)
        await default_writer.add_messages(
            [{"role": "user", "content": [{"text": "Normal extraction event."}]}],
            AddMessagesContext(sequence_numbers=[1]),
        )
        assert "extractionMode" not in captured

        readonly = AgentCoreMemoryStore(**identity)
        assert readonly.writable is False
        with pytest.raises(ValueError, match="not writable"):
            await readonly.add_messages([{"role": "user", "content": [{"text": "x"}]}])

    async def test_exact_and_subtree_retrieval_fields_are_accepted_live(
        self, semantic_memory: Any, memory_client: Any
    ) -> None:
        """Exercise both AgentCore retrieval target arms against the service."""
        actor_id = f"read-actor-{uuid.uuid4().hex}"
        identity = {
            "memory_id": semantic_memory["id"],
            "actor_id": actor_id,
            "session_id": f"read-session-{uuid.uuid4().hex}",
            "client": memory_client,
        }
        exact = AgentCoreMemoryStore(**identity, namespace=FACTS_NAMESPACE)
        subtree = AgentCoreMemoryStore(**identity, namespace_path=f"/facts/{actor_id}")
        assert isinstance(await exact.search("anything"), list)
        assert isinstance(await subtree.search("anything"), list)

    async def test_direct_store_and_factory_work_with_memory_manager(
        self, semantic_memory: Any, memory_client: Any
    ) -> None:
        """Validate the direct primitive and factory output against real MemoryManager."""
        actor_id = f"manager-actor-{uuid.uuid4().hex}"
        session_id = f"manager-session-{uuid.uuid4().hex}"
        stores = create_agentcore_memory_stores(
            memory_id=semantic_memory["id"],
            actor_id=actor_id,
            session_id=session_id,
            namespaces=[{"namespace": FACTS_NAMESPACE}],
            extraction=True,
            client=memory_client,
        )
        manager = MemoryManager(stores=stores)
        assert len(stores) == 1 and stores[0].writable
        assert isinstance(await manager.search("anything"), list)

        direct = AgentCoreMemoryStore(
            memory_id=semantic_memory["id"],
            actor_id=f"direct-{uuid.uuid4().hex}",
            session_id=f"direct-session-{uuid.uuid4().hex}",
            namespace=FACTS_NAMESPACE,
            writable=True,
            extraction=True,
            client=memory_client,
        )
        await direct.add_messages(
            [{"role": "user", "content": [{"text": "I collect vinyl records."}]}],
            AddMessagesContext(sequence_numbers=[0]),
        )
        assert isinstance(await direct.search("what does the user collect"), list)


@pytest.mark.integration
async def test_session_scoped_namespace_drift(semantic_memory: Any, memory_client: Any) -> None:
    """A ``{sessionId}`` namespace does not leak records across sessions."""
    actor_id = f"drift-actor-{uuid.uuid4().hex}"
    session_a = f"drift-session-a-{uuid.uuid4().hex}"
    session_b = f"drift-session-b-{uuid.uuid4().hex}"
    writer = AgentCoreMemoryStore(
        memory_id=semantic_memory["id"],
        actor_id=actor_id,
        session_id=session_a,
        namespace=SUMMARY_NAMESPACE,
        writable=True,
        extraction=True,
        client=memory_client,
    )
    await writer.add_messages(
        [
            {"role": "user", "content": [{"text": "We are planning a spring trip to Japan."}]},
            {"role": "assistant", "content": [{"text": "Cherry blossom season is lovely."}]},
            {"role": "user", "content": [{"text": "Book a Kyoto ryokan for two nights."}]},
        ],
        AddMessagesContext(sequence_numbers=[0, 1, 2]),
    )
    store_a = AgentCoreMemoryStore(
        memory_id=semantic_memory["id"],
        actor_id=actor_id,
        session_id=session_a,
        namespace=SUMMARY_NAMESPACE,
        client=memory_client,
    )
    store_b = AgentCoreMemoryStore(
        memory_id=semantic_memory["id"],
        actor_id=actor_id,
        session_id=session_b,
        namespace=SUMMARY_NAMESPACE,
        client=memory_client,
    )
    from_a = await poll_for_records(lambda: store_a.search("What trip is planned?"))
    assert from_a, "No session-A summary surfaced before the AgentCore extraction timeout"
    assert await store_b.search("What trip is planned?") == []


@pytest.mark.integration
async def test_real_agent_memory_manager_round_trip(semantic_memory: Any, memory_client: Any) -> None:
    """Drive extraction through a real Strands agent and poll manager recall."""
    actor_id = f"e2e-actor-{uuid.uuid4().hex}"
    stores = create_agentcore_memory_stores(
        memory_id=semantic_memory["id"],
        actor_id=actor_id,
        session_id=f"e2e-session-{uuid.uuid4().hex}",
        namespaces=[{"namespace": FACTS_NAMESPACE}],
        extraction=True,
        client=memory_client,
    )
    manager = MemoryManager(stores=stores)
    agent = Agent(
        model=BedrockModel(region_name=REGION, model_id=MODEL_ID),
        system_prompt="Use long-term memory to personalize answers.",
        memory_manager=manager,
    )
    await agent.invoke_async("Remember this: my dog Pixel is a corgi.")
    await manager.flush()
    results = await poll_for_records(lambda: manager.search("What is the user's dog named?"))
    assert results, "No E2E records surfaced before the AgentCore extraction timeout"
    assert "pixel" in " ".join(result.content.lower() for result in results)
    expected_namespace = FACTS_NAMESPACE.replace("{actorId}", actor_id)
    assert any(expected_namespace in (result.metadata or {}).get("_namespaces", []) for result in results), (
        f"Expected recalled _namespaces to contain {expected_namespace}"
    )
