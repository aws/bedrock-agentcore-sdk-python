# Strands AgentCore MemoryStore

`AgentCoreMemoryStore` plugs AgentCore long-term memory directly into Strands' `MemoryManager` for
long-term recall and extraction. It requires `strands-agents>=1.46.0`.

## One namespace

A store is recall-only by default. Set `writable=True` on exactly one store when Strands should send
messages to AgentCore for server-side long-term extraction:

```python
import os

from strands import Agent
from strands.memory import MemoryManager

from bedrock_agentcore.memory.integrations.strands.memorystore import AgentCoreMemoryStore

store = AgentCoreMemoryStore(
    memory_id=os.environ["AGENTCORE_MEMORY_ID"],
    actor_id="demo-user",
    session_id="demo-session",
    namespace="/facts/{actorId}/",
    writable=True,
    extraction=True,
    region_name="us-east-1",
)
manager = MemoryManager(stores=[store])
agent = Agent(memory_manager=manager)
agent("Remember that I prefer window seats.")
```

`namespace` performs exact-prefix retrieval. Use `namespace_path` instead to search a namespace
subtree. The integration resolves `{actorId}` and `{sessionId}` client-side; substitute other
placeholders and malformed braces before constructing the store.

## Multiple namespaces

`create_agentcore_memory_stores` returns `list[MemoryStore]` for direct `MemoryManager` composition.
Each item is a concrete `AgentCoreMemoryStore`; the factory shares one boto3 client and prevents
duplicate writes by allowing at most one writer:

```python
import os

from strands.memory import IntervalTrigger, MemoryManager, MemoryMessageFilter

from bedrock_agentcore.memory.integrations.strands.memorystore import create_agentcore_memory_stores

stores = create_agentcore_memory_stores(
    memory_id=os.environ["AGENTCORE_MEMORY_ID"],
    actor_id="demo-user",
    session_id="demo-session",
    namespaces=[
        {
            "namespace": "/preferences/{actorId}/",
            "max_search_results": 5,
            "min_score": 0.7,
        },
        {
            "namespace": "/facts/{actorId}/",
            "max_search_results": 10,
            "min_score": 0.3,
        },
    ],
    extraction={
        "cadence": IntervalTrigger(turns=10),
        "filter": MemoryMessageFilter(exclude=["toolUse", "toolResult", "image"]),
    },
    region_name="us-east-1",
)
manager = MemoryManager(stores=stores)
```

With extraction enabled, the first namespace not explicitly marked `writable=False` becomes the
writer. Set `writable=True` on one namespace to choose it explicitly. Omit `extraction` or pass
`False` for recall-only stores.

## Search and write behavior

- Search defaults to 5 results. `min_score` enables client-side score filtering and over-fetches by
  a factor of 4 (configurable with `over_fetch_factor`); only the over-fetched `topK` is capped at 100.
- Returned metadata uses reserved keys `_id`, `_score`, `_namespaces`, and `_createdAt`.
- Writes preserve user/assistant roles, ignore blank and tool-only messages, and batch up to 50
  consecutive turns per AgentCore event by default. `max_turns_per_event` accepts any positive integer.
- `metadata_provider` returns scalar strings, finite numbers, or booleans. Strings pass through;
  other finite scalars use Python `json.dumps` formatting. `None`, arrays, objects, non-finite numbers,
  and values outside AgentCore's allowed character set are rejected locally.
- Direct `AgentCoreMemoryStore(...)` construction accepts `extraction_mode="SKIP"` to omit long-term
  extraction for its events. The multi-namespace factory intentionally does not expose this option.
- `add_messages()` is the supported write interface. The flat-string Strands `add()` API is not
  implemented because it loses role and turn information.

## Batching, cadence, and flush

Three separate controls determine write timing and cost:

1. **Batching is always on.** Each flush packs its role-tagged messages into as few `create_event`
   requests as `max_turns_per_event` allows.
2. **Cadence controls when buffered messages are dispatched across turns.** `extraction=True` uses
   Strands' default trigger. Pass an extraction config with an `IntervalTrigger` or another Strands
   trigger to tune cadence.
3. **`flush()` lets pending write attempts settle; it does not acknowledge durability or server-side
   extraction.** Strands 1.46 logs and swallows sender failures, rolls back its high-water mark, and
   retains the failed batch for a later retry.

Synchronous `agent(...)` invocations flush automatically. After async invocation or streaming, call
`await manager.flush()` at a lifecycle or shutdown boundary to let pending writes settle. Monitor logs
or telemetry for failures rather than treating `flush()` as proof that data was persisted. AgentCore's
server-side extraction remains eventually consistent, so newly written records may not be immediately
searchable.

Reuse one manager per `(actor_id, session_id)` while that session is active. Reuse keeps trigger state
and buffered turns alive, allowing a coarser cadence to reduce calls. The application owns manager
caching and eviction.

## Namespace and error contract

Recall works only when the query namespace matches the concrete namespace where AgentCore stored the
extracted record. Writes append to the shared `(memory_id, actor_id, session_id)` stream; the memory
resource's strategies decide which namespaces receive extracted records. That is why a store set must
have at most one writer.

- AgentCore resolves strategy placeholders at extraction time, but retrieval does not. The store
  resolves only `{actorId}` and `{sessionId}` and rejects remaining braces at construction.
- Match the namespace template used when provisioning the strategy. `namespace` queries one exact
  prefix; `namespace_path` queries a parent subtree.
- A namespace containing `{sessionId}` is session-scoped. Use a stable session id or actor-only
  namespace for cross-session recall.
- The store consumes an existing memory resource; it does not provision strategies or the resource.
- Retrieval failures propagate to `MemoryManager`, which applies its per-store partial-failure behavior.
- Sender failures remain buffered for retry and are logged by Strands rather than propagated by
  `flush()`.

The integration calls the boto3 `bedrock-agentcore` data-plane client directly. AWS credentials use
boto3's normal credential chain; no credentials are stored by the integration.
