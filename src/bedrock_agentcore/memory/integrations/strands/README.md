# Strands AgentCore Memory Examples

This directory documents the Strands `AgentCoreMemorySessionManager` for persistent conversation storage and the native `AgentCoreMemoryStore` integration for Strands-managed long-term memory (LTM).

## Quick Setup

```bash
pip install 'bedrock-agentcore[strands-agents]'
```

or to develop locally:
```bash
git clone https://github.com/aws/bedrock-agentcore-sdk-python.git
cd bedrock-agentcore-sdk-python
uv sync
source .venv/bin/activate
```

## Examples Overview

### 1. Short-Term Memory (STM)
Basic memory functionality for conversation persistence within a session.

### 2. Long-Term Memory (LTM)
Advanced memory with multiple strategies for user preferences, facts, and session summaries.

---

## Native Strands long-term memory stores

`AgentCoreMemoryStore` plugs AgentCore long-term memory directly into Strands' `MemoryManager`.
It complements `AgentCoreMemorySessionManager`: use the session manager for AgentCore-backed
conversation persistence (STM), and use memory stores for Strands-native long-term recall and
extraction. This integration requires `strands-agents>=1.46.0`.

### One namespace

A store is recall-only by default. Set `writable=True` on exactly one store when Strands should
send messages to AgentCore for server-side long-term extraction:

```python
from strands import Agent
from strands.memory import MemoryManager

from bedrock_agentcore.memory.integrations.strands import AgentCoreMemoryStore

store = AgentCoreMemoryStore(
    memory_id=MEM_ID,
    actor_id=ACTOR_ID,
    session_id=SESSION_ID,
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
subtree. The integration resolves `{actorId}` and `{sessionId}` client-side; other placeholders and
malformed braces must be substituted before constructing the store.

### Multiple namespaces

`create_agentcore_memory_stores` returns `list[MemoryStore]` for direct `MemoryManager` composition. Each
item is a concrete `AgentCoreMemoryStore`; the factory shares one boto3 client and prevents duplicate
writes by allowing at most one writer:

```python
from strands.memory import IntervalTrigger, MemoryManager, MemoryMessageFilter

from bedrock_agentcore.memory.integrations.strands import create_agentcore_memory_stores

stores = create_agentcore_memory_stores(
    memory_id=MEM_ID,
    actor_id=ACTOR_ID,
    session_id=SESSION_ID,
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
writer. Set `writable=True` on one namespace to choose it explicitly. Omit `extraction` (or pass
`False`) for recall-only stores.

### Search and write behavior

- Search defaults to 5 results. `min_score` enables client-side score filtering and over-fetches by
  a factor of 4 (configurable with `over_fetch_factor`); only the over-fetched `topK` is capped at 100.
- Returned metadata uses reserved keys `_id`, `_score`, `_namespaces`, and `_createdAt`.
- Writes preserve user/assistant roles, ignore blank and tool-only messages, and batch up to 50
  consecutive turns per AgentCore event by default. `max_turns_per_event` accepts any positive integer.
- `metadata_provider` returns scalar strings, finite numbers, or booleans. Values are serialized with
  ECMAScript-compatible scalar text before AgentCore's value-character validation; `None`, arrays,
  objects, non-finite numbers, and strings outside `[a-zA-Z0-9\s._:/=+@-]` are rejected locally.
- Direct `AgentCoreMemoryStore(...)` construction accepts `extraction_mode="SKIP"` to omit long-term
  extraction for its events. The multi-namespace factory intentionally does not expose this option.
- `add_messages()` is the supported write interface. The flat-string Strands `add()` API is not
  implemented because it loses role and turn information.

### Batching, cadence, and flush

Three separate controls determine write timing and cost:

1. **Batching is always on.** Each flush packs its role-tagged messages into as few `create_event`
   requests as `max_turns_per_event` allows. A turn with several messages is not one request per message.
2. **Cadence controls when buffered messages are dispatched across turns.** `extraction=True` uses
   Strands' default trigger. Pass an extraction config with an `IntervalTrigger` (or another Strands
   trigger) to tune cadence. Cadence only batches across invocations when the same `MemoryManager` is
   reused and the application does not flush after every turn.
3. **`flush()` is the durability boundary, not a server-extraction barrier.** It waits for in-flight
   client `create_event` calls to settle. AgentCore's server-side extraction remains eventually
   consistent, so newly written long-term records may not be immediately searchable.

A synchronous `agent(...)` invocation automatically flushes at its boundary in Strands 1.46. Async
entry points do not: after `await agent.invoke_async(...)` or async streaming, explicitly call
`await manager.flush()` at the application boundary where writes must be durable.

Reuse one manager per `(actor_id, session_id)` while that session is active. Reuse keeps trigger state
and buffered turns alive, allowing a coarser cadence to reduce calls. Flushing after every invocation
is the durable default (roughly one batched event per turn); flushing less often can lower write cost
but risks losing the unflushed tail if the process or runtime is reclaimed. The application owns
manager caching and eviction.

### Namespace and error contract

Recall works only when the query namespace matches the concrete namespace where AgentCore stored the
extracted record. Writes do not target the store's read namespace: `create_event` appends to the shared
`(memory_id, actor_id, session_id)` stream, and the resource's strategies decide which namespaces receive
extracted records. That is why a store set must have at most one writer.

- AgentCore resolves strategy placeholders at extraction/write time, but retrieval does not resolve
  placeholders. The store resolves only `{actorId}` and `{sessionId}` and rejects remaining braces at
  construction. Pre-resolve values such as `{memoryStrategyId}` yourself.
- Match the namespace template used when provisioning the strategy. `namespace` queries one exact
  prefix; `namespace_path` queries a parent subtree.
- A namespace containing `{sessionId}` is session-scoped. A store for session B cannot recall records
  extracted under session A; use a stable session id or actor-only namespace for cross-session recall.
- The store consumes an existing memory resource. Strategy definitions, expiry, encryption, indexed keys,
  and record streaming are control-plane concerns configured separately; the store does not provision them.
- Writes and extraction are eventually consistent. A successful flush proves client writes settled,
  not that extraction is complete.
- Write failures propagate so Strands can roll back and retry the buffered batch. Retrieval failures
  propagate to `MemoryManager`, which applies its normal per-store partial-failure behavior.

The integration calls the boto3 `bedrock-agentcore` data-plane client directly. AWS credentials use
boto3's normal credential chain; no credentials are stored by the integration.

---

## Short-Term Memory Example

### Basic Setup

```python
import uuid
import boto3
from datetime import date
from strands import Agent
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
```

### Create a Basic Memory

```python
client = MemoryClient(region_name="us-east-1")
basic_memory = client.create_memory(
    name="BasicTestMemory",
    description="Basic memory for testing short-term functionality"
)
print(basic_memory.get('id'))
```

### Configure and Use Agent

```python
MEM_ID = basic_memory.get('id')
ACTOR_ID = "actor_id_test_%s" % datetime.now().strftime("%Y%m%d%H%M%S")
SESSION_ID = "testing_session_id_%s" % datetime.now().strftime("%Y%m%d%H%M%S")


# Configure memory
agentcore_memory_config = AgentCoreMemoryConfig(
    memory_id=MEM_ID,
    session_id=SESSION_ID,
    actor_id=ACTOR_ID
)

# Create session manager
session_manager = AgentCoreMemorySessionManager(
    agentcore_memory_config=agentcore_memory_config,
    region_name="us-east-1"
)

# Create agent
agent = Agent(
    system_prompt="You are a helpful assistant. Use all you know about the user to provide helpful responses.",
    session_manager=session_manager,
)
```

### Example Conversation

```python
agent("I like sushi with tuna")
# Agent remembers this preference

agent("I like pizza")
# Agent acknowledges both preferences

agent("What should I buy for lunch today?")
# Agent suggests options based on remembered preferences
```

---

## Long-Term Memory Example

### Create LTM Memory with Strategies

```python
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from datetime import datetime

# Create comprehensive memory with all built-in strategies
client = MemoryClient(region_name="us-east-1")
comprehensive_memory = client.create_memory_and_wait(
    name="ComprehensiveAgentMemory",
    description="Full-featured memory with all built-in strategies",
    strategies=[
        {
            "summaryMemoryStrategy": {
                "name": "SessionSummarizer",
                "namespaceTemplates": ["/summaries/{actorId}/{sessionId}/"]
            }
        },
        {
            "userPreferenceMemoryStrategy": {
                "name": "PreferenceLearner",
                "namespaceTemplates": ["/preferences/{actorId}/"]
            }
        },
        {
            "semanticMemoryStrategy": {
                "name": "FactExtractor",
                "namespaceTemplates": ["/facts/{actorId}/"]
            }
        }
    ]
)
MEM_ID = comprehensive_memory.get('id')
ACTOR_ID = "actor_id_test_%s" % datetime.now().strftime("%Y%m%d%H%M%S")
SESSION_ID = "testing_session_id_%s" % datetime.now().strftime("%Y%m%d%H%M%S")

```

### Single Namespace Retrieval

```python
config = AgentCoreMemoryConfig(
    memory_id=MEM_ID,
    session_id=SESSION_ID,
    actor_id=ACTOR_ID,
    retrieval_config={
        "/preferences/{actorId}/": RetrievalConfig(
            top_k=5,
            relevance_score=0.7
        )
    }
)
session_manager = AgentCoreMemorySessionManager(config, region_name='us-east-1')
ltm_agent = Agent(session_manager=session_manager)
```

### Multiple Namespace Retrieval

```python
config = AgentCoreMemoryConfig(
    memory_id=MEM_ID,
    session_id=SESSION_ID,
    actor_id=ACTOR_ID,
    retrieval_config={
        "/preferences/{actorId}/": RetrievalConfig(
            top_k=5,
            relevance_score=0.7
        ),
        "/facts/{actorId}/": RetrievalConfig(
            top_k=10,
            relevance_score=0.3
        ),
        "/summaries/{actorId}/{sessionId}/": RetrievalConfig(
            top_k=5,
            relevance_score=0.5
        )
    }
)
session_manager = AgentCoreMemorySessionManager(config, region_name='us-east-1')
agent_with_multiple_namespaces = Agent(session_manager=session_manager)
```

---

## Large Payload example processing an Image using the [strands_tools](https://github.com/strands-agents/tools) library

### Agent with Image Processing

```python
from strands import Agent, tool
from strands_tools import generate_image, image_reader

ACTOR_ID = "actor_id_test_%s" % datetime.now().strftime("%Y%m%d%H%M%S")
SESSION_ID = "testing_session_id_%s" % datetime.now().strftime("%Y%m%d%H%M%S")

config = AgentCoreMemoryConfig(
    memory_id=MEM_ID,
    session_id=SESSION_ID,
    actor_id=ACTOR_ID,
)
session_manager = AgentCoreMemorySessionManager(config, region_name='us-east-1')
agent_with_tools = Agent(
    tools=[image_reader],
    system_prompt="You will be provided with a filesystem path to an image. Describe the image in detail.",
    session_manager=session_manager,
    agent_id='my_test_agent_id'
)
# Use with image
result = agent_with_tools("/path/to/image.png")
```

---

## Key Configuration Options

### AgentCoreMemoryConfig Parameters

- `memory_id`: ID of the Bedrock AgentCore Memory resource
- `session_id`: Unique identifier for the conversation session
- `actor_id`: Unique identifier for the user/actor
- `retrieval_config`: Dictionary mapping namespaces to RetrievalConfig objects
- `batch_size`: Number of messages to buffer before sending to AgentCore Memory (1-100, default: 1). A value of 1 sends immediately (no batching).
- `default_metadata`: Optional dictionary of key-value metadata to attach to every message event. Maximum 15 total keys per event (including internal keys). Example: `{"location": {"stringValue": "NYC"}}`
- `metadata_provider`: Optional callable returning a metadata dictionary. Called at each event creation for dynamic values (e.g., traceId). Merged after `default_metadata`.

### RetrievalConfig Parameters

- `top_k`: Number of top results to retrieve (default: 5)
- `relevance_score`: Minimum relevance threshold (0.0-1.0)

### Memory Strategies
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-strategies.html

1. **summaryMemoryStrategy**: Summarizes conversation sessions
2. **userPreferenceMemoryStrategy**: Learns and stores user preferences
3. **semanticMemoryStrategy**: Extracts and stores factual information

### Namespace Patterns

- `/preferences/{actorId}/`: User-specific preferences
- `/facts/{actorId}/`: User-specific facts
- `/summaries/{actorId}/{sessionId}/`: Session-specific summaries


---

## Event Metadata

You can attach custom key-value metadata to every message event. This is useful for tagging
conversations with contextual information (e.g., location, project, case type) that can later
be used to filter events with `list_events`.

### Default Metadata (applied to all messages)

```python
config = AgentCoreMemoryConfig(
    memory_id=MEM_ID,
    session_id=SESSION_ID,
    actor_id=ACTOR_ID,
    default_metadata={
        "project": "atlas",
        "env": "production",
    },
)
session_manager = AgentCoreMemorySessionManager(config, region_name='us-east-1')
agent = Agent(session_manager=session_manager)
agent("Hello!")  # This event will have project=atlas and env=production metadata
```

> Plain strings are auto-wrapped to `{"stringValue": "..."}`. The explicit form
> `{"project": {"stringValue": "atlas"}}` also works.

### Dynamic Metadata (metadata_provider)

For values that change per invocation (e.g., traceId for Langfuse), use `metadata_provider` —
a callable invoked at each event creation:

```python
from langfuse.decorators import langfuse_context

def get_trace_metadata():
    return {"traceId": langfuse_context.get_current_trace_id() or ""}

config = AgentCoreMemoryConfig(
    memory_id=MEM_ID,
    session_id=SESSION_ID,
    actor_id=ACTOR_ID,
    metadata_provider=get_trace_metadata,
)
session_manager = AgentCoreMemorySessionManager(config, region_name='us-east-1')
agent = Agent(session_manager=session_manager)
agent("Hello!")  # Event gets the current traceId automatically
```

### Per-call Metadata

You can also pass metadata on individual `create_message` calls. Per-call metadata is merged
with `default_metadata` and `metadata_provider` (per-call values override both for the same key):

```python
session_manager.create_message(
    session_id, agent_id, message,
    metadata={"priority": "high"},
)
```

> **Note:** The API allows a maximum of 15 metadata key-value pairs per event.
> The keys `stateType` and `agentId` are reserved for internal use.

---

## Message Batching

When `batch_size` is greater than 1, messages are buffered in memory and sent to AgentCore Memory
in a single API call once the buffer reaches the configured size. This reduces the number of API
requests in high-throughput conversations.

> **Important:** When using `batch_size > 1`, you **must** use a `with` block or call `close()`
> when the session is complete. Otherwise, any buffered messages that have not yet reached the
> batch threshold will be lost.

### Recommended: Context Manager

```python
config = AgentCoreMemoryConfig(
    memory_id=MEM_ID,
    session_id=SESSION_ID,
    actor_id=ACTOR_ID,
    batch_size=10,  # Buffer up to 10 messages before sending
)

# The `with` block guarantees all buffered messages are flushed on exit
with AgentCoreMemorySessionManager(config, region_name='us-east-1') as session_manager:
    agent = Agent(
        system_prompt="You are a helpful assistant.",
        session_manager=session_manager,
    )
    agent("Hello!")
    agent("Tell me about AWS")
# All remaining buffered messages are automatically flushed here
```

### Alternative: Explicit close()

If you cannot use a `with` block, call `close()` manually:

```python
session_manager = AgentCoreMemorySessionManager(config, region_name='us-east-1')
try:
    agent = Agent(
        system_prompt="You are a helpful assistant.",
        session_manager=session_manager,
    )
    agent("Hello!")
finally:
    session_manager.close()  # Flush any remaining buffered messages
```

---

## Important Notes

### Session Management
- Only **one** agent per session is currently supported
- Creating multiple agents with the same session will show a warning

### Memory Types
- **STM (Short-Term Memory)**: Basic conversation persistence within a session
- **LTM (Long-Term Memory)**: Advanced memory with multiple strategies for learning user preferences, facts, and summaries

### Best Practices
- Use unique `session_id` for each conversation
- Use consistent `actor_id` for the same user across sessions
- Configure appropriate `relevance_score` thresholds for your use case
- Test with different `top_k` values to optimize retrieval performance
- When using `batch_size > 1`, always use a `with` block or call `close()` to ensure buffered messages are flushed before the session ends
