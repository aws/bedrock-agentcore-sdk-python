# Bedrock AgentCore Memory SDK

High-level Python SDK for AWS Bedrock AgentCore Memory service with flexible conversation handling and complete branch management.

## Key Features

### Flexible Conversation API
- Save any number of messages in a single call
- Support for USER, ASSISTANT, TOOL, OTHER roles
- Natural conversation flow representation

### Complete Branch Management
- List all branches in a session
- Navigate specific branches
- Get conversation tree structure
- Build context from any branch
- Continue conversations in existing branches

### Simplified Memory Operations
- Semantic search with vector store
- Automatic namespace handling
- Polling helpers for async operations

### LLM Integration Support
- Callback pattern for any LLM (Bedrock, OpenAI, etc.)
- Separated retrieve/generate/save pattern for flexibility
- Complete conversation turn in one method call


## Quick Start

```python
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.constants import StrategyType

client = MemoryClient()

# Create memory with strategies
# MemoryStrategies determine how memory records are extracted from conversations
memory = client.create_memory_and_wait(
    name="MyAgentMemory",
    strategies=[{
        StrategyType.SEMANTIC.value: {
            "name": "FactExtractor",
            "namespaces": ["/food/{actorId}"]
        }
    }]
)

# Save conversations, which will be used for memory extraction (if memory strategies are configured when calling create_memory)
event = client.create_event(
    memory_id=memory['id'],
    actor_id="user-123",
    session_id="session-456",
    messages=[
        ("I love eating apples and cherries", "USER"),
        ("Apples are very good.", "ASSISTANT"),
        ("What is your favorite thing about apples", "USER"),
        ("I enjoy their flavor -- and their nutritional benefits", "ASSISTANT")
    ]
)

# Then after some time has passed and memory records are extracted, you can do
memory_records = client.retrieve_memories(
    memory_id=memory['id'],
    namespace="/food/user-123",
    query="what food does the user like"
)

# Or if you have multiple namespaces (say you have multiple users (denoted by actor_id)) and want to search across all of them:
memory_records = client.retrieve_memories(
    memory_id=memory['id'],
    namespace="/", # we can use any prefix of the namespace that we defined in create_memory_and_wait
    query="Food"
)

```

## Core Usage Examples

### Natural Conversation Flow

```python
# Multiple user messages, tool usage, flexible patterns
event = client.create_event(
    memory_id=memory_id,
    actor_id=actor_id,
    session_id=session_id,
    messages=[
        ("I need help with my order", "USER"),
        ("Order #12345", "USER"),
        ("Let me look that up", "ASSISTANT"),
        ("lookup_order('12345')", "TOOL"),
        ("Found it! Your order ships tomorrow.", "ASSISTANT")
    ]
)
```

### Branch Management

```python
# Create branches for different scenarios
branch = client.fork_conversation(
    memory_id=memory_id,
    actor_id=actor_id,
    session_id=session_id,
    root_event_id=event_id,
    branch_name="premium-option",
    new_messages=[
        ("What about expedited shipping?", "USER"),
        ("I can upgrade you to overnight delivery for $20", "ASSISTANT")
    ]
)

# Navigate branches
branches = client.list_branches(memory_id, actor_id, session_id)
events = client.list_branch_events(
    memory_id=memory_id,
    actor_id=actor_id,
    session_id=session_id,
    branch_name="premium-option"
)
```

### LLM Integration Patterns

#### Pattern 1: Callback-based (Simple cases)

```python
def my_llm(user_input: str, memories: List[Dict]) -> str:
    # Your LLM logic here
    context = "\n".join([m['content']['text'] for m in memories])
    # Call Bedrock, OpenAI, etc.
    return "AI response based on context"

memories, response, event = client.process_turn_with_llm(
    memory_id=memory_id,
    actor_id="user-123",
    session_id="session-456",
    user_input="What did we discuss?",
    llm_callback=my_llm,
    retrieval_namespace="support/facts/{sessionId}"
)
```

#### Pattern 2: Separated calls (More control)

```python
# Step 1: Retrieve
memories = client.retrieve_memories(
    memory_id=memory_id,
    namespace="support/facts/{sessionId}",
    query="previous discussion"
)

# Step 2: Your LLM logic
response = your_llm_logic(user_input, memories)

# Step 3: Save
event = client.create_event(
    memory_id=memory_id,
    actor_id="user-123",
    session_id="session-456",
    messages=[(user_input, "USER"), (response, "ASSISTANT")]
)
```

### Environment Variables

- AGENTCORE_MEMORY_ROLE_ARN - IAM role for memory execution
- AGENTCORE_CONTROL_ENDPOINT - Override control plane endpoint
- AGENTCORE_DATA_ENDPOINT - Override data plane endpoint

### Best Practices

- Separate retrieval and storage: Use retrieve_memories() and create_event() as separate steps
- Wait for extraction: Use wait_for_memories() after creating events
- Handle service errors: Retry on ServiceException errors
- Use branches: Create branches for different scenarios or A/B testing
