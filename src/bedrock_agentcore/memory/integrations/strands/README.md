# Strands AgentCore Memory Examples

This directory contains comprehensive examples demonstrating how to use the Strands AgentCoreMemorySessionManager with Amazon Bedrock AgentCore Memory for persistent conversation storage and intelligent retrieval (Supports STM and LTM).

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
                "namespaces": ["/summaries/{actorId}/{sessionId}"]
            }
        },
        {
            "userPreferenceMemoryStrategy": {
                "name": "PreferenceLearner",
                "namespaces": ["/preferences/{actorId}"]
            }
        },
        {
            "semanticMemoryStrategy": {
                "name": "FactExtractor",
                "namespaces": ["/facts/{actorId}"]
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
        "/preferences/{actorId}": RetrievalConfig(
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
        "/preferences/{actorId}": RetrievalConfig(
            top_k=5,
            relevance_score=0.7
        ),
        "/facts/{actorId}": RetrievalConfig(
            top_k=10,
            relevance_score=0.3
        ),
        "/summaries/{actorId}/{sessionId}": RetrievalConfig(
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

### RetrievalConfig Parameters

- `top_k`: Number of top results to retrieve (default: 5)
- `relevance_score`: Minimum relevance threshold (0.0-1.0)

### Memory Strategies
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-strategies.html

1. **summaryMemoryStrategy**: Summarizes conversation sessions
2. **userPreferenceMemoryStrategy**: Learns and stores user preferences
3. **semanticMemoryStrategy**: Extracts and stores factual information

### Namespace Patterns

- `/preferences/{actorId}`: User-specific preferences
- `/facts/{actorId}`: User-specific facts
- `/summaries/{actorId}/{sessionId}`: Session-specific summaries


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
