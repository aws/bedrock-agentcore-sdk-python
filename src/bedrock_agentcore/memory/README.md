# Bedrock AgentCore Memory SDK

High-level Python SDK for AWS Bedrock AgentCore Memory service with streamlined session management and flexible
conversation handling.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
  - [Installation](#installation)
  - [Authentication](#authentication)
  - [Environment Variables](#environment-variables)
- [Recommended Classes](#recommended-classes)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Enhanced LLM Integration with Memory Context](#enhanced-llm-integration-with-memory-context)
  - [Natural Conversation Flow](#natural-conversation-flow)
  - [Branch Management](#branch-management)
  - [Session and Actor Management](#session-and-actor-management)
  - [Memory Record Management](#memory-record-management)
  - [Event Management with Metadata](#event-management-with-metadata)
  - [Alternative Pattern: Separated Operations](#alternative-pattern-separated-operations)
- [Error Handling](#error-handling)
  - [Common Exceptions](#common-exceptions)
  - [Best Practices for Error Handling](#best-practices-for-error-handling)
- [Migration from MemoryClient](#migration-from-memoryclient)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

The Bedrock AgentCore Memory SDK provides a comprehensive solution for managing conversational AI memory with both short-term (conversational events) and long-term (semantic memory) storage capabilities. The SDK is designed around three main components:

### Core Components

1. **MemorySessionManager** - The primary interface for managing multiple sessions and actors
2. **MemorySession** - Session-scoped interface that simplifies operations by automatically handling memory_id, actor_id, and session_id parameters
3. **MemoryClient** - Legacy client interface (still supported but not recommended for new projects)

### Architecture

The memory system operates on a hierarchical structure:

- **Memory** - Top-level container for all data
- **Actor** - Represents individual users or entities
- **Session** - Conversation contexts within an actor
- **Events** - Individual conversation turns or actions
- **Branches** - Alternative conversation paths for A/B testing or exploration

## Setup

### Installation

Install the Bedrock AgentCore SDK using pip:

```bash
pip install bedrock-agentcore
```

### Authentication

The SDK uses AWS credentials for authentication. Ensure you have one of the following configured:

1. **AWS CLI credentials** (recommended for development):

   ```bash
   aws configure
   ```
2. **Environment variables**:

   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-east-1
   ```
3. **IAM roles** (recommended for production):

   - EC2 instance roles
   - ECS task roles
   - Lambda execution roles
4. **AWS credentials file**:

   ```ini
   [default]
   aws_access_key_id = your_access_key
   aws_secret_access_key = your_secret_key
   region = us-east-1
   ```

### Environment Variables

The following environment variables can be used to configure the SDK:

- `AGENTCORE_MEMORY_ROLE_ARN` - IAM role for memory execution (legacy)
- `AGENTCORE_CONTROL_ENDPOINT` - Override control plane endpoint
- `AGENTCORE_DATA_ENDPOINT` - Override data plane endpoint
- `AWS_REGION` - AWS region (e.g., us-east-1)
- `AWS_DEFAULT_REGION` - Alternative AWS region variable (e.g., us-east-1)

**Region Resolution Order:**
The SDK resolves the AWS region in the following priority order:
1. `region_name` parameter passed to `MemorySessionManager`
2. Region from `boto3_session` if provided
3. `AWS_REGION` environment variable
4. `boto3.Session().region_name` (which checks `AWS_DEFAULT_REGION` and AWS config)
5. Default fallback: `us-west-2`

## Recommended Classes

### MemorySessionManager (Recommended)

The primary interface for managing conversational AI sessions with both short-term (conversational events) and
long-term (semantic memory) storage. Provides a clean, session-oriented API for memory operations.

### MemorySession (Recommended)

Session-scoped interface that simplifies operations by automatically handling memory_id, actor_id, and session_id parameters.

### MemoryClient (Legacy)

The original client interface. While still supported, we recommend migrating to MemorySessionManager for new projects.

## Key Features

### Streamlined Session Management

- Session-scoped operations with automatic parameter handling
- Create MemorySession instances for simplified API calls
- Built-in actor and session tracking

### Flexible Conversation API

- Save any number of messages in a single call with `add_turns()`
- Support for USER, ASSISTANT, TOOL, OTHER roles via `ConversationalMessage`
- Support for binary data via `BlobMessage`
- Natural conversation flow representation

### Complete Branch Management

- List all branches in a session
- Fork conversations from specific events
- Navigate specific branches with simplified API
- Build context from any branch

### Enhanced LLM Integration

- Built-in `process_turn_with_llm()` method for complete conversation turns
- Callback pattern for any LLM (Bedrock, OpenAI, etc.)
- Automatic memory retrieval, LLM processing, and response storage
- Flexible retrieval configuration with namespace templating

### Simplified Memory Operations

- Semantic search with `search_long_term_memories()`
- Automatic namespace handling with template variables
- List and manage memory records
- Actor and session management

## Quick Start

```python
from bedrock_agentcore.memory import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

# Initialize the session manager
manager = MemorySessionManager(
    memory_id="your-memory-id",  # Use existing memory id
    region_name="us-east-1"
)

# Create a session for a specific actor
session = manager.create_memory_session(
    actor_id="user-123",
    session_id="session-456"  # Optional - will generate UUID if not provided
)

# Add conversation turns
session.add_turns([
    ConversationalMessage("I love eating apples and cherries", MessageRole.USER),
    ConversationalMessage("Apples are very good for you!", MessageRole.ASSISTANT),
    ConversationalMessage("What's your favorite thing about apples?", MessageRole.USER),
    ConversationalMessage("I enjoy their flavor and nutritional benefits", MessageRole.ASSISTANT)
])

# Search long-term memories (after memory extraction has occurred)
memories = session.search_long_term_memories(
    query="what food does the user like",
    namespace_prefix="/food/user-123",
    top_k=5
)

# Or search across multiple users
memories = manager.search_long_term_memories(
    query="Food preferences",
    namespace_prefix="/food/",  # Search all food-related memories
    top_k=10
)
```

## Usage

### Enhanced LLM Integration with Memory Context

```python
from bedrock_agentcore.memory.constants import RetrievalConfig

def my_llm(user_input: str, memories: List[Dict]) -> str:
    # Format context from retrieved memories
    context = "\n".join([
        m.get('content', {}).get('text', '')
        for m in memories
    ])

    # Call your LLM (Bedrock, OpenAI, etc.)
    # This is just an example - use your actual LLM integration
    response = f"Based on our previous discussions about {context}, here's my response to: {user_input}"
    return response

# Configure memory retrieval with multiple namespaces
retrieval_config = {
    "support/facts/{sessionId}": RetrievalConfig(top_k=5, relevance_score=0.3),
    "user/preferences/{actorId}": RetrievalConfig(top_k=3, relevance_score=0.5)
}

# Process complete conversation turn with automatic memory integration
memories, response, event = session.process_turn_with_llm(
    user_input="What did we discuss about my preferences?",
    llm_callback=my_llm,
    retrieval_config=retrieval_config
)

print(f"Retrieved {len(memories)} relevant memories")
print(f"LLM Response: {response}")
print(f"Stored event ID: {event.event_id}")
```

### Natural Conversation Flow

```python
from bedrock_agentcore.memory.constants import ConversationalMessage, BlobMessage, MessageRole

# Multiple message types in a single turn
session.add_turns([
    ConversationalMessage("I need help with my order", MessageRole.USER),
    ConversationalMessage("Order #12345", MessageRole.USER),
    BlobMessage({"image_data": "base64_encoded_receipt"}),  # Binary data
    ConversationalMessage("Let me look that up", MessageRole.ASSISTANT),
    ConversationalMessage("lookup_order('12345')", MessageRole.TOOL),
    ConversationalMessage("Found it! Your order ships tomorrow.", MessageRole.ASSISTANT)
])
```

### Branch Management

```python
# Get conversation history
turns = session.get_last_k_turns(k=3)
print(f"Last 3 conversation turns: {len(turns)}")

# Fork conversation for alternative scenario
branch_event = session.fork_conversation(
    root_event_id="event-123",
    branch_name="premium-option",
    messages=[
        ConversationalMessage("What about expedited shipping?", MessageRole.USER),
        ConversationalMessage("I can upgrade you to overnight delivery for $20", MessageRole.ASSISTANT)
    ]
)

# List all branches in the session
branches = session.list_branches()
for branch in branches:
    print(f"Branch: {branch.name}, Events: {branch.event_count}")

# Get events from specific branch
branch_events = session.list_events(branch_name="premium-option")
```

### Session and Actor Management

```python
# Manager-level operations
actors = manager.list_actors()
print(f"Found {len(actors)} actors in memory")

# Actor-specific operations
actor = session.get_actor()
actor_sessions = actor.list_sessions()
print(f"Actor has {len(actor_sessions)} sessions")

# Create multiple sessions for the same actor
session2 = manager.create_memory_session(
    actor_id="user-123",
    session_id="session-789"
)
```

### Memory Record Management

```python
# List all memory records in a namespace
records = session.list_long_term_memory_records(
    namespace_prefix="/user/preferences/user-123",
    max_results=20
)

# Get specific memory record
record = session.get_memory_record("record-id-123")
print(f"Record content: {record.content}")

# Delete memory record
session.delete_memory_record("record-id-123")
```

### Event Management with Metadata

Events can now be managed by defining custom metadata.

Learn more here!: [Working example](src/bedrock_agentcore/memory/metadata-workflow.ipynb)

### Alternative Pattern: Separated Operations

```python
# For more control, you can separate the steps:

# Step 1: Retrieve relevant memories
memories = session.search_long_term_memories(
    query="previous discussion",
    namespace_prefix="support/facts/session-456",
    top_k=5
)

# Step 2: Process with your LLM
user_input = "What did we discuss?"
response = your_llm_logic(user_input, memories)

# Step 3: Save the conversation
event = session.add_turns([
    ConversationalMessage(user_input, MessageRole.USER),
    ConversationalMessage(response, MessageRole.ASSISTANT)
])
```

## Error Handling

### Common Exceptions

The SDK raises specific exceptions for different error conditions:

```python
from bedrock_agentcore.memory import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

try:
    manager = MemorySessionManager(
        memory_id="your-memory-id",
        region_name="us-east-1"
    )

    session = manager.create_memory_session(
        actor_id="user-123",
        session_id="session-456"
    )

    # Add conversation turns
    event = session.add_turns([
        ConversationalMessage("Hello", MessageRole.USER),
        ConversationalMessage("Hi there!", MessageRole.ASSISTANT)
    ])

except NoCredentialsError:
    print("AWS credentials not found. Please configure your credentials.")

except ClientError as e:
    error_code = e.response['Error']['Code']
    error_message = e.response['Error']['Message']

    if error_code == 'ResourceNotFoundException':
        print(f"Memory not found: {error_message}")
    elif error_code == 'ValidationException':
        print(f"Invalid input: {error_message}")
    elif error_code == 'AccessDeniedException':
        print(f"Access denied: {error_message}")
    elif error_code == 'ThrottlingException':
        print(f"Request throttled: {error_message}")
    else:
        print(f"AWS error ({error_code}): {error_message}")

except Exception as e:
    print(f"Unexpected error: {str(e)}")
```

### Best Practices for Error Handling

1. **Always handle authentication errors**:

   ```python
   try:
       manager = MemorySessionManager(memory_id="test")
   except NoCredentialsError:
       # Guide user to configure credentials
       print("Please run 'aws configure' or set AWS environment variables")
   ```
2. **Validate inputs before API calls**:

   ```python
   def validate_user_input(user_input: str) -> bool:
       if validate_input(user_input)
           raise ValueError("user_input must be a non-empty string")
       return True

   validate_memory_id(memory_id)
   ```
3. **Handle rate limiting gracefully**:

   ```python
   try:
       memories = session.search_long_term_memories(query="test")
   except ClientError as e:
       if e.response['Error']['Code'] == 'ThrottlingException':
           print("Request rate exceeded. Please reduce request frequency.")
           time.sleep(5)  # Wait before retrying
   ```
4. **Log errors for debugging**:

   ```python
   import logging

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   try:
       event = session.add_turns(messages)
   except Exception as e:
       logger.error(f"Failed to add turns: {str(e)}", exc_info=True)
       raise
   ```
5. **Use context managers for cleanup**:

   ```python
   from contextlib import contextmanager

   @contextmanager
   def memory_session_context(manager, actor_id, session_id):
       session = None
       try:
           session = manager.create_memory_session(actor_id, session_id)
           yield session
       except Exception as e:
           logger.error(f"Error in memory session: {str(e)}")
           raise
       finally:
           # Cleanup if needed
           if session:
               logger.info(f"Session {session_id} operations completed")

   # Usage
   with memory_session_context(manager, "user-123", "session-456") as session:
       session.add_turns(messages)
   ```

## Migration from MemoryClient

If you're currently using MemoryClient, here's how to migrate:

### Before (MemoryClient)

```python
from bedrock_agentcore.memory import MemoryClient

client = MemoryClient()
event = client.create_event(
    memory_id="memory-123",
    actor_id="user-456",
    session_id="session-789",
    messages=[("Hello", "USER"), ("Hi there", "ASSISTANT")]
)
```

### After (MemorySessionManager)

```python
from bedrock_agentcore.memory import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

manager = MemorySessionManager(memory_id="memory-123")
session = manager.create_memory_session(
    actor_id="user-456",
    session_id="session-789"
)

event = session.add_turns([
    ConversationalMessage("Hello", MessageRole.USER),
    ConversationalMessage("Hi there", MessageRole.ASSISTANT)
])
```

### Key Migration Benefits

- **Cleaner API**: No need to pass memory_id, actor_id, session_id to every method
- **Type Safety**: Use `ConversationalMessage` and `BlobMessage` instead of tuples
- **Better Organization**: Session-scoped vs manager-scoped operations
- **Enhanced Features**: Built-in LLM integration with `process_turn_with_llm()`

## Best Practices

### Session Management

- Use `MemorySessionManager` for multi-session, multi-actor scenarios
- Use `MemorySession` for session-specific operations to avoid parameter repetition
- Create separate sessions for different conversation contexts

### Memory Operations

- Use `process_turn_with_llm()` for integrated LLM workflows
- Separate retrieval and storage with `search_long_term_memories()` and `add_turns()` for custom workflows
- Use namespace prefixes effectively for organized memory retrieval
- Handle service errors with appropriate retry logic

### Message Handling

- Use `ConversationalMessage` for text-based interactions
- Use `BlobMessage` for binary data (images, files, etc.)
- Group related messages in single `add_turns()` calls for logical conversation units

### Branch Management

- Create branches for A/B testing different responses
- Use descriptive branch names for easier navigation
- Fork from specific events to maintain conversation context

### Performance Optimization

- Batch operations when possible using `add_turns()` with multiple messages
- Use appropriate `top_k` values for memory searches to balance relevance and performance
- Implement caching for frequently accessed memory records
- Monitor and optimize namespace structures for efficient retrieval

### Security

- Use IAM roles instead of hardcoded credentials in production
- Implement proper access controls for memory resources
- Validate and sanitize user inputs before storing in memory
- Use encryption for sensitive data in memory records

## API Reference

### Core Classes

- **MemorySessionManager**: Primary interface for managing sessions and actors
- **MemorySession**: Session-scoped operations interface
- **MemoryClient**: Legacy client interface (deprecated)

### Data Models

- **ConversationalMessage**: Text-based conversation messages
- **BlobMessage**: Binary data messages
- **Event**: Individual conversation events
- **Branch**: Alternative conversation paths
- **ActorSummary**: Actor information summary
- **SessionSummary**: Session information summary
- **MemoryRecord**: Long-term memory records

### Configuration Classes

- **RetrievalConfig**: Configuration for memory retrieval operations
- **MessageRole**: Enumeration of message roles (USER, ASSISTANT, TOOL, OTHER)
- **MemoryStatus**: Memory resource status enumeration
- **StrategyType**: Memory strategy type enumeration

For detailed API documentation, refer to the inline docstrings and type hints in the source code.
