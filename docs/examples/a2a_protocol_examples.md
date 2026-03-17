# A2A Protocol Support

This document explains how to serve your agent using the [A2A (Agent-to-Agent) protocol](https://google.github.io/A2A/) on Bedrock AgentCore Runtime.

## Installation

A2A support requires the optional `a2a` extra:

```bash
pip install "bedrock-agentcore[a2a]"
```

## Quick Start

### Strands Agent

Strands provides a built-in `StrandsA2AExecutor` that wraps a Strands `Agent` as an A2A executor. When no `AgentCard` is provided, one is auto-built from the agent's `name` and `description`.

```python
from strands import Agent
from strands.a2a import StrandsA2AExecutor
from bedrock_agentcore.runtime import serve_a2a

agent = Agent(
    model="us.anthropic.claude-sonnet-4-20250514",
    system_prompt="You are a helpful calculator.",
)

if __name__ == "__main__":
    serve_a2a(StrandsA2AExecutor(agent))
```

### LangGraph Agent

LangGraph requires a thin `AgentExecutor` wrapper (~15 lines) and an explicit `AgentCard`:

```python
from langchain_aws import ChatBedrockConverse
from langgraph.prebuilt import create_react_agent

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Part, TextPart
from a2a.utils import new_task
from bedrock_agentcore.runtime import serve_a2a

llm = ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-20250514")
graph = create_react_agent(llm, tools=[], prompt="You are a helpful calculator.")


class LangGraphA2AExecutor(AgentExecutor):
    def __init__(self, graph):
        self.graph = graph

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        user_text = context.get_user_input()
        result = await self.graph.ainvoke({"messages": [("user", user_text)]})
        response = result["messages"][-1].content
        await updater.add_artifact([Part(root=TextPart(text=response))])
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


card = AgentCard(
    name="langgraph-agent",
    description="A LangGraph agent on Bedrock AgentCore",
    url="http://localhost:9000/",
    version="0.1.0",
    capabilities=AgentCapabilities(streaming=True),
    skills=[AgentSkill(id="calc", name="calculator", description="Arithmetic", tags=["math"])],
    default_input_modes=["text"],
    default_output_modes=["text"],
)

if __name__ == "__main__":
    serve_a2a(LangGraphA2AExecutor(graph), card)
```

### Google ADK Agent

Google ADK provides `A2aAgentExecutor` built-in. You supply an explicit `AgentCard`:

```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.a2a import A2aAgentExecutor

from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from bedrock_agentcore.runtime import serve_a2a

agent = LlmAgent(
    model="gemini-2.0-flash",
    name="calculator",
    description="A calculator agent",
    instruction="You are a helpful calculator.",
)
runner = Runner(agent=agent, app_name="calculator", session_service=None)

card = AgentCard(
    name="adk-agent",
    description="A Google ADK agent on Bedrock AgentCore",
    url="http://localhost:9000/",
    version="0.1.0",
    capabilities=AgentCapabilities(streaming=True),
    skills=[AgentSkill(id="calc", name="calculator", description="Arithmetic", tags=["math"])],
    default_input_modes=["text"],
    default_output_modes=["text"],
)

if __name__ == "__main__":
    serve_a2a(A2aAgentExecutor(runner=runner), card)
```

## API Reference

### `serve_a2a(executor, agent_card=None, *, port=9000, host=None, ...)`

Starts a Bedrock-compatible A2A server with `uvicorn`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `executor` | `AgentExecutor` | required | An a2a-sdk `AgentExecutor` that implements the agent logic |
| `agent_card` | `AgentCard` | `None` | Agent metadata. Auto-built from executor if omitted (works best with Strands) |
| `port` | `int` | `9000` | Port to serve on |
| `host` | `str` | `None` | Host to bind to. Auto-detected: `0.0.0.0` in Docker, `127.0.0.1` otherwise |
| `task_store` | `TaskStore` | `None` | Custom task store; defaults to `InMemoryTaskStore` |
| `context_builder` | `CallContextBuilder` | `None` | Custom context builder; defaults to `BedrockCallContextBuilder` |
| `ping_handler` | `Callable[[], PingStatus]` | `None` | Custom health check callback |
| `**kwargs` | | | Additional arguments forwarded to `uvicorn.run()` |

### `build_a2a_app(executor, agent_card=None, *, task_store=None, context_builder=None, ping_handler=None)`

Builds a Starlette ASGI application without starting a server. Useful for testing or embedding in a larger app.

Returns a `Starlette` application with routes:
- `POST /` — A2A JSON-RPC endpoint (`message/send`, `message/stream`, `tasks/get`, `tasks/cancel`)
- `GET /.well-known/agent-card.json` — Agent card discovery
- `GET /ping` — Bedrock health check

### `build_runtime_url(agent_arn, region=None)`

Builds the Bedrock AgentCore runtime invocation URL from an agent ARN.

```python
from bedrock_agentcore.runtime import build_runtime_url

url = build_runtime_url("arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-agent-abc123")
# https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/arn%3Aaws%3A.../invocations
```

### `BedrockCallContextBuilder`

Extracts Bedrock runtime headers from incoming requests and propagates them into `BedrockAgentCoreContext` contextvars. This is the default `context_builder` used by `build_a2a_app` and `serve_a2a`.

Headers extracted:
- `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` — session ID
- `X-Amzn-Bedrock-AgentCore-Runtime-Request-Id` — request ID (auto-generated UUID if missing)
- `WorkloadAccessToken` — workload access token
- `OAuth2CallbackUrl` — OAuth2 callback URL
- `Authorization` — authorization header
- `X-Amzn-Bedrock-AgentCore-Runtime-Custom-*` — custom headers

## Behavior Details

### Agent Card Auto-Population

When deployed on Bedrock AgentCore, the `AGENTCORE_RUNTIME_URL` environment variable is set automatically. The agent card's `url` field is updated to match, so you don't need to hardcode the deployed URL.

### Docker Host Detection

When `host` is not specified, `serve_a2a` automatically binds to `0.0.0.0` inside Docker containers (detected via `/.dockerenv` or `DOCKER_CONTAINER` env var) and `127.0.0.1` otherwise.

### Custom Ping Handler

```python
from bedrock_agentcore.runtime import serve_a2a
from bedrock_agentcore.runtime.models import PingStatus

def my_ping():
    if is_overloaded():
        return PingStatus.HEALTHY_BUSY
    return PingStatus.HEALTHY

serve_a2a(executor, ping_handler=my_ping)
```

If the ping handler raises an exception, the server falls back to `PingStatus.HEALTHY`.
