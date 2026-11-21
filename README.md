<div align="center">
  <div>
    <a href="https://aws.amazon.com/bedrock/agentcore/">
      <img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/b8b9456d-c9e2-45e1-ac5b-760f21f1ac18" />
   </a>
  </div>

  <h1>
    Bedrock AgentCore SDK
  </h1>

  <h2>
    Deploy your local AI agent to Bedrock AgentCore with zero infrastructure
  </h2>

  <div align="center">
    <a href="https://github.com/aws/bedrock-agentcore-sdk-python/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/aws/bedrock-agentcore-sdk-python"/></a>
    <a href="https://github.com/aws/bedrock-agentcore-sdk-python/issues"><img alt="GitHub open issues" src="https://img.shields.io/github/issues/aws/bedrock-agentcore-sdk-python"/></a>
    <a href="https://github.com/aws/bedrock-agentcore-sdk-python/pulls"><img alt="GitHub open pull requests" src="https://img.shields.io/github/issues-pr/aws/bedrock-agentcore-sdk-python"/></a>
    <a href="https://github.com/aws/bedrock-agentcore-sdk-python/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/aws/bedrock-agentcore-sdk-python"/></a>
    <a href="https://pypi.org/project/bedrock-agentcore"><img alt="PyPI version" src="https://img.shields.io/pypi/v/bedrock-agentcore"/></a>
    <a href="https://python.org"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/bedrock-agentcore"/></a>
  </div>

  <p>
  <a href="https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html">Documentation</a>
    ◆ <a href="https://github.com/awslabs/amazon-bedrock-agentcore-samples">Samples</a>
    ◆ <a href="https://discord.gg/strands">Discord</a>
    ◆ <a href="https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agentcore-control.html">Boto3 Python SDK</a>
    ◆ <a href="https://github.com/aws/bedrock-agentcore-sdk-python">Runtime Python SDK</a>

  </p>
</div>

## Overview
Amazon Bedrock AgentCore enables you to deploy and operate highly effective agents securely, at scale using any framework and model. With Amazon Bedrock AgentCore, developers can accelerate AI agents into production with the scale, reliability, and security, critical to real-world deployment. AgentCore provides tools and capabilities to make agents more effective and capable, purpose-built infrastructure to securely scale agents, and controls to operate trustworthy agents. Amazon Bedrock AgentCore services are composable and work with popular open-source frameworks and any model, so you don’t have to choose between open-source flexibility and enterprise-grade security and reliability.

## 🚀 From Local Development to Bedrock AgentCore

```python
from bedrock_agentcore import BedrockAgentCoreApp
app = BedrockAgentCoreApp()

from strands import Agent # or bring your agent.

@app.entrypoint
async def handler(request):
    prompt = request.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")

    agent = Agent()

    async for event in agent.stream_async(prompt):
        yield (event)

app.run()
```

Validate agent input before forwarding it to an agent framework. Keep prompts typed as strings and pass only prompt
text to the agent.

**What you get with Bedrock AgentCore:**
- ✅ **Keep your agent logic** - Works with Strands, LangGraph, CrewAI, Autogen, or custom frameworks
- ✅ **Zero infrastructure management** - No servers, containers, or scaling concerns
- ✅ **Enterprise-grade platform** - Built-in auth, memory, observability, security
- ✅ **Production-ready deployment** - Reliable, scalable, compliant hosting

## Amazon Bedrock AgentCore services
- 🚀 **Runtime** - Secure and session isolated compute: **[Get started](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-getting-started.html)**
- 🧠 **Memory** - Persistent knowledge across sessions: **[Get started](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-get-started.html)**
- 🔗 **Gateway** - Transform APIs into MCP tools: **[Get started](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-quick-start.html)**
- 💻 **Code Interpreter** - Secure sandboxed execution: **[Get started](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html)**
- 🌐 **Browser** - Cloud-based web automation: **[Get started](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/browser-quickstart.html)**
- 📊 **Observability** - OpenTelemetry tracing: **[Get started](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-get-started.html)**
- 🔐 **Identity** - AWS & third-party auth: **[Get started](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/identity-getting-started.html)**

## Build Warmup Hook

On platform version V2, AgentCore Runtime can call a [build warmup hook](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-lifecycle-hooks.html) while it prepares your agent, before the runtime is `READY`. It sends two `POST` requests, first `memory` (load what every session needs, such as imports, models and caches) and then `disk` (read the files that sessions read at start). Register a handler for either, both or none:

```python
from bedrock_agentcore import BedrockAgentCoreApp, HookConfig, WarmupType

app = BedrockAgentCoreApp(warmup_hook=HookConfig(path="/warmup"))  # serves the hook at POST /warmup
MODEL = None

@app.warmup(WarmupType.MEMORY)
def load_model():
    global MODEL
    MODEL = load_my_model()  # runs before the runtime is READY; sessions see what's loaded here

@app.warmup(WarmupType.DISK)
def read_startup_files():
    for path in ["/app/config/agent.json"]:
        with open(path, "rb") as f:
            while f.read(1 << 20):
                pass

@app.entrypoint
def invoke(payload):
    return MODEL.predict(payload)

app.run()
```

- Handlers run off the event loop, so `/ping` keeps answering.
- A type without a handler, or one this SDK doesn't recognize, returns 200 (`"result": "skipped"`). Only a missing type header returns 400.
- An exception returns 500, which fails the create or update.
- Handlers can't be generators: `@app.warmup` raises `ValueError` for a function with `yield`.
- Both calls share a 120-second limit.

Configure the hook on the runtime:

```python
client.create_agent_runtime(
    ...,
    platformVersion="V2",
    hooksConfiguration={"build": {"warmup": {"action": {"httpPost": {"path": "/warmup"}}}}},
)
```

- `HookConfig.path` must equal `hooksConfiguration.build.warmup.action.httpPost.path`. The API requires the path, and the SDK has no default. Without `warmup_hook`, the app adds no warmup route and `@app.warmup` raises `ValueError`.
- `HookConfig.port` must equal `httpPost.port`. Omit both to serve the hook on the port the app serves on (`app.run(port=...)`, default 8080), next to `/invocations` and `/ping`.
- To serve the hook on its own port (8000-12000), for example when you run your own server on 8080, set both:

  ```python
  app = BedrockAgentCoreApp(warmup_hook=HookConfig(path="/warmup", port=9000))
  # hooksConfiguration={"build": {"warmup": {"action": {"httpPost": {"path": "/warmup", "port": 9000}}}}}
  ```

  The app then serves `POST /warmup` on port 9000 only, from a second listener on `127.0.0.1` in the same process that `app.run()` opens before the serving port and closes after it. A separate port requires starting the agent with `app.run()`: started another way (for example `uvicorn agent:app`), the app fails its startup with a message saying so, and `app.run(lifespan="off")` raises `ValueError`. If the hook port is taken, the app exits before the serving port opens. A `HookConfig.port` equal to the serving port is served on the serving port, as if it were unset.
- Serve the hook from one server process. The memory call reaches one process. That process may start child processes (for example a browser or a model server) and keep them, but other worker processes (for example `uvicorn --workers N`) don't receive the call, and would each try to open the hook port.

## AG-UI Protocol Support

Deploy agents using the [AG-UI protocol](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-agui.html) with zero boilerplate. A single `entrypoint` handler is served over both SSE (`POST /invocations`) and WebSocket (`/ws`).

```python
from bedrock_agentcore.runtime import serve_ag_ui

# Framework agent with a .run() method — one line
serve_ag_ui(agui_agent)
```

Or write a custom agent with the decorator form:

```python
from bedrock_agentcore.runtime import AGUIApp
from ag_ui.core import RunAgentInput, RunStartedEvent, RunFinishedEvent

app = AGUIApp()

@app.entrypoint
async def my_agent(input_data: RunAgentInput):
    yield RunStartedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)
    # ... your agent logic, yield AG-UI events ...
    yield RunFinishedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)

app.run()
```

Install with: `pip install "bedrock-agentcore[ag-ui]"`

See the [AG-UI protocol contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-agui-protocol-contract.html) for full details.

## 🏗️ Deployment

### AgentCore CLI (Recommended)

The fastest way to deploy your agent to Bedrock AgentCore. Install via [npm](https://www.npmjs.com/package/@aws/agentcore):

```bash
npm i -g @aws/agentcore
```

Then create and deploy your agent:

```bash
agentcore create --name MyAgent --defaults
cd MyAgent
agentcore deploy
```

The CLI handles packaging, infrastructure provisioning, and deployment automatically.
See the [AgentCore CLI documentation](https://github.com/aws/agentcore-cli) for full details.

### AWS CDK (Advanced)

The AgentCore CLI generates AWS CDK under the hood. For full infrastructure-as-code control, customize the generated CDK project directly or refer to the [L1 CDK documentation](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_bedrockagentcore-readme.html).

## A2A Protocol Support

Serve your agent using the [A2A (Agent-to-Agent) protocol](https://google.github.io/A2A/) on Bedrock AgentCore Runtime. Works with any framework that provides an a2a-sdk `AgentExecutor` (Strands, LangGraph, Google ADK, or custom).

Strands currently uses a2a-sdk 0.3:

```bash
pip install "bedrock-agentcore[a2a]"
```

For A2A protocol v1, install the mutually exclusive `a2a-v1` extra:

```bash
pip install "bedrock-agentcore[a2a-v1]"
```

```python
from strands import Agent
from strands.a2a import StrandsA2AExecutor
from bedrock_agentcore.runtime import serve_a2a

agent = Agent(model="us.anthropic.claude-sonnet-4-20250514", system_prompt="You are a helpful assistant.")
serve_a2a(StrandsA2AExecutor(agent))
```

See [A2A Protocol Examples](docs/examples/a2a_protocol_examples.md) for LangGraph, Google ADK, and advanced usage.

## 📝 License & Contributing

- **License:** Apache 2.0 - see [LICENSE.txt](LICENSE.txt)
- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Security:** Report vulnerabilities via [SECURITY.md](SECURITY.md)
