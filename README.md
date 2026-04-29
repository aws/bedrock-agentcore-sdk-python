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
    ◆ <a href="https://discord.gg/bedrockagentcore-preview">Discord</a>
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

    agent = Agent()

    async for event in agent.stream_async(prompt):
        yield (event)

app.run()
```

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

```bash
pip install "bedrock-agentcore[a2a]"
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
