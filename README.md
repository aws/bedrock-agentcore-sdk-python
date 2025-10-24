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
    ◆ <a href="https://github.com/aws/bedrock-agentcore-starter-toolkit">Starter Toolkit</a>

  </p>
</div>

## Overview
Amazon Bedrock AgentCore enables you to deploy and operate highly effective agents securely, at scale using any framework and model. With Amazon Bedrock AgentCore, developers can accelerate AI agents into production with the scale, reliability, and security, critical to real-world deployment. AgentCore provides tools and capabilities to make agents more effective and capable, purpose-built infrastructure to securely scale agents, and controls to operate trustworthy agents. Amazon Bedrock AgentCore services are composable and work with popular open-source frameworks and any model, so you don’t have to choose between open-source flexibility and enterprise-grade security and reliability.

## 🚀 From Local Development to Bedrock AgentCore

```python
# Your existing agent (any framework)
from strands import Agent
# or LangGraph, CrewAI, Autogen, custom logic - doesn't matter

def my_local_agent(query):
    # Your carefully crafted agent logic
    return agent.process(query)

# Deploy to Bedrock AgentCore
from bedrock_agentcore import BedrockAgentCoreApp
app = BedrockAgentCoreApp()

@app.entrypoint
def production_agent(request):
    return my_local_agent(request.get("prompt"))  # Same logic, enterprise platform

app.run()  # Ready to run on Bedrock AgentCore
```

**What you get with Bedrock AgentCore:**
- ✅ **Keep your agent logic** - Works with Strands, LangGraph, CrewAI, Autogen, custom frameworks
- ✅ **Zero infrastructure management** - No servers, containers, or scaling concerns
- ✅ **Enterprise-grade platform** - Built-in auth, memory, observability, security
- ✅ **Production-ready deployment** - Reliable, scalable, compliant hosting

## Amazon Bedrock AgentCore services
- 🚀 **Runtime** - Secure and session isolated compute: **[Runtime Quick Start](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-toolkit.html)**
- 🧠 **Memory** - Persistent knowledge across sessions: **[Memory Quick Start](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-get-started.html)**
- 🔗 **Gateway** - Transform APIs into MCP tools: **[Gateway Quick Start](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-quick-start.html)**
- 💻 **Code Interpreter** - Secure sandboxed execution: **[Code Interpreter Quick Start](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-getting-started.html)**
- 🌐 **Browser** - Cloud-based web automation: **[Browser Quick Start](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/browser-onboarding.html)**
- 📊 **Observability** - OpenTelemetry tracing: **[Observability Quick Start](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-get-started.html)**
- 🔐 **Identity** - AWS & third-party auth: **[Identity Quick Start](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/identity-getting-started-cognito.html)**

## 🏗️ Deployment

**Quick Start:** Use the [Bedrock AgentCore Starter Toolkit](https://github.com/aws/bedrock-agentcore-starter-toolkit) for rapid prototyping.

**Production:** [AWS CDK](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_bedrockagentcore-readme.html).


## 📝 License & Contributing

- **License:** Apache 2.0 - see [LICENSE.txt](LICENSE.txt)
- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Security:** Report vulnerabilities via [SECURITY.md](SECURITY.md)
