<div align="center">
  <h1>
    Bedrock AgentCore SDK
  </h1>

  <h2>
    Transform any function into a production API in 3 lines. Your code stays unchanged.
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
    <a href="https://github.com/aws/bedrock-agentcore-sdk-python">Python SDK</a>
    ◆ <a href="https://github.com/aws/bedrock-agentcore-starter-toolkit">Starter Toolkit</a>
    ◆ <a href="https://github.com/awslabs/amazon-bedrock-agentcore-samples">Samples</a>
  </p>
</div>

Bedrock AgentCore SDK is a lightweight Python SDK that transforms any AI agent function into a production-ready HTTP API server. Just add 3 lines to your existing code.

## ⚠️ Preview Status

Bedrock AgentCore SDK is currently in public preview. APIs may change as we refine the SDK.

## The 3-Line Transformation

**Before** - Your existing function:
```python
def invoke(payload):
    user_message = payload.get("prompt", "Hello")
    response = agent(user_message)
    return response
```

**After** - Add 3 lines to make it an API:
```python
from bedrock_agentcore.runtime import BedrockAgentCoreApp  # +1
app = BedrockAgentCoreApp()                                # +2

@app.entrypoint                                           # +3
def invoke(payload):  # ← Your function stays EXACTLY the same
    user_message = payload.get("prompt", "Hello")
    response = agent(user_message)
    return response
```

Your function is now a production-ready API server with health monitoring, streaming support, and AWS integration.

## Features

- **Zero Code Changes**: Your existing function remains untouched
- **Production Ready**: Automatic `/invocations` and `/ping` endpoints with health monitoring
- **Streaming Support**: Native support for generators and async generators
- **Async Task Tracking**: Built-in monitoring for long-running background tasks
- **Framework Agnostic**: Works with any AI framework (Strands, LangChain, custom)
- **AWS Optimized**: Ready for deployment to AWS infrastructure

## Quick Start

```bash
pip install bedrock-agentcore
```

```python
# my_agent.py
from strands import Agent  # Or any AI framework
from bedrock_agentcore.runtime import BedrockAgentCoreApp

agent = Agent()
app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    """Your existing function - unchanged"""
    user_message = payload.get("prompt", "Hello")
    response = agent(user_message)
    return response

if __name__ == "__main__":
    app.run()  # Starts server on http://localhost:8080
```

Test your API:
```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world!"}'
```

## Core Capabilities

### Streaming Responses
```python
@app.entrypoint
def invoke(payload):
    # Yields are automatically converted to Server-Sent Events
    for chunk in agent.stream(payload.get("prompt")):
        yield chunk
```

### Custom Health Checks
```python
@app.ping
def health_check():
    return PingStatus.HEALTHY if model_loaded else PingStatus.UNHEALTHY
```

### Async Task Tracking
```python
@app.async_task
async def background_task():
    # Automatically tracked for health monitoring
    await long_running_operation()
```

### Request Context
```python
@app.entrypoint
def invoke(payload, context: RequestContext):
    # Access session info and auth
    return agent(payload.get("prompt"), session_id=context.session_id)
```

## What's Created Automatically

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/invocations` | POST | Calls your `invoke` function |
| `/ping` | GET | Health checks for load balancers |

BedrockAgentCoreApp handles:
- HTTP request/response formatting
- Content-type headers (`application/json` or `text/event-stream`)
- Error handling and logging
- Async task health monitoring

## Deployment

For production deployments, use [AWS CDK](https://aws.amazon.com/cdk/) for infrastructure as code.

For quick prototyping and deployment tools, see the [Bedrock AgentCore Starter Toolkit](https://github.com/aws/bedrock-agentcore-starter-toolkit).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 License. See [LICENSE.txt](LICENSE.txt).

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
