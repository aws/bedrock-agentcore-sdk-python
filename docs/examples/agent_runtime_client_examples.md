# AgentRuntimeClient Examples

This document provides practical examples for using the `AgentRuntimeClient` to authenticate WebSocket connections to AgentCore Runtime.

## Basic Usage

### Backend Service (SigV4 Headers)

```python
from bedrock_agentcore.runtime import AgentRuntimeClient
import websockets
import asyncio

async def main():
    # Initialize client
    client = AgentRuntimeClient(region="us-west-2")

    # Generate WebSocket connection with authentication
    ws_url, headers = client.generate_ws_connection(
        runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime"
    )

    # Connect using any WebSocket library
    async with websockets.connect(ws_url, extra_headers=headers) as ws:
        # Send message
        await ws.send('{"inputText": "Hello!"}')

        # Receive response
        response = await ws.recv()
        print(f"Received: {response}")

asyncio.run(main())
```

### Frontend Client (Presigned URL)

```python
from bedrock_agentcore.runtime import AgentRuntimeClient

# Backend: Generate presigned URL
client = AgentRuntimeClient(region="us-west-2")

presigned_url = client.generate_presigned_url(
    runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime",
    expires=300  # 5 minutes
)

# Share presigned_url with frontend
# Frontend JavaScript: new WebSocket(presigned_url)
```

## Advanced Usage

### With Endpoint Qualifier

```python
client = AgentRuntimeClient(region="us-west-2")

# For generate_ws_connection (header-based auth)
ws_url, headers = client.generate_ws_connection(
    runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime",
    endpoint_name="DEFAULT"
)
# URL will include: ?qualifier=DEFAULT

# For generate_presigned_url (query-based auth)
presigned_url = client.generate_presigned_url(
    runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime",
    endpoint_name="DEFAULT"
)
# URL will include: ?qualifier=DEFAULT&X-Amz-Algorithm=...
```

### With Custom Query Parameters (Presigned URL only)

```python
client = AgentRuntimeClient(region="us-west-2")

# custom_headers parameter is only available for presigned URLs
presigned_url = client.generate_presigned_url(
    runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime",
    custom_headers={"custom_param": "value", "another": "param"}
)

# URL will include: ?custom_param=value&another=param&X-Amz-Algorithm=...
```

### With Explicit Session ID

```python
client = AgentRuntimeClient(region="us-west-2")

ws_url, headers = client.generate_ws_connection(
    runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime",
    session_id="my-custom-session-id"
)
```

## Error Handling

```python
from bedrock_agentcore.runtime import AgentRuntimeClient

client = AgentRuntimeClient(region="us-west-2")

try:
    ws_url, headers = client.generate_ws_connection(
        runtime_arn="invalid-arn"
    )
except ValueError as e:
    print(f"Invalid ARN format: {e}")
except RuntimeError as e:
    print(f"AWS credentials error: {e}")
```

## Using Different WebSocket Libraries

### With websockets library

```python
import websockets

ws_url, headers = client.generate_ws_connection(runtime_arn)
async with websockets.connect(ws_url, extra_headers=headers) as ws:
    await ws.send(message)
```

### With aiohttp library

```python
import aiohttp

ws_url, headers = client.generate_ws_connection(runtime_arn)
async with aiohttp.ClientSession() as session:
    async with session.ws_connect(ws_url, headers=headers) as ws:
        await ws.send_str(message)
```
