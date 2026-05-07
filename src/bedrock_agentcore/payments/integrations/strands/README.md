# Strands AgentCore Payments Plugin

The AgentCore Payments Plugin leverages Amazon Bedrock AgentCore Payments to provide automated payment processing
capabilities for Strands Agents. It supports the [x402 Payment Required](https://www.x402.org/) protocol, enabling
agents to automatically handle HTTP 402 responses by processing microtransaction payments to access paid APIs,
MCP servers, and premium content.

## Overview

- **Automatic x402 Payment Handling** — intercepts HTTP 402 responses from tools, processes payment requirements, and retries requests with payment headers
- **Payment Query Tools** — built-in tools for agents to query payment instruments and sessions at runtime
- **Multi-Protocol Support** — handles x402 v1 and v2 payment protocols
- **Multi-Handler Architecture** — supports generic tools, `http_request` tools, and MCP Gateway proxy tools
- **Interrupt-Based Error Handling** — raises Strands SDK interrupts on payment failures so the agent (or application) can respond dynamically
- **Configurable Auto-Payment** — enable or disable automatic payment processing per plugin instance

## How It Works

### x402 Payment Flow

```
┌─────────┐     ┌──────────┐     ┌──────────────┐     ┌────────────────┐
│  Agent  │────▶│   Tool   │────▶│  Paid API    │────▶│  402 Response  │
│         │     └──────────┘     └──────────────┘     └────────────────┘
│         │                                                   │
│         │     ┌──────────┐     ┌──────────────┐             │
│         │◀────│   Tool   │◀────│  Plugin      │◀────────────┘
│ (result)│     │ (retry)  │     │  processes   │
└─────────┘     └──────────┘     │  payment     │
                                 └──────────────┘
```

1. Agent calls a tool (e.g., `http_request`) that hits a paid API
2. The API returns HTTP 402 with x402 payment requirements
3. The plugin's `after_tool_call` hook intercepts the 402 response
4. The plugin extracts payment requirements using the appropriate handler
5. The plugin calls `PaymentManager.generate_payment_header()` to process the payment
6. The payment header is applied to the tool input
7. The tool is automatically retried with the payment credentials
8. The API returns a successful response

## Installation

```bash
pip install 'bedrock-agentcore[strands-agents]'
```

Or to develop locally:

```bash
git clone https://github.com/aws/bedrock-agentcore-sdk-python.git
cd bedrock-agentcore-sdk-python
uv sync
source .venv/bin/activate
```

## Quick Start

Once your payment resources are ready, wire up the plugin:

```python
import os
from strands import Agent
from strands_tools import http_request
from bedrock_agentcore.payments.integrations.strands import (
    AgentCorePaymentsPlugin,
    AgentCorePaymentsPluginConfig,
)

plugin = AgentCorePaymentsPlugin(config=AgentCorePaymentsPluginConfig(
    payment_manager_arn=os.environ["PAYMENT_MANAGER_ARN"],
    user_id="test-user-123",
    payment_instrument_id=os.environ["PAYMENT_INSTRUMENT_ID"],
    payment_session_id=os.environ["PAYMENT_SESSION_ID"],
    region="us-east-1",
))

agent = Agent(
    system_prompt="You are a helpful assistant that can access paid APIs.",
    tools=[http_request],
    plugins=[plugin],
)

# 402 responses are automatically handled
agent("Fetch a joke from https://premium-api.example.com/joke")
```

The plugin intercepts x402 payment requests automatically, processes the payment, and retries the
request with payment proof for the agent.

---

## Built-in Agent Tools

The plugin registers three tools that agents can use to query payment information at runtime:

| Tool | Description |
|------|-------------|
| `get_payment_instrument` | Retrieve details about a specific payment instrument |
| `list_payment_instruments` | List all payment instruments for a user |
| `get_payment_session` | Retrieve details about a payment session (budget, status, expiry) |

All three tools accept an optional `user_id` parameter. If not provided, the tool falls back to the
`user_id` configured in the plugin config.

These tools enable agents to make informed decisions about payment methods and payment limits during
conversations.

---

## Prerequisites

Before using the plugin, you need:

1. **A Payment Manager** — created via the `PaymentClient` control plane API or AWS Console
2. **A Payment Connector** — linked to a credential provider for a supported payment vendor
3. **A Payment Instrument** — a user's registered payment instrument (funded and signing-enabled)
4. **A Payment Session** — a time-bounded session with optional payment limits

For more details on the Payments SDK, see the [Payments SDK README](../../README.md).

## Setup

### Creating Payment Manager and Connector

> **One-time setup:** The payment resource creation shown below is typically done once, separately
> from your agent application. In production, you would create the payment resources through the
> AWS Console or a separate setup script using the AgentCore SDK, then use `PaymentManagerArn` and
> `PaymentConnectorId` in your agent application.

```python
import os
from bedrock_agentcore.payments.client import PaymentClient

# This is typically done once, separately from your agent application
payment_client = PaymentClient(region_name="us-east-1")

response = payment_client.create_payment_manager_with_connector(
    payment_manager_name="AgentCorePaymentManager",
    payment_manager_description="Payment Manager for Agent Core",
    authorizer_type="AWS_IAM",
    role_arn="arn:aws:iam::123456789012:role/BedrockAgentCoreFullAccess",
    payment_connector_config={
        "name": "agent-core-connector",
        "description": "Payment Connector for Agent Core",
        "payment_credential_provider_config": {
            "name": "agent-core-provider",
            "credential_provider_vendor": "CoinbaseCDP",
            "credentials": {
                "api_key_id": "<your-coinbase-api-key-id>",
                "api_key_secret": "<your-coinbase-api-key-secret>",
                "wallet_secret": "<your-coinbase-wallet-secret>",
            },
        },
    },
    wait_for_ready=True,
    max_wait=300,
    poll_interval=5,
)

# Export for reuse in your agent application
payment_manager_arn = response["paymentManager"]["paymentManagerArn"]
payment_connector_id = response["paymentConnector"]["paymentConnectorId"]
os.environ["PAYMENT_MANAGER_ARN"] = payment_manager_arn
os.environ["PAYMENT_CONNECTOR_ID"] = payment_connector_id
print(f"Payment Manager ARN: {payment_manager_arn}")
print(f"Payment Connector ID: {payment_connector_id}")
```

The `wait_for_ready=True` parameter causes the method to poll until all resources reach READY status.
If any step fails, previously created resources are automatically rolled back.

### Creating a Payment Instrument

Create a payment instrument for a given user to process payments. Below is an example creating an
Ethereum chain-compatible embedded crypto wallet:

```python
instrument = manager.create_payment_instrument(
    user_id="test-user-123",
    payment_connector_id=os.environ["PAYMENT_CONNECTOR_ID"],
    payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
    payment_instrument_details={
        "embeddedCryptoWallet": {
            "network": "ETHEREUM",
            "linkedAccounts": [
                {"email": {"emailAddress": "email@example.com"}}
            ],
        }
    },
)

payment_instrument_id = instrument["paymentInstrumentId"]
os.environ["PAYMENT_INSTRUMENT_ID"] = payment_instrument_id
print(f"Payment Instrument ID: {payment_instrument_id}")
```

For Solana-compatible chains, use `"SOLANA"` for the network input. Once created, the instrument
must be funded and permission granted for signing before the agent can use it. These are end-user
actions that should be completed before using the payment instrument in your agent.

If you are using Coinbase as wallet provider, you'll receive a `redirectUrl` in the payment
instrument response, pointing to the Coinbase-hosted WalletHub. Redirect your user there to grant
signing permission and transfer funds.

For Stripe, developers use a provided URL template to host a frontend page where end users can take
the same actions.

### Creating a Payment Session

You also need a payment session before processing payments:

```python
session = manager.create_payment_session(
    user_id="test-user-123",
    limits={"maxSpendAmount": {"value": "100.00", "currency": "USD"}},
    expiry_time_in_minutes=60,
)

payment_session_id = session["paymentSessionId"]
os.environ["PAYMENT_SESSION_ID"] = payment_session_id
print(f"Payment Session ID: {payment_session_id}")
```

---

## Advanced Usage

### Dynamic Instrument/Session Selection

Dynamic Instrument/Session selection is useful when the payment instrument and the payment session aren’t known upfront and need to be resolved dynamically during execution. You can initialize the plugin without a payment instrument or payment session, then set them later based on runtime logic or agent interrupts:

```python
config = AgentCorePaymentsPluginConfig(
    payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-abc123",
    user_id="user-123",
    region="us-east-1",
    # payment_instrument_id and payment_session_id omitted
)

plugin = AgentCorePaymentsPlugin(config=config)
agent = Agent(
    system_prompt="You are a helpful assistant.",
    tools=[http_request],
    plugins=[plugin],
)

# Later, update configuration dynamically
config.update_payment_instrument_id("payment-instrument-xyz789")
config.update_payment_session_id("payment-session-def456")
```

When the plugin encounters a 402 response without these values configured, it raises a
`PaymentInstrumentConfigurationRequired` or `PaymentSessionConfigurationRequired` interrupt
that your application can handle.

---

### Handling Payment Interrupts

When payment processing fails, the plugin stores the failure and raises an interrupt. Your application
should handle these interrupts to provide autonomous functionality:

```python
result = agent("Access the premium endpoint at https://api.example.com/premium")

while result.stop_reason == "interrupt":
    responses = []
    for interrupt in result.interrupts:
        reason = interrupt.reason
        match reason.get("exceptionType"):
            case "PaymentInstrumentConfigurationRequired":
                plugin.config.update_payment_instrument_id("payment-instrument-new123")
                msg = "Payment instrument configured. Please retry."
            case "PaymentSessionConfigurationRequired":
                plugin.config.update_payment_session_id("payment-session-new456")
                msg = "Payment session configured. Please retry."
            case _:
                msg = f"Payment failed: {reason.get('exceptionMessage')}"

        responses.append({"interruptResponse": {"interruptId": interrupt.id, "response": msg}})

    result = agent(responses)
```

---

### Disabling Auto-Payment

If you want the plugin to only provide payment query tools without automatic 402 handling:

```python
config = AgentCorePaymentsPluginConfig(
    payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-abc123",
    user_id="user-123",
    region="us-east-1",
    auto_payment=False,  # Disable automatic 402 processing
)
```

---

### Network Preferences

You can specify preferred blockchain networks for payment processing:

```python
config = AgentCorePaymentsPluginConfig(
    payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-abc123",
    user_id="user-123",
    payment_instrument_id="payment-instrument-xyz789",
    payment_session_id="payment-session-def456",
    region="us-east-1",
    network_preferences_config=["eip155:8453", "base-sepolia", "solana-mainnet"],
)
```

If not specified, the system uses a default preference order prioritizing Solana mainnet and Base
(Ethereum L2) for low transaction fees.

---

### Payment Tool Allowlist

You can restrict which tools are eligible for automatic x402 payment processing using the
`payment_tool_allowlist` parameter. When set, only tool calls whose name appears in this list
will trigger payment processing; all others are skipped:

```python
config = AgentCorePaymentsPluginConfig(
    payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-abc123",
    user_id="user-123",
    payment_instrument_id="payment-instrument-xyz789",
    payment_session_id="payment-session-def456",
    region="us-east-1",
    payment_tool_allowlist=["http_request", "mcp_proxy_tool_call"],
)
```

When `payment_tool_allowlist` is `None` (default), all tools are eligible for payment processing.

---

### Using CUSTOM_JWT (Bearer Token) Authentication

When your payment manager uses `CUSTOM_JWT` authorizer type, configure the plugin with a bearer
token or token provider instead of SigV4 credentials. The service derives the `userId` from the
JWT `sub` claim, so `user_id` is optional.

#### Static Bearer Token

```python
from bedrock_agentcore.payments.integrations.strands import (
    AgentCorePaymentsPlugin,
    AgentCorePaymentsPluginConfig,
)

config = AgentCorePaymentsPluginConfig(
    payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-jwt",
    bearer_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
    # user_id is optional with bearer auth — derived from JWT 'sub' claim
    payment_instrument_id="payment-instrument-xyz789",
    payment_session_id="payment-session-def456",
    region="us-east-1",
)

plugin = AgentCorePaymentsPlugin(config=config)
agent = Agent(
    system_prompt="You are a helpful assistant that can access paid APIs.",
    tools=[http_request],
    plugins=[plugin],
)
```

#### Dynamic Token Provider (Recommended for Production)

Use a callable token provider for automatic token refresh before each request:

```python
import requests

def get_fresh_token() -> str:
    """Fetch a fresh JWT from your identity provider."""
    resp = requests.post(
        "https://your-domain.auth.us-east-1.amazoncognito.com/oauth2/token",
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

config = AgentCorePaymentsPluginConfig(
    payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-jwt",
    token_provider=get_fresh_token,  # Called before each request
    payment_instrument_id="payment-instrument-xyz789",
    payment_session_id="payment-session-def456",
    region="us-east-1",
)

plugin = AgentCorePaymentsPlugin(config=config)
agent = Agent(
    system_prompt="You are a helpful assistant that can access paid APIs.",
    tools=[http_request],
    plugins=[plugin],
)

# 402 responses are handled automatically using JWT auth
agent("Fetch data from https://premium-api.example.com/data")
```

> **Note:** `bearer_token` and `token_provider` are mutually exclusive. Use `token_provider` in
> production for automatic token refresh. Use `bearer_token` for quick testing with a known token.

---

## Configuration Reference

### AgentCorePaymentsPluginConfig Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `payment_manager_arn` | `str` | Yes | — | ARN of the Bedrock AgentCore Payment Manager resource |
| `user_id` | `Optional[str]` | Conditional | `None` | Unique identifier for the user. Required for SigV4 auth; optional with bearer token auth (derived from JWT `sub` claim) |
| `payment_instrument_id` | `Optional[str]` | No | `None` | Payment instrument ID. Can be set later via `update_payment_instrument_id()` |
| `payment_session_id` | `Optional[str]` | No | `None` | Payment session ID. Can be set later via `update_payment_session_id()` |
| `region` | `Optional[str]` | No | `None` | AWS region for the payment manager |
| `network_preferences_config` | `Optional[list[str]]` | No | `None` | List of network CAIP-2 identifiers in order of preference |
| `auto_payment` | `bool` | No | `True` | Whether to automatically process 402 payment requirements |
| `max_interrupt_retries` | `int` | No | `5` | Maximum interrupt retries per tool use. Set to 0 to disable interrupts |
| `agent_name` | `Optional[str]` | No | `None` | Agent name propagated via HTTP header on API calls |
| `bearer_token` | `Optional[str]` | No | `None` | Static JWT bearer token for CUSTOM_JWT auth. Mutually exclusive with `token_provider` |
| `token_provider` | `Optional[Callable[[], str]]` | No | `None` | Callable returning a fresh JWT token string. Mutually exclusive with `bearer_token` |
| `payment_tool_allowlist` | `Optional[List[str]]` | No | `None` | List of tool names eligible for automatic payment processing. When `None`, all tools are eligible |

---

## End-to-End Examples

### Calling Coinbase Bazaar Tools via MCP Client

This example shows automatic 402 payment handling with Strands and a direct MCP connection to Coinbase Bazaar.

**Environment Variables:**

```bash
PAYMENT_MANAGER_ARN=arn:aws:bedrock-agentcore:<region>:<account>:payment-manager/<name>
USER_ID=<user-id>
PAYMENT_INSTRUMENT_ID=<instrument-id>
PAYMENT_SESSION_ID=<session-id>
AWS_REGION=us-west-2
MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0
```

**Agent Code:**

```python
import os
from dotenv import load_dotenv
load_dotenv()

from datetime import timedelta
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from bedrock_agentcore.payments.integrations.strands import (
    AgentCorePaymentsPlugin,
    AgentCorePaymentsPluginConfig,
)

MODEL_ID              = os.environ.get("MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
PAYMENT_MANAGER_ARN   = os.environ["PAYMENT_MANAGER_ARN"]
USER_ID               = os.environ["USER_ID"]
PAYMENT_INSTRUMENT_ID = os.environ["PAYMENT_INSTRUMENT_ID"]
PAYMENT_SESSION_ID    = os.environ["PAYMENT_SESSION_ID"]
REGION                = os.environ.get("AWS_REGION", "us-west-2")

COINBASE_BAZAAR_URL = "https://api.cdp.coinbase.com/platform/v2/x402/discovery/mcp"

def main():
    # 1. Connect to Coinbase Bazaar MCP server
    mcp_client = MCPClient(lambda: streamablehttp_client(
        COINBASE_BAZAAR_URL,
        timeout=timedelta(seconds=120),
    ))

    # 2. Configure payment plugin
    payment_plugin = AgentCorePaymentsPlugin(config=AgentCorePaymentsPluginConfig(
        payment_manager_arn=PAYMENT_MANAGER_ARN,
        user_id=USER_ID,
        payment_instrument_id=PAYMENT_INSTRUMENT_ID,
        payment_session_id=PAYMENT_SESSION_ID,
        region=REGION,
    ))

    # 3. Create agent — plugin handles 402 payments automatically
    with mcp_client:
        agent = Agent(
            model=BedrockModel(model_id=MODEL_ID, streaming=True),
            tools=mcp_client.list_tools_sync(),
            plugins=[payment_plugin],
        )
        result = agent("Get me the latest crypto news")
        print(result.message)

if __name__ == "__main__":
    main()
```

### Calling Coinbase Bazaar Tools via AgentCore Gateway

This example demonstrates how to leverage AgentCore Gateway to interact with Coinbase Bazaar MCP tools.

**Prerequisite:** Add "Coinbase x402 Bazaar" as a target in your Gateway.

**Environment Variables:**

```bash
GATEWAY_URL=https://<gateway-id>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
CLIENT_ID=<cognito-client-id>
CLIENT_SECRET=<cognito-client-secret>
TOKEN_URL=https://<domain>.auth.<region>.amazoncognito.com/oauth2/token
PAYMENT_MANAGER_ARN=arn:aws:bedrock-agentcore:<region>:<account>:payment-manager/<name>
USER_ID=<user-id>
PAYMENT_INSTRUMENT_ID=<instrument-id>
PAYMENT_SESSION_ID=<session-id>
AWS_REGION=us-west-2
MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0
```

**Agent Code:**

```python
import os
from dotenv import load_dotenv
load_dotenv()

from datetime import timedelta
import requests as http_requests
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from bedrock_agentcore.payments.integrations.strands import (
    AgentCorePaymentsPlugin,
    AgentCorePaymentsPluginConfig,
)

GATEWAY_URL           = os.environ["GATEWAY_URL"]
CLIENT_ID             = os.environ["CLIENT_ID"]
CLIENT_SECRET         = os.environ["CLIENT_SECRET"]
TOKEN_URL             = os.environ["TOKEN_URL"]
MODEL_ID              = os.environ.get("MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
PAYMENT_MANAGER_ARN   = os.environ["PAYMENT_MANAGER_ARN"]
USER_ID               = os.environ["USER_ID"]
PAYMENT_INSTRUMENT_ID = os.environ["PAYMENT_INSTRUMENT_ID"]
PAYMENT_SESSION_ID    = os.environ["PAYMENT_SESSION_ID"]
REGION                = os.environ.get("AWS_REGION", "us-west-2")

def get_oauth_token():
    resp = http_requests.post(TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }, headers={"Content-Type": "application/x-www-form-urlencoded"})
    resp.raise_for_status()
    return resp.json()["access_token"]

def main():
    token = get_oauth_token()

    # 1. Connect to Gateway MCP server
    mcp_client = MCPClient(lambda: streamablehttp_client(
        GATEWAY_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timedelta(seconds=120),
    ))

    # 2. Configure payment plugin
    payment_plugin = AgentCorePaymentsPlugin(config=AgentCorePaymentsPluginConfig(
        payment_manager_arn=PAYMENT_MANAGER_ARN,
        user_id=USER_ID,
        payment_instrument_id=PAYMENT_INSTRUMENT_ID,
        payment_session_id=PAYMENT_SESSION_ID,
        region=REGION,
    ))

    # 3. Create agent — plugin handles 402 payments automatically
    with mcp_client:
        agent = Agent(
            model=BedrockModel(model_id=MODEL_ID, streaming=True),
            tools=mcp_client.list_tools_sync(),
            plugins=[payment_plugin],
        )
        result = agent("Get me the latest crypto news")
        print(result.message)

if __name__ == "__main__":
    main()
```

The developer writes no payment logic. The plugin intercepts 402 responses, generates payment proofs
via AgentCore, and retries the tool call automatically.

---

## Supported Response Formats

The plugin supports multiple tool response formats:

- **Spec-compliant marker format**: Tools return `PAYMENT_REQUIRED: {json}` in content blocks
- **Legacy http_request format**: `Status Code:`, `Headers:`, `Body:` text blocks
- **MCP Gateway format**: `structuredContent` with `x402Version` and `accepts` fields

### Payment Handler Resolution

The plugin selects the appropriate handler based on tool characteristics:

| Strategy | Condition | Handler |
|----------|-----------|---------|
| Name-based registry | `http_request` tool | `HttpRequestPaymentHandler` |
| Shape detection | Tools with `toolName` + `parameters` input | `MCPRequestPaymentHandler` |
| Generic fallback | All other tools | `GenericPaymentHandler` |

---

## Important Notes

### Payment Session Limits

Payment sessions have configurable spending limits and expiry times (15–480 minutes). Monitor session
budgets using the `get_payment_session` tool to avoid `InsufficientBudget` errors.

### Retry Limits and Post-Payment Failure Detection

The plugin enforces a maximum of 3 payment retry attempts per tool use and a configurable maximum of
5 interrupt retries. These limits are checked independently — interrupt retry limits do not gate
402 payment processing.

Additionally, the plugin detects **post-payment failures**: if a 402 response is received *after*
a payment retry was already attempted (e.g., due to insufficient balance or invalid signature),
the plugin propagates the failure as an interrupt instead of retrying again. This prevents infinite
loops where the plugin keeps signing and retrying against a server that rejects the payment for
non-retryable reasons.

### Custom Tools

To make your custom tools compatible with automatic payment processing, return responses using the
spec-compliant `PAYMENT_REQUIRED:` marker format:

```python
import json

payment_required = {
    "statusCode": 402,
    "headers": response_headers,
    "body": response_body,
}

return {
    "status": "error",
    "content": [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_required)}"}],
}
```

### Thread Safety

`AgentCorePaymentsPlugin` is not thread-safe. Create separate plugin instances for concurrent agents.

### x402 Protocol Support

- **v1**: Payment header returned as `X-PAYMENT` (base64-encoded JSON)
- **v2**: Payment header returned as `PAYMENT-SIGNATURE` (base64-encoded JSON with resource and extension fields)

### Supported Blockchain Networks

- **Ethereum**: Base, Ethereum mainnet, Arbitrum, Optimism, Sepolia testnets
- **Solana**: Mainnet, Devnet, Testnet (identified by CAIP-2 genesis hashes or simplified names)
