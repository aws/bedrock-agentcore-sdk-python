# Bedrock AgentCore payments SDK

High-level Python SDK for AWS Bedrock AgentCore payments service with support for payment instrument management,
session-based payment limits, and x402 payment processing for AI agent microtransactions.

## Overview

The Bedrock AgentCore Payments SDK enables AI agents to process microtransaction payments to access paid APIs,
MCP servers, and premium content. The SDK supports the [x402 Payment Required](https://www.x402.org/) protocol,
allowing agents to automatically handle HTTP 402 responses and complete cryptocurrency transactions on behalf of users.

### Architecture

The payments system operates on a hierarchical structure:

```
PaymentClient (Control Plane)
  └── Payment Manager
        └── Payment Connector ──▶ Payment Credential Provider (vendor credentials)
              └── Payment Instrument (user's wallet)
                    └── Payment Session (time-bounded spending context)
```

- **Payment Credential Provider** — stores vendor credentials (e.g., Coinbase CDP API keys, Stripe Privy credentials) securely
- **Payment Manager** — top-level resource that owns connectors and orchestrates payment operations
- **Payment Connector** — links a payment manager to a credential provider for a specific payment vendor
- **Payment Instrument** — a user's registered payment method (e.g., embedded crypto wallet) created through a connector
- **Payment Session** — a time-bounded spending context with configurable limits

### Core Components

| Component | Layer | Purpose |
|-----------|-------|---------|
| `PaymentClient` | Control plane | Create and manage payment infrastructure (managers, connectors, credential providers) |
| `PaymentManager` | Data plane | Payment operations (instruments, sessions, payment processing, header generation) |
| `AgentCorePaymentsPlugin` | Framework integration | Strands Agents plugin for automatic x402 payment handling ([see Strands README](integrations/strands/README.md)) |

## Installation

```bash
pip install bedrock-agentcore
```

For Strands Agents integration:

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

## Prerequisites

AgentCore Payments connects to external payment providers for wallet operations. You must obtain
credentials from at least one supported provider before creating a Payment Connector.

**Supported providers:**
- **Coinbase CDP** — API key ID, API key secret, and wallet secret
- **Stripe Privy** — App ID, app secret, and optional authorization key

## Authentication

The SDK supports two authentication modes:

### AWS IAM (Default)

Uses standard AWS credentials via any of:
- AWS CLI credentials (`aws configure`)
- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- IAM roles (EC2 instance roles, ECS task roles, Lambda execution roles)

### Custom JWT (Bearer Token)

For OAuth/CUSTOM_JWT authentication, provide a bearer token or token provider:

```python
from bedrock_agentcore.payments import PaymentManager

# Static bearer token (for testing)
manager = PaymentManager(
    payment_manager_arn="arn:...",
    bearer_token="your-jwt-token",
)

# Dynamic token provider (recommended for production)
manager = PaymentManager(
    payment_manager_arn="arn:...",
    token_provider=lambda: get_fresh_token(),
)
```

> **Note:** `bearer_token` and `token_provider` are mutually exclusive. When using bearer token auth,
> the service derives `userId` from the JWT `sub` claim, so `user_id` is optional on data plane calls.

### Region Resolution Order

1. `region_name` parameter passed to `PaymentManager` or `PaymentClient`
2. Region from `boto3_session` if provided
3. `AWS_REGION` environment variable
4. `boto3.Session().region_name` (checks `AWS_DEFAULT_REGION` and AWS config)
5. Default fallback: `us-west-2`

## Quick Start

```python
import os
from bedrock_agentcore.payments import PaymentManager

manager = PaymentManager(
    payment_manager_arn=os.environ["PAYMENT_MANAGER_ARN"],
    region_name="us-east-1",
)

# Create a payment instrument (embedded crypto wallet)
instrument = manager.create_payment_instrument(
    payment_connector_id=os.environ["PAYMENT_CONNECTOR_ID"],
    payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
    payment_instrument_details={
        "embeddedCryptoWallet": {
            "network": "ETHEREUM",
            "linkedAccounts": [
                {"email": {"emailAddress": "user@example.com"}}
            ],
        }
    },
    user_id="user-123",
)

# Create a payment session with spending limits
session = manager.create_payment_session(
    expiry_time_in_minutes=60,
    user_id="user-123",
    limits={"maxSpendAmount": {"value": "100.00", "currency": "USD"}},
)

# Check instrument balance
balance = manager.get_payment_instrument_balance(
    payment_connector_id=os.environ["PAYMENT_CONNECTOR_ID"],
    payment_instrument_id=instrument["paymentInstrumentId"],
    chain="BASE_SEPOLIA",
    token="USDC",
    user_id="user-123",
)
print(f"Balance: {balance}")
```

## Usage

### Creating Payment Manager and Connector

> **Note:** Payment resource creation is typically done once, separately from your agent application.
> In production, create these resources through the AWS Console or a separate setup script, then use
> the `paymentManagerArn` and `paymentConnectorId` in your agent application.

```python
import os
from bedrock_agentcore.payments.client import PaymentClient

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
                "api_key_id": os.environ["COINBASE_API_KEY_ID"],
                "api_key_secret": os.environ["COINBASE_API_KEY_SECRET"],
                "wallet_secret": os.environ["COINBASE_WALLET_SECRET"],
            },
        },
    },
    wait_for_ready=True,
    max_wait=300,
    poll_interval=5,
)

# Extract details from response
payment_manager_arn = response["paymentManager"]["paymentManagerArn"]
payment_connector_id = response["paymentConnector"]["paymentConnectorId"]
credential_provider_arn = response["credentialProvider"]["credentialProviderArn"]
print(f"Payment Manager ARN: {payment_manager_arn}")
print(f"Payment Connector ID: {payment_connector_id}")
print(f"Credential Provider ARN: {credential_provider_arn}")
```

The `wait_for_ready=True` parameter causes the method to poll until all resources reach READY status.
If any step fails, previously created resources are automatically rolled back.

For Stripe Privy, use `"StripePrivy"` as the `credential_provider_vendor` with the appropriate credentials:

```python
"credentials": {
    "app_id": os.environ["STRIPE_PRIVY_APP_ID"],
    "app_secret": os.environ["STRIPE_PRIVY_APP_SECRET"],
    "authorization_key": os.environ.get("STRIPE_PRIVY_AUTH_KEY", ""),  # optional
}
```

---

### Creating a Payment Instrument

Create a payment instrument for a user. Below is an example creating an Ethereum-compatible embedded crypto wallet:

```python
from bedrock_agentcore.payments import PaymentManager

manager = PaymentManager(
    payment_manager_arn=payment_manager_arn,
    region_name="us-east-1",
)

instrument = manager.create_payment_instrument(
    payment_connector_id=payment_connector_id,
    payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
    payment_instrument_details={
        "embeddedCryptoWallet": {
            "network": "ETHEREUM",
            "linkedAccounts": [
                {"email": {"emailAddress": "user@example.com"}}
            ],
        }
    },
    user_id="test-user-123",
)

payment_instrument_id = instrument["paymentInstrumentId"]
```

For Solana-compatible chains, use `"SOLANA"` for the network input:

```python
instrument = manager.create_payment_instrument(
    payment_connector_id=payment_connector_id,
    payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
    payment_instrument_details={
        "embeddedCryptoWallet": {
            "network": "SOLANA",
            "linkedAccounts": [
                {"email": {"emailAddress": "user@example.com"}}
            ],
        }
    },
    user_id="test-user-123",
)
```

> **Important:** Once created, the instrument must be funded and permission granted for signing
> before the agent can use it. These are end-user actions that should be completed before using
> the payment instrument in your agent.
>
> - **Coinbase**: You'll receive a `redirectUrl` in the response pointing to the Coinbase-hosted
>   WalletHub. Redirect your user there to grant signing permission and transfer funds.
> - **Stripe**: Developers use a provided URL template to host a frontend page where end users
>   can take the same actions.

---

### Querying Instrument Balance

Check the token balance for a payment instrument on a specific chain:

```python
balance = manager.get_payment_instrument_balance(
    payment_connector_id=payment_connector_id,
    payment_instrument_id=payment_instrument_id,
    chain="BASE_SEPOLIA",
    token="USDC",
    user_id="test-user-123",
)
print(f"Balance: {balance}")
```

Supported chains include `BASE_SEPOLIA`, `BASE`, `SOLANA_DEVNET`, `SOLANA_MAINNET`, etc.

---

### Creating a Payment Session

Create a payment session before processing payments:

```python
session = manager.create_payment_session(
    expiry_time_in_minutes=60,
    user_id="test-user-123",
    limits={"maxSpendAmount": {"value": "100.00", "currency": "USD"}},
)

payment_session_id = session["paymentSessionId"]
```

Check session status and remaining payment limits:

```python
session_details = manager.get_payment_session(
    payment_session_id=payment_session_id,
    user_id="test-user-123",
)
print(f"Available: {session_details.get('availableLimits', {}).get('availableSpendAmount')}")
```

List all sessions for a user:

```python
sessions = manager.list_payment_sessions(user_id="test-user-123")
```

---

### Processing Payments

Process a payment transaction directly:

```python
payment = manager.process_payment(
    payment_session_id=payment_session_id,
    payment_instrument_id=payment_instrument_id,
    payment_type="CRYPTO_X402",
    payment_input={
        "cryptoX402": {
            "version": "1",
            "payload": {
                "scheme": "exact",
                "network": "base-sepolia",
                "maxAmountRequired": "5000",
                "resource": "https://example.com/premium-api",
                "description": "Premium API access",
                "mimeType": "application/json",
                "payTo": "0x6813749E1eB9E0001A44C2684695FE8AD676cdD9",
                "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF71",
            },
        }
    },
    user_id="test-user-123",
)
```

---

### Payment Header Generation

Generate x402 payment headers for HTTP 402 Payment Required responses. This is the core method
used by the Strands plugin under the hood:

```python
header = manager.generate_payment_header(
    payment_instrument_id=payment_instrument_id,
    payment_session_id=payment_session_id,
    payment_required_request={
        "statusCode": 402,
        "headers": {"content-type": "application/json"},
        "body": {
            "x402Version": 1,
            "accepts": [
                {
                    "scheme": "exact",
                    "network": "base-sepolia",
                    "maxAmountRequired": "5000",
                    "resource": "https://example.com/api",
                    "payTo": "0x...",
                    "asset": "0x...",
                }
            ],
        },
    },
    user_id="test-user-123",
    network_preferences=["base-sepolia", "eip155:8453", "solana-mainnet"],
)
# Returns: {"X-PAYMENT": "base64..."} (v1) or {"PAYMENT-SIGNATURE": "base64..."} (v2)
```

**Network selection process:**
1. Filter accepts to those matching the instrument's blockchain type (Ethereum or Solana)
2. Use provided `network_preferences` or fall back to the default `NETWORK_PREFERENCES`
3. Pick the first network from preferences that matches a filtered accept
4. If no match found, return the first filtered accept

---

### Deleting Resources

#### Data Plane (PaymentManager)

Delete a payment session (hard delete — permanently removes the record):

```python
result = manager.delete_payment_session(
    payment_session_id="payment-session-abc123",
    user_id="user-123",
)
# result: {"status": "DELETED"}
```

Delete a payment instrument (soft delete — marks as DELETED, preserved for audit):

```python
result = manager.delete_payment_instrument(
    payment_instrument_id="payment-instrument-xyz789",
    payment_connector_id="connector-456",
    user_id="user-123",
)
# result: {"status": "DELETED"}
```

> **Note:** Deleting a non-existent or already-deleted resource raises `PaymentSessionNotFound`
> or `PaymentInstrumentNotFound`.

#### Control Plane (PaymentClient)

```python
from bedrock_agentcore.payments.client import PaymentClient

client = PaymentClient(region_name="us-east-1")

# Delete connector first, then manager
client.delete_payment_connector(
    payment_manager_id="pm-123",
    payment_connector_id="connector-456",
)

client.delete_payment_manager(payment_manager_id="pm-123")
```

> **Important:** Delete resources in the correct order to avoid dependency errors:
> 1. Delete payment instruments first
> 2. Delete payment sessions
> 3. Delete payment connectors
> 4. Delete the payment manager last

---

### Using CUSTOM_JWT (Bearer Token) Authentication

When your payment manager uses `CUSTOM_JWT` authorizer type, configure the `PaymentManager` with a
bearer token or token provider. The service derives `userId` from the JWT `sub` claim, so `user_id`
is optional on all data plane calls.

#### Setting Up a CUSTOM_JWT Payment Manager

```python
from bedrock_agentcore.payments.client import PaymentClient

client = PaymentClient(region_name="us-east-1")

manager_response = client.create_payment_manager(
    name="JWTPaymentManager",
    role_arn="arn:aws:iam::123456789012:role/BedrockAgentCoreFullAccess",
    authorizer_type="CUSTOM_JWT",
    authorizer_configuration={
        "customJWTConfiguration": {
            "issuer": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_EXAMPLE",
            "audiences": ["your-client-id"],
        }
    },
    wait_for_ready=True,
)
```

#### Using PaymentManager with a Token Provider

```python
import requests
from bedrock_agentcore.payments import PaymentManager

def get_fresh_token() -> str:
    """Fetch a fresh JWT from your identity provider."""
    resp = requests.post(
        "https://your-domain.auth.us-east-1.amazoncognito.com/oauth2/token",
        data={
            "grant_type": "client_credentials",
            "client_id": "your-client-id",
            "client_secret": "your-client-secret",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

manager = PaymentManager(
    payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-jwt",
    region_name="us-east-1",
    token_provider=get_fresh_token,  # Called before each request
)

# user_id is not required — the service extracts it from the JWT 'sub' claim
instrument = manager.create_payment_instrument(
    payment_connector_id="connector-456",
    payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
    payment_instrument_details={
        "embeddedCryptoWallet": {
            "network": "ETHEREUM",
            "linkedAccounts": [
                {"email": {"emailAddress": "user@example.com"}}
            ],
        }
    },
)
```

> **Note:** `bearer_token` and `token_provider` are mutually exclusive. Use `token_provider` in
> production for automatic token refresh. Use `bearer_token` for quick testing with a known token.

---

### Control Plane Operations

For individual resource management (alternative to `create_payment_manager_with_connector`):

```python
from bedrock_agentcore.payments.client import PaymentClient

client = PaymentClient(region_name="us-east-1")

# Create a payment manager
manager_response = client.create_payment_manager(
    name="MyPaymentManager",
    role_arn="arn:aws:iam::123456789012:role/PaymentRole",
    authorizer_type="AWS_IAM",
    wait_for_ready=True,
)

# Create a payment connector
connector_response = client.create_payment_connector(
    payment_manager_id=manager_response["paymentManagerId"],
    name="MyCoinbaseConnector",
    connector_type="CoinbaseCDP",
    credential_provider_configurations=[
        {"coinbaseCDP": {"credentialProviderArn": "arn:..."}}
    ],
    wait_for_ready=True,
)

# List payment managers
managers = client.list_payment_managers(max_results=10)

# List connectors for a manager
connectors = client.list_payment_connectors(
    payment_manager_id=manager_response["paymentManagerId"],
)

# Update a payment manager
client.update_payment_manager(
    payment_manager_id=manager_response["paymentManagerId"],
    description="Updated description",
)
```

---

## Error Handling

### Common Exceptions

```python
from bedrock_agentcore.payments import (
    PaymentError,
    PaymentInstrumentNotFound,
    PaymentSessionNotFound,
    InvalidPaymentInstrument,
    InsufficientBudget,
    PaymentSessionExpired,
    PaymentInstrumentConfigurationRequired,
    PaymentSessionConfigurationRequired,
)

try:
    payment = manager.process_payment(
        payment_session_id="session-456",
        payment_instrument_id="instrument-789",
        payment_type="CRYPTO_X402",
        payment_input={...},
        user_id="user-123",
    )
except PaymentInstrumentNotFound:
    print("Payment instrument not found. Create one first.")
except PaymentSessionNotFound:
    print("Payment session not found or expired.")
except PaymentSessionExpired:
    print("Payment session has expired. Create a new session.")
except InsufficientBudget:
    print("Payment amount exceeds remaining session budget.")
except InvalidPaymentInstrument:
    print("Payment instrument is invalid or inactive.")
except PaymentError as e:
    print(f"Payment operation failed: {e}")
```

### Best Practices for Error Handling

1. **Handle specific exceptions first** — catch `PaymentInstrumentNotFound`, `InsufficientBudget`, etc. before the generic `PaymentError`
2. **Handle rate limiting gracefully** — catch `ClientError` with `ThrottlingException` code and retry with backoff
3. **Log errors for debugging** — use structured logging with `exc_info=True` for full tracebacks

---

## Best Practices

### Infrastructure Setup

- Use `create_payment_manager_with_connector()` for one-step setup with automatic rollback
- Use `PaymentClient` only for control plane operations (creating/managing managers and connectors)
- Use `PaymentManager` for all data plane operations (instruments, sessions, payments)

### Instrument Management

- Use `EMBEDDED_CRYPTO_WALLET` as the instrument type
- Ensure instruments are funded and signing permissions are granted before use
- Use `"ETHEREUM"` or `"SOLANA"` for the network field

### Session Management

- Set appropriate `expiry_time_in_minutes` values (15–480 minutes)
- Configure spending limits to control maximum transaction amounts
- Monitor remaining payment limits via `get_payment_session` before processing payments

### Network Preferences

- Provide `network_preferences` to control blockchain network selection order
- Default preferences prioritize Solana (fast, low cost) then Ethereum networks
- Ensure your payment instrument's network matches at least one accept in the x402 payload

### Security

- Use IAM roles instead of hardcoded credentials in production
- Use `token_provider` (callable) over static `bearer_token` for automatic token refresh
- Never log or expose bearer tokens or API key secrets
- Use `client_token` for idempotent payment operations

### Thread Safety

- `PaymentManager` is **not** thread-safe — create separate instances for concurrent operations
- Reuse instances within a single thread for connection pooling benefits

---

## API Reference

### PaymentManager Methods

| Method | Description |
|--------|-------------|
| `create_payment_instrument()` | Create a payment instrument (embedded crypto wallet) |
| `get_payment_instrument()` | Retrieve payment instrument details |
| `get_payment_instrument_balance()` | Query token balance for an instrument on a specific chain |
| `list_payment_instruments()` | List payment instruments for a user |
| `delete_payment_instrument()` | Delete a payment instrument (soft delete) |
| `create_payment_session()` | Create a time-bounded payment session with spending limits |
| `get_payment_session()` | Retrieve payment session details |
| `list_payment_sessions()` | List payment sessions for a user |
| `delete_payment_session()` | Delete a payment session (hard delete) |
| `process_payment()` | Process a payment transaction |
| `generate_payment_header()` | Generate x402 payment headers for 402 responses |

### PaymentManager Constructor Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `payment_manager_arn` | `str` | Yes | ARN of the payment manager instance |
| `region_name` | `Optional[str]` | No | AWS region for the client |
| `boto3_session` | `Optional[boto3.Session]` | No | Custom boto3 session |
| `boto_client_config` | `Optional[BotocoreConfig]` | No | Custom botocore client config |
| `agent_name` | `Optional[str]` | No | Agent name propagated via HTTP header |
| `bearer_token` | `Optional[str]` | No | Static JWT for CUSTOM_JWT auth |
| `token_provider` | `Optional[Callable[[], str]]` | No | Callable returning fresh JWT |

### PaymentClient Methods

| Method | Description |
|--------|-------------|
| `create_payment_manager()` | Create a payment manager resource |
| `get_payment_manager()` | Retrieve payment manager details |
| `list_payment_managers()` | List payment managers |
| `update_payment_manager()` | Update a payment manager |
| `delete_payment_manager()` | Delete a payment manager |
| `create_payment_connector()` | Create a payment connector |
| `get_payment_connector()` | Retrieve payment connector details |
| `list_payment_connectors()` | List payment connectors for a manager |
| `update_payment_connector()` | Update a payment connector |
| `delete_payment_connector()` | Delete a payment connector |
| `create_payment_manager_with_connector()` | One-step setup with automatic rollback |

### Exception Classes

| Exception | Description |
|-----------|-------------|
| `PaymentError` | Base exception for all payment operations |
| `PaymentInstrumentNotFound` | Payment instrument does not exist |
| `PaymentSessionNotFound` | Payment session does not exist |
| `InvalidPaymentInstrument` | Payment instrument is invalid or inactive |
| `InsufficientBudget` | Payment amount exceeds remaining payment limits |
| `PaymentSessionExpired` | Payment session has expired |
| `PaymentInstrumentConfigurationRequired` | Payment instrument ID not configured |
| `PaymentSessionConfigurationRequired` | Payment session ID not configured |

### Constants

| Constant | Description |
|----------|-------------|
| `PaymentManagerStatus` | Payment manager resource statuses (CREATING, READY, etc.) |
| `PaymentConnectorStatus` | Payment connector statuses |
| `PaymentConnectorType` | Supported connector types (CoinbaseCDP, StripePrivy) |
| `PaymentsAuthorizerType` | Authorizer types (AWS_IAM, CUSTOM_JWT) |
| `NETWORK_PREFERENCES` | Default blockchain network preference order |
| `DEFAULT_MAX_RESULTS` | Default pagination limit (100) |

---

## Strands Agents Integration

For automatic x402 payment handling in Strands Agents, see the dedicated
[Strands AgentCore Payments Plugin README](integrations/strands/README.md).

The plugin provides:
- Automatic interception and processing of HTTP 402 responses
- Built-in payment query tools for agents
- Interrupt-based error handling for payment failures
- Configurable auto-payment and tool allowlists
