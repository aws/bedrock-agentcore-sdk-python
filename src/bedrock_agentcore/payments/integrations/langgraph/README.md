# LangGraph AgentCore Payments Middleware

The AgentCore Payments Middleware enables LangGraph agents to autonomously handle [x402 Payment Required](https://www.x402.org/) responses. When a tool hits a paid API that returns HTTP 402, the middleware automatically detects the payment requirement, signs the payment via PaymentManager, and retries the request with payment credentials — all transparent to the LLM.

## Overview

- **Automatic x402 Payment Handling** — intercepts 402 responses, processes payment, retries with proof
- **Zero Wrapper Code** — no manual tool wrapping needed; just pass `middleware=[payments]` to `create_agent`
- **Multi-Format Detection** — handles `PAYMENT_REQUIRED:` marker, raw JSON `statusCode: 402`, and x402 payloads
- **Custom Handler Registry** — register handlers for tools with non-standard response formats
- **Built-in Tools** — payment-aware `http_request` + payment query tools auto-registered
- **Deterministic Error Messages** — tailored, actionable error messages returned to the LLM on failure
- **Async Support** — non-blocking `asyncio.sleep` and `asyncio.to_thread` for the async path
- **Auto-Session** — optionally create payment sessions lazily on first payment

## How It Works

```
┌─────────┐     ┌──────────────────────────────┐     ┌────────────┐
│  Agent  │────▶│  wrap_tool_call (middleware)  │────▶│    Tool    │──── HTTP ───▶ Paid API
│         │     │                              │     └────────────┘              │
│         │     │  1. Execute tool              │                               │
│         │     │  2. Detect 402               │◀── 402 + x402 payload ─────────┘
│         │     │  3. Sign payment (PM)         │
│         │     │  4. Inject header            │
│         │     │  5. Wait (blockchain delay)   │
│         │     │  6. Retry tool               │──── HTTP + payment header ──▶ Paid API
│         │◀────│  7. Return 200 to agent      │◀── 200 + content ──────────────┘
└─────────┘     └──────────────────────────────┘
```

## Quick Start

```python
from langchain.agents import create_agent
from bedrock_agentcore.payments.integrations.langgraph import (
    AgentCorePaymentsConfig,
    AgentCorePaymentsMiddleware,
)

# 1. Config
config = AgentCorePaymentsConfig(
    payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123",
    user_id="user-123",
    payment_instrument_id="instrument-456",
    region="us-east-1",
    auto_session=True,  # session created automatically on first payment
)

# 2. Middleware
payments = AgentCorePaymentsMiddleware(config)

# 3. Agent — that's it
agent = create_agent(
    model="claude-sonnet-4-20250514",
    tools=[],  # middleware auto-registers http_request + payment query tools
    middleware=[payments],
)

# 402 responses are handled automatically
result = agent.invoke({"messages": [{"role": "user", "content": "Fetch data from https://paid-api.example.com/data"}]})
```

## Built-in Tools

The middleware automatically registers these tools (available to the LLM):

| Tool | Description |
|------|-------------|
| `http_request` | Call any HTTP endpoint. 402 responses are paid automatically. |
| `get_payment_instrument` | Query details about a payment instrument |
| `list_payment_instruments` | List all instruments for a user |
| `get_payment_instrument_balance` | Check wallet balance on a chain |
| `get_payment_session` | Query session budget, status, expiry |

Set `provide_http_request=False` if you bring your own HTTP tool.

## Custom Tool Integration Contract

For your own tools to work with auto-payment, they need two things:

### 1. Signal 402 (output)

The tool must indicate a 402 response in its return value. Three formats are supported:

**Option A: `PAYMENT_REQUIRED:` marker (recommended)**
```python
@tool
def my_api(query: str, headers: dict = None) -> str:
    resp = httpx.get(URL, headers=headers or {})
    if resp.status_code == 402:
        payload = {"statusCode": 402, "headers": dict(resp.headers), "body": resp.json()}
        return f"PAYMENT_REQUIRED: {json.dumps(payload)}"
    return resp.text
```

**Option B: Raw JSON with `statusCode: 402` (fallback detection)**
```python
@tool
def my_api(query: str, headers: dict = None) -> str:
    resp = httpx.get(URL, headers=headers or {})
    return json.dumps({"statusCode": resp.status_code, "headers": dict(resp.headers), "body": resp.json()})
```

**Option C: Custom handler (for non-standard formats)**
```python
config = AgentCorePaymentsConfig(
    ...,
    custom_handlers={"my_tool": MyCustomHandler()},
)
```

### 2. Accept and forward `headers` (input)

The tool **must** have a `headers` parameter and use it in its HTTP request. The middleware injects the payment header into `tool_args["headers"]` before retry:

```python
@tool
def my_api(query: str, headers: dict = None) -> str:
    resp = httpx.get(URL, headers=headers or {})  # ← forwards headers
    ...
```

Without this, the payment header is injected but never sent to the server.

### Minimal custom tool template

```python
@tool
def my_paid_tool(query: str, headers: dict = None) -> str:
    """Access a paid API. Payments handled automatically."""
    resp = httpx.get("https://paid-api.example.com/data", headers=headers or {})
    if resp.status_code == 402:
        payload = {"statusCode": 402, "headers": dict(resp.headers), "body": resp.json()}
        return f"PAYMENT_REQUIRED: {json.dumps(payload)}"
    return json.dumps(resp.json())
```

## Detection Priority

When a tool returns, the middleware checks for 402 in this order:

1. **Custom handler** (if registered for this tool name) — full control over detection
2. **`PAYMENT_REQUIRED:` marker** — explicit opt-in signal in content
3. **Lenient fallback** — parses raw JSON for `statusCode: 402` or `x402Version` + `accepts` fields

This means MCP tools and other tools that return raw JSON are handled automatically without needing the marker or a custom handler.

## Error Handling

When payment processing fails, the middleware gives you two layers of control:

1. **Error handler callback** (`on_payment_error`) — your code resolves the issue programmatically, and the middleware retries
2. **Deterministic error ToolMessages** — if no callback is set (or it returns `PROPAGATE`), the LLM receives a tailored error message

### Error Handler Callback (Recommended)

Register a callback to handle payment errors programmatically — auto-provision missing resources, refresh expired sessions, or create new instruments — **without the LLM ever seeing an error**.

```python
from bedrock_agentcore.payments.integrations.langgraph import (
    AgentCorePaymentsConfig,
    AgentCorePaymentsMiddleware,
    PaymentErrorContext,
    ErrorResolution,
)

def handle_payment_error(ctx: PaymentErrorContext) -> ErrorResolution | str:
    if ctx.exception_type in ("PaymentSessionNotFound", "PaymentSessionExpired"):
        # Create a fresh session and retry
        session = pm.create_payment_session(
            user_id=ctx.config.user_id,
            limits={"maxSpendAmount": {"value": "5.00", "currency": "USD"}},
            expiry_time_in_minutes=60,
        )
        ctx.config.payment_session_id = session["paymentSessionId"]
        return ErrorResolution.RETRY

    if ctx.exception_type == "InsufficientBudget":
        # Create a new session with higher budget
        session = pm.create_payment_session(
            user_id=ctx.config.user_id,
            limits={"maxSpendAmount": {"value": "10.00", "currency": "USD"}},
            expiry_time_in_minutes=60,
        )
        ctx.config.payment_session_id = session["paymentSessionId"]
        return ErrorResolution.RETRY

    if ctx.exception_type == "PaymentInstrumentConfigurationRequired":
        # Custom message — direct the user to your setup page
        return "Payment instrument not configured. Please visit https://myapp.com/wallet/setup to set up your wallet."

    # Can't handle — use the default deterministic error message
    return ErrorResolution.PROPAGATE

config = AgentCorePaymentsConfig(
    payment_manager_arn="arn:...",
    user_id="user-1",
    payment_instrument_id="instr-1",
    region="us-east-1",
    on_payment_error=handle_payment_error,
    max_error_retries=3,
)
```

The callback can return:

| Return | Behavior |
|--------|----------|
| `ErrorResolution.RETRY` | Retry payment with updated config |
| `ErrorResolution.PROPAGATE` | Use default deterministic error message |
| `str` | Custom message sent to the LLM as `"PAYMENT ERROR: {your string}"` |

#### How It Works

```
Payment exception occurs
    │
    ├── on_payment_error is None? → deterministic error ToolMessage (default behavior)
    │
    ▼
    Invoke callback(PaymentErrorContext)
    │
    ├── Returns PROPAGATE → deterministic error ToolMessage to LLM
    │
    ├── Returns RETRY → re-attempt payment with (potentially updated) config
    │       │
    │       ├── Success → return paid content to LLM ✅
    │       └── Fails again → loop back to callback (up to max_error_retries)
    │
    └── Callback raises exception → fall through to error ToolMessage (no crash)
```

#### PaymentErrorContext Fields

| Field | Type | Description |
|-------|------|-------------|
| `exception` | `Exception` | The exception instance |
| `exception_type` | `str` | Class name (e.g., `"PaymentSessionExpired"`) |
| `exception_message` | `str` | `str(exception)` |
| `tool_name` | `str` | Tool that triggered the 402 |
| `tool_args` | `dict` | The tool call arguments |
| `payment_required_request` | `dict \| None` | The 402 payload (None if error before extraction) |
| `config` | `AgentCorePaymentsConfig` | Mutable reference — modify to fix the issue |
| `retry_count` | `int` | How many times we've retried (starts at 0) |

#### Async Callbacks

The callback can be `async def` — automatically awaited in the async path:

```python
async def async_handler(ctx: PaymentErrorContext) -> ErrorResolution:
    session = await create_session_async(ctx.config.user_id)
    ctx.config.payment_session_id = session["id"]
    return ErrorResolution.RETRY
```

#### Safety Guarantees

- **Max retries**: `max_error_retries=3` (default) prevents infinite loops. Set to 0 to disable.
- **Exception safety**: If the callback raises, the middleware falls through to the error ToolMessage — never crashes.
- **Backward compatible**: `on_payment_error=None` (default) preserves existing behavior.

#### Recommended Resolution Patterns

| Exception | Typical Resolution |
|---|---|
| `PaymentSessionNotFound` / `PaymentSessionExpired` | Create a new session via `pm.create_payment_session(...)`, set `ctx.config.payment_session_id`, return RETRY |
| `InsufficientBudget` | Create a new session with higher limits, or PROPAGATE and let the user decide |
| `PaymentInstrumentConfigurationRequired` | Set `ctx.config.payment_instrument_id` from your app's user → instrument mapping, return RETRY |
| `PaymentInstrumentNotFound` | Likely a config error — PROPAGATE (instrument IDs shouldn't change at runtime) |
| `PaymentSessionConfigurationRequired` | Create a session and set it, or enable `auto_session=True` instead of using the callback for this |
| Generic `PaymentError` | Log it, PROPAGATE — usually transient or unrecoverable |

### Deterministic Error Messages (Default / Fallback)

When no callback is configured, or the callback returns `PROPAGATE`, the LLM receives a tailored error message with instructions not to retry:

| Failure | ToolMessage Content |
|---------|-------------------|
| No instrument configured | `PAYMENT ERROR: No payment instrument configured. Do not retry this call. Inform the user they need to configure a payment instrument before making paid requests.` |
| No session configured | `PAYMENT ERROR: No payment session configured. Do not retry this call. Inform the user they need to create a payment session before making paid requests.` |
| Instrument not found | `PAYMENT ERROR: Payment instrument not found. Do not retry this call. Inform the user their payment instrument ID is invalid or has been deleted.` |
| Session not found | `PAYMENT ERROR: Payment session not found. Do not retry this call. Inform the user their payment session ID is invalid or has expired.` |
| Session expired | `PAYMENT ERROR: Payment session has expired. Do not retry this call. Inform the user they need to create a new payment session.` |
| Insufficient budget | `PAYMENT ERROR: Insufficient budget. The payment amount exceeds the remaining session limit. Do not retry this call. Inform the user they need to increase their session budget or create a new session with higher limits.` |
| Payment rejected by server | `PAYMENT ERROR: Payment was signed but rejected by the server ({detail}). Do not retry this call. Inform the user that the payment was not accepted by the merchant.` |
| Generic payment failure | `PAYMENT ERROR: Payment processing failed ({message}). Do not retry this call. Inform the user that payment could not be completed.` |
| Incompatible tool format | `PAYMENT ERROR: Could not apply payment credentials to this tool's request format. Do not retry this call. Inform the user this tool is not compatible with automatic payment processing.` |


## Custom Handlers

Register custom `PaymentResponseHandler` implementations for tools with non-standard output formats. The custom handler is used for **all three phases**: detection, extraction, and injection.

> **Input contract:** a custom handler's `extract_*` methods receive the **raw `ToolMessage.content`** — a `str` or a list of content blocks — exactly as the tool returned it, not the middleware's internal wrapped shape. Parse it yourself. The built-in handlers (`GenericPaymentHandler`, `HttpRequestPaymentHandler`, `MCPRequestPaymentHandler`) expect a different, normalized shape, so passing one of them directly as a custom handler will silently fail to detect 402s — subclass `PaymentResponseHandler` (or wrap a built-in) and parse the raw content.

```python
from bedrock_agentcore.payments.integrations.handlers import PaymentResponseHandler

class MyMCPHandler(PaymentResponseHandler):
    def extract_status_code(self, result):
        # `result` is the raw ToolMessage.content (str or list of blocks).
        # Parse your tool's output format to detect 402.
        ...

    def extract_headers(self, result):
        # Extract HTTP headers from the 402 response
        ...

    def extract_body(self, result):
        # Extract the x402 payment body
        ...

    def validate_tool_input(self, tool_input):
        # Check that tool_input is suitable for header injection
        return isinstance(tool_input, dict)

    def apply_payment_header(self, tool_input, payment_header):
        # Put the payment header where your tool reads it from
        tool_input["headers"] = tool_input.get("headers", {})
        tool_input["headers"].update(payment_header)
        return True

config = AgentCorePaymentsConfig(
    ...,
    custom_handlers={"my_mcp_tool": MyMCPHandler()},
)
```

## MCP Server Tools

MCP tools connected via `langchain-mcp-adapters` work with the middleware. Since the adapter serializes MCP responses into `ToolMessage.content` as strings, the fallback detection handles the common case (raw JSON with `statusCode: 402`). For non-standard MCP response formats, register a custom handler.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({"paid_api": {"transport": "stdio", "command": "python", "args": ["server.py"]}})
mcp_tools = await client.get_tools()

agent = create_agent(
    model=model,
    tools=mcp_tools,
    middleware=[payments],  # auto-detects 402 from MCP tool responses
)
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `payment_manager_arn` | `str` | *required* | ARN of the payment manager resource |
| `user_id` | `str \| None` | `None` | User ID. Required for SigV4 auth; optional with bearer token |
| `payment_instrument_id` | `str \| None` | `None` | Instrument ID for x402 signing |
| `payment_session_id` | `str \| None` | `None` | Session ID for budget enforcement |
| `payment_connector_id` | `str \| None` | `None` | Connector ID (optional) |
| `region` | `str \| None` | `None` | AWS region |
| `network_preferences_config` | `list[str] \| None` | `None` | Ordered CAIP-2 network identifiers |
| `auto_payment` | `bool` | `True` | Enable/disable automatic 402 processing |
| `auto_session` | `bool` | `False` | Auto-create session on first 402 |
| `auto_session_budget` | `str` | `"1.00"` | Budget (USD) for auto-created sessions |
| `auto_session_expiry_minutes` | `int` | `60` | Expiry for auto-created sessions |
| `agent_name` | `str \| None` | `None` | Agent name for data-plane headers |
| `bearer_token` | `str \| None` | `None` | Static JWT. Mutually exclusive with `token_provider` |
| `token_provider` | `Callable \| None` | `None` | Callable returning fresh JWT. Mutually exclusive with `bearer_token` |
| `payment_tool_allowlist` | `list[str] \| None` | `None` | Tools eligible for payment. `None` = all |
| `provide_http_request` | `bool` | `True` | Register built-in `http_request` tool |
| `post_payment_retry_delay_seconds` | `float` | `3.0` | Delay after signing before retry |
| `custom_handlers` | `dict[str, Handler] \| None` | `None` | Custom handlers keyed by tool name |
| `on_payment_error` | `Callable \| None` | `None` | Error callback for programmatic recovery. See Error Handler Callback. |
| `max_error_retries` | `int` | `3` | Max times callback can return RETRY per tool call. 0 disables. |

## Payment Tool Allowlist

Restrict which tools get payment processing:

```python
config = AgentCorePaymentsConfig(
    ...,
    payment_tool_allowlist=["http_request", "my_paid_api"],
)
```

Modify the allowlist at runtime:

```python
# Add tools
config.add_to_allowlist("new_paid_tool", "another_tool")

# Remove tools (reverts to all-eligible if list becomes empty)
config.remove_from_allowlist("my_paid_api")
```

Tools not in the list pass through untouched. When `None` (default), all tools are eligible.

## Auto-Session

Skip manual session creation — the middleware creates one lazily on the first 402:

```python
config = AgentCorePaymentsConfig(
    payment_manager_arn="arn:...",
    user_id="user-1",
    payment_instrument_id="instr-1",
    region="us-east-1",
    auto_session=True,           # enable lazy creation
    auto_session_budget="5.00",  # $5 budget
    auto_session_expiry_minutes=120,  # 2 hours
)
```

The session is created once and reused for all subsequent payments in that middleware instance.

> **Instance lifecycle:** Create one `AgentCorePaymentsMiddleware` per agent invocation (or per user request in a server). The middleware is not thread-safe — sharing a single instance across concurrent invocations can cause races on session creation and config mutations.

## Disabling Auto-Payment

Use the middleware only for its built-in query tools without 402 interception:

```python
config = AgentCorePaymentsConfig(
    ...,
    auto_payment=False,
)
```

## Bearer Token Authentication

For payment managers using `CUSTOM_JWT` authorizer:

```python
# Static token
config = AgentCorePaymentsConfig(
    payment_manager_arn="arn:...",
    bearer_token="eyJhbGciOiJSUzI1NiJ9...",
    payment_instrument_id="instr-1",
    auto_session=True,
)

# Dynamic token provider (recommended for production)
config = AgentCorePaymentsConfig(
    payment_manager_arn="arn:...",
    token_provider=lambda: fetch_fresh_jwt(),
    payment_instrument_id="instr-1",
    auto_session=True,
)
```

With bearer auth, `user_id` is optional (derived from JWT `sub` claim).

## Sync vs Async

The middleware provides both sync (`wrap_tool_call`) and async (`awrap_tool_call`) paths. You don't choose between them — LangGraph calls the right one automatically based on how you invoke the agent:

| Invocation | Path used | When to use |
|---|---|---|
| `agent.invoke(...)` | Sync — uses `time.sleep`, direct calls | Scripts, CLI tools, simple applications |
| `agent.ainvoke(...)` / `await agent.ainvoke(...)` | Async — uses `asyncio.sleep`, `asyncio.to_thread` | FastAPI, Jupyter notebooks, web servers, any `async def` context |

### What the async path does differently

- **Non-blocking delay:** Uses `await asyncio.sleep()` instead of `time.sleep()` for the post-payment blockchain timing delay. This avoids blocking the event loop while waiting for on-chain settlement.
- **Threaded payment signing:** Runs `generate_payment_header()` via `asyncio.to_thread()` since the PaymentManager SDK is synchronous. This keeps the event loop free during the signing call.
- **Async error callbacks:** If your `on_payment_error` handler is `async def`, it's automatically awaited in the async path (sync callbacks also work).

### Example: async in FastAPI

```python
from fastapi import FastAPI
from langchain.agents import create_agent

app = FastAPI()

@app.post("/chat")
async def chat(message: str):
    config = AgentCorePaymentsConfig(...)
    payments = AgentCorePaymentsMiddleware(config)
    agent = create_agent(model="claude-sonnet-4-20250514", tools=[], middleware=[payments])

    # Uses awrap_tool_call automatically — won't block other requests
    result = await agent.ainvoke({"messages": [{"role": "user", "content": message}]})
    return result
```

### Example: sync in a script

```python
# Uses wrap_tool_call automatically — simple and straightforward
result = agent.invoke({"messages": [{"role": "user", "content": "Fetch paid data"}]})
```

No configuration needed — just use `.invoke()` or `.ainvoke()` and the middleware adapts.

## Comparison: With vs Without Middleware

**Without middleware** (manual wrapping):
- Write a wrapper function per tool type (~30-50 lines each)
- Handle 402 detection, x402 parsing, signing, retry manually
- No error handling — exceptions crash the tool call
- No blockchain timing delay — fast facilitators may reject
- No budget error messages to the LLM
- Adding a new tool = another wrapper

**With middleware:**
```python
config = AgentCorePaymentsConfig(payment_manager_arn="...", user_id="...", payment_instrument_id="...", auto_session=True)
agent = create_agent(model=model, tools=[my_tools], middleware=[AgentCorePaymentsMiddleware(config)])
```

Done. All tools handled automatically.
