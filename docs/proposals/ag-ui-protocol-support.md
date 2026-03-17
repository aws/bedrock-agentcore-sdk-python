# Proposal: AG-UI Protocol Support

## Problem

Bedrock AgentCore Runtime supports AG-UI as a protocol mode. Today, users must manually wire FastAPI/Starlette, understand `EventEncoder`, construct `StreamingResponse`, parse `RunAgentInput`, add `/ping`, extract Bedrock headers, and configure uvicorn — roughly 25 lines of boilerplate that every AG-UI agent repeats.

The current example from the docs:

```python
app = FastAPI()

@app.post("/invocations")
async def invocations(input_data: dict, request: Request):
    accept_header = request.headers.get("accept")
    encoder = EventEncoder(accept=accept_header)

    async def event_generator():
        run_input = RunAgentInput(**input_data)
        async for event in agui_agent.run(run_input):
            yield encoder.encode(event)

    return StreamingResponse(
        event_generator(),
        media_type=encoder.get_content_type()
    )

@app.get("/ping")
async def ping():
    return JSONResponse({"status": "Healthy"})

uvicorn.run(app, host="0.0.0.0", port=8080)
```

Users shouldn't need to know about `EventEncoder`, `StreamingResponse`, or SSE formatting.

---

## Approach: New App Class — `AGUIApp`

### Why a separate class, not extending `BedrockAgentCoreApp`

Although both use port 8080 and `POST /invocations`, the code paths share almost nothing:

| | HTTP mode | AG-UI mode |
|---|---|---|
| Input | raw dict | `RunAgentInput` (parsed + validated) |
| Output | JSON, Response, sync/async generator | always async iterable of typed Events |
| SSE encoding | `json.dumps` | `EventEncoder` (camelCase, `exclude_none`) |
| Error path | `JSONResponse` 500 | `RunErrorEvent` in SSE stream |
| Handler dispatch | 5 branches (sync, async, gen, async gen, Response) | 1 branch (async iterable of events) |

Adding AG-UI to `BedrockAgentCoreApp` would mean:
- `_handle_invocation` grows a branch with completely different parsing, encoding, and error handling
- The class mixes two protocols that happen to share a URL path
- Users see debug actions, task tracking — none of which AG-UI uses
- `ag-ui-protocol` is optional — conditionals scattered through `app.py`

A focused class (~100 lines) that only does "parse `RunAgentInput`, stream events, handle errors" is easier to understand, test, and maintain.

---

## Interface

### Framework agents (common case) — no decorator

The agent already exists and has a `.run()` method. Just hand it over:

```python
from bedrock_agentcore.runtime import serve_ag_ui

serve_ag_ui(agui_agent)
```

Or with more control:

```python
from bedrock_agentcore.runtime import AGUIApp

app = AGUIApp()
app.entrypoint(agui_agent)
app.run()
```

### Custom agents — decorator form

When the user IS the agent and writes the logic directly:

```python
app = AGUIApp()

@app.entrypoint
async def my_agent(input_data: RunAgentInput):
    yield RunStartedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)
    result = await call_my_llm(input_data.messages)
    yield TextMessageChunkEvent(delta=result)
    yield RunFinishedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)

app.run()
```

### With optional context (Bedrock headers)

```python
@app.entrypoint
async def my_agent(input_data: RunAgentInput, context: RequestContext):
    token = BedrockAgentCoreContext.get_workload_access_token()
    yield RunStartedEvent(...)
```

### Custom ping

```python
@app.ping
def health():
    return PingStatus.HEALTHY_BUSY
```

### Testing

```python
from bedrock_agentcore.runtime import build_ag_ui_app
from starlette.testclient import TestClient

app = build_ag_ui_app(my_agent)
client = TestClient(app)

# SSE transport
response = client.post("/invocations", json={...})

# WebSocket transport
with client.websocket_connect("/ws") as ws:
    ws.send_json({...})
    msg = ws.receive_text()
```

---

## Dual Transport: SSE and WebSocket

The [AG-UI protocol contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-agui-protocol-contract.html) specifies SSE and WebSocket as **alternative transports** for the same AG-UI event stream. The runtime picks one based on the `--protocol` flag at deployment time.

A single `@app.entrypoint` handler is automatically wired to **both**:

| Endpoint | Transport | Flow |
|----------|-----------|------|
| `POST /invocations` | SSE | Parse `RunAgentInput` from body → stream events as `text/event-stream` |
| `/ws` | WebSocket | Accept → receive `RunAgentInput` as first JSON message → stream events as text frames → close |

The same agent handler works identically over either transport — no separate handler needed.

### WebSocket protocol

1. Client connects to `/ws` (Bedrock headers available on the upgrade request)
2. Client sends one JSON message: a `RunAgentInput` payload
3. Server streams AG-UI events as text frames (same `EventEncoder` output)
4. Server closes the connection on completion
5. Errors during streaming → `RunErrorEvent` sent before close
6. Invalid input → close with code 1003
7. No entrypoint registered → close with code 1011

---

## How `entrypoint` Works

`entrypoint` accepts either:

1. **A callable** (async generator function) — decorator form for custom agents
2. **An object with `.run()`** — non-decorator form for framework agents

Detection: if the arg has a `.run()` attribute, store `agent.run` as the handler. Otherwise treat the arg as the handler directly.

When `POST /invocations` arrives (SSE):

1. Extract Bedrock headers → set `BedrockAgentCoreContext`
2. Parse request body → `RunAgentInput` (HTTP 400 on failure)
3. Call registered handler, iterate yielded events
4. Encode each event via `EventEncoder`, stream as `StreamingResponse`
5. On error during streaming → emit `RunErrorEvent`, close stream

When `/ws` connects (WebSocket):

1. Accept WebSocket connection
2. Extract Bedrock headers from upgrade request → set `BedrockAgentCoreContext`
3. Receive first JSON message → `RunAgentInput` (close 1003 on failure)
4. Call same handler, send each event as a text frame
5. On error during streaming → send `RunErrorEvent`, close connection

### Pre-stream vs mid-stream errors

The AG-UI spec treats agent errors as `RunErrorEvent` events on the stream. However, if the request is malformed enough that we can't even start streaming (invalid JSON, failed `RunAgentInput` validation), returning an HTTP 400 (SSE) or closing with code 1003 (WebSocket) is more appropriate than starting a stream just to immediately error.

---

## How AG-UI Differs from A2A

The `a2a-sdk` is a **complete server framework** (app builder, request handler, executor ABC, task store). Our `serve_a2a` wires those components together.

The `ag-ui-protocol` SDK is **just types and an encoder**:
- `ag_ui.core` — Pydantic event/input types (`RunAgentInput`, `Event`, etc.)
- `ag_ui.encoder` — `EventEncoder` (SSE formatting)

No server builder, no executor ABC. Our app class owns the full request/response path — but it's thin since the protocol is "payload in, events out."

---

## What We Ship

### `src/bedrock_agentcore/runtime/ag_ui.py` (~280 lines)

**`AGUIApp` (extends Starlette)**

Constructor:
- `debug: bool = False`
- `lifespan: Optional[Lifespan] = None`
- `middleware: Sequence[Middleware] | None = None`
- Checks for `ag-ui-protocol` package at init time
- Sets up routes: `POST /invocations`, `GET /ping`, `WebSocketRoute /ws`

Methods:
- `entrypoint(agent_or_func)` — registers handler for both SSE and WebSocket; works as decorator or direct call
- `ping(func)` — registers custom ping handler (decorator)
- `run(port=8080, host=None, **kwargs)` — Docker detection + `uvicorn.run()`

**`serve_ag_ui(agent, *, port=8080, host=None, ping_handler=None, **kwargs)`**
- Creates `AGUIApp`, registers agent, runs uvicorn
- One-liner for framework agents

**`build_ag_ui_app(agent, *, ping_handler=None)` → `AGUIApp`**
- Same but returns app without starting — for `TestClient`, mounting, etc.

### Optional dependency in `pyproject.toml`

```toml
[project.optional-dependencies]
ag-ui = ["ag-ui-protocol>=0.1.10"]
```

---

## What We Handle Automatically

| Concern | Before (manual) | After |
|---------|-----------------|-------|
| `EventEncoder` setup | User creates and calls `.encode()` | Auto-handled |
| `StreamingResponse` | User writes generator + wraps | Auto-handled |
| `RunAgentInput` parsing | User parses from dict | Auto-handled |
| Bedrock headers | User doesn't do it | Auto → `BedrockAgentCoreContext` |
| `/ping` health check | User writes endpoint | Built-in, customizable via `@app.ping` |
| `/ws` WebSocket transport | User implements separately | Same handler, auto-wired |
| Docker host detection | User hardcodes `0.0.0.0` | `app.run()` auto-detects |
| Error handling | User writes try/except | Auto → `RunErrorEvent` |

---

## Design Decisions

1. **New class, not extending `BedrockAgentCoreApp`** — Different input parsing, SSE encoding, error semantics, handler dispatch. The shared URL path is Bedrock's routing, not a reason to couple code. Optional dep stays cleanly isolated.

2. **`AGUIApp` class name** — Brief and clear. `BedrockAgentCoreAGUIApp` is too verbose for an app users construct directly.

3. **Dual-mode `entrypoint`** — Accepts callable (decorator for custom agents) or object with `.run()` (framework agents). The common case (framework agent) needs no decorator — just pass the agent object.

4. **Single entrypoint, dual transport** — The AG-UI contract defines SSE and WebSocket as alternative transports. One `@app.entrypoint` wires the handler to both `/invocations` and `/ws` automatically.

5. **Lazy imports** — `ag_ui.core` and `ag_ui.encoder` only imported when the code path executes (inside handlers). The SDK presence is checked once at `AGUIApp.__init__` time.

6. **Pre-stream validation returns HTTP errors** — Invalid JSON or `RunAgentInput` get HTTP 400 (SSE) / close 1003 (WebSocket) rather than starting a stream to immediately emit `RunErrorEvent`. Agent errors *during* streaming produce `RunErrorEvent` per spec.

7. **`serve_ag_ui` one-liner** — For the common case. Thin wrapper around the app class.

8. **Port 8080** — Per Bedrock AG-UI contract. (A2A uses 9000.)

---

## Files

| Action | Path |
|--------|------|
| Modify | `pyproject.toml` — add `ag-ui` optional dep |
| Create | `src/bedrock_agentcore/runtime/ag_ui.py` — `AGUIApp`, `serve_ag_ui`, `build_ag_ui_app` |
| Modify | `src/bedrock_agentcore/runtime/__init__.py` — export new symbols |
| Create | `tests/bedrock_agentcore/runtime/test_ag_ui.py` — unit tests |
| Create | `tests/integration/runtime/test_ag_ui_integration.py` — integration tests |
