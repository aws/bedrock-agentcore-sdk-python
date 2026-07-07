"""Integration test for the streaming-bridge deadlock.

Boots a real uvicorn server running a BedrockAgentCoreApp with a streaming
(async generator) entrypoint, then drives it with real HTTP requests:

  1. Session A opens the SSE stream, reads a few chunks, then DISCONNECTS.
  2. Session B (fresh) must still receive a normal response.

Before the fix, A's disconnect fills the bridge's bounded queue and blocks the
shared worker loop, so B hangs and its handler never runs. After the fix, the
orphaned producer is torn down and B succeeds.

No mocks: full ASGI stack over a loopback socket.
"""

import socket
import threading
import time

import httpx
import pytest
import uvicorn

from bedrock_agentcore.runtime import BedrockAgentCoreApp


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def live_server():
    """Run a streaming BedrockAgentCoreApp on a real uvicorn server in a thread."""
    app = BedrockAgentCoreApp()

    @app.entrypoint
    async def handler(payload):
        n = int(payload.get("chunks", 5000))
        for i in range(n):
            yield f"data: chunk {i}\n\n"

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the server to accept connections.
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            httpx.get(f"{base}/ping", timeout=1.0)
            break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        pytest.fail("live server did not start")

    yield base

    server.should_exit = True
    thread.join(timeout=5)


def test_ping_stays_healthy_and_fresh_session_works_after_disconnect(live_server):
    """After a mid-stream client disconnect: /ping stays healthy AND a fresh session responds."""
    base = live_server

    # Session A: open stream, read a few chunks, then disconnect by leaving the context.
    read = 0
    with httpx.Client(timeout=30.0) as c:
        with c.stream("POST", f"{base}/invocations", json={"sessionId": "A", "chunks": 5000}) as r:
            for _ in r.iter_lines():
                read += 1
                if read >= 3:
                    break
    assert read >= 3

    time.sleep(2)  # let the (now-orphaned) producer fill the queue if it's going to block

    # The main loop must still answer health checks (it always did — proves the split).
    ping = httpx.get(f"{base}/ping", timeout=5.0)
    assert ping.status_code == 200

    # The worker loop must still serve a fresh session. Before the fix this times out.
    resp = httpx.post(
        f"{base}/invocations",
        json={"sessionId": "B", "chunks": 3},
        headers={"Accept": "text/event-stream"},
        timeout=15.0,
    )
    assert resp.status_code == 200
    assert "chunk 0" in resp.text
