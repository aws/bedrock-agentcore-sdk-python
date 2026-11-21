"""The warmup hook on its own port (``HookConfig.port``), served next to the app by ``app.run()``."""

import contextlib
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time

import requests

logger = logging.getLogger("sdk-runtime-warmup-hook-port-test")

WARMUP_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Warmup-Type"
MEMORY_WARMUP_SECONDS = 2
NEEDS_RUN_MESSAGE = (
    "Hook ports are only served by app.run(); start the agent with app.run(), or leave HookConfig.port unset"
)

AGENT = """
import os
import time

from bedrock_agentcore import BedrockAgentCoreApp, HookConfig, WarmupType

APP_PORT = int(os.environ["APP_PORT"])
HOOK_PORT = int(os.environ["HOOK_PORT"])

app = BedrockAgentCoreApp(debug=True, warmup_hook=HookConfig(path="/warmup", port=HOOK_PORT))
WARMED = False


@app.warmup(WarmupType.MEMORY)
def load_shared_state():
    global WARMED
    time.sleep({seconds})  # blocking work; /ping on the serving port must keep answering
    WARMED = True


@app.entrypoint
def invoke(payload):
    return {{"warmed": WARMED}}


if __name__ == "__main__":
    app.run(port=APP_PORT)
"""


def _free_ports(count):
    """Pick free ports in the range HookConfig.port accepts (8000-12000)."""
    ports = []
    for port in random.sample(range(8000, 12001), 4001):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        ports.append(port)
        if len(ports) == count:
            return ports
    raise RuntimeError("no free ports")  # pragma: no cover


def _is_listening(port):
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_for(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@contextlib.contextmanager
def _agent(tmp_path, app_port, hook_port, command):
    (tmp_path / "hook_port_agent.py").write_text(textwrap.dedent(AGENT.format(seconds=MEMORY_WARMUP_SECONDS)))
    log_path = tmp_path / "agent.log"
    env = {**os.environ, "APP_PORT": str(app_port), "HOOK_PORT": str(hook_port), "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, cwd=tmp_path, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        yield process, (lambda: log_path.read_text())
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        logger.info("agent output:\n%s", log_path.read_text())


def test_hook_on_own_port(tmp_path):
    app_port, hook_port = _free_ports(2)
    command = [sys.executable, "-m", "hook_port_agent"]
    with _agent(tmp_path, app_port, hook_port, command) as (process, output):
        assert _wait_for(lambda: f"Uvicorn running on http://127.0.0.1:{app_port}" in output(), 15), output()
        assert _is_listening(hook_port)
        app_url, hook_url = f"http://127.0.0.1:{app_port}", f"http://127.0.0.1:{hook_port}"

        assert requests.get(f"{app_url}/ping", timeout=2).json()["status"] == "Healthy"
        # The hook isn't served on the serving port, and the app's routes aren't served on the hook port.
        assert requests.post(f"{app_url}/warmup", headers={WARMUP_HEADER: "memory"}, timeout=2).status_code == 404
        assert requests.get(f"{hook_url}/ping", timeout=2).status_code == 404

        pings = []
        stop = threading.Event()

        def poll_ping():
            while not stop.is_set():
                start = time.monotonic()
                response = requests.get(f"{app_url}/ping", timeout=1)
                pings.append((start, response.status_code, response.json()["status"], time.monotonic() - start))
                time.sleep(0.1)

        poller = threading.Thread(target=poll_ping, daemon=True)
        poller.start()
        try:
            memory_start = time.monotonic()
            memory = requests.post(f"{hook_url}/warmup", headers={WARMUP_HEADER: "memory"}, timeout=10)
            memory_end = time.monotonic()
        finally:
            stop.set()
            poller.join(timeout=5)

        logger.info(
            "memory warmup on hook port: %s %s (%.2fs)", memory.status_code, memory.text, memory_end - memory_start
        )
        assert memory.status_code == 200
        assert memory.json() == {"status": "ok", "type": "memory", "result": "completed"}
        assert memory_end - memory_start >= MEMORY_WARMUP_SECONDS
        during = [ping for ping in pings if memory_start <= ping[0] <= memory_end]
        assert len(during) >= 5
        assert all(code == 200 and status == "Healthy" for _, code, status, _ in during)
        assert max(duration for _, _, _, duration in during) < 0.5

        disk = requests.post(f"{hook_url}/warmup", headers={WARMUP_HEADER: "disk"}, timeout=10)
        assert disk.json() == {"status": "ok", "type": "disk", "result": "skipped"}
        assert requests.post(f"{app_url}/invocations", json={}, timeout=5).json() == {"warmed": True}

        # One SIGTERM stops both servers: the main server shuts down, then its lifespan stops the hook server.
        process.send_signal(signal.SIGTERM)
        returncode = process.wait(timeout=15)
        logger.info("exit code after SIGTERM: %s", returncode)
        # uvicorn re-raises the signal it handled once it has shut down, so the process ends by SIGTERM
        # (-15) after a graceful shutdown, exactly as it does without a hook port.
        assert returncode in (0, -signal.SIGTERM)
        text = output()
        assert "Application shutdown complete" in text
        assert "Hook servers stopped" in text
        assert not _is_listening(app_port)
        assert not _is_listening(hook_port)


def test_uvicorn_cli_launch_fails_startup(tmp_path):
    app_port, hook_port = _free_ports(2)
    command = [sys.executable, "-m", "uvicorn", "hook_port_agent:app", "--port", str(app_port)]
    with _agent(tmp_path, app_port, hook_port, command) as (process, output):
        returncode = process.wait(timeout=15)
        logger.info("exit code under uvicorn CLI: %s", returncode)
        assert returncode == 3  # uvicorn's startup-failure exit code
        assert NEEDS_RUN_MESSAGE in output()
        assert not _is_listening(app_port)
        assert not _is_listening(hook_port)


def test_taken_hook_port_exits_before_serving_port_opens(tmp_path):
    app_port, hook_port = _free_ports(2)
    command = [sys.executable, "-m", "hook_port_agent"]
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", hook_port))
        taken.listen()
        with _agent(tmp_path, app_port, hook_port, command) as (process, output):
            serving_port_opened = False
            deadline = time.monotonic() + 15
            while process.poll() is None and time.monotonic() < deadline:
                serving_port_opened = serving_port_opened or _is_listening(app_port)
                time.sleep(0.02)
            returncode = process.wait(timeout=15)
            logger.info("exit code with taken hook port: %s", returncode)
            assert returncode == 3
            assert not serving_port_opened
            text = output()
            assert f"Can't serve the hook on HookConfig.port {hook_port}" in text
            assert "Uvicorn running on" not in text
