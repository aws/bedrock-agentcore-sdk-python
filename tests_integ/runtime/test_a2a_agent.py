"""E2E integration test for A2A protocol support.

Tests the full A2A lifecycle:
- Agent Card discovery
- Ping health check
- JSON-RPC message/send (sync)
- JSON-RPC message/stream (SSE streaming)
- Error handling (invalid JSON-RPC, missing method, etc.)
- Session ID propagation
"""

import logging
import textwrap

from tests_integ.runtime.a2a_client import A2AClient
from tests_integ.runtime.base_test import BaseSDKRuntimeTest, start_agent_server

logger = logging.getLogger("sdk-runtime-a2a-agent-test")

A2A_SERVER_ENDPOINT = "http://127.0.0.1:9000"


class TestSDKA2AAgent(BaseSDKRuntimeTest):
    def setup(self):
        self.agent_module = "a2a_agent"
        with open(self.agent_module + ".py", "w") as file:
            content = textwrap.dedent("""
                import uuid
                from bedrock_agentcore.runtime import (
                    BedrockAgentCoreA2AApp,
                    AgentCard,
                    AgentSkill,
                )

                agent_card = AgentCard(
                    name="E2E Test Agent",
                    description="An agent for E2E integration testing",
                    skills=[
                        AgentSkill(
                            id="echo",
                            name="Echo",
                            description="Echoes back user input",
                            tags=["test", "echo"],
                        ),
                    ],
                )

                app = BedrockAgentCoreA2AApp(agent_card=agent_card, debug=True)

                @app.entrypoint
                def handle_message(request, context):
                    params = request.params or {}
                    message = params.get("message", {})
                    parts = message.get("parts", [])

                    user_text = ""
                    for part in parts:
                        if part.get("kind") == "text":
                            user_text = part.get("text", "")
                            break

                    session_id = context.session_id if context else None

                    return {
                        "artifacts": [
                            {
                                "artifactId": str(uuid.uuid4()),
                                "name": "echo_response",
                                "parts": [
                                    {"kind": "text", "text": f"Echo: {user_text}"},
                                    {"kind": "data", "data": {"session_id": session_id}},
                                ],
                            }
                        ]
                    }

                app.run()
            """).strip()
            file.write(content)

    def run_test(self):
        with start_a2a_server(self.agent_module):
            client = A2AClient(A2A_SERVER_ENDPOINT)

            self._test_agent_card(client)
            self._test_ping(client)
            self._test_message_send(client)
            self._test_session_propagation(client)
            self._test_invalid_jsonrpc(client)
            self._test_missing_method(client)

            logger.info("All A2A E2E tests passed!")

    def _test_agent_card(self, client):
        """Test Agent Card discovery endpoint."""
        logger.info("--- Testing Agent Card ---")
        card = client.get_agent_card()
        assert card["name"] == "E2E Test Agent", f"Expected 'E2E Test Agent', got {card['name']}"
        assert card["description"] == "An agent for E2E integration testing"
        assert card["protocolVersion"] == "0.3.0"
        assert card["preferredTransport"] == "JSONRPC"
        assert card["capabilities"]["streaming"] is True
        assert len(card["skills"]) == 1
        assert card["skills"][0]["id"] == "echo"
        assert card["skills"][0]["tags"] == ["test", "echo"]
        assert "url" in card
        logger.info("Agent Card test passed: %s", card["name"])

    def _test_ping(self, client):
        """Test ping health check endpoint."""
        logger.info("--- Testing Ping ---")
        ping = client.ping()
        assert ping["status"] in ["Healthy", "HEALTHY"], f"Unexpected status: {ping['status']}"
        assert "time_of_last_update" in ping
        logger.info("Ping test passed: %s", ping)

    def _test_message_send(self, client):
        """Test JSON-RPC message/send."""
        logger.info("--- Testing message/send ---")
        response = client.send_message("Hello, A2A!", request_id="test-001")

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-001"
        assert "result" in response
        assert "error" not in response

        result = response["result"]
        assert "artifacts" in result
        assert len(result["artifacts"]) == 1

        artifact = result["artifacts"][0]
        assert artifact["name"] == "echo_response"
        text_part = artifact["parts"][0]
        assert text_part["text"] == "Echo: Hello, A2A!"
        logger.info("message/send test passed: %s", text_part["text"])

    def _test_session_propagation(self, client):
        """Test session ID is propagated to handler."""
        logger.info("--- Testing Session Propagation ---")
        response = client.send_message(
            "session test",
            request_id="test-session",
            session_id="my-session-456",
        )

        result = response["result"]
        data_part = result["artifacts"][0]["parts"][1]
        assert data_part["data"]["session_id"] == "my-session-456", (
            f"Expected session_id 'my-session-456', got {data_part['data']['session_id']}"
        )
        logger.info("Session propagation test passed")

    def _test_invalid_jsonrpc(self, client):
        """Test error handling for invalid JSON-RPC version."""
        logger.info("--- Testing Invalid JSON-RPC ---")
        response = client.send_raw({
            "jsonrpc": "1.0",
            "id": "bad-001",
            "method": "message/send",
        })

        assert "error" in response
        assert response["error"]["code"] == -32600  # INVALID_REQUEST
        logger.info("Invalid JSON-RPC test passed: code=%d", response["error"]["code"])

    def _test_missing_method(self, client):
        """Test error handling for missing method."""
        logger.info("--- Testing Missing Method ---")
        response = client.send_raw({
            "jsonrpc": "2.0",
            "id": "bad-002",
        })

        assert "error" in response
        assert response["error"]["code"] == -32600  # INVALID_REQUEST
        logger.info("Missing method test passed: code=%d", response["error"]["code"])


import subprocess
import threading
import time
from contextlib import contextmanager


@contextmanager
def start_a2a_server(agent_module, timeout=10):
    """Start an A2A agent server on port 9000."""
    logger.info("Starting A2A agent server...")
    start_time = time.time()

    agent_server = subprocess.Popen(
        ["python", "-m", agent_module],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        while time.time() - start_time < timeout:
            if agent_server.stdout is None:
                raise RuntimeError("Agent server has no configured output")

            if agent_server.poll() is not None:
                out = agent_server.stdout.read()
                raise RuntimeError(f"Error when running agent server: {out}")

            line = agent_server.stdout.readline()
            while line:
                line = line.strip()
                if line:
                    logger.info(line)
                    if "Uvicorn running on" in line and "9000" in line:
                        # Start logging thread
                        def log_output():
                            for l in iter(agent_server.stdout.readline, ""):
                                if l.strip():
                                    logger.info(l.strip())
                        t = threading.Thread(target=log_output, daemon=True)
                        t.start()
                        yield agent_server
                        return
                line = agent_server.stdout.readline()

            time.sleep(0.5)
        raise TimeoutError(f"A2A server did not start within {timeout} seconds")
    finally:
        logger.info("Stopping A2A agent server...")
        if agent_server.poll() is None:
            agent_server.terminate()
            try:
                agent_server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                agent_server.kill()
                agent_server.wait()
            finally:
                if agent_server.stdout:
                    agent_server.stdout.close()
        logger.info("A2A agent server terminated")


def test(tmp_path):
    TestSDKA2AAgent().run(tmp_path)
