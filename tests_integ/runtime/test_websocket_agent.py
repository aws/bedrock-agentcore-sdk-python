import asyncio
import json
import logging
import textwrap

import websockets

from tests_integ.runtime.base_test import AGENT_SERVER_ENDPOINT, BaseSDKRuntimeTest, start_agent_server

logger = logging.getLogger("sdk-runtime-websocket-test")


class TestSDKWebSocketAgent(BaseSDKRuntimeTest):
    def setup(self):
        self.agent_module = "websocket_agent"
        with open(self.agent_module + ".py", "w") as file:
            content = textwrap.dedent("""
                from bedrock_agentcore import BedrockAgentCoreApp

                app = BedrockAgentCoreApp(debug=True)

                @app.websocket
                async def websocket_handler(websocket, context):
                    await websocket.accept()

                    # Echo server - receive and respond to messages
                    try:
                        while True:
                            data = await websocket.receive_json()

                            # Handle different message types
                            if data.get("action") == "echo":
                                await websocket.send_json({
                                    "type": "echo_response",
                                    "message": data.get("message"),
                                    "session_id": context.session_id
                                })
                            elif data.get("action") == "stream":
                                # Stream multiple messages
                                count = data.get("count", 3)
                                for i in range(count):
                                    await websocket.send_json({
                                        "type": "stream_chunk",
                                        "chunk_id": i,
                                        "data": f"Chunk {i+1} of {count}"
                                    })
                                await websocket.send_json({"type": "stream_complete"})
                            elif data.get("action") == "close":
                                await websocket.send_json({"type": "closing"})
                                break
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": str(e)})
                    finally:
                        await websocket.close()

                app.run()
            """).strip()
            file.write(content)

    def run_test(self):
        with start_agent_server(self.agent_module):
            # Replace http:// with ws:// for WebSocket connection
            ws_endpoint = AGENT_SERVER_ENDPOINT.replace("http://", "ws://") + "/ws"

            # Run async WebSocket tests
            asyncio.run(self._test_websocket_echo(ws_endpoint))
            asyncio.run(self._test_websocket_streaming(ws_endpoint))
            asyncio.run(self._test_websocket_with_session(ws_endpoint))

    async def _test_websocket_echo(self, ws_endpoint):
        """Test basic WebSocket echo functionality."""
        logger.info("Testing WebSocket echo...")

        async with websockets.connect(ws_endpoint) as websocket:
            # Send echo request
            await websocket.send(json.dumps({"action": "echo", "message": "Hello WebSocket!"}))

            # Receive echo response
            response = await websocket.recv()
            data = json.loads(response)

            logger.info("Echo response: %s", data)
            assert data["type"] == "echo_response"
            assert data["message"] == "Hello WebSocket!"

            # Close connection
            await websocket.send(json.dumps({"action": "close"}))
            closing_msg = await websocket.recv()
            assert json.loads(closing_msg)["type"] == "closing"

    async def _test_websocket_streaming(self, ws_endpoint):
        """Test WebSocket streaming functionality."""
        logger.info("Testing WebSocket streaming...")

        async with websockets.connect(ws_endpoint) as websocket:
            # Request stream of 5 messages
            await websocket.send(json.dumps({"action": "stream", "count": 5}))

            # Receive streamed chunks
            chunks = []
            for _ in range(5):
                response = await websocket.recv()
                chunk = json.loads(response)
                logger.info("Received chunk: %s", chunk)
                assert chunk["type"] == "stream_chunk"
                chunks.append(chunk)

            # Receive completion message
            complete_msg = await websocket.recv()
            completion = json.loads(complete_msg)
            assert completion["type"] == "stream_complete"

            # Verify all chunks received
            assert len(chunks) == 5
            assert chunks[0]["chunk_id"] == 0
            assert chunks[4]["chunk_id"] == 4

            # Close connection
            await websocket.send(json.dumps({"action": "close"}))

    async def _test_websocket_with_session(self, ws_endpoint):
        """Test WebSocket with session ID in headers."""
        logger.info("Testing WebSocket with session ID...")

        # Add session ID header
        extra_headers = [("X-Amzn-Bedrock-AgentCore-Runtime-Session-Id", "test-session-123")]

        async with websockets.connect(ws_endpoint, additional_headers=extra_headers) as websocket:
            # Send echo request
            await websocket.send(json.dumps({"action": "echo", "message": "Session test"}))

            # Receive response with session ID
            response = await websocket.recv()
            data = json.loads(response)

            logger.info("Response with session: %s", data)
            assert data["session_id"] == "test-session-123"

            # Close connection
            await websocket.send(json.dumps({"action": "close"}))


def test(tmp_path):
    TestSDKWebSocketAgent().run(tmp_path)
