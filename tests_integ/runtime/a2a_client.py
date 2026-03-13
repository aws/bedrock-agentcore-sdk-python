import json
import logging

import requests


class A2AClient:
    """Local A2A client for invoking JSON-RPC endpoints."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.logger = logging.getLogger("sdk-runtime-test-a2a-client")

    def get_agent_card(self):
        """GET /.well-known/agent-card.json"""
        url = f"{self.endpoint}/.well-known/agent-card.json"
        self.logger.info("Fetching Agent Card from %s", url)
        return requests.get(url, timeout=5).json()

    def ping(self):
        """GET /ping"""
        url = f"{self.endpoint}/ping"
        self.logger.info("Pinging A2A server")
        return requests.get(url, timeout=5).json()

    def send_message(self, text: str, request_id: str = "req-001", session_id: str = None):
        """POST / with JSON-RPC message/send"""
        url = self.endpoint + "/"
        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] = session_id

        body = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": text}],
                }
            },
        }
        self.logger.info("Sending message: %s", text)
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        return resp.json()

    def stream_message(self, text: str, request_id: str = "req-001"):
        """POST / with JSON-RPC message/stream (SSE response)"""
        url = self.endpoint + "/"
        body = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": text}],
                }
            },
        }
        self.logger.info("Streaming message: %s", text)
        resp = requests.post(url, headers={"Content-Type": "application/json"}, json=body, timeout=30, stream=True)

        events = []
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                events.append(json.loads(line[6:]))
        return events

    def send_raw(self, body: dict):
        """POST / with raw JSON-RPC body"""
        url = self.endpoint + "/"
        resp = requests.post(url, headers={"Content-Type": "application/json"}, json=body, timeout=10)
        return resp.json()
