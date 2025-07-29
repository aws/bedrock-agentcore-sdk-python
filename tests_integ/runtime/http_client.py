import json
import logging

import requests


class HttpClient:
    """Local HTTP client for invoking endpoints."""

    def __init__(self, endpoint: str):
        """Initialize the local client with the given endpoint."""
        self.endpoint = endpoint
        self.logger = logging.getLogger("sdk-runtime-test-http-client")

    def invoke_endpoint(self, payload: str):
        """Invoke the endpoint with the given parameters."""
        self.logger.info("Sending request to agent with payload: %s", payload)

        url = f"{self.endpoint}/invocations"

        headers = {
            "Content-Type": "application/json",
        }

        try:
            body = json.loads(payload) if isinstance(payload, str) else payload
        except json.JSONDecodeError:
            # Fallback for non-JSON strings - wrap in payload object
            self.logger.warning("Failed to parse payload as JSON, wrapping in payload object")
            body = {"message": payload}

        try:
            # Make request with timeout
            return requests.post(url, headers=headers, json=body, timeout=100, stream=True).text
        except requests.exceptions.RequestException as e:
            self.logger.error("Failed to invoke agent endpoint: %s", str(e))
            raise

    def ping(self):
        self.logger.info("Pinging agent server")

        url = f"{self.endpoint}/ping"
        try:
            return requests.get(url, timeout=2).text
        except requests.exceptions.RequestException as e:
            self.logger.error("Failed to ping agent endpoint: %s", str(e))
            raise
