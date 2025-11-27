"""Integration tests for AgentRuntimeClient.

These tests validate that the client generates valid credentials
that can be used to connect to actual AgentCore Runtime endpoints.
"""

import pytest
import websockets

from bedrock_agentcore.runtime import AgentRuntimeClient


@pytest.mark.integration
class TestAgentRuntimeClientIntegration:
    """Integration tests for AgentRuntimeClient."""

    def test_generate_ws_connection_returns_valid_format(self):
        """Test that generate_ws_connection returns properly formatted credentials."""
        client = AgentRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test-runtime"

        ws_url, headers = client.generate_ws_connection(runtime_arn)

        # Verify URL format
        assert ws_url.startswith("wss://")
        assert "runtimes" in ws_url
        assert "/ws" in ws_url

        # Verify required headers are present
        assert "Authorization" in headers
        assert "X-Amz-Date" in headers
        assert "Host" in headers

    def test_generate_presigned_url_returns_valid_format(self):
        """Test that generate_presigned_url returns properly formatted URL."""
        client = AgentRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test-runtime"

        presigned_url = client.generate_presigned_url(runtime_arn)

        # Verify URL format
        assert presigned_url.startswith("wss://")
        assert "runtimes" in presigned_url
        assert "X-Amz-Algorithm" in presigned_url
        assert "X-Amz-Signature" in presigned_url

    @pytest.mark.skip(reason="Requires actual runtime endpoint")
    async def test_connect_with_generated_headers(self):
        """Test connecting to actual runtime with generated headers.

        This test is skipped by default. To run it, provide a valid runtime ARN
        and remove the skip decorator.
        """
        client = AgentRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test-runtime"

        ws_url, headers = client.generate_ws_connection(runtime_arn)

        # Attempt to connect
        async with websockets.connect(ws_url, extra_headers=headers) as ws:
            # Send test message
            await ws.send('{"type": "test"}')

            # Receive response
            response = await ws.recv()
            assert response is not None
