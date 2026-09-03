"""Integration tests for WebSearchClient.

These tests call the real gateway, so they need a gateway that already has a web
search connector target on it. The web search connector is enabled per account, so
they skip rather than fail when the account is not entitled.

Run with:
    uv run pytest tests_integ/tools/test_web_search_client.py -xvs

Requires environment variables:
    WEB_SEARCH_GATEWAY_ID: ID of a gateway with a web search connector target
    BEDROCK_TEST_REGION: AWS region (default: us-east-1). The connector is only
        offered in us-east-1, eu-west-1 and ap-northeast-1.
    WEB_SEARCH_TARGET_NAME: Optional. The target name, if it is not the SDK default.
"""

import os

import pytest

from bedrock_agentcore.tools.web_search_client import WebSearchClient, WebSearchError


@pytest.mark.integration
class TestWebSearchClient:
    """Integration tests for WebSearchClient over a gateway target."""

    @classmethod
    def setup_class(cls):
        cls.gateway_id = os.environ.get("WEB_SEARCH_GATEWAY_ID")
        if not cls.gateway_id:
            pytest.skip("WEB_SEARCH_GATEWAY_ID must be set")
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-east-1")
        cls.target_name = os.environ.get("WEB_SEARCH_TARGET_NAME")

    def _client(self):
        return WebSearchClient(
            region=self.region,
            gateway_id=self.gateway_id,
            target_name=self.target_name,
        )

    def _search(self, client, query, **kwargs):
        """Search, skipping the test when the account is not entitled to the connector."""
        try:
            return client.search(query, **kwargs)
        except WebSearchError as e:
            if "not available for this account" in str(e):
                pytest.skip(f"web-search connector not enabled for this account: {e}")
            raise

    def test_search_returns_results(self):
        with self._client() as client:
            response = self._search(client, "what is amazon bedrock agentcore", max_results=3)

        assert len(response) > 0
        assert len(response) <= 3
        first = response.results[0]
        assert first.text
        # Citations must be retained for any output shown to an end user, so the
        # client has to surface the source URL.
        assert first.url

    def test_search_respects_max_results(self):
        with self._client() as client:
            response = self._search(client, "python urllib3 release notes", max_results=1)

        assert len(response) == 1

    def test_search_with_domain_filter(self):
        """Needs connector version 1.2.0 or later on the target."""
        with self._client() as client:
            response = self._search(
                client,
                "agentcore gateway connector targets",
                max_results=5,
                include_domains=["docs.aws.amazon.com"],
            )

        assert len(response) > 0
        for result in response:
            assert result.url is None or "aws.amazon.com" in result.url

    def test_tool_name_discovery(self):
        """Without target_name the client finds the tool through tools/list."""
        with WebSearchClient(region=self.region, gateway_id=self.gateway_id) as client:
            self._search(client, "bedrock agentcore gateway", max_results=1)

            assert client.backend._tool_name.endswith("WebSearch")

    def test_session_is_reused_across_searches(self):
        with self._client() as client:
            self._search(client, "first query", max_results=1)
            self._search(client, "second query", max_results=1)

            assert client.backend._mcp_session_id

    def test_oversized_query_is_rejected_locally(self):
        with self._client() as client:
            with pytest.raises(ValueError, match="200 characters or fewer"):
                client.search("x" * 201)
