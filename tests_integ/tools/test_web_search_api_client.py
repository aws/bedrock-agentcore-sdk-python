"""Integration tests for WebSearchClient over the Web Search API (InvokeWebSearch).

These tests call the real API with the default backend, so they need credentials allowed to
call bedrock-agentcore:InvokeWebSearch.

Run with:
    uv run pytest tests_integ/tools/test_web_search_api_client.py -xvs

Requires environment variables:
    BEDROCK_TEST_REGION: AWS region (default: us-east-1). The API is offered in us-east-1,
        us-east-2, us-west-2, eu-west-1, ap-northeast-1 and ap-southeast-2.
"""

import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import pytest

from bedrock_agentcore.tools.web_search_client import WebSearchApiBackend, WebSearchClient


@pytest.mark.integration
class TestWebSearchApiClient:
    """Integration tests for WebSearchClient over the Web Search API."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-east-1")

    def test_default_backend_is_the_api(self):
        with WebSearchClient(region=self.region) as client:
            assert isinstance(client.backend, WebSearchApiBackend)

    def test_search_returns_results(self):
        with WebSearchClient(region=self.region) as client:
            response = client.search("what is amazon bedrock agentcore", max_results=3)

        assert 0 < len(response) <= 3
        assert all(result.text for result in response)

    def test_published_dates_are_parsed(self):
        with WebSearchClient(region=self.region) as client:
            response = client.search("aws news this week", max_results=10)

        dated = [result for result in response if result.published_date]
        assert dated, "expected at least one dated result"
        assert all(result.published_at is not None and result.published_at.tzinfo for result in dated)

    def test_excluded_domain_is_not_returned(self):
        with WebSearchClient(region=self.region) as client:
            response = client.search("kubernetes release notes", max_results=10, exclude_domains=["kubernetes.io"])

        hosts = [urlparse(result.url).hostname or "" for result in response if result.url]
        assert not any(host == "kubernetes.io" or host.endswith(".kubernetes.io") for host in hosts)

    def test_fixed_date_floor_is_not_widened_by_a_call(self):
        floor = datetime.now(timezone.utc) - timedelta(days=7)
        with WebSearchClient(region=self.region, published_after=floor.strftime("%Y-%m-%dT%H:%M:%SZ")) as client:
            response = client.search("security advisory", max_results=10, published_after="2020-01-01T00:00:00Z")

        assert all(result.published_at is None or result.published_at >= floor for result in response)

    def test_oversized_query_is_rejected_locally(self):
        with WebSearchClient(region=self.region) as client:
            with pytest.raises(ValueError):
                client.search("x" * 201)
