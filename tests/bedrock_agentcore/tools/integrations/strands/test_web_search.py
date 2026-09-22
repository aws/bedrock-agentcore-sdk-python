"""Tests for AgentCoreWebSearch."""

from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.tools.integrations.strands.web_search import (
    DEFAULT_REGION,
    AgentCoreWebSearch,
    _format_response,
)
from bedrock_agentcore.tools.web_search_client import WebSearchError, WebSearchResponse, WebSearchResult

ARN = "arn:aws:bedrock-agentcore:eu-west-1:111122223333:gateway/my-gw-abc123"


def _response(*results):
    return WebSearchResponse(results=list(results), search_id="search-1")


def _result(**kwargs):
    fields = {"text": "", "url": None, "title": None, "published_date": None}
    fields.update(kwargs)
    return WebSearchResult(**fields)


@pytest.fixture
def client():
    """A stand-in WebSearchClient whose searches return one result."""
    client = MagicMock()
    client.search.return_value = _response(_result(title="t", url="https://example.com", text="body"))
    return client


class TestClientConstruction:
    """How the underlying WebSearchClient gets built."""

    @patch("bedrock_agentcore.tools.integrations.strands.web_search.WebSearchClient")
    def test_leaves_region_unset_when_a_gateway_arn_carries_one(self, mock_client):
        AgentCoreWebSearch(gateway_arn=ARN)

        # Filling in a default here would send a eu-west-1 gateway to us-east-1,
        # because the client prefers an explicit region over the ARN's.
        assert "region" not in mock_client.call_args.kwargs

    @patch("bedrock_agentcore.tools.integrations.strands.web_search.WebSearchClient")
    def test_an_explicit_region_still_wins_over_a_gateway_arn(self, mock_client):
        AgentCoreWebSearch("us-east-1", gateway_arn=ARN)

        assert mock_client.call_args.kwargs["region"] == "us-east-1"

    @patch("bedrock_agentcore.tools.integrations.strands.web_search.WebSearchClient")
    def test_defaults_the_region_when_no_arn_is_given(self, mock_client):
        AgentCoreWebSearch(gateway_id="my-gw-abc123")

        assert mock_client.call_args.kwargs["region"] == DEFAULT_REGION

    @patch("bedrock_agentcore.tools.integrations.strands.web_search.WebSearchClient")
    def test_forwards_only_the_arguments_that_were_given(self, mock_client):
        AgentCoreWebSearch(gateway_id="my-gw-abc123")

        # An explicit gateway_id=None would tie this module to the gateway
        # transport, so absent arguments must not be forwarded at all.
        assert set(mock_client.call_args.kwargs) == {"integration_source", "region", "gateway_id"}

    @patch("bedrock_agentcore.tools.integrations.strands.web_search.WebSearchClient")
    def test_attributes_usage_to_strands(self, mock_client):
        AgentCoreWebSearch(gateway_id="my-gw-abc123")

        assert mock_client.call_args.kwargs["integration_source"] == "strands"

    @patch("bedrock_agentcore.tools.integrations.strands.web_search.WebSearchClient")
    def test_forwards_the_discovery_shortcuts(self, mock_client):
        AgentCoreWebSearch(gateway_id="my-gw-abc123", target_name="amazon-web-search", tool_name="x___WebSearch")

        assert mock_client.call_args.kwargs["target_name"] == "amazon-web-search"
        assert mock_client.call_args.kwargs["tool_name"] == "x___WebSearch"

    def test_rejects_a_client_alongside_a_gateway_argument(self, client):
        with pytest.raises(ValueError, match="not both"):
            AgentCoreWebSearch(gateway_id="my-gw-abc123", client=client)


class TestToolSpec:
    """The tool spec Strands derives from the decorated method."""

    def test_is_named_web_search(self, client):
        assert AgentCoreWebSearch(client=client).web_search.tool_name == "web_search"

    def test_takes_the_query_plus_every_filter_and_requires_only_the_query(self, client):
        schema = AgentCoreWebSearch(client=client).web_search.tool_spec["inputSchema"]["json"]

        assert schema["required"] == ["query"]
        assert set(schema["properties"]) == {
            "query",
            "max_results",
            "include_domains",
            "exclude_domains",
            "published_after",
            "published_before",
        }

    def test_asks_the_model_to_cite_its_sources(self, client):
        # Citing sources is a condition of use for this connector, so losing this
        # sentence from the description is a compliance problem, not a wording one.
        description = AgentCoreWebSearch(client=client).web_search.tool_spec["description"]

        assert "Cite the URLs" in description


class TestSearching:
    """What reaches the client, and what comes back."""

    def test_passes_the_query_through(self, client):
        AgentCoreWebSearch(client=client).web_search("who maintains urllib3")

        assert client.search.call_args.args == ("who maintains urllib3",)

    def test_forwards_every_filter(self, client):
        AgentCoreWebSearch(client=client).web_search(
            "python releases",
            max_results=5,
            include_domains=["python.org"],
            exclude_domains=["spam.example"],
            published_after="2026-01-01T00:00:00Z",
            published_before="2026-06-01T00:00:00Z",
        )

        assert client.search.call_args.kwargs == {
            "max_results": 5,
            "include_domains": ["python.org"],
            "exclude_domains": ["spam.example"],
            "published_after": "2026-01-01T00:00:00Z",
            "published_before": "2026-06-01T00:00:00Z",
        }

    def test_returns_the_formatted_results(self, client):
        result = AgentCoreWebSearch(client=client).web_search("anything")

        assert "1. t" in result
        assert "https://example.com" in result

    def test_raises_rather_than_returning_the_failure_as_text(self, client):
        # Returning the message as a successful result would make a failed search
        # look like a search that found nothing.
        client.search.side_effect = WebSearchError("gateway said no")

        with pytest.raises(WebSearchError, match="gateway said no"):
            AgentCoreWebSearch(client=client).web_search("anything")

    def test_refuses_to_search_once_closed(self, client):
        search = AgentCoreWebSearch(client=client)
        search.close()

        with pytest.raises(ValueError, match="closed"):
            search.web_search("anything")


class TestFormatting:
    """How results are rendered for the model."""

    def test_numbers_each_result(self):
        text = _format_response(_response(_result(title="first"), _result(title="second")))

        assert text.startswith("1. first")
        assert "2. second" in text

    def test_shows_the_publication_date(self):
        # Recency is most of why an agent searches, so the date has to reach the model
        # rather than being dropped on the way through.
        text = _format_response(_response(_result(title="t", published_date="2026-09-14T00:00:00Z")))

        assert "Published: 2026-09-14T00:00:00Z" in text

    def test_omits_fields_the_index_did_not_report(self):
        text = _format_response(_response(_result(text="just an extract")))

        assert "URL:" not in text
        assert "Published:" not in text
        assert "just an extract" in text

    def test_names_untitled_results(self):
        assert "1. Untitled" in _format_response(_response(_result(url="https://example.com")))

    def test_explains_an_empty_result_set(self):
        # An empty list is what an over-narrow filter returns, and the model
        # cannot tell that from a failure unless the text says so.
        text = _format_response(_response())

        assert "No results" in text
        assert "filter" in text


class TestLifecycle:
    """Ownership of the client."""

    @patch("bedrock_agentcore.tools.integrations.strands.web_search.WebSearchClient")
    def test_closes_a_client_it_created(self, mock_client):
        search = AgentCoreWebSearch(gateway_id="my-gw-abc123")
        search.close()

        mock_client.return_value.close.assert_called_once()

    def test_leaves_an_injected_client_open(self, client):
        AgentCoreWebSearch(client=client).close()

        client.close.assert_not_called()

    @patch("bedrock_agentcore.tools.integrations.strands.web_search.WebSearchClient")
    def test_closing_twice_closes_once(self, mock_client):
        search = AgentCoreWebSearch(gateway_id="my-gw-abc123")
        search.close()
        search.close()

        mock_client.return_value.close.assert_called_once()

    @patch("bedrock_agentcore.tools.integrations.strands.web_search.WebSearchClient")
    def test_closes_on_leaving_a_with_block(self, mock_client):
        with AgentCoreWebSearch(gateway_id="my-gw-abc123") as search:
            assert search is not None

        mock_client.return_value.close.assert_called_once()
