"""Integration tests for AgentCoreWebSearch.

These call a real gateway that already has a web search connector target on it. The
connector is enabled per account, so they skip rather than fail when the account is not
entitled to it.

Run with:
    uv run pytest tests_integ/tools/integrations/strands/test_web_search_integration.py -xvs

Requires environment variables:
    WEB_SEARCH_GATEWAY_ID: ID of a gateway with a web search connector target
    BEDROCK_TEST_REGION: AWS region (default: us-east-1). The connector is offered in
        us-east-1, eu-west-1 and ap-northeast-1.
    WEB_SEARCH_TARGET_NAME: Optional. Name of the web search target on that gateway.
        Supplying it saves a tools/list call, since Gateway prefixes every tool with the
        name of the target it came from.
    STRANDS_TEST_MODEL_ID: Optional. Model to run the agent test against.
        Default: us.anthropic.claude-sonnet-4-5-20250929-v1:0. That test also needs
        bedrock:InvokeModel on the model and skips if the model is not reachable.
"""

import os

import pytest
from strands import Agent

from bedrock_agentcore.tools.integrations.strands import AgentCoreWebSearch
from bedrock_agentcore.tools.web_search_client import WebSearchError

DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


@pytest.mark.integration
class TestAgentCoreWebSearchIntegration:
    """AgentCoreWebSearch against a live gateway target."""

    @classmethod
    def setup_class(cls):
        cls.gateway_id = os.environ.get("WEB_SEARCH_GATEWAY_ID")
        if not cls.gateway_id:
            pytest.skip("WEB_SEARCH_GATEWAY_ID must be set")
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-east-1")
        cls.target_name = os.environ.get("WEB_SEARCH_TARGET_NAME")
        cls.model_id = os.environ.get("STRANDS_TEST_MODEL_ID", DEFAULT_MODEL_ID)

    def _search_tool(self):
        return AgentCoreWebSearch(
            region=self.region,
            gateway_id=self.gateway_id,
            target_name=self.target_name,
        )

    def _search(self, tool, query, **kwargs):
        """Search, skipping when the account is not entitled to the connector."""
        try:
            return tool.web_search(query, **kwargs)
        except WebSearchError as e:
            if "not available for this account" in str(e):
                pytest.skip(f"web-search connector not enabled for this account: {e}")
            raise

    def test_search_returns_citable_results(self):
        with self._search_tool() as tool:
            text = self._search(tool, "what is amazon bedrock agentcore", max_results=3)

        assert text.startswith("1. ")
        # Citations must be retained for any output shown to an end user, so a URL has
        # to survive into the text the model reads.
        assert "URL: http" in text
        assert "No results" not in text

    def test_search_respects_max_results(self):
        with self._search_tool() as tool:
            text = self._search(tool, "python urllib3 release notes", max_results=2)

        assert "3. " not in text

    def test_search_with_domain_filter(self):
        """Needs connector version 1.2.0 or later on the target.

        A request-level include filter is only accepted from 1.2.0 on, and the version of
        an existing gateway's target is not this test's to choose, so an older target
        skips rather than fails.
        """
        with self._search_tool() as tool:
            try:
                text = self._search(
                    tool,
                    "agentcore gateway connector targets",
                    max_results=5,
                    include_domains=["docs.aws.amazon.com"],
                )
            except WebSearchError as e:
                if "domainFilter" in str(e) or "include" in str(e):
                    pytest.skip(f"target's connector version does not accept an include filter: {e}")
                raise

        if "No results" in text:
            pytest.skip("include filter returned nothing; check the target's own domain rules")
        for line in text.splitlines():
            if line.strip().startswith("URL: "):
                assert "aws.amazon.com" in line

    def test_tool_name_discovery_without_a_target_name(self):
        """Without target_name the client finds the tool through tools/list."""
        with AgentCoreWebSearch(region=self.region, gateway_id=self.gateway_id) as tool:
            self._search(tool, "bedrock agentcore gateway", max_results=1)

            assert tool._client.backend._tool_name.endswith("WebSearch")

    def test_the_client_is_reused_across_searches(self):
        with self._search_tool() as tool:
            self._search(tool, "first query", max_results=1)
            self._search(tool, "second query", max_results=1)

            assert tool._client.backend._mcp_session_id

    def test_an_agent_calls_the_tool_and_cites_a_url(self):
        """The end to end path: a model decides to search, and the URL reaches its answer.

        This is the only test here that exercises the Strands tool contract rather than
        the method. It asserts a URL appears in the tool result recorded in the agent's
        messages, not that the model's prose cites one, because the model's wording is
        not this test's to assert.

        A search failure does not propagate out of ``agent(...)``: the Strands executor
        turns a tool exception into a ``status: "error"`` tool result, which is why the
        entitlement check reads the results rather than catching. A model that cannot be
        reached does propagate, since that is not a tool call.

        The model is resolved by Strands through ``BedrockModel``, so it uses the ambient
        AWS region rather than ``BEDROCK_TEST_REGION``. Those two are independent: web
        search is offered in fewer regions than the models are.
        """
        with self._search_tool() as tool:
            agent = Agent(model=self.model_id, tools=[tool.web_search])
            try:
                agent("Search the web for the newest boto3 release and give me the source URL.")
            except Exception as e:
                if "AccessDenied" in str(e) or "ValidationException" in str(e):
                    pytest.skip(f"test model {self.model_id} is not reachable: {e}")
                raise

            tool_results = [
                block["toolResult"]
                for message in agent.messages
                for block in message.get("content", [])
                if isinstance(block, dict) and "toolResult" in block
            ]

        assert tool_results, "the model did not call the search tool"
        returned = str(tool_results)
        if "not available for this account" in returned:
            pytest.skip(f"web-search connector not enabled for this account: {returned}")

        # A raised search failure arrives here as status error, so this also pins down
        # that the failure stays distinguishable from a search that found nothing.
        assert all(result.get("status") != "error" for result in tool_results), returned
        assert "URL: http" in returned
