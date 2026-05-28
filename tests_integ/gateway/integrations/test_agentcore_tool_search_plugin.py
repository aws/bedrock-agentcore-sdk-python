"""Integration tests for AgentCoreToolSearchPlugin.

Requires environment variables:
    BEDROCK_TEST_REGION: AWS region (default: us-west-2)
    GATEWAY_ROLE_ARN: IAM role ARN with AgentCore gateway trust policy
    GATEWAY_LAMBDA_ARN: Lambda ARN for the gateway target (must implement MCP tool handler)

Prerequisites:
    1. Deploy the Lambda in tests_integ/gateway/integrations/lambda_function/lambda_function.py
       (Python 3.10+ runtime, handler: lambda_function.lambda_handler)

    2. The GATEWAY_ROLE_ARN must have:
       - Trust policy for bedrock-agentcore.amazonaws.com
       - lambda:InvokeFunction permission on the GATEWAY_LAMBDA_ARN

       Example inline policy:
       {
           "Version": "2012-10-17",
           "Statement": [{
               "Effect": "Allow",
               "Action": "lambda:InvokeFunction",
               "Resource": "<GATEWAY_LAMBDA_ARN>"
           }]
       }

    3. Install mcp-proxy-for-aws: uv pip install mcp-proxy-for-aws

"""

import logging
import os
import time

import pytest

from bedrock_agentcore.gateway.client import GatewayClient
from bedrock_agentcore.gateway.integrations.strands.plugins import AgentCoreToolSearchPlugin
from bedrock_agentcore.gateway.integrations.strands.plugins.agentcore_tool_search.intent_providers import (
    StrandsIntentProvider,
    IntentProvider,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FixedIntentProvider(IntentProvider):
    """Intent provider that returns a fixed string for deterministic testing."""

    def __init__(self, intent: str):
        self._intent = intent

    def derive_intent(self, messages: list[dict], model=None) -> str:
        return self._intent


@pytest.mark.integration
class TestAgentCoreToolSearchPluginIntegration:
    """Integration tests for AgentCoreToolSearchPlugin with a live gateway.

    Creates a gateway with a Lambda target exposing test tools, then verifies
    the plugin can search and load those tools.
    """

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.role_arn = os.environ.get("GATEWAY_ROLE_ARN")
        cls.lambda_arn = os.environ.get("GATEWAY_LAMBDA_ARN")

        if not cls.role_arn or not cls.lambda_arn:
            pytest.fail("GATEWAY_ROLE_ARN and GATEWAY_LAMBDA_ARN must be set")

        cls.gw_client = GatewayClient(region_name=cls.region)
        cls.test_prefix = f"sdk-integ-plugin-{int(time.time())}"
        cls.gateway_id = None
        cls.target_id = None

        # Create gateway with semantic search enabled
        gw = cls.gw_client.create_gateway_and_wait(
            name=f"{cls.test_prefix}-gw",
            roleArn=cls.role_arn,
            authorizerType="NONE",
            protocolType="MCP",
            protocolConfiguration={
                "mcp": {
                    "searchType": "SEMANTIC",
                },
            },
        )
        cls.gateway_id = gw["gatewayId"]
        logger.info("Created gateway: %s", cls.gateway_id)

        # Create target with test tools
        target = cls.gw_client.create_gateway_target_and_wait(
            gatewayIdentifier=cls.gateway_id,
            name=f"{cls.test_prefix}-target",
            targetConfiguration={
                "mcp": {
                    "lambda": {
                        "lambdaArn": cls.lambda_arn,
                        "toolSchema": {
                            "inlinePayload": [
                                {
                                    "name": "get_weather",
                                    "description": "Get current weather for a city",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "city": {"type": "string", "description": "City name"},
                                        },
                                        "required": ["city"],
                                    },
                                },
                                {
                                    "name": "send_email",
                                    "description": "Send an email to a recipient",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "to": {"type": "string"},
                                            "subject": {"type": "string"},
                                            "body": {"type": "string"},
                                        },
                                        "required": ["to", "subject", "body"],
                                    },
                                },
                            ]
                        },
                    }
                },
            },
            credentialProviderConfigurations=[
                {"credentialProviderType": "GATEWAY_IAM_ROLE"},
            ],
        )
        cls.target_id = target["targetId"]
        logger.info("Created target: %s", cls.target_id)

        # Wait for target search indexing to complete (can take up to 60s)
        time.sleep(60)

    @classmethod
    def teardown_class(cls):
        if cls.gateway_id:
            if cls.target_id:
                try:
                    cls.gw_client.delete_gateway_target_and_wait(
                        gatewayIdentifier=cls.gateway_id,
                        targetId=cls.target_id,
                    )
                except Exception as e:
                    logger.warning("Failed to delete target %s: %s", cls.target_id, e)
            try:
                cls.gw_client.delete_gateway_and_wait(
                    gatewayIdentifier=cls.gateway_id,
                )
            except Exception as e:
                logger.warning("Failed to delete gateway %s: %s", cls.gateway_id, e)

    def _make_mcp_client(self):
        """Create an MCPClient connected to the test gateway via Streamable HTTP with IAM auth."""
        from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
        from strands.tools.mcp import MCPClient

        endpoint = f"https://{self.gateway_id}.gateway.bedrock-agentcore.{self.region}.amazonaws.com/mcp"
        return MCPClient(
            lambda: aws_iam_streamablehttp_client(
                endpoint=endpoint,
                aws_region=self.region,
                aws_service="bedrock-agentcore",
            )
        )

    @pytest.mark.order(1)
    def test_plugin_with_default_intent_provider(self):
        """Plugin initializes correctly with StrandsIntentProvider."""
        mcp_client = self._make_mcp_client()
        plugin = AgentCoreToolSearchPlugin(mcp_client=mcp_client)
        assert isinstance(plugin._intent_provider, StrandsIntentProvider)
        assert plugin.name == "agentcore-tool-search-plugin"

    @pytest.mark.order(2)
    def test_plugin_with_custom_intent_provider(self):
        """Plugin accepts a custom IntentProvider."""
        mcp_client = self._make_mcp_client()
        provider = FixedIntentProvider("weather query")
        plugin = AgentCoreToolSearchPlugin(mcp_client=mcp_client, intent_provider=provider)
        assert plugin._intent_provider is provider

    @pytest.mark.order(3)
    def test_gateway_search_returns_results(self):
        """Calling x_amz_bedrock_agentcore_search on the gateway returns tool definitions."""
        mcp_client = self._make_mcp_client()

        with mcp_client:
            result = mcp_client.call_tool_sync(
                tool_use_id="test-search",
                name="x_amz_bedrock_agentcore_search",
                arguments={"query": "get weather information"},
            )

        assert result is not None
        logger.info("Search result keys: %s", result.keys() if isinstance(result, dict) else type(result))

    @pytest.mark.order(4)
    def test_plugin_loads_tools_via_hook(self):
        """Plugin loads matching tools into the agent via the before_invocation hook."""
        from strands import Agent

        mcp_client = self._make_mcp_client()
        provider = FixedIntentProvider("get weather information")
        plugin = AgentCoreToolSearchPlugin(mcp_client=mcp_client, intent_provider=provider)

        with mcp_client:
            # First verify the search endpoint returns tools
            result = mcp_client.call_tool_sync(
                tool_use_id="debug-search",
                name="x_amz_bedrock_agentcore_search",
                arguments={"query": "get weather information"},
            )
            logger.info("Raw search result: %s", result)

            agent = Agent(
                system_prompt="You are a helpful assistant. Use available tools to help the user.",
                tools=[],
                plugins=[plugin],
            )
            # Trigger an invocation so the hook fires
            agent("What is the weather in Seattle?")

        logger.info("Loaded tool names: %s", plugin._loaded_tool_names)
        # The gateway should have returned the get_weather tool
        assert len(plugin._loaded_tool_names) > 0, (
            f"Expected tools to be loaded but got none. Raw search result was: {result}"
        )

    @pytest.mark.order(5)
    def test_empty_intent_loads_no_tools(self):
        """Plugin does not search gateway when intent is empty."""
        from unittest.mock import Mock

        from strands.hooks import BeforeInvocationEvent

        mcp_client = self._make_mcp_client()
        provider = FixedIntentProvider("")
        plugin = AgentCoreToolSearchPlugin(mcp_client=mcp_client, intent_provider=provider)

        # Simulate a before_invocation event
        event = Mock(spec=BeforeInvocationEvent)
        event.messages = [{"role": "user", "content": [{"text": "hello"}]}]
        event.agent = Mock()
        event.agent.model = None
        event.agent.tool_registry = Mock()
        event.agent.tool_registry.registry = {}

        plugin.on_before_invocation(event)

        assert len(plugin._loaded_tool_names) == 0

    @pytest.mark.order(6)
    def test_tools_cleared_between_invocations(self):
        """Previously loaded tools are cleared before each new search."""
        from unittest.mock import Mock

        from strands.hooks import BeforeInvocationEvent

        mcp_client = self._make_mcp_client()
        provider = FixedIntentProvider("get weather information")
        plugin = AgentCoreToolSearchPlugin(mcp_client=mcp_client, intent_provider=provider)

        with mcp_client:
            # First: simulate invocation with a real intent
            event = Mock(spec=BeforeInvocationEvent)
            event.messages = [{"role": "user", "content": [{"text": "weather"}]}]
            event.agent = Mock()
            event.agent.model = None
            event.agent.tool_registry = Mock()
            event.agent.tool_registry.registry = {}

            plugin.on_before_invocation(event)
            first_tools = set(plugin._loaded_tool_names)
            logger.info("First invocation tools: %s", first_tools)
            assert len(first_tools) > 0

            # Second: switch to empty intent — tools should be cleared
            provider._intent = ""
            event.agent.tool_registry.registry = {name: Mock() for name in first_tools}

            plugin.on_before_invocation(event)
            second_tools = set(plugin._loaded_tool_names)
            logger.info("Second invocation tools: %s", second_tools)

        assert len(second_tools) == 0
        # Verify old tools were removed from registry
        for name in first_tools:
            assert name not in event.agent.tool_registry.registry


@pytest.mark.integration
class TestStrandsIntentProviderIntegration:
    """Integration tests for StrandsIntentProvider with a real LLM."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")

    def test_derive_intent_from_messages(self):
        """StrandsIntentProvider produces a non-empty intent string from messages."""
        provider = StrandsIntentProvider(message_window=3)
        messages = [
            {"role": "user", "content": [{"text": "What's the weather like in Seattle today?"}]},
            {"role": "assistant", "content": [{"text": "Let me check the weather for you."}]},
            {"role": "user", "content": [{"text": "Also check tomorrow's forecast."}]},
        ]

        intent = provider.derive_intent(messages)

        logger.info("Derived intent: %s", intent)
        assert isinstance(intent, str)
        assert len(intent) > 0

    def test_derive_intent_empty_messages(self):
        """StrandsIntentProvider returns empty string for empty messages."""
        provider = StrandsIntentProvider()
        intent = provider.derive_intent([])
        assert intent == ""

    def test_derive_intent_with_custom_model(self):
        """StrandsIntentProvider works with an explicitly provided model."""
        from strands.models.bedrock import BedrockModel

        model = BedrockModel(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name=self.region,
        )
        provider = StrandsIntentProvider(model=model)
        messages = [
            {"role": "user", "content": [{"text": "I need to send an email to my team about the project update."}]},
        ]

        intent = provider.derive_intent(messages)

        logger.info("Derived intent with custom model: %s", intent)
        assert isinstance(intent, str)
        assert len(intent) > 0

    def test_derive_intent_respects_message_window(self):
        """StrandsIntentProvider only considers the last N messages."""
        provider = StrandsIntentProvider(message_window=2)
        messages = [
            {"role": "user", "content": [{"text": "Tell me about dogs."}]},
            {"role": "assistant", "content": [{"text": "Dogs are great pets."}]},
            {"role": "user", "content": [{"text": "Now tell me about the stock market."}]},
            {"role": "assistant", "content": [{"text": "The stock market is complex."}]},
            {"role": "user", "content": [{"text": "What are the best investment strategies?"}]},
        ]

        intent = provider.derive_intent(messages)

        logger.info("Derived intent (window=2): %s", intent)
        assert isinstance(intent, str)
        assert len(intent) > 0
