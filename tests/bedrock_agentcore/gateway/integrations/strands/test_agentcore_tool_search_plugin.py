"""Tests for AgentCoreToolSearchPlugin."""

import json
from unittest.mock import Mock

import pytest

from bedrock_agentcore.gateway.integrations.strands.plugins.agentcore_tool_search.intent_providers import (
    StrandsIntentProvider,
    IntentProvider,
)
from bedrock_agentcore.gateway.integrations.strands.plugins.agentcore_tool_search.plugin import (
    AgentCoreToolSearchPlugin,
)


class FakeIntentProvider(IntentProvider):
    """Test intent provider that returns a fixed intent string."""

    def __init__(self, intent: str = "test intent"):
        self._intent = intent

    def derive_intent(self, messages: list[dict], model=None) -> str:
        return self._intent


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCPClient."""
    client = Mock()
    client.call_tool_sync.return_value = {"content": []}
    return client


@pytest.fixture
def fixed_intent_provider():
    """Create a fixed intent provider."""
    return FakeIntentProvider("get weather")


@pytest.fixture
def plugin(mock_mcp_client, fixed_intent_provider):
    """Create an AgentCoreToolSearchPlugin with mocked dependencies."""
    return AgentCoreToolSearchPlugin(mcp_client=mock_mcp_client, intent_provider=fixed_intent_provider)


@pytest.fixture
def mock_event():
    """Create a mock BeforeInvocationEvent."""
    event = Mock()
    event.messages = [{"role": "user", "content": [{"text": "hello"}]}]
    event.agent = Mock()
    event.agent.model = None
    event.agent.tool_registry = Mock()
    event.agent.tool_registry.registry = {}
    return event


class TestAgentCoreToolSearchPluginInit:
    """Test AgentCoreToolSearchPlugin initialization."""

    def test_init_with_custom_intent_provider(self, mock_mcp_client):
        """Test initialization with a custom intent provider."""
        provider = FakeIntentProvider("custom")
        plugin = AgentCoreToolSearchPlugin(mcp_client=mock_mcp_client, intent_provider=provider)
        assert plugin._intent_provider is provider
        assert plugin._mcp_client is mock_mcp_client

    def test_init_default_intent_provider(self, mock_mcp_client):
        """Test initialization uses StrandsIntentProvider when none provided."""
        plugin = AgentCoreToolSearchPlugin(mcp_client=mock_mcp_client)
        assert isinstance(plugin._intent_provider, StrandsIntentProvider)

    def test_plugin_name(self, plugin):
        """Test plugin has correct name."""
        assert plugin.name == "agentcore-tool-search-plugin"

    def test_tools_property_returns_empty(self, plugin):
        """Test tools property returns empty list."""
        assert plugin.tools == []


class TestOnBeforeInvocation:
    """Test on_before_invocation hook behavior."""

    def test_empty_intent_skips_search(self, mock_mcp_client, mock_event):
        """Test that empty intent does not call gateway search."""
        provider = FakeIntentProvider("")
        plugin = AgentCoreToolSearchPlugin(mcp_client=mock_mcp_client, intent_provider=provider)

        plugin.on_before_invocation(mock_event)

        mock_mcp_client.call_tool_sync.assert_not_called()

    def test_calls_gateway_search_with_intent(self, plugin, mock_mcp_client, mock_event):
        """Test that derived intent is passed to gateway search."""
        plugin.on_before_invocation(mock_event)

        mock_mcp_client.call_tool_sync.assert_called_once_with(
            tool_use_id="intent-search",
            name="x_amz_bedrock_agentcore_search",
            arguments={"query": "get weather"},
        )

    def test_passes_agent_model_to_intent_provider(self, mock_mcp_client, mock_event):
        """Test that the agent's model is passed to derive_intent."""
        provider = Mock(spec=IntentProvider)
        provider.derive_intent.return_value = ""
        plugin = AgentCoreToolSearchPlugin(mcp_client=mock_mcp_client, intent_provider=provider)
        mock_event.agent.model = Mock(name="test-model")

        plugin.on_before_invocation(mock_event)

        provider.derive_intent.assert_called_once_with(mock_event.messages, model=mock_event.agent.model)

    def test_registers_tools_from_structured_content(self, plugin, mock_mcp_client, mock_event):
        """Test tools are registered from structuredContent response."""
        mock_mcp_client.call_tool_sync.return_value = {
            "structuredContent": {
                "tools": [
                    {
                        "name": "weather_tool",
                        "description": "Get weather",
                        "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
                    }
                ]
            }
        }

        plugin.on_before_invocation(mock_event)

        mock_event.agent.tool_registry.register_tool.assert_called_once()
        registered_tool = mock_event.agent.tool_registry.register_tool.call_args[0][0]
        assert registered_tool.tool_name == "weather_tool"
        assert "weather_tool" in plugin._loaded_tool_names

    def test_registers_tools_from_text_content(self, plugin, mock_mcp_client, mock_event):
        """Test tools are registered from JSON text content response."""
        tools_json = json.dumps(
            {
                "tools": [
                    {
                        "name": "calc_tool",
                        "description": "Calculator",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ]
            }
        )
        mock_mcp_client.call_tool_sync.return_value = {"content": [{"text": tools_json}]}

        plugin.on_before_invocation(mock_event)

        mock_event.agent.tool_registry.register_tool.assert_called_once()
        registered_tool = mock_event.agent.tool_registry.register_tool.call_args[0][0]
        assert registered_tool.tool_name == "calc_tool"

    def test_clears_previously_loaded_tools(self, plugin, mock_mcp_client, mock_event):
        """Test previously loaded tools are removed from registry."""
        mock_mcp_client.call_tool_sync.return_value = {"content": []}
        plugin._loaded_tool_names = {"old_tool_1", "old_tool_2"}
        mock_event.agent.tool_registry.registry = {
            "old_tool_1": Mock(),
            "old_tool_2": Mock(),
            "permanent_tool": Mock(),
        }

        plugin.on_before_invocation(mock_event)

        assert "old_tool_1" not in mock_event.agent.tool_registry.registry
        assert "old_tool_2" not in mock_event.agent.tool_registry.registry
        assert "permanent_tool" in mock_event.agent.tool_registry.registry
        assert len(plugin._loaded_tool_names) == 0

    def test_gateway_search_failure_logs_and_returns(self, plugin, mock_mcp_client, mock_event):
        """Test gateway search failure is handled gracefully."""
        mock_mcp_client.call_tool_sync.side_effect = RuntimeError("connection failed")

        plugin.on_before_invocation(mock_event)

        mock_event.agent.tool_registry.register_tool.assert_not_called()

    def test_skips_invalid_tool_defs(self, plugin, mock_mcp_client, mock_event):
        """Test malformed tool definitions are skipped."""
        mock_mcp_client.call_tool_sync.return_value = {
            "structuredContent": {
                "tools": [
                    {"description": "no name field"},
                    "not a dict",
                    {"name": "valid_tool", "description": "ok", "inputSchema": {"type": "object", "properties": {}}},
                ]
            }
        }

        plugin.on_before_invocation(mock_event)

        mock_event.agent.tool_registry.register_tool.assert_called_once()
        registered_tool = mock_event.agent.tool_registry.register_tool.call_args[0][0]
        assert registered_tool.tool_name == "valid_tool"

    def test_register_tool_failure_continues(self, plugin, mock_mcp_client, mock_event):
        """Test that failure to register one tool doesn't block others."""
        mock_mcp_client.call_tool_sync.return_value = {
            "structuredContent": {
                "tools": [
                    {"name": "tool_a", "description": "A", "inputSchema": {"type": "object", "properties": {}}},
                    {"name": "tool_b", "description": "B", "inputSchema": {"type": "object", "properties": {}}},
                ]
            }
        }
        mock_event.agent.tool_registry.register_tool.side_effect = [RuntimeError("fail"), None]

        plugin.on_before_invocation(mock_event)

        assert mock_event.agent.tool_registry.register_tool.call_count == 2
        assert "tool_a" not in plugin._loaded_tool_names
        assert "tool_b" in plugin._loaded_tool_names

    def test_none_result_loads_no_tools(self, plugin, mock_mcp_client, mock_event):
        """Test None result from gateway loads no tools."""
        mock_mcp_client.call_tool_sync.return_value = None

        plugin.on_before_invocation(mock_event)

        mock_event.agent.tool_registry.register_tool.assert_not_called()

    def test_empty_messages_with_intent(self, mock_mcp_client):
        """Test plugin works with empty messages list."""
        provider = FakeIntentProvider("")
        plugin = AgentCoreToolSearchPlugin(mcp_client=mock_mcp_client, intent_provider=provider)
        event = Mock()
        event.messages = []
        event.agent = Mock()
        event.agent.model = None
        event.agent.tool_registry = Mock()
        event.agent.tool_registry.registry = {}

        plugin.on_before_invocation(event)

        mock_mcp_client.call_tool_sync.assert_not_called()
