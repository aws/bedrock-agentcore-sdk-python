"""AgentCore tool search plugin for Strands Agents."""

import json
import logging

from mcp.types import Tool as MCPTool
from strands.hooks import BeforeInvocationEvent
from strands.plugins import Plugin, hook
from strands.tools.mcp import MCPClient
from strands.tools.mcp.mcp_agent_tool import MCPAgentTool

from .intent_providers import IntentProvider, StrandsIntentProvider

logger = logging.getLogger(__name__)


class AgentCoreToolSearchPlugin(Plugin):
    """Plugin that dynamically loads tools from AgentCore Gateway based on semantic intent.

    Args:
        mcp_client: MCPClient connected to an AgentCore Gateway.
        intent_provider: Strategy for deriving intent. Defaults to StrandsIntentProvider.
    """

    name = "agentcore-tool-search-plugin"

    def __init__(
        self,
        mcp_client: MCPClient,
        intent_provider: IntentProvider | None = None,
    ):
        """Initialize the plugin.

        Args:
            mcp_client: MCPClient connected to an AgentCore Gateway.
            intent_provider: Strategy for deriving intent. Defaults to StrandsIntentProvider.
        """
        super().__init__()
        self._intent_provider = intent_provider or StrandsIntentProvider()
        self._mcp_client = mcp_client
        self._loaded_tool_names: set[str] = set()

    @property
    def tools(self):
        """Return empty list; tools are loaded dynamically via the hook."""
        return []

    @hook
    def on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Derive intent, search gateway, and load matching tools."""
        messages = event.messages or []

        # Pass the agent's model to the intent provider
        intent = self._intent_provider.derive_intent(messages, model=event.agent.model)
        logger.info("Derived intent: %s", intent)

        # Clear all previously loaded conditional tools
        for name in list(self._loaded_tool_names):
            event.agent.tool_registry.registry.pop(name, None)
        self._loaded_tool_names.clear()

        if not intent:
            return

        try:
            result = self._mcp_client.call_tool_sync(
                tool_use_id="intent-search",
                name="x_amz_bedrock_agentcore_search",
                arguments={"query": intent},
            )
            agent_tools = self._build_tools_from_search_result(result)
        except Exception as e:
            logger.error("AgentCore Gateway search failed: %s", e)
            return

        for agent_tool in agent_tools:
            try:
                # Skip if a non-dynamic tool with this name already exists
                if (
                    agent_tool.tool_name in event.agent.tool_registry.registry
                    and agent_tool.tool_name not in self._loaded_tool_names
                ):
                    logger.debug("Skipping tool %s: already registered as a static tool", agent_tool.tool_name)
                    continue
                event.agent.tool_registry.register_tool(agent_tool)
                self._loaded_tool_names.add(agent_tool.tool_name)
            except Exception as e:
                logger.error("Failed to register tool %s: %s", agent_tool.tool_name, e)

        logger.info("Loaded tools: %s", self._loaded_tool_names)

    def _build_tools_from_search_result(self, result) -> list[MCPAgentTool]:
        """Build MCPAgentTool objects from the gateway search response."""
        tools = []
        if not result or not isinstance(result, dict):
            return tools

        tool_defs = []
        structured = result.get("structuredContent")
        if isinstance(structured, dict) and "tools" in structured:
            tool_defs = structured["tools"]
        else:
            for block in result.get("content", []):
                if isinstance(block, dict) and "text" in block:
                    try:
                        data = json.loads(block["text"])
                        if isinstance(data, dict) and "tools" in data:
                            tool_defs = data["tools"]
                            break
                    except (json.JSONDecodeError, TypeError):
                        continue

        for tool_def in tool_defs:
            if not isinstance(tool_def, dict) or "name" not in tool_def:
                continue
            mcp_tool = MCPTool(
                name=tool_def["name"],
                description=tool_def.get("description", ""),
                inputSchema=tool_def.get("inputSchema", {"type": "object", "properties": {}}),
            )
            tools.append(MCPAgentTool(mcp_tool=mcp_tool, mcp_client=self._mcp_client))

        return tools
