"""Strands-native message-shape converter adapter."""

from ..bedrock_converter import AgentCoreMemoryConverter


class BedrockConverseConverter(AgentCoreMemoryConverter):
    """Alias adapter for the default Strands-native-shape converter."""
