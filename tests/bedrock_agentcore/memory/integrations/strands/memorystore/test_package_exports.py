"""Package export tests for the contained MemoryStore integration."""

import bedrock_agentcore.memory.integrations.strands as strands_integration
import bedrock_agentcore.memory.integrations.strands.memorystore as memorystore


def test_memorystore_package_explicitly_exports_public_surface() -> None:
    """Expose MemoryStore APIs from their contained canonical package."""
    assert set(memorystore.__all__) == {
        "RESERVED_METADATA_PREFIX",
        "AgentCoreEventSender",
        "AgentCoreEventSenderConfig",
        "AgentCoreExtractionConfig",
        "AgentCoreExactNamespaceStoreConfig",
        "AgentCoreMemoryStore",
        "AgentCoreMemoryStoreConfig",
        "AgentCoreNamespaceConfig",
        "AgentCoreSubtreeStoreConfig",
        "CreateAgentCoreMemoryStoresInput",
        "ExtractionMode",
        "MetadataProvider",
        "MetadataValue",
        "assert_writable_topology",
        "create_agentcore_memory_stores",
        "resolve_namespace",
        "slugify_namespace",
    }


def test_strands_root_preserves_converter_exports_only() -> None:
    """Do not add MemoryStore APIs to the existing Strands package root."""
    assert strands_integration.__all__ == ["MemoryConverter", "OpenAIConverseConverter"]
    assert strands_integration.MemoryConverter is not None
    assert strands_integration.OpenAIConverseConverter is not None
    assert not hasattr(strands_integration, "AgentCoreMemoryStore")
