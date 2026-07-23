"""Strands-native long-term memory stores backed by Bedrock AgentCore Memory."""

from .factory import assert_writable_topology, create_agentcore_memory_stores
from .sender import AgentCoreEventSender
from .store import AgentCoreMemoryStore
from .types import (
    RESERVED_METADATA_PREFIX,
    AgentCoreEventSenderConfig,
    AgentCoreExactNamespaceStoreConfig,
    AgentCoreExtractionConfig,
    AgentCoreMemoryStoreConfig,
    AgentCoreNamespaceConfig,
    AgentCoreSubtreeStoreConfig,
    CreateAgentCoreMemoryStoresInput,
    ExtractionMode,
    MetadataProvider,
    MetadataValue,
    resolve_namespace,
    slugify_namespace,
)

__all__ = [
    "RESERVED_METADATA_PREFIX",
    "AgentCoreEventSender",
    "AgentCoreEventSenderConfig",
    "AgentCoreExactNamespaceStoreConfig",
    "AgentCoreExtractionConfig",
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
]
