"""Strands integrations for Bedrock AgentCore Memory."""

import re
from importlib import metadata, util
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .converters import MemoryConverter, OpenAIConverseConverter  # noqa: F401
    from .factory import assert_writable_topology, create_agentcore_memory_stores  # noqa: F401
    from .sender import AgentCoreEventSender  # noqa: F401
    from .store import AgentCoreMemoryStore  # noqa: F401
    from .types import (  # noqa: F401
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

_LEGACY_EXPORTS = {"MemoryConverter", "OpenAIConverseConverter"}
_STORE_EXPORTS = {
    "AgentCoreEventSender",
    "AgentCoreMemoryStore",
    "assert_writable_topology",
    "create_agentcore_memory_stores",
}
_TYPE_EXPORTS = {
    "RESERVED_METADATA_PREFIX",
    "AgentCoreEventSenderConfig",
    "AgentCoreExtractionConfig",
    "AgentCoreExactNamespaceStoreConfig",
    "AgentCoreMemoryStoreConfig",
    "AgentCoreNamespaceConfig",
    "AgentCoreSubtreeStoreConfig",
    "CreateAgentCoreMemoryStoresInput",
    "ExtractionMode",
    "MetadataProvider",
    "MetadataValue",
    "resolve_namespace",
    "slugify_namespace",
}
_NATIVE_EXPORTS = _STORE_EXPORTS | _TYPE_EXPORTS
_MINIMUM_STRANDS_VERSION = (1, 46, 0)


def _module_spec_exists(name: str) -> bool:
    """Return whether a module exists without importing it or its package parent."""
    try:
        root, _, child = name.partition(".")
        root_spec = util.find_spec(root)
        if root_spec is None:
            return False
        if not child:
            return True
        locations = root_spec.submodule_search_locations
        if locations is None:
            return False
        from importlib.machinery import PathFinder

        return PathFinder.find_spec(name, locations) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _has_native_memory_support() -> bool:
    """Check for Strands native memory without importing the new integration modules."""
    try:
        installed = metadata.version("strands-agents")
        match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?([^+]*)", installed)
        if match is None:
            return False
        parsed = tuple(int(part or 0) for part in match.groups()[:3])
        suffix = match.group(4).lower()
        if parsed < _MINIMUM_STRANDS_VERSION:
            return False
        if parsed == _MINIMUM_STRANDS_VERSION and any(marker in suffix for marker in ("a", "b", "rc", "dev")):
            return False
        return _module_spec_exists("strands.memory")
    except (ImportError, ModuleNotFoundError, ValueError, metadata.PackageNotFoundError):
        return False


_STRANDS_AVAILABLE = _module_spec_exists("strands")
_NATIVE_MEMORY_AVAILABLE = _has_native_memory_support()
__all__ = sorted(
    (_LEGACY_EXPORTS if _STRANDS_AVAILABLE else set()) | (_NATIVE_EXPORTS if _NATIVE_MEMORY_AVAILABLE else set())
)


def _unsupported_native_memory_error(name: str) -> ImportError:
    return ImportError(
        f"'{name}' requires strands-agents>=1.46.0. Install it with: pip install 'bedrock-agentcore[strands-agents]'"
    )


def __getattr__(name: str) -> Any:
    """Lazy-load package-root exports while preserving optional-dependency compatibility."""
    if name in _LEGACY_EXPORTS:
        from .converters import MemoryConverter, OpenAIConverseConverter

        value: Any = {
            "MemoryConverter": MemoryConverter,
            "OpenAIConverseConverter": OpenAIConverseConverter,
        }[name]
        globals()[name] = value
        return value

    if name not in _NATIVE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if not _NATIVE_MEMORY_AVAILABLE:
        raise _unsupported_native_memory_error(name)

    try:
        if name == "AgentCoreEventSender":
            from .sender import AgentCoreEventSender

            value = AgentCoreEventSender
        elif name == "AgentCoreMemoryStore":
            from .store import AgentCoreMemoryStore

            value = AgentCoreMemoryStore
        elif name in {"assert_writable_topology", "create_agentcore_memory_stores"}:
            from .factory import assert_writable_topology, create_agentcore_memory_stores

            value = {
                "assert_writable_topology": assert_writable_topology,
                "create_agentcore_memory_stores": create_agentcore_memory_stores,
            }[name]
        else:
            from . import types as _types

            value = getattr(_types, name)
    except ImportError as error:
        raise _unsupported_native_memory_error(name) from error

    globals()[name] = value
    return value
