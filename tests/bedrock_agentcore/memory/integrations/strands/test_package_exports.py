"""Package-root export and lazy compatibility tests for the Strands integration."""

import builtins
import importlib
import sys
from types import ModuleType
from typing import Any

import pytest

MODULE = "bedrock_agentcore.memory.integrations.strands"


def _fresh_module() -> ModuleType:
    sys.modules.pop(MODULE, None)
    return importlib.import_module(MODULE)


def test_package_root_exports_only_supported_surface() -> None:
    """Keep service caps, protocols, helpers, and defaults off the package root."""
    module = _fresh_module()
    assert set(module.__all__) == {
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
        "MemoryConverter",
        "MetadataProvider",
        "MetadataValue",
        "OpenAIConverseConverter",
        "assert_writable_topology",
        "create_agentcore_memory_stores",
        "resolve_namespace",
        "slugify_namespace",
    }
    for name in (
        "AgentCoreDataPlaneClient",
        "AgentCoreMemoryConfig",
        "DEFAULT_MAX_SEARCH_RESULTS",
        "DEFAULT_MAX_TURNS_PER_EVENT",
        "DEFAULT_OVERFETCH_FACTOR",
        "MAX_TOPK",
        "assert_resolved_namespace",
    ):
        with pytest.raises(AttributeError):
            getattr(module, name)


def test_existing_converter_exports_remain_available_without_new_store_modules() -> None:
    """Legacy converter imports do not touch the new store modules."""
    for name in (MODULE, f"{MODULE}.store", f"{MODULE}.factory", f"{MODULE}.types"):
        sys.modules.pop(name, None)
    module = importlib.import_module(MODULE)
    assert module.MemoryConverter is not None
    assert module.OpenAIConverseConverter is not None
    assert f"{MODULE}.store" not in sys.modules
    assert f"{MODULE}.factory" not in sys.modules
    assert f"{MODULE}.types" not in sys.modules


def test_new_symbol_import_error_is_actionable_for_old_or_missing_strands(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lazy store access explains the extra and minimum supported version."""
    for name in (MODULE, f"{MODULE}.store", f"{MODULE}.factory", f"{MODULE}.types"):
        sys.modules.pop(name, None)
    module = importlib.import_module(MODULE)
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "strands.memory" or name.startswith("strands.memory."):
            raise ImportError("simulated old strands")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"strands-agents>=1\.46\.0.*bedrock-agentcore\[strands-agents\]"):
        _ = module.AgentCoreMemoryStore


def test_old_strands_wildcard_import_keeps_legacy_exports_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy wildcard imports succeed when native-memory Strands APIs are unavailable."""
    for name in (MODULE, f"{MODULE}.store", f"{MODULE}.factory", f"{MODULE}.types"):
        sys.modules.pop(name, None)

    real_version = importlib.metadata.version

    def old_strands_version(distribution_name: str) -> str:
        if distribution_name == "strands-agents":
            return "1.45.0"
        return real_version(distribution_name)

    monkeypatch.setattr(importlib.metadata, "version", old_strands_version)
    namespace: dict[str, Any] = {}
    exec(f"from {MODULE} import *", namespace)

    assert namespace["MemoryConverter"] is not None
    assert namespace["OpenAIConverseConverter"] is not None
    assert "AgentCoreMemoryStore" not in namespace
    assert f"{MODULE}.store" not in sys.modules
    assert f"{MODULE}.factory" not in sys.modules
    assert f"{MODULE}.types" not in sys.modules

    module = sys.modules[MODULE]
    assert module.__all__ == ["MemoryConverter", "OpenAIConverseConverter"]
    with pytest.raises(ImportError, match=r"strands-agents>=1\.46\.0.*bedrock-agentcore\[strands-agents\]"):
        _ = module.AgentCoreMemoryStore
    with pytest.raises(ImportError, match=r"strands-agents>=1\.46\.0.*bedrock-agentcore\[strands-agents\]"):
        exec(f"from {MODULE} import AgentCoreMemoryStore", {})


def test_missing_strands_wildcard_import_is_empty_and_new_symbols_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional dependency does not make a wildcard import fail."""
    for name in (MODULE, f"{MODULE}.store", f"{MODULE}.factory", f"{MODULE}.types"):
        sys.modules.pop(name, None)

    real_find_spec = importlib.util.find_spec

    def missing_strands_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "strands" or name.startswith("strands."):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", missing_strands_spec)
    namespace: dict[str, Any] = {}
    exec(f"from {MODULE} import *", namespace)

    module = sys.modules[MODULE]
    assert module.__all__ == []
    assert set(namespace) == {"__builtins__"}
    with pytest.raises(ImportError, match=r"strands-agents>=1\.46\.0.*bedrock-agentcore\[strands-agents\]"):
        _ = module.AgentCoreMemoryStore
