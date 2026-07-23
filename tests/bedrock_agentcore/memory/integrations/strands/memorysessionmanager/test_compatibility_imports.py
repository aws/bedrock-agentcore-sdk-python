"""Compatibility tests for pre-existing SessionManager import paths."""

import importlib
import subprocess
import sys
from unittest.mock import Mock, patch

from bedrock_agentcore.memory.integrations.strands import MemoryConverter, OpenAIConverseConverter
from bedrock_agentcore.memory.integrations.strands.bedrock_converter import AgentCoreMemoryConverter
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, PersistenceMode, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.converters import (
    MemoryConverter as CompatibilityMemoryConverter,
)
from bedrock_agentcore.memory.integrations.strands.converters import (
    OpenAIConverseConverter as CompatibilityOpenAIConverseConverter,
)
from bedrock_agentcore.memory.integrations.strands.memorysessionmanager import (
    AgentCoreMemoryConfig as CanonicalAgentCoreMemoryConfig,
)
from bedrock_agentcore.memory.integrations.strands.memorysessionmanager import (
    AgentCoreMemoryConverter as CanonicalAgentCoreMemoryConverter,
)
from bedrock_agentcore.memory.integrations.strands.memorysessionmanager import (
    AgentCoreMemorySessionManager as CanonicalAgentCoreMemorySessionManager,
)
from bedrock_agentcore.memory.integrations.strands.memorysessionmanager import (
    MemoryConverter as CanonicalMemoryConverter,
)
from bedrock_agentcore.memory.integrations.strands.memorysessionmanager import (
    OpenAIConverseConverter as CanonicalOpenAIConverseConverter,
)
from bedrock_agentcore.memory.integrations.strands.memorysessionmanager import (
    PersistenceMode as CanonicalPersistenceMode,
)
from bedrock_agentcore.memory.integrations.strands.memorysessionmanager import (
    RetrievalConfig as CanonicalRetrievalConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

OLD_MODULE_ROOT = "bedrock_agentcore.memory.integrations.strands"
CANONICAL_MODULE_ROOT = f"{OLD_MODULE_ROOT}.memorysessionmanager"


def test_preexisting_import_paths_are_explicit_aliases() -> None:
    """Keep existing imports working without warning or duplicate implementations."""
    assert AgentCoreMemorySessionManager is CanonicalAgentCoreMemorySessionManager
    assert AgentCoreMemoryConfig is CanonicalAgentCoreMemoryConfig
    assert AgentCoreMemoryConverter is CanonicalAgentCoreMemoryConverter
    assert PersistenceMode is CanonicalPersistenceMode
    assert RetrievalConfig is CanonicalRetrievalConfig
    assert MemoryConverter is CompatibilityMemoryConverter is CanonicalMemoryConverter
    assert OpenAIConverseConverter is CompatibilityOpenAIConverseConverter is CanonicalOpenAIConverseConverter


MODULE_SUFFIXES = (
    "session_manager",
    "bedrock_converter",
    "config",
    "converters",
    "converters.protocol",
    "converters.openai",
)


def test_preexisting_module_paths_are_canonical_module_objects() -> None:
    """Compatibility paths must share globals with their canonical modules."""
    for suffix in MODULE_SUFFIXES:
        old_module = importlib.import_module(f"{OLD_MODULE_ROOT}.{suffix}")
        canonical_module = importlib.import_module(f"{CANONICAL_MODULE_ROOT}.{suffix}")
        assert old_module is canonical_module


def test_preexisting_module_paths_are_canonical_when_imported_first() -> None:
    """Old paths must alias canonical modules even before an explicit canonical import."""
    script = f"""
import importlib

old_root = {OLD_MODULE_ROOT!r}
canonical_root = {CANONICAL_MODULE_ROOT!r}
suffixes = {MODULE_SUFFIXES!r}
for suffix in suffixes:
    old_module = importlib.import_module(f"{{old_root}}.{{suffix}}")
    canonical_module = importlib.import_module(f"{{canonical_root}}.{{suffix}}")
    assert old_module is canonical_module, suffix
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_patching_preexisting_memory_client_path_affects_session_manager_runtime() -> None:
    """Patching MemoryClient through the old path must affect canonical construction."""
    config = AgentCoreMemoryConfig(memory_id="memory-id", session_id="session-id", actor_id="actor-id")
    memory_client = Mock()
    boto_session = Mock(region_name="us-west-2")
    boto_session.client.return_value = Mock()

    with (
        patch(f"{OLD_MODULE_ROOT}.session_manager.MemoryClient", return_value=memory_client) as memory_client_class,
        patch("boto3.Session", return_value=boto_session),
        patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
    ):
        manager = CanonicalAgentCoreMemorySessionManager(config)

    assert manager.memory_client is memory_client
    memory_client_class.assert_called_once_with(region_name=None)


def test_patching_preexisting_bedrock_converter_logger_affects_canonical_global() -> None:
    """Patching logger through the old path must affect canonical converter behavior."""
    canonical_module = importlib.import_module(f"{CANONICAL_MODULE_ROOT}.bedrock_converter")

    with patch(f"{OLD_MODULE_ROOT}.bedrock_converter.logger") as old_path_logger:
        assert canonical_module.logger is old_path_logger
        CanonicalAgentCoreMemoryConverter.events_to_messages([{"payload": [{"blob": "invalid json"}]}])

    old_path_logger.error.assert_called_once_with("Failed to parse blob content: %s", {"blob": "invalid json"})
