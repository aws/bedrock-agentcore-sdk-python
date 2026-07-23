"""Compatibility alias for the Strands AgentCore Memory session manager configuration."""

import sys
from typing import TYPE_CHECKING

from .memorysessionmanager import config as _canonical_module

if TYPE_CHECKING:
    from .memorysessionmanager.config import AgentCoreMemoryConfig as AgentCoreMemoryConfig
    from .memorysessionmanager.config import PersistenceMode as PersistenceMode
    from .memorysessionmanager.config import RetrievalConfig as RetrievalConfig
    from .memorysessionmanager.config import normalize_metadata as normalize_metadata

sys.modules[__name__] = _canonical_module
