"""Compatibility alias for the Strands AgentCore Memory session manager."""

import sys
from typing import TYPE_CHECKING

from .memorysessionmanager import session_manager as _canonical_module

if TYPE_CHECKING:
    from .memorysessionmanager.session_manager import (
        AGENT_ID_KEY as AGENT_ID_KEY,
    )
    from .memorysessionmanager.session_manager import (
        LEGACY_AGENT_PREFIX as LEGACY_AGENT_PREFIX,
    )
    from .memorysessionmanager.session_manager import (
        LEGACY_SESSION_PREFIX as LEGACY_SESSION_PREFIX,
    )
    from .memorysessionmanager.session_manager import (
        MAX_FETCH_ALL_RESULTS as MAX_FETCH_ALL_RESULTS,
    )
    from .memorysessionmanager.session_manager import (
        MAX_METADATA_KEYS as MAX_METADATA_KEYS,
    )
    from .memorysessionmanager.session_manager import (
        RESERVED_METADATA_KEYS as RESERVED_METADATA_KEYS,
    )
    from .memorysessionmanager.session_manager import (
        STATE_TYPE_KEY as STATE_TYPE_KEY,
    )
    from .memorysessionmanager.session_manager import (
        AgentCoreMemorySessionManager as AgentCoreMemorySessionManager,
    )
    from .memorysessionmanager.session_manager import BufferedMessage as BufferedMessage
    from .memorysessionmanager.session_manager import StateType as StateType

sys.modules[__name__] = _canonical_module
