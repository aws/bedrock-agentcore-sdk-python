"""Compatibility alias for the Strands AgentCore Memory converter."""

import sys
from typing import TYPE_CHECKING

from .memorysessionmanager import bedrock_converter as _canonical_module

if TYPE_CHECKING:
    from .memorysessionmanager.bedrock_converter import CONVERSATIONAL_MAX_SIZE as CONVERSATIONAL_MAX_SIZE
    from .memorysessionmanager.bedrock_converter import AgentCoreMemoryConverter as AgentCoreMemoryConverter

sys.modules[__name__] = _canonical_module
