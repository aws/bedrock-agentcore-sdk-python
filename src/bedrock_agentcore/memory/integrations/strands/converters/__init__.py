"""Compatibility aliases for Strands session manager converters."""

import sys
from typing import TYPE_CHECKING

from ..memorysessionmanager import converters as _canonical_module
from ..memorysessionmanager.converters import openai as _openai_module
from ..memorysessionmanager.converters import protocol as _protocol_module

if TYPE_CHECKING:
    from ..memorysessionmanager.converters import MemoryConverter as MemoryConverter
    from ..memorysessionmanager.converters import OpenAIConverseConverter as OpenAIConverseConverter

sys.modules[f"{__name__}.openai"] = _openai_module
sys.modules[f"{__name__}.protocol"] = _protocol_module
sys.modules[__name__] = _canonical_module
