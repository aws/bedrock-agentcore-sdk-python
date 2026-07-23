"""Compatibility alias for the session manager converter protocol."""

import sys
from typing import TYPE_CHECKING

from ..memorysessionmanager.converters import protocol as _canonical_module

if TYPE_CHECKING:
    from ..memorysessionmanager.converters.protocol import CONVERSATIONAL_MAX_SIZE as CONVERSATIONAL_MAX_SIZE
    from ..memorysessionmanager.converters.protocol import MemoryConverter as MemoryConverter
    from ..memorysessionmanager.converters.protocol import exceeds_conversational_limit as exceeds_conversational_limit

sys.modules[__name__] = _canonical_module
