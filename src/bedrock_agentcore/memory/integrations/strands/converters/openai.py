"""Compatibility alias for the OpenAI Converse session manager converter."""

import sys
from typing import TYPE_CHECKING

from ..memorysessionmanager.converters import openai as _canonical_module

if TYPE_CHECKING:
    from ..memorysessionmanager.converters.openai import OpenAIConverseConverter as OpenAIConverseConverter

sys.modules[__name__] = _canonical_module
