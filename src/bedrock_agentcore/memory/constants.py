"""Constants for Bedrock AgentCore Memory SDK."""

from enum import Enum
from typing import Dict, List


class StrategyType(Enum):
    """Memory strategy types."""

    SEMANTIC = "semanticMemoryStrategy"
    SUMMARY = "summaryMemoryStrategy"
    USER_PREFERENCE = "userPreferenceMemoryStrategy"
    CUSTOM = "customMemoryStrategy"


class MemoryStrategyTypeEnum(Enum):
    """Internal strategy type enum."""

    SEMANTIC = "SEMANTIC"
    SUMMARIZATION = "SUMMARIZATION"
    USER_PREFERENCE = "USER_PREFERENCE"
    CUSTOM = "CUSTOM"


class OverrideType(Enum):
    """Custom strategy override types."""

    SEMANTIC_OVERRIDE = "SEMANTIC_OVERRIDE"
    SUMMARY_OVERRIDE = "SUMMARY_OVERRIDE"
    USER_PREFERENCE_OVERRIDE = "USER_PREFERENCE_OVERRIDE"


class MemoryStatus(Enum):
    """Memory resource statuses."""

    CREATING = "CREATING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    UPDATING = "UPDATING"
    DELETING = "DELETING"


class MemoryStrategyStatus(Enum):
    """Memory strategy statuses (new from API update)."""

    CREATING = "CREATING"
    ACTIVE = "ACTIVE"
    DELETING = "DELETING"
    FAILED = "FAILED"


class Role(Enum):
    """Conversation roles."""

    USER = "USER"
    ASSISTANT = "ASSISTANT"


class MessageRole(Enum):
    """Extended message roles including tool usage."""

    USER = "USER"
    ASSISTANT = "ASSISTANT"
    TOOL = "TOOL"
    OTHER = "OTHER"


# Default namespaces for each strategy type
DEFAULT_NAMESPACES: Dict[StrategyType, List[str]] = {
    StrategyType.SEMANTIC: ["/actor/{actorId}/strategy/{strategyId}/{sessionId}"],
    StrategyType.SUMMARY: ["/actor/{actorId}/strategy/{strategyId}/{sessionId}"],
    StrategyType.USER_PREFERENCE: ["/actor/{actorId}/strategy/{strategyId}"],
}


# Configuration wrapper keys for update operations
# These are still needed for wrapping configurations during updates
EXTRACTION_WRAPPER_KEYS: Dict[MemoryStrategyTypeEnum, str] = {
    MemoryStrategyTypeEnum.SEMANTIC: "semanticExtractionConfiguration",
    MemoryStrategyTypeEnum.USER_PREFERENCE: "userPreferenceExtractionConfiguration",
}

CUSTOM_EXTRACTION_WRAPPER_KEYS: Dict[OverrideType, str] = {
    OverrideType.SEMANTIC_OVERRIDE: "semanticExtractionOverride",
    OverrideType.USER_PREFERENCE_OVERRIDE: "userPreferenceExtractionOverride",
}

CUSTOM_CONSOLIDATION_WRAPPER_KEYS: Dict[OverrideType, str] = {
    OverrideType.SEMANTIC_OVERRIDE: "semanticConsolidationOverride",
    OverrideType.SUMMARY_OVERRIDE: "summaryConsolidationOverride",
    OverrideType.USER_PREFERENCE_OVERRIDE: "userPreferenceConsolidationOverride",
}


# ConfigLimits class - keeping minimal version for any validation needs
class ConfigLimits:
    """Configuration limits (most are deprecated but keeping class for compatibility)."""

    # These specific limits are being deprecated but might still be used in some places
    MIN_TRIGGER_EVERY_N_MESSAGES = 1
    MAX_TRIGGER_EVERY_N_MESSAGES = 16
    MIN_HISTORICAL_CONTEXT_WINDOW = 0
    MAX_HISTORICAL_CONTEXT_WINDOW = 12
