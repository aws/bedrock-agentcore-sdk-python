"""Constants for Bedrock AgentCore Memory SDK."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StrategyType(Enum):
    """Memory strategy types."""

    SEMANTIC = "semanticMemoryStrategy"
    SUMMARY = "summaryMemoryStrategy"
    USER_PREFERENCE = "userPreferenceMemoryStrategy"
    EPISODIC = "episodicMemoryStrategy"
    CUSTOM = "customMemoryStrategy"


class MemoryStrategyTypeEnum(Enum):
    """Internal strategy type enum."""

    SEMANTIC = "SEMANTIC"
    SUMMARIZATION = "SUMMARIZATION"
    USER_PREFERENCE = "USER_PREFERENCE"
    EPISODIC = "EPISODIC"
    CUSTOM = "CUSTOM"


class OverrideType(Enum):
    """Custom strategy override types."""

    SEMANTIC_OVERRIDE = "SEMANTIC_OVERRIDE"
    SUMMARY_OVERRIDE = "SUMMARY_OVERRIDE"
    USER_PREFERENCE_OVERRIDE = "USER_PREFERENCE_OVERRIDE"
    EPISODIC_OVERRIDE = "EPISODIC_OVERRIDE"


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
    StrategyType.SEMANTIC: ["/strategies/{memoryStrategyId}/actors/{actorId}"],
    StrategyType.SUMMARY: ["/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}"],
    StrategyType.USER_PREFERENCE: ["/strategies/{memoryStrategyId}/actors/{actorId}"],
    StrategyType.EPISODIC: ["/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}"],
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
    OverrideType.EPISODIC_OVERRIDE: "episodicExtractionOverride",
}

CUSTOM_CONSOLIDATION_WRAPPER_KEYS: Dict[OverrideType, str] = {
    OverrideType.SEMANTIC_OVERRIDE: "semanticConsolidationOverride",
    OverrideType.SUMMARY_OVERRIDE: "summaryConsolidationOverride",
    OverrideType.USER_PREFERENCE_OVERRIDE: "userPreferenceConsolidationOverride",
    OverrideType.EPISODIC_OVERRIDE: "episodicConsolidationOverride",
}

CUSTOM_REFLECTION_WRAPPER_KEYS: Dict[OverrideType, str] = {
    OverrideType.EPISODIC_OVERRIDE: "episodicReflectionOverride",
}


# ConfigLimits class - keeping minimal version for any validation needs
class ConfigLimits:
    """Configuration limits (most are deprecated but keeping class for compatibility)."""

    # These specific limits are being deprecated but might still be used in some places
    MIN_TRIGGER_EVERY_N_MESSAGES = 1
    MAX_TRIGGER_EVERY_N_MESSAGES = 16
    MIN_HISTORICAL_CONTEXT_WINDOW = 0
    MAX_HISTORICAL_CONTEXT_WINDOW = 12


@dataclass
class ConversationalMessage:
    """Represents a conversational message with text and role.

    Args:
        text: The message content
        role: The role of the message sender (e.g., 'USER', 'ASSISTANT')
    """

    text: str
    role: MessageRole

    def __post_init__(self):
        """Validate message fields after initialization."""
        if not isinstance(self.text, str):
            raise ValueError("ConversationalMessage.text must be a string")
        if not isinstance(self.role, MessageRole):
            raise ValueError("ConversationalMessage.role must be a MessageRole")


@dataclass
class BlobMessage:
    """Represents a blob message containing arbitrary data.

    Args:
        data: Any arbitrary data to be stored as a blob
    """

    data: Any


class RetrievalConfig(BaseModel):
    """Configuration for memory retrieval operations.

    Attributes:
        top_k: Number of top-scoring records to return from semantic search (default: 10)
        relevance_score: Relevance score to filter responses from semantic search (default: 0.0)
        strategy_id: Optional parameter to filter memory strategies (default: None)
        retrieval_query: Optional custom query for semantic search (default: None)
    """

    top_k: int = Field(default=10, gt=1, le=100)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy_id: Optional[str] = None
    retrieval_query: Optional[str] = None
