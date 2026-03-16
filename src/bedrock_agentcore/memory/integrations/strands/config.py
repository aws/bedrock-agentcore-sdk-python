"""Configuration for AgentCore Memory Session Manager."""

from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field, field_validator


def normalize_metadata(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize metadata values: plain strings become {"stringValue": value}."""
    return {k: {"stringValue": v} if isinstance(v, str) else v for k, v in raw.items()}


class RetrievalConfig(BaseModel):
    """Configuration for memory retrieval operations.

    Attributes:
        top_k: Number of top-scoring records to return from semantic search (default: 10)
        relevance_score: Relevance score to filter responses from semantic search (default: 0.2)
        strategy_id: Optional parameter to filter memory strategies (default: None)
        initialization_query: Optional custom query for initialization retrieval (default: None)
    """

    top_k: int = Field(default=10, gt=0, le=1000)
    relevance_score: float = Field(default=0.2, ge=0.0, le=1.0)
    strategy_id: Optional[str] = None
    initialization_query: Optional[str] = None


class AgentCoreMemoryConfig(BaseModel):
    """Configuration for AgentCore Memory Session Manager.

    Attributes:
        memory_id: Required Bedrock AgentCore Memory ID
        session_id: Required unique ID for the session
        actor_id: Required unique ID for the agent instance/user
        retrieval_config: Optional dictionary mapping namespaces to retrieval configurations
        batch_size: Number of messages to batch before sending to AgentCore Memory.
            Default of 1 means immediate sending (no batching). Max 100.
        flush_interval_seconds: Optional interval in seconds for automatic buffer flushing.
            Useful for long-running agents to ensure messages are persisted regularly.
            Default is None (disabled).
        context_tag: XML tag name used to wrap retrieved memory context injected into messages.
            Default is "user_context".
        filter_restored_tool_context: When True, strip historical toolUse/toolResult blocks from
            restored messages before loading them into Strands runtime memory. Default is False.
        default_metadata: Optional default metadata key-value pairs to attach to every message event.
            Merged with any per-call metadata. Maximum 15 total keys per event (including internal keys).
            Accepts plain strings (auto-wrapped) or explicit MetadataValue dicts.
            Example: {"location": "NYC"} or {"location": {"stringValue": "NYC"}}
        metadata_provider: Optional callable that returns metadata key-value pairs. Called at each
            event creation, so it can return dynamic values (e.g. current traceId). The returned
            dict is merged after default_metadata but before per-call metadata.
            Accepts plain strings (auto-wrapped) or explicit MetadataValue dicts.
    """

    memory_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    retrieval_config: Optional[Dict[str, RetrievalConfig]] = None
    batch_size: int = Field(default=1, ge=1, le=100)
    flush_interval_seconds: Optional[float] = Field(default=None, gt=0)
    context_tag: str = Field(default="user_context", min_length=1)
    filter_restored_tool_context: bool = Field(default=False)
    default_metadata: Optional[Dict[str, Any]] = None
    metadata_provider: Optional[Callable[[], Dict[str, Any]]] = None

    @field_validator("default_metadata", mode="before")
    @classmethod
    def _normalize_default_metadata(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        return normalize_metadata(v)
