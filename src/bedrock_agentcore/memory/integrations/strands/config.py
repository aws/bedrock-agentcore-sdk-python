"""Configuration for AgentCore Memory Session Manager."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from bedrock_agentcore.memory.constants import MessageRole
from bedrock_agentcore.memory.models import StringValue


class BranchConfig(BaseModel):
    """Configuration for AgentCore Memory branching.

    Attributes:
        name: Descriptive name for the branch
        root_event_id: ID of the event from which this branch originates
    """

    name: str = Field(min_length=1)
    root_event_id: Optional[str] = ""

    def to_agentcore_format(self) -> dict:
        """Convert to AgentCore Memory API format."""
        return {"name": self.name, "rootEventId": self.root_event_id}


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


class ShortTermRetrievalConfig(BaseModel):
    """Configuration for Short term memory retrieval operations"""

    branch_filter: Optional[bool] = True
    metadata: Optional[Dict[str, StringValue]] = None


class AgentCoreMemoryConfig(BaseModel):
    """Configuration for AgentCore Memory Session Manager.

    Attributes:
        memory_id: Required Bedrock AgentCore Memory ID
        session_id: Required unique ID for the session
        actor_id: Required unique ID for the agent instance/user
        retrieval_config: Optional dictionary mapping namespaces to retrieval configurations
        default_branch: Optional default branch configuration for the session
        message_types: Optional list of message types to filter
        metadata: Optional dictionary of metadata to include with events
    """

    memory_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    retrieval_config: Optional[Dict[str, RetrievalConfig]] = None
    default_branch: Optional[BranchConfig] = Field(
        default_factory=lambda: BranchConfig(name="main", root_event_id="")
    )
    short_term_retrieval_config: Optional[ShortTermRetrievalConfig] = (
        ShortTermRetrievalConfig()
    )
    message_types: Optional[List[str]] = Field(default=["user", "assistant"])
    metadata: Optional[Dict[str, StringValue]] = (
        None  # Currently only supports agent_id. Will be extended further.
    )

    @field_validator("memory_id", "session_id", "actor_id")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v
