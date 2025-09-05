"""Configuration for AgentCore Memory Session Manager."""

from typing import Dict, Optional

from pydantic import BaseModel, Field


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
    """

    memory_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    retrieval_config: Optional[Dict[str, RetrievalConfig]] = None
