"""Request context models for Bedrock AgentCore Server.

Contains metadata extracted from HTTP requests that handlers can optionally access.
"""

from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel, Field


class RequestContext(BaseModel):
    """Request context containing metadata from HTTP requests."""

    session_id: Optional[str] = Field(None)


class BedrockAgentCoreContext:
    """Context manager for Bedrock AgentCore."""

    _workload_access_token: ContextVar[str] = ContextVar("workload_access_token")

    @classmethod
    def set_workload_access_token(cls, token: str):
        """Set the workload access token in the context."""
        cls._workload_access_token.set(token)

    @classmethod
    def get_workload_access_token(cls) -> Optional[str]:
        """Get the workload access token from the context."""
        try:
            return cls._workload_access_token.get()
        except LookupError:
            return None
