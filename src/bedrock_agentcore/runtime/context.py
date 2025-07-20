"""Request context models for Bedrock AgentCore Server.

Contains metadata extracted from HTTP requests that handlers can optionally access.
"""

import time
from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel, Field


class RequestContext(BaseModel):
    """Request context containing metadata from HTTP requests."""

    session_id: Optional[str] = Field(None)


class BedrockAgentCoreContext:
    """Context manager for Bedrock AgentCore."""

    _workload_access_token: ContextVar[str] = ContextVar("workload_access_token")
    _token_expiry: ContextVar[float] = ContextVar("token_expiry")
    _token_id: ContextVar[str] = ContextVar("token_id")

    @classmethod
    def set_workload_access_token(cls, token: str, expiry_seconds: int = 3600, token_id: Optional[str] = None):
        """Set the workload access token in the context with expiration."""
        if not token or not token.strip():
            raise ValueError("Token cannot be empty")
        
        cls._workload_access_token.set(token)
        cls._token_expiry.set(time.time() + expiry_seconds)
        if token_id:
            cls._token_id.set(token_id)

    @classmethod
    def get_workload_access_token(cls) -> Optional[str]:
        """Get the workload access token from the context if not expired."""
        try:
            token = cls._workload_access_token.get()
            expiry = cls._token_expiry.get()
            
            # Check if token is expired
            if time.time() > expiry:
                cls.clear_workload_access_token()
                return None
                
            return token
        except LookupError:
            return None
    
    @classmethod
    def clear_workload_access_token(cls):
        """Clear the workload access token from context."""
        try:
            cls._workload_access_token.set("")
            cls._token_expiry.set(0)
            cls._token_id.set("")
        except LookupError:
            pass
    
    @classmethod
    def is_token_expired(cls) -> bool:
        """Check if the current token is expired."""
        try:
            expiry = cls._token_expiry.get()
            return time.time() > expiry
        except LookupError:
            return True