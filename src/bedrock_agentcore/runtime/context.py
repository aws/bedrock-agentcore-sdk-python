"""Request context models for Bedrock AgentCore Server.

Contains metadata extracted from HTTP requests that handlers can optionally access.
"""

from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel, Field


class RequestContext(BaseModel):
    """Request context containing metadata from HTTP requests."""

    session_id: str | None = Field(None)
    request_headers: dict[str, str] | None = Field(None)
    request: Any | None = Field(None, description="The underlying Starlette request object")

    class Config:
        """Allow non-serializable types like Starlette Request."""

        arbitrary_types_allowed = True


class BedrockAgentCoreContext:
    """Unified context manager for Bedrock AgentCore."""

    _workload_access_token: ContextVar[str | None] = ContextVar("workload_access_token")
    _oauth2_callback_url: ContextVar[str | None] = ContextVar("oauth2_callback_url")
    _request_id: ContextVar[str | None] = ContextVar("request_id")
    _session_id: ContextVar[str | None] = ContextVar("session_id")
    _request_headers: ContextVar[dict[str, str] | None] = ContextVar("request_headers")

    @classmethod
    def set_workload_access_token(cls, token: str):
        """Set the workload access token in the context."""
        cls._workload_access_token.set(token)

    @classmethod
    def get_workload_access_token(cls) -> str | None:
        """Get the workload access token from the context."""
        try:
            return cls._workload_access_token.get()
        except LookupError:
            return None

    @classmethod
    def set_oauth2_callback_url(cls, workload_callback_url: str):
        """Set the oauth2 callback url in the context."""
        cls._oauth2_callback_url.set(workload_callback_url)

    @classmethod
    def get_oauth2_callback_url(cls) -> str | None:
        """Get the oauth2 callback url from the context."""
        try:
            return cls._oauth2_callback_url.get()
        except LookupError:
            return None

    @classmethod
    def set_request_context(cls, request_id: str, session_id: str | None = None):
        """Set request-scoped identifiers."""
        cls._request_id.set(request_id)
        cls._session_id.set(session_id)

    @classmethod
    def get_request_id(cls) -> str | None:
        """Get current request ID."""
        try:
            return cls._request_id.get()
        except LookupError:
            return None

    @classmethod
    def get_session_id(cls) -> str | None:
        """Get current session ID."""
        try:
            return cls._session_id.get()
        except LookupError:
            return None

    @classmethod
    def set_request_headers(cls, headers: dict[str, str]):
        """Set request headers in the context."""
        cls._request_headers.set(headers)

    @classmethod
    def get_request_headers(cls) -> dict[str, str] | None:
        """Get request headers from the context."""
        try:
            return cls._request_headers.get()
        except LookupError:
            return None
