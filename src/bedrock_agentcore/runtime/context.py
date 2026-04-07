"""Request context models for Bedrock AgentCore Server.

Contains metadata extracted from HTTP requests that handlers can optionally access.
"""

import logging
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field

from ..config_bundle.bundle import ConfigBundleRef

logger = logging.getLogger(__name__)



class RequestContext(BaseModel):
    """Request context containing metadata from HTTP requests."""

    session_id: Optional[str] = Field(None)
    request_headers: Optional[Dict[str, str]] = Field(None)
    request: Optional[Any] = Field(None, description="The underlying Starlette request object")

    class Config:
        """Allow non-serializable types like Starlette Request."""

        arbitrary_types_allowed = True


class BedrockAgentCoreContext:
    """Unified context manager for Bedrock AgentCore."""

    _workload_access_token: ContextVar[Optional[str]] = ContextVar("workload_access_token")
    _oauth2_callback_url:   ContextVar[Optional[str]] = ContextVar("oauth2_callback_url")
    _request_id:            ContextVar[Optional[str]] = ContextVar("request_id")
    _session_id:            ContextVar[Optional[str]] = ContextVar("session_id")
    _request_headers:       ContextVar[Optional[Dict[str, str]]] = ContextVar("request_headers")

    # Config bundle — ref identifies the bundle for this request.
    # _bundle_fetcher is the lru_cache-wrapped app._resolve_bundle_config(ref),
    # set per-request by the app. Calling it fetches from the API on first use
    # for a given bundle version, then returns the cached result on subsequent calls.
    _config_bundle_ref: ContextVar[Optional[ConfigBundleRef]] = ContextVar("config_bundle_ref", default=None)
    _bundle_fetcher: ContextVar[Optional[Callable[[], Dict[str, Any]]]] = ContextVar(
        "bundle_fetcher", default=None
    )

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

    @classmethod
    def set_oauth2_callback_url(cls, workload_callback_url: str):
        """Set the oauth2 callback url in the context."""
        cls._oauth2_callback_url.set(workload_callback_url)

    @classmethod
    def get_oauth2_callback_url(cls) -> Optional[str]:
        """Get the oauth2 callback url from the context."""
        try:
            return cls._oauth2_callback_url.get()
        except LookupError:
            return None

    @classmethod
    def set_request_context(cls, request_id: str, session_id: Optional[str] = None):
        """Set request-scoped identifiers."""
        cls._request_id.set(request_id)
        cls._session_id.set(session_id)

    @classmethod
    def get_request_id(cls) -> Optional[str]:
        """Get current request ID."""
        try:
            return cls._request_id.get()
        except LookupError:
            return None

    @classmethod
    def get_session_id(cls) -> Optional[str]:
        """Get current session ID."""
        try:
            return cls._session_id.get()
        except LookupError:
            return None

    @classmethod
    def set_request_headers(cls, headers: Dict[str, str]):
        """Set request headers in the context."""
        cls._request_headers.set(headers)

    @classmethod
    def get_request_headers(cls) -> Optional[Dict[str, str]]:
        """Get request headers from the context."""
        try:
            return cls._request_headers.get()
        except LookupError:
            return None

    @classmethod
    def set_config_bundle_ref(cls, ref: Optional[ConfigBundleRef]) -> None:
        """Set the configuration bundle reference for the current request."""
        cls._config_bundle_ref.set(ref)

    @classmethod
    def get_config_bundle_ref(cls) -> Optional[ConfigBundleRef]:
        """Get the configuration bundle reference for the current request."""
        return cls._config_bundle_ref.get()

    @classmethod
    def _set_bundle_loader(cls, fetcher: Callable[[], Dict[str, Any]]) -> None:
        """Register the config fetcher for this request. Called by the app.

        The fetcher is lru_cache-wrapped app._resolve_bundle_config(ref), so the
        underlying API call is made at most once per unique bundle version across
        all requests on this app instance.
        """
        cls._bundle_fetcher.set(fetcher)

    @classmethod
    def _clear_bundle_loader(cls) -> None:
        """Clear the config fetcher. Called by the app when no bundle ref is present."""
        cls._bundle_fetcher.set(None)

    @classmethod
    def get_bundle_config(cls) -> Dict[str, Any]:
        """Return this runtime's config from the current request's bundle.

        Fetches from the API on the first call for a given bundle version, then
        serves from the per-app-instance LRU cache on all subsequent calls.
        Returns {} if no bundle ref is present in the request baggage.

        Raises:
            Exception: Propagated from the underlying API call if the config
                bundle service is unavailable. Callers that require graceful
                degradation should catch and fall back to their own defaults.
        """
        fetcher = cls._bundle_fetcher.get()
        if fetcher is None:
            return {}
        return fetcher()
