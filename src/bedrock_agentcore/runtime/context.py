"""Request context models for Bedrock AgentCore Server.

Contains metadata extracted from HTTP requests that handlers can optionally access.
"""

import re
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any, ClassVar, Dict, List, Mapping, Optional

from pydantic import BaseModel, Field


class RequestContext(BaseModel):
    """Immutable request context containing metadata from HTTP requests.

    This represents what the client sent and should never be modified.
    Uses MappingProxyType via @property for truly immutable headers.

    Attributes:
        session_id: Session identifier from X-Amzn-Bedrock-AgentCore-Runtime-Session-Id header
        request_headers: Immutable headers sent by client (exposed as MappingProxyType)
    """

    session_id: Optional[str] = Field(None, description="Session identifier")
    request_headers_dict: Optional[Dict[str, str]] = Field(
        None, alias="request_headers", description="Internal headers storage"
    )

    class Config:
        """Pydantic model configuration for immutability."""

        frozen = True
        arbitrary_types_allowed = True

    @property
    def request_headers(self) -> Optional[Mapping[str, str]]:
        """Return headers as immutable MappingProxyType.

        Returns:
            MappingProxyType wrapping headers dict, or None if no headers
        """
        if self.request_headers_dict is None:
            return None
        return MappingProxyType(self.request_headers_dict)


class ProcessingContext(BaseModel):
    """Mutable processing context with namespace isolation.

    This is where middleware can inject data that handlers need to see.
    Uses namespaces to prevent accidental collisions and make malicious overwrites obvious.

    Attributes:
        middleware_data: Namespaced data injected by middleware layers
        processing_metadata: Metadata about request processing (timestamps, traces, etc)
    """

    middleware_data: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Namespaced middleware-injected data"
    )
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")

    # ClassVar for Pydantic v2 compatibility
    DENY_PATTERNS: ClassVar[List[str]] = [
        r"\x00",
        r"[\r\n].*?:",
        r"<script[>\s]",
        r"javascript:",
        r"\$\{",
    ]

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        max_size: int = 100_000,
        validate: bool = False,
    ) -> None:
        """Set middleware data in a specific namespace with optional validation.

        Args:
            namespace: Namespace to isolate this data (e.g., 'auth', 'metrics', 'timing')
            key: Data key within the namespace
            value: Data value (must be JSON-serializable)
            max_size: Maximum size for string values (default 100KB)
            validate: Whether to perform security validation (default False for performance)

        Raises:
            ValueError: If validation is enabled and value fails checks

        Example:
            context.processing.set('auth', 'user_id', 'user123')
        """
        if validate and isinstance(value, str):
            if len(value) > max_size:
                raise ValueError(f"Value too large: {len(value)} bytes > {max_size} bytes")

            for pattern in self.DENY_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    raise ValueError("Potentially unsafe pattern detected in value")

        if namespace not in self.middleware_data:
            self.middleware_data[namespace] = {}

        self.middleware_data[namespace][key] = value

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """Get middleware data from a specific namespace.

        Args:
            namespace: Namespace to read from
            key: Data key within the namespace
            default: Default value if key not found

        Returns:
            Value associated with key in namespace, or default

        Example:
            user_id = context.processing.get('auth', 'user_id')
        """
        return self.middleware_data.get(namespace, {}).get(key, default)

    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        """Get all data from a specific namespace.

        Args:
            namespace: Namespace to retrieve

        Returns:
            Dictionary of all key-value pairs in the namespace

        Example:
            auth_data = context.processing.get_namespace('auth')
        """
        return self.middleware_data.get(namespace, {}).copy()

    def has_namespace(self, namespace: str) -> bool:
        """Check if a namespace exists.

        Args:
            namespace: Namespace to check

        Returns:
            True if namespace exists, False otherwise
        """
        return namespace in self.middleware_data

    def add_metadata(self, key: str, value: Any) -> None:
        """Add processing metadata for observability.

        Args:
            key: Metadata key
            value: Metadata value

        Example:
            context.processing.add_metadata('handler_start_time', time.time())
        """
        self.processing_metadata[key] = value


class AgentContext(BaseModel):
    """Combined context that handlers receive.

    Provides access to both immutable request data and mutable processing data.

    Attributes:
        request: Immutable request data (what client sent)
        processing: Mutable processing data (what middleware added, namespaced)
    """

    request: RequestContext
    processing: ProcessingContext

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True

    @property
    def session_id(self) -> Optional[str]:
        """Session ID (backward compatibility property).

        Returns:
            Session ID from request context
        """
        return self.request.session_id

    @property
    def request_headers(self) -> Optional[Mapping[str, str]]:
        """Request headers (backward compatibility property).

        Returns:
            Immutable request headers as MappingProxyType
        """
        return self.request.request_headers


class BedrockAgentCoreContext:
    """Unified context manager for Bedrock AgentCore.

    Uses Python's contextvars for thread-safe, request-scoped storage.
    """

    _workload_access_token: ContextVar[Optional[str]] = ContextVar("workload_access_token", default=None)
    _oauth2_callback_url: ContextVar[Optional[str]] = ContextVar("oauth2_callback_url", default=None)
    _request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
    _session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
    _request_headers: ContextVar[Optional[Dict[str, str]]] = ContextVar("request_headers", default=None)
    _processing_context: ContextVar[Optional[ProcessingContext]] = ContextVar("processing_context", default=None)

    @classmethod
    def set_workload_access_token(cls, token: str):
        """Set the workload access token in the context.

        Args:
            token: Workload access token
        """
        cls._workload_access_token.set(token)

    @classmethod
    def get_workload_access_token(cls) -> Optional[str]:
        """Get the workload access token from the context.

        Returns:
            Workload access token or None if not set
        """
        try:
            return cls._workload_access_token.get()
        except LookupError:
            return None

    @classmethod
    def set_oauth2_callback_url(cls, workload_callback_url: str):
        """Set the OAuth2 callback URL in the context.

        Args:
            workload_callback_url: OAuth2 callback URL
        """
        cls._oauth2_callback_url.set(workload_callback_url)

    @classmethod
    def get_oauth2_callback_url(cls) -> Optional[str]:
        """Get the OAuth2 callback URL from the context.

        Returns:
            OAuth2 callback URL or None if not set
        """
        try:
            return cls._oauth2_callback_url.get()
        except LookupError:
            return None

    @classmethod
    def set_request_context(cls, request_id: str, session_id: Optional[str] = None):
        """Set request-scoped identifiers.

        Args:
            request_id: Unique request identifier
            session_id: Optional session identifier
        """
        cls._request_id.set(request_id)
        cls._session_id.set(session_id)

    @classmethod
    def get_request_id(cls) -> Optional[str]:
        """Get current request ID.

        Returns:
            Request ID or None if not set
        """
        try:
            return cls._request_id.get()
        except LookupError:
            return None

    @classmethod
    def get_session_id(cls) -> Optional[str]:
        """Get current session ID.

        Returns:
            Session ID or None if not set
        """
        try:
            return cls._session_id.get()
        except LookupError:
            return None

    @classmethod
    def set_request_headers(cls, headers: Dict[str, str]):
        """Set request headers in the context.

        Args:
            headers: Dictionary of request headers
        """
        cls._request_headers.set(headers)

    @classmethod
    def get_request_headers(cls) -> Optional[Dict[str, str]]:
        """Get request headers from the context.

        Returns:
            Request headers dictionary or None if not set
        """
        try:
            return cls._request_headers.get()
        except LookupError:
            return None

    @classmethod
    def set_processing_context(cls, context: ProcessingContext):
        """Set processing context for current request.

        Args:
            context: ProcessingContext instance with middleware data
        """
        cls._processing_context.set(context)

    @classmethod
    def get_processing_context(cls) -> Optional[ProcessingContext]:
        """Get processing context for current request.

        Returns:
            ProcessingContext instance or None if not set
        """
        try:
            return cls._processing_context.get()
        except LookupError:
            return None


class StandardNamespaces:
    """Standard namespaces for common middleware patterns.

    Using these constants helps prevent typos and ensures consistency.
    """

    AUTH = "auth"
    TIMING = "timing"
    AUDIT = "audit"
    METRICS = "metrics"
    OBSERVABILITY = "observability"
    FEATURE_FLAGS = "features"
    RATE_LIMIT = "rate_limit"
    CUSTOM = "custom"
