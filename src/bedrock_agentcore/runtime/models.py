"""Models for BedrockAgentCore runtime.

Contains data models and enums used throughout the runtime system.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PingStatus(str, Enum):
    """Ping status enum for health check responses."""

    HEALTHY = "Healthy"
    HEALTHY_BUSY = "HealthyBusy"


class WarmupType(str, Enum):
    """Build warmup call type, sent by AgentCore Runtime in the ``WARMUP_TYPE_HEADER`` header.

    AgentCore Runtime calls the build warmup hook twice, first with ``memory`` and then with ``disk``. The app
    answers a type it doesn't know with 200 ``skipped``, so a type added later doesn't fail the call.
    See https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-lifecycle-hooks.html
    """

    MEMORY = "memory"
    DISK = "disk"


# Mirrors the public API constraints for HttpPostLifecycleHookAction.
_LIFECYCLE_HOOK_PATH_MAX_LENGTH = 32
_LIFECYCLE_HOOK_PATH_PATTERN = re.compile(r"^/[A-Za-z0-9._~/-]*$")
_LIFECYCLE_HOOK_PORT_MIN = 8000
_LIFECYCLE_HOOK_PORT_MAX = 12000


@dataclass(frozen=True)
class HookConfig:
    """Where the app serves a lifecycle hook, as ``POST`` on ``path``.

    Without ``port`` the hook is served on the serving port. With ``port`` it's served on that port by a
    separate listener on 127.0.0.1 in the same process, which needs ``app.run()``.

    See https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-lifecycle-hooks.html

    Attributes:
        path: Path of the hook route: 1-32 characters from ``A-Z a-z 0-9 . _ ~ / -``, starting with ``/``,
            with no trailing ``/`` (except ``/`` itself), no ``//``, and no ``.`` or ``..`` segments.
        port: Port of the hook route, 8000-12000. ``None`` (the default) serves it on the serving port.

    Raises:
        ValueError: If ``path`` or ``port`` is invalid.
    """

    path: str
    port: Optional[int] = None

    def __post_init__(self):
        """Validate ``path`` and ``port`` against the API constraints."""
        path = self.path
        if not isinstance(path, str):
            raise ValueError(f"HookConfig.path must be a string, got {path!r}")
        if not 1 <= len(path) <= _LIFECYCLE_HOOK_PATH_MAX_LENGTH:
            raise ValueError(
                f"HookConfig.path must be between 1 and {_LIFECYCLE_HOOK_PATH_MAX_LENGTH} characters, got {path!r}"
            )
        if not _LIFECYCLE_HOOK_PATH_PATTERN.fullmatch(path):
            raise ValueError(
                "HookConfig.path must start with '/' and contain only letters, numbers, '.', '_', '~', '/' and '-', "
                f"got {path!r}"
            )
        segments = path[1:].split("/")
        if path != "/" and (path.endswith("/") or "//" in path or any(seg in (".", "..") for seg in segments)):
            raise ValueError(
                "HookConfig.path must be canonical: no trailing '/' (except '/' itself), no '//', "
                f"and no '.' or '..' segments, got {path!r}"
            )
        port = self.port
        if port is not None and (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not _LIFECYCLE_HOOK_PORT_MIN <= port <= _LIFECYCLE_HOOK_PORT_MAX
        ):
            raise ValueError(
                f"HookConfig.port must be an integer between {_LIFECYCLE_HOOK_PORT_MIN} and "
                f"{_LIFECYCLE_HOOK_PORT_MAX}, got {port!r}"
            )


# Header constants
SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
SHELL_ID_HEADER = "X-Amzn-Bedrock-AgentCore-Shell-Id"
REQUEST_ID_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Request-Id"
ACCESS_TOKEN_HEADER = "WorkloadAccessToken"  # nosec
IDENTITY_WAT_HEADER = "X-Amz-Bedrock-AgentCore-Identity-WAT"  # nosec
OAUTH2_CALLBACK_URL_HEADER = "OAuth2CallbackUrl"
AUTHORIZATION_HEADER = "Authorization"
CUSTOM_HEADER_PREFIX = "X-Amzn-Bedrock-AgentCore-Runtime-Custom-"
WARMUP_TYPE_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Warmup-Type"
AGENTCORE_RUNTIME_URL_ENV = "AGENTCORE_RUNTIME_URL"

# Headers that cannot be forwarded to agent code.
# Source: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-header-allowlist.html
RESTRICTED_HEADERS: frozenset[str] = frozenset(
    h.lower()
    for h in [
        # Authentication & Authorization
        "Proxy-Authorization",
        "WWW-Authenticate",
        # Content Negotiation
        "Accept",
        "Accept-Charset",
        "Accept-Encoding",
        "Accept-Language",
        "Content-Type",
        "Content-Length",
        "Content-Encoding",
        "Content-Language",
        "Content-Location",
        "Content-Range",
        # Caching
        "Cache-Control",
        "ETag",
        "Expires",
        "If-Match",
        "If-Modified-Since",
        "If-None-Match",
        "If-Range",
        "If-Unmodified-Since",
        "Last-Modified",
        "Pragma",
        "Vary",
        # Connection Management
        "Connection",
        "Keep-Alive",
        "Proxy-Connection",
        "Upgrade",
        # Request Context
        "Host",
        "User-Agent",
        "Referer",
        "From",
        # Range / Transfer
        "Range",
        "Accept-Ranges",
        "Transfer-Encoding",
        "TE",
        "Trailer",
        # Server Information
        "Server",
        "Date",
        "Location",
        "Retry-After",
        # Cookies
        "Set-Cookie",
        "Cookie",
        # Security
        "Content-Security-Policy",
        "Content-Security-Policy-Report-Only",
        "Strict-Transport-Security",
        "X-Content-Type-Options",
        "X-Frame-Options",
        "X-XSS-Protection",
        "Referrer-Policy",
        "Permissions-Policy",
        "Cross-Origin-Embedder-Policy",
        "Cross-Origin-Opener-Policy",
        "Cross-Origin-Resource-Policy",
        # CORS
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Methods",
        "Access-Control-Allow-Headers",
        "Access-Control-Allow-Credentials",
        "Access-Control-Expose-Headers",
        "Access-Control-Max-Age",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
        "Origin",
        # Client Hints
        "Accept-CH",
        "Accept-CH-Lifetime",
        "DPR",
        "Width",
        "Viewport-Width",
        "Downlink",
        "ECT",
        "RTT",
        "Save-Data",
        # Experimental / Proposed
        "Clear-Site-Data",
        "Feature-Policy",
        "Expect-CT",
        "Public-Key-Pins",
        "Public-Key-Pins-Report-Only",
        # Proxy
        "Via",
        "Forwarded",
        "X-Forwarded-For",
        "X-Forwarded-Host",
        "X-Forwarded-Proto",
        "X-Real-IP",
        "X-Requested-With",
        "X-CSRF-Token",
        # IP Spoofing / URL Manipulation
        "True-Client-IP",
        "X-Client-IP",
        "X-Cluster-Client-IP",
        "X-Originating-IP",
        "X-Source-IP",
        "X-Original-URL",
        "X-Original-Host",
        "X-Rewrite-URL",
        # CDN / Proxy
        "CF-Ray",
        "CF-Connecting-IP",
        "X-Amz-Cf-Id",
        "X-Cache",
        "X-Served-By",
        # HTTP/2 Pseudo Headers
        ":method",
        ":path",
        ":scheme",
        ":authority",
        ":status",
        # Server Push
        "Link",
        # WebSocket
        "Sec-WebSocket-Key",
        "Sec-WebSocket-Accept",
        "Sec-WebSocket-Version",
        "Sec-WebSocket-Protocol",
        "Sec-WebSocket-Extensions",
    ]
)

# Baggage keys for routing experiment span attributes
BAGGAGE_KEY_EXPERIMENT_ARN = "aws.agentcore.gateway.routing_experiment_arn"
BAGGAGE_KEY_EXPERIMENT_VARIANT = "aws.agentcore.gateway.routing_experiment_variant_name"

# Task action constants
TASK_ACTION_PING_STATUS = "ping_status"
TASK_ACTION_JOB_STATUS = "job_status"
TASK_ACTION_FORCE_HEALTHY = "force_healthy"
TASK_ACTION_FORCE_BUSY = "force_busy"
TASK_ACTION_CLEAR_FORCED_STATUS = "clear_forced_status"


_CUSTOM_HEADER_PREFIX_LOWER = CUSTOM_HEADER_PREFIX.lower()
_AUTHORIZATION_HEADER_LOWER = AUTHORIZATION_HEADER.lower()
_IDENTITY_WAT_HEADER_LOWER = IDENTITY_WAT_HEADER.lower()


def is_forwardable_header(header_name: str) -> bool:
    """Return True if the header may be forwarded to agent code.

    Rules (from the AgentCore runtime header allowlist docs):
    - Not in the restricted headers list
    - Does not start with ``x-amz-`` (reserved for AWS SigV4 signing)
      unless it is the Identity WAT header used for workload identity chain propagation
    - Does not start with ``x-amzn-`` unless it starts with the legacy
      ``X-Amzn-Bedrock-AgentCore-Runtime-Custom-`` prefix
    """
    lower = header_name.lower()
    if lower in RESTRICTED_HEADERS:
        return False
    if lower.startswith("x-amz-"):
        # Allow the Identity WAT header through for workload identity chain propagation
        if lower == _IDENTITY_WAT_HEADER_LOWER:
            return True
        return False
    if lower.startswith("x-amzn-") and not lower.startswith(_CUSTOM_HEADER_PREFIX_LOWER):
        return False
    return True
