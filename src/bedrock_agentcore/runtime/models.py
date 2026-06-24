"""Models for BedrockAgentCore runtime.

Contains data models and enums used throughout the runtime system.
"""

from enum import Enum


class PingStatus(str, Enum):
    """Ping status enum for health check responses."""

    HEALTHY = "Healthy"
    HEALTHY_BUSY = "HealthyBusy"


# Header constants
SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
SHELL_ID_HEADER = "X-Amzn-Bedrock-AgentCore-Shell-Id"
REQUEST_ID_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Request-Id"
ACCESS_TOKEN_HEADER = "WorkloadAccessToken"  # nosec
OAUTH2_CALLBACK_URL_HEADER = "OAuth2CallbackUrl"
AUTHORIZATION_HEADER = "Authorization"
CUSTOM_HEADER_PREFIX = "X-Amzn-Bedrock-AgentCore-Runtime-Custom-"
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


def is_forwardable_header(header_name: str) -> bool:
    """Return True if the header may be forwarded to agent code.

    Rules (from the AgentCore runtime header allowlist docs):
    - Not in the restricted headers list
    - Does not start with ``x-amz-`` (reserved for AWS SigV4 signing)
    - Does not start with ``x-amzn-`` unless it starts with the legacy
      ``X-Amzn-Bedrock-AgentCore-Runtime-Custom-`` prefix
    """
    lower = header_name.lower()
    if lower in RESTRICTED_HEADERS:
        return False
    if lower.startswith("x-amz-"):
        return False
    if lower.startswith("x-amzn-") and not lower.startswith(_CUSTOM_HEADER_PREFIX_LOWER):
        return False
    return True
