"""Client for AgentCore Web Search.

Web Search is reachable today as an AgentCore Gateway connector target, which the
agent calls as an MCP tool. This module wraps that so callers get a plain
``search()`` method and a normalized result type instead of MCP content blocks:

    >>> from bedrock_agentcore.tools import WebSearchClient
    >>>
    >>> client = WebSearchClient(region="us-east-1", gateway_id="my-gateway-abc123")
    >>> response = client.search("what shipped in python 3.13", max_results=5)
    >>> for result in response.results:
    ...     print(result.title, result.url)

The transport lives behind :class:`WebSearchBackend` so the same ``search()``
signature and the same :class:`WebSearchResult` can be served by a different
backend later without changing callers.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

import urllib3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from bedrock_agentcore._utils.endpoints import get_gateway_mcp_endpoint
from bedrock_agentcore._utils.user_agent import SDK_VERSION, build_user_agent_suffix

logger = logging.getLogger(__name__)

#: Name of the MCP tool the web search connector exposes. Fixed by the service.
WEB_SEARCH_TOOL_NAME = "WebSearch"

#: Gateway prefixes every tool it exposes with the name of the target it came from.
GATEWAY_TOOL_NAME_DELIMITER = "___"

#: Default target name used by ``GatewayClient.create_web_search_target``.
DEFAULT_TARGET_NAME = "amazon-web-search"

#: Service name the gateway data plane signs as.
GATEWAY_SIGNING_SERVICE = "bedrock-agentcore"

#: Documented input limits for the WebSearch tool.
MAX_QUERY_LENGTH = 200
MIN_MAX_RESULTS = 1
MAX_MAX_RESULTS = 25

#: Regions where the web search connector is offered. Used for a warning only, never
#: to block a call, so that a newly added region does not require an SDK release.
KNOWN_REGIONS = ("us-east-1", "eu-west-1", "ap-northeast-1")

_MCP_PROTOCOL_VERSION = "2025-06-18"
_DEFAULT_TIMEOUT = 30


class WebSearchError(RuntimeError):
    """Raised when a web search call fails."""


@dataclass(frozen=True)
class WebSearchResult:
    """A single web search result.

    Attributes:
        text: The extracted snippet relevant to the query. Always present.
        url: URL of the source page.
        title: Title of the source page.
        published_date: Publication date of the page, as reported by the index.
    """

    text: str
    url: Optional[str] = None
    title: Optional[str] = None
    published_date: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "WebSearchResult":
        """Build a result from one entry of a search response."""
        return cls(
            text=payload.get("text") or "",
            url=payload.get("url"),
            title=payload.get("title"),
            published_date=payload.get("publishedDate"),
        )


@dataclass(frozen=True)
class WebSearchResponse:
    """The results of one web search.

    Attributes:
        results: The results, in the order the service returned them.
        search_id: Service-assigned identifier for the search, when present.
    """

    results: List[WebSearchResult] = field(default_factory=list)
    search_id: Optional[str] = None

    def __len__(self) -> int:
        """Number of results."""
        return len(self.results)

    def __iter__(self) -> Iterator[WebSearchResult]:
        """Iterate over the results."""
        return iter(self.results)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "WebSearchResponse":
        """Build a response from the decoded search payload."""
        raw_results = payload.get("results") or []
        return cls(
            results=[WebSearchResult.from_payload(item) for item in raw_results if isinstance(item, dict)],
            search_id=payload.get("id"),
        )


def _build_arguments(
    query: str,
    max_results: Optional[int] = None,
    include_domains: Optional[Sequence[str]] = None,
    exclude_domains: Optional[Sequence[str]] = None,
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate search inputs and shape them into the tool's argument object.

    Raises:
        ValueError: If the query is empty or over the documented length limit, or
            if max_results falls outside the documented range.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"query must be {MAX_QUERY_LENGTH} characters or fewer, got {len(query)}")

    arguments: Dict[str, Any] = {"query": query}

    if max_results is not None:
        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise ValueError(f"max_results must be an integer, got {type(max_results).__name__}")
        if not MIN_MAX_RESULTS <= max_results <= MAX_MAX_RESULTS:
            raise ValueError(f"max_results must be between {MIN_MAX_RESULTS} and {MAX_MAX_RESULTS}, got {max_results}")
        arguments["maxResults"] = max_results

    filters: Dict[str, Any] = {}
    domain_filter: Dict[str, List[str]] = {}
    if include_domains:
        domain_filter["include"] = list(include_domains)
    if exclude_domains:
        domain_filter["exclude"] = list(exclude_domains)
    if domain_filter:
        filters["domainFilter"] = domain_filter

    published_filter: Dict[str, str] = {}
    if published_after:
        published_filter["from"] = published_after
    if published_before:
        published_filter["to"] = published_before
    if published_filter:
        filters["publishedDateFilter"] = published_filter

    if filters:
        arguments["filters"] = filters

    return arguments


def _extract_search_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the search payload out of an MCP ``tools/call`` result.

    The connector returns the results as a JSON document inside a text content
    block, so the text has to be decoded rather than read directly.

    Raises:
        WebSearchError: If the tool reported an error or returned no decodable
            text content.
    """
    if result.get("isError"):
        raise WebSearchError(f"Web search tool reported an error: {_first_text(result) or result}")

    text = _first_text(result)
    if text is None:
        raise WebSearchError(f"Web search response contained no text content: {result}")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WebSearchError(f"Could not decode web search response as JSON: {text[:200]!r}") from exc

    if not isinstance(payload, dict):
        raise WebSearchError(f"Expected a JSON object in the web search response, got {type(payload).__name__}")
    return payload


def _first_text(result: Dict[str, Any]) -> Optional[str]:
    """Return the first text content block of an MCP result, if any."""
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            return block["text"]
    return None


class WebSearchBackend:
    """How a :class:`WebSearchClient` reaches web search.

    A backend takes the tool's argument object and returns the decoded search
    payload, meaning a dict shaped ``{"id": ..., "results": [...]}``. Everything
    above this line is transport independent.
    """

    def search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Run one search and return the decoded payload."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources held by the backend."""


class GatewayMcpBackend(WebSearchBackend):
    """Reaches web search through an AgentCore Gateway target over MCP.

    Speaks the subset of MCP streamable HTTP that one tool call needs -- initialize,
    the initialized notification, optionally ``tools/list``, then ``tools/call`` --
    signing each request with SigV4. It is deliberately narrow: it is not a general
    MCP client, and it holds no dependency beyond what the SDK already requires.

    Both response framings the transport allows are handled, since a gateway may
    answer a POST with either ``application/json`` or ``text/event-stream``.
    """

    def __init__(
        self,
        endpoint: str,
        region: str,
        *,
        boto3_session: Optional[Any] = None,
        tool_name: Optional[str] = None,
        target_name: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        integration_source: Optional[str] = None,
        signing_service: str = GATEWAY_SIGNING_SERVICE,
    ):
        """Initialize the backend.

        Args:
            endpoint: The gateway's MCP endpoint URL.
            region: Region to sign for.
            boto3_session: Session to take credentials from. Defaults to a new session.
            tool_name: Fully qualified tool name. Skips discovery when given.
            target_name: Target the connector was added under. Used to derive the tool
                name without a ``tools/list`` round trip.
            timeout: Per-request timeout in seconds.
            integration_source: Optional framework identifier for the User-Agent.
            signing_service: SigV4 service name.
        """
        import boto3

        self._endpoint = endpoint
        self._region = region
        self._session = boto3_session or boto3.Session()
        self._signing_service = signing_service
        self._timeout = timeout
        self._user_agent = f"python-urllib3/{urllib3.__version__} {build_user_agent_suffix(integration_source)}"

        self._tool_name = tool_name
        self._target_name = target_name

        # A single signed POST per call, so retries are left to the caller: replaying a
        # tools/call is not always safe and the signature is only valid for a few minutes.
        self._http = urllib3.PoolManager(retries=urllib3.Retry(total=0, redirect=0))

        self._lock = threading.Lock()
        self._mcp_session_id: Optional[str] = None
        self._protocol_version = _MCP_PROTOCOL_VERSION
        self._initialized = False
        self._request_id = 0

    # Transport
    # -------------------------------------------------------------------------
    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _signed_headers(self, body: bytes, extra: Dict[str, str]) -> Dict[str, str]:
        """Sign a request body with SigV4 and return the headers to send."""
        credentials = self._session.get_credentials()
        if credentials is None:
            raise WebSearchError("No AWS credentials available. Configure credentials before calling web search.")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Content-Length": str(len(body)),
            "User-Agent": self._user_agent,
            **extra,
        }
        request = AWSRequest(method="POST", url=self._endpoint, data=body, headers=headers)
        SigV4Auth(credentials.get_frozen_credentials(), self._signing_service, self._region).add_auth(request)
        return dict(request.headers)

    def _post(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send one JSON-RPC message and return the decoded reply, if there is one."""
        body = json.dumps(message).encode("utf-8")

        extra: Dict[str, str] = {}
        if self._mcp_session_id:
            extra["Mcp-Session-Id"] = self._mcp_session_id
        if self._initialized:
            extra["MCP-Protocol-Version"] = self._protocol_version

        response = self._http.request(
            "POST",
            self._endpoint,
            body=body,
            headers=self._signed_headers(body, extra),
            timeout=urllib3.Timeout(total=self._timeout),
            preload_content=True,
        )

        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._mcp_session_id = session_id

        if response.status >= 400:
            body_text = response.data.decode("utf-8", "replace")[:500]
            raise WebSearchError(f"Web search request failed with HTTP {response.status}: {body_text}")

        if not response.data:
            return None

        reply = _decode_jsonrpc(response.headers.get("Content-Type", ""), response.data)
        if reply is None:
            return None
        if "error" in reply:
            error = reply["error"] or {}
            raise WebSearchError(f"Gateway returned a JSON-RPC error {error.get('code')}: {error.get('message')}")
        return reply

    # MCP session
    # -------------------------------------------------------------------------
    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        reply = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "bedrock-agentcore-python", "version": SDK_VERSION},
                },
            }
        )
        if reply is None:
            raise WebSearchError("Gateway did not answer the MCP initialize request")

        negotiated = (reply.get("result") or {}).get("protocolVersion")
        if isinstance(negotiated, str) and negotiated:
            self._protocol_version = negotiated

        self._initialized = True
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _ensure_tool_name(self) -> str:
        """Resolve the fully qualified tool name, discovering it if necessary."""
        if self._tool_name:
            return self._tool_name

        if self._target_name:
            self._tool_name = f"{self._target_name}{GATEWAY_TOOL_NAME_DELIMITER}{WEB_SEARCH_TOOL_NAME}"
            return self._tool_name

        candidates = [
            name
            for name in self._list_tool_names()
            if name == WEB_SEARCH_TOOL_NAME or name.endswith(f"{GATEWAY_TOOL_NAME_DELIMITER}{WEB_SEARCH_TOOL_NAME}")
        ]
        if not candidates:
            raise WebSearchError(
                f"No {WEB_SEARCH_TOOL_NAME} tool found on {self._endpoint}. "
                "Add a web search connector target to the gateway, or pass target_name."
            )
        if len(candidates) > 1:
            raise WebSearchError(
                f"Gateway exposes more than one {WEB_SEARCH_TOOL_NAME} tool ({', '.join(sorted(candidates))}). "
                "Pass target_name to choose one."
            )

        self._tool_name = candidates[0]
        logger.debug("Resolved web search tool name to %s", self._tool_name)
        return self._tool_name

    def _list_tool_names(self) -> List[str]:
        """List every tool the gateway exposes, following pagination."""
        names: List[str] = []
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"cursor": cursor} if cursor else {}
            reply = self._post({"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list", "params": params})
            result = (reply or {}).get("result") or {}
            for tool in result.get("tools") or []:
                if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                    names.append(tool["name"])
            cursor = result.get("nextCursor")
            if not cursor:
                return names

    # WebSearchBackend
    # -------------------------------------------------------------------------
    def search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call the WebSearch tool and return the decoded search payload."""
        with self._lock:
            self._ensure_initialized()
            tool_name = self._ensure_tool_name()
            reply = self._post(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                }
            )
        if reply is None:
            raise WebSearchError("Gateway did not answer the web search tool call")
        return _extract_search_payload(reply.get("result") or {})

    def close(self) -> None:
        """Close the connection pool."""
        self._http.clear()
        self._initialized = False
        self._mcp_session_id = None


def _decode_jsonrpc(content_type: str, data: bytes) -> Optional[Dict[str, Any]]:
    """Decode a JSON-RPC reply from either a JSON body or an SSE stream.

    Returns None when the body carries no JSON-RPC message, which is what a
    notification acknowledgement looks like.
    """
    text = data.decode("utf-8", "replace")

    if "text/event-stream" in content_type.lower():
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            chunk = line[len("data:") :].strip()
            if not chunk:
                continue
            try:
                message = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and "jsonrpc" in message:
                return message
        return None

    try:
        message = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WebSearchError(f"Could not decode gateway response as JSON: {text[:200]!r}") from exc
    return message if isinstance(message, dict) else None


class WebSearchClient:
    """Client for AgentCore Web Search.

    Attributes:
        region (str): The region being used.
        backend (WebSearchBackend): The transport in use.

    Basic Usage:
        >>> from bedrock_agentcore.tools import WebSearchClient
        >>>
        >>> client = WebSearchClient(region="us-east-1", gateway_id="my-gateway-abc123")
        >>> response = client.search("latest boto3 release notes")
        >>> response.results[0].url

    Context Manager:
        >>> with WebSearchClient(region="us-east-1", gateway_id="my-gateway-abc123") as client:
        ...     response = client.search("who maintains urllib3")
    """

    def __init__(
        self,
        region: Optional[str] = None,
        *,
        gateway_id: Optional[str] = None,
        gateway_arn: Optional[str] = None,
        gateway_endpoint: Optional[str] = None,
        target_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        backend: Optional[WebSearchBackend] = None,
        boto3_session: Optional[Any] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        integration_source: Optional[str] = None,
    ):
        """Initialize the client.

        Exactly one of ``gateway_id``, ``gateway_arn``, ``gateway_endpoint`` or
        ``backend`` identifies where the search goes. The gateway arguments are
        keyword only and optional so that a future transport needing none of them
        is an addition rather than a breaking change.

        Args:
            region: Region to call. Defaults to the session's region.
            gateway_id: ID of a gateway with a web search connector target.
            gateway_arn: ARN of that gateway. The ID and region are read from it.
            gateway_endpoint: A gateway MCP endpoint URL, if you already have one.
            target_name: Name of the connector target. Supplying it avoids a
                ``tools/list`` round trip on the first search.
            tool_name: Fully qualified tool name, if you already know it.
            backend: A backend to use as is. Overrides every gateway argument.
            boto3_session: Session to take credentials and region from.
            timeout: Per-request timeout in seconds.
            integration_source: Optional framework identifier for the User-Agent.

        Raises:
            ValueError: If no gateway is identified, or more than one is.
        """
        import boto3

        self._session = boto3_session or boto3.Session()
        self._owns_backend = backend is None

        if backend is not None:
            if any(value is not None for value in (gateway_id, gateway_arn, gateway_endpoint)):
                raise ValueError("Pass either backend or one of gateway_id/gateway_arn/gateway_endpoint, not both")
            self.region = region or self._session.region_name
            self.backend: WebSearchBackend = backend
            return

        given = [
            name
            for name, value in (
                ("gateway_id", gateway_id),
                ("gateway_arn", gateway_arn),
                ("gateway_endpoint", gateway_endpoint),
            )
            if value
        ]
        if len(given) > 1:
            raise ValueError(f"Pass only one of gateway_id, gateway_arn or gateway_endpoint, got {', '.join(given)}")

        if gateway_arn:
            gateway_id, arn_region = _parse_gateway_arn(gateway_arn)
            region = region or arn_region

        self.region = region or self._session.region_name
        if not self.region:
            raise ValueError("region could not be determined. Pass region= or configure a default region.")
        if self.region not in KNOWN_REGIONS:
            logger.warning(
                "Web search is offered in %s. Calling %s may fail if the connector is not available there.",
                ", ".join(KNOWN_REGIONS),
                self.region,
            )

        if gateway_id:
            gateway_endpoint = get_gateway_mcp_endpoint(gateway_id, self.region)
        if not gateway_endpoint:
            raise ValueError("One of gateway_id, gateway_arn, gateway_endpoint or backend is required")

        self.backend = GatewayMcpBackend(
            endpoint=gateway_endpoint,
            region=self.region,
            boto3_session=self._session,
            tool_name=tool_name,
            target_name=target_name,
            timeout=timeout,
            integration_source=integration_source,
        )

    def search(
        self,
        query: str,
        *,
        max_results: Optional[int] = None,
        include_domains: Optional[Sequence[str]] = None,
        exclude_domains: Optional[Sequence[str]] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
    ) -> WebSearchResponse:
        """Search the web.

        The filter arguments need connector version 1.2.0 or later on the target.
        On an earlier version the tool accepts only ``query`` and ``max_results``.
        Target level domain rules always apply on top and cannot be relaxed here.

        Args:
            query: What to search for. 200 characters or fewer.
            max_results: How many results to return, 1 to 25. Service default is 10.
            include_domains: Restrict results to these domains.
            exclude_domains: Drop results from these domains.
            published_after: Earliest publication date, ISO-8601 UTC, inclusive.
            published_before: Latest publication date, ISO-8601 UTC, inclusive.

        Returns:
            The search results.

        Raises:
            ValueError: If the query or max_results is outside the documented limits.
            WebSearchError: If the call fails or the response cannot be decoded.
        """
        arguments = _build_arguments(
            query=query,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            published_after=published_after,
            published_before=published_before,
        )
        return WebSearchResponse.from_payload(self.backend.search(arguments))

    def close(self) -> None:
        """Release the backend, if this client created it."""
        if self._owns_backend:
            self.backend.close()

    def __enter__(self) -> "WebSearchClient":
        """Enter the context manager."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the client on exit."""
        self.close()


def _parse_gateway_arn(arn: str) -> tuple:
    """Pull the gateway ID and region out of a gateway ARN.

    Raises:
        ValueError: If the ARN is not a gateway ARN.
    """
    parts = arn.split(":")
    if len(parts) < 6 or parts[0] != "arn" or not parts[5].startswith("gateway/"):
        raise ValueError(
            f"Not a gateway ARN: {arn!r}. Expected 'arn:aws:bedrock-agentcore:<region>:<account>:gateway/<id>'."
        )
    gateway_id = parts[5].split("/", 1)[1]
    if not gateway_id:
        raise ValueError(f"Gateway ARN carries no gateway ID: {arn!r}")
    return gateway_id, parts[3]
