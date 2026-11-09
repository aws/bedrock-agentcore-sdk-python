"""Client for AgentCore Web Search.

Web Search is reachable two ways: directly, through the ``InvokeWebSearch`` operation on the
AgentCore data plane, and as an AgentCore Gateway connector target that agents call as an MCP
tool. This module wraps both, so callers get a plain ``search()`` method and a normalized
result type instead of API shapes or MCP content blocks:

    >>> from bedrock_agentcore.tools import WebSearchClient
    >>>
    >>> client = WebSearchClient(region="us-east-1")  # the Web Search API
    >>> response = client.search("what shipped in python 3.13", max_results=5)
    >>> for result in response.results:
    ...     print(result.title, result.url, result.published_at)

Pass ``gateway_id`` (or ``gateway_arn`` or ``gateway_endpoint``) to search through a gateway
target instead. The transport lives behind :class:`WebSearchBackend`, so the same
``search()`` signature and the same :class:`WebSearchResult` are served by either one.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import urllib3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from bedrock_agentcore._utils.endpoints import (
    DP_ENDPOINT_OVERRIDE,
    _validate_endpoint_url,
    get_gateway_mcp_endpoint,
    validate_region,
)
from bedrock_agentcore._utils.sse import iter_sse_data
from bedrock_agentcore._utils.user_agent import SDK_VERSION, build_user_agent_suffix

logger = logging.getLogger(__name__)

#: Name of the MCP tool the web search connector exposes. Fixed by the service.
WEB_SEARCH_TOOL_NAME = "WebSearch"

#: Gateway prefixes every tool it exposes with the name of the target it came from.
GATEWAY_TOOL_NAME_DELIMITER = "___"

#: Service name the gateway data plane signs as.
GATEWAY_SIGNING_SERVICE = "bedrock-agentcore"

#: Documented input limits for the WebSearch tool.
MAX_QUERY_LENGTH = 200
MIN_MAX_RESULTS = 1
MAX_MAX_RESULTS = 25
MAX_DOMAIN_FILTER_ENTRIES = 100

#: Identifier of the AWS-managed web search tool that the Web Search API invokes.
DEFAULT_WEB_SEARCH_ID = "web-search.v1"

#: Regions where the web search connector is offered. Used for a warning only, never
#: to block a call, so that a newly added region does not require an SDK release.
KNOWN_REGIONS = ("us-east-1", "eu-west-1", "ap-northeast-1")

#: Regions where the Web Search API is offered. Also used for a warning only.
KNOWN_API_REGIONS = ("us-east-1", "us-east-2", "us-west-2", "eu-west-1", "ap-northeast-1", "ap-southeast-2")

#: UTC offsets, in hours, for the zone names the gateway connector writes into its dates.
_ZONE_OFFSETS = {
    "UTC": 0,
    "GMT": 0,
    "PDT": -7,
    "PST": -8,
    "MDT": -6,
    "MST": -7,
    "CDT": -5,
    "CST": -6,
    "EDT": -4,
    "EST": -5,
    "BST": 1,
    "CET": 1,
    "CEST": 2,
    "JST": 9,
    "KST": 9,
    "AEST": 10,
    "AEDT": 11,
}

#: English names in the gateway connector's dates, parsed without the process locale.
_MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_GATEWAY_DATE = re.compile(
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<half>AM|PM),\s*(?P<weekday>[A-Za-z]+),\s*"
    r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})\s+(?P<year>\d{4})",
    re.IGNORECASE,
)
_DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")
_FRACTION = re.compile(r"(T\d{2}:\d{2}:\d{2})\.(\d+)")

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
        published_date: Publication date of the page, as the service reported it. The Web
            Search API reports ISO 8601. The gateway connector reports a readable form such
            as ``"05:00PM, Tuesday, October 06 2026, PDT"``.
        published_at: ``published_date`` parsed into a timezone-aware UTC datetime, the same
            for both transports. ``None`` when the service reported no date or one that
            can't be parsed.
        acquired_at: When the index acquired the page, as a timezone-aware UTC datetime, if
            the service reports it.
    """

    text: str
    url: Optional[str] = None
    title: Optional[str] = None
    published_date: Optional[str] = None
    published_at: Optional[datetime] = None
    acquired_at: Optional[datetime] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "WebSearchResult":
        """Build a result from one entry of a search response, from either transport."""
        published = payload.get("publishedDate", payload.get("publishedAt"))
        acquired = payload.get("acquiredDate", payload.get("acquiredAt"))
        return cls(
            text=payload.get("text") or "",
            url=payload.get("url"),
            title=payload.get("title"),
            published_date=_format_timestamp(published) if isinstance(published, datetime) else published,
            published_at=_parse_timestamp(published),
            acquired_at=_parse_timestamp(acquired),
        )


def _as_aware_utc(value: datetime) -> datetime:
    """Convert to UTC, taking a naive datetime as already UTC."""
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _parse_iso(text: str) -> Optional[datetime]:
    """Parse ISO 8601, including a ``Z`` suffix and any number of fractional digits."""
    text = _FRACTION.sub(lambda m: f"{m.group(1)}.{(m.group(2) + '000000')[:6]}", text)
    try:
        return datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None


def _parse_gateway_date(text: str) -> Optional[datetime]:
    """Parse the connector's form, "05:00PM, Tuesday, October 06 2026, PDT", in any locale."""
    stamp, _, zone = text.rpartition(", ")
    match = _GATEWAY_DATE.fullmatch(stamp.strip())
    month = _MONTHS.get(match.group("month").lower()) if match else None
    if not match or month is None or zone.upper() not in _ZONE_OFFSETS:
        return None
    if match.group("weekday").lower() not in _WEEKDAYS:
        return None
    hour = int(match.group("hour"))
    if not 1 <= hour <= 12:
        return None
    hour = hour % 12 + (12 if match.group("half").upper() == "PM" else 0)
    try:
        local = datetime(int(match.group("year")), month, int(match.group("day")), hour, int(match.group("minute")))
    except ValueError:
        return None
    return (local - timedelta(hours=_ZONE_OFFSETS[zone.upper()])).replace(tzinfo=timezone.utc)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse a date from either transport into an aware UTC datetime, or ``None``."""
    if isinstance(value, datetime):
        return _as_aware_utc(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    parsed = _parse_iso(text)
    if parsed is not None:
        return _as_aware_utc(parsed)
    return _parse_gateway_date(text)


def _format_timestamp(value: datetime) -> str:
    """Format a datetime as ISO 8601 UTC with a ``Z`` suffix, keeping fractional seconds."""
    return _as_aware_utc(value).isoformat().replace("+00:00", "Z")


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
        ValueError: If the query is empty or over the documented length limit, if
            max_results falls outside the documented range, or if either domain
            list is longer than the documented maximum.
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
        domain_filter["include"] = _checked_domains("include_domains", include_domains)
    if exclude_domains:
        domain_filter["exclude"] = _checked_domains("exclude_domains", exclude_domains)
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


def _checked_domains(argument: str, domains: Sequence[str]) -> List[str]:
    """Return the domain list, rejecting one longer than the documented maximum."""
    values = list(domains)
    if len(values) > MAX_DOMAIN_FILTER_ENTRIES:
        raise ValueError(f"{argument} accepts at most {MAX_DOMAIN_FILTER_ENTRIES} domains, got {len(values)}")
    return values


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
        """Send one JSON-RPC message and return the reply to it, if there is one.

        Raises:
            WebSearchError: If the transport fails, the gateway answers with an HTTP
                error or a JSON-RPC error, or the reply carries neither a result nor
                an error.
        """
        body = json.dumps(message).encode("utf-8")
        expected_id = message.get("id")

        extra: Dict[str, str] = {}
        if self._mcp_session_id:
            extra["Mcp-Session-Id"] = self._mcp_session_id
        if self._initialized:
            extra["MCP-Protocol-Version"] = self._protocol_version

        try:
            response = self._http.request(
                "POST",
                self._endpoint,
                body=body,
                headers=self._signed_headers(body, extra),
                timeout=urllib3.Timeout(total=self._timeout),
                preload_content=True,
            )
        except urllib3.exceptions.HTTPError as exc:
            # Connection refused, DNS failure, TLS errors and read timeouts all land
            # here. search() promises WebSearchError, so they cannot escape as urllib3
            # exceptions.
            raise WebSearchError(f"Web search request to {self._endpoint} failed: {exc}") from exc

        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._mcp_session_id = session_id

        if response.status >= 400:
            body_text = response.data.decode("utf-8", "replace")[:500]
            # 404 against a session we hold is the spec's signal that the session is
            # gone. Forget it, so the next call re-initializes rather than failing
            # this way until someone calls close().
            if response.status == 404 and self._mcp_session_id:
                self._reset_session()
            raise WebSearchError(f"Web search request failed with HTTP {response.status}: {body_text}")

        if not response.data:
            return None

        reply = _decode_jsonrpc(response.headers.get("Content-Type", ""), response.data, expected_id)
        if reply is None:
            return None
        if "error" in reply:
            error = reply["error"] or {}
            raise WebSearchError(f"Gateway returned a JSON-RPC error {error.get('code')}: {error.get('message')}")
        if expected_id is not None and "result" not in reply:
            raise WebSearchError(f"Gateway reply carried neither a result nor an error: {reply}")
        return reply

    def _reset_session(self) -> None:
        """Forget the MCP session, so the next call performs the handshake again."""
        self._initialized = False
        self._mcp_session_id = None

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

        # Set before the notification is sent, because from here on every request
        # carries MCP-Protocol-Version. If the notification fails, the handshake did
        # not complete, so the flag is rolled back rather than left claiming a session
        # the gateway never acknowledged.
        self._initialized = True
        try:
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except Exception:
            self._reset_session()
            raise

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
        self._reset_session()


class WebSearchApiBackend(WebSearchBackend):
    """Reaches web search through the ``InvokeWebSearch`` operation on the AgentCore data plane.

    Calls are signed with the caller's AWS credentials by boto3. The calling identity needs
    ``bedrock-agentcore:InvokeWebSearch`` on
    ``arn:<partition>:bedrock-agentcore:<region>:aws:tool/web-search.v1``. There is no
    resource to create first. boto3 clients are thread-safe, so unlike the gateway backend
    this one holds no session state and one instance can serve concurrent searches.
    """

    def __init__(
        self,
        region: str,
        *,
        boto3_session: Optional[Any] = None,
        client: Optional[Any] = None,
        web_search_id: str = DEFAULT_WEB_SEARCH_ID,
        integration_source: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        """Initialize the backend.

        Args:
            region: Region to call.
            boto3_session: Session to build the ``bedrock-agentcore`` client from. Defaults
                to a new session.
            client: A ``bedrock-agentcore`` boto3 client to use as is, instead of building one.
            web_search_id: Identifier of the web search tool to invoke.
            integration_source: Optional framework identifier for the User-Agent.
            timeout: Connect and read timeout in seconds for the client this backend builds.
        """
        self._web_search_id = web_search_id
        if client is None:
            import boto3
            from botocore.config import Config

            # Same data plane client, endpoint override and User-Agent as the browser and
            # code interpreter clients.
            kwargs: Dict[str, Any] = {
                "region_name": validate_region(region),
                "config": Config(
                    user_agent_extra=build_user_agent_suffix(integration_source),
                    connect_timeout=timeout,
                    read_timeout=timeout,
                ),
            }
            if DP_ENDPOINT_OVERRIDE:
                kwargs["endpoint_url"] = DP_ENDPOINT_OVERRIDE
            client = (boto3_session or boto3.Session()).client("bedrock-agentcore", **kwargs)
        self._client = client

    def search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Run one search and return the decoded payload.

        Raises:
            botocore.exceptions.ClientError: If the service rejects the call, for example
                with ``ThrottlingException`` or ``AccessDeniedException``. It is not wrapped,
                matching the browser and code interpreter clients.
        """
        request: Dict[str, Any] = {"webSearchId": self._web_search_id, "searchQuery": arguments["query"]}
        if "maxResults" in arguments:
            request["maxResults"] = arguments["maxResults"]

        given = arguments.get("filters") or {}
        filters: Dict[str, Any] = {}
        if given.get("domainFilter"):
            filters["domainFilter"] = dict(given["domainFilter"])
        # The argument object names the date bounds "from" and "to", as the gateway tool does.
        dates = given.get("publishedDateFilter") or {}
        published: Dict[str, Any] = {}
        # The API accepts only UTC date-times, so a bound given with an offset is converted.
        if "from" in dates:
            published["publishedAfter"] = _as_utc(dates["from"])
        if "to" in dates:
            published["publishedBefore"] = _exclusive_upper_bound(dates["to"])
        if published:
            filters["publishedDateFilter"] = published
        if filters:
            request["filters"] = filters

        response = self._client.invoke_web_search(**request)

        results = []
        for item in response.get("results") or []:
            entry = {key: item[key] for key in ("title", "url", "text") if key in item}
            for key in ("publishedAt", "acquiredAt"):
                value = item.get(key)
                if value is not None:
                    entry[key] = _format_timestamp(value) if isinstance(value, datetime) else value
            results.append(entry)
        return {"results": results}


def _as_utc(value: Any) -> Any:
    """Return a date bound as an ISO 8601 UTC string when it can be parsed, else unchanged."""
    parsed = _parse_timestamp(value)
    return _format_timestamp(parsed) if parsed is not None else value


def _exclusive_upper_bound(value: Any) -> Any:
    """Return ``published_before`` for the API, whose upper bound is exclusive.

    A date-only bound means "through that day", as the gateway connector reads it, so it
    becomes the start of the next day in UTC instead of the start of the same day.
    """
    if isinstance(value, str) and _DATE_ONLY.fullmatch(value.strip()):
        next_day = datetime.fromisoformat(value.strip()).replace(tzinfo=timezone.utc) + timedelta(days=1)
        return _format_timestamp(next_day)
    return _as_utc(value)


def _decode_jsonrpc(content_type: str, data: bytes, expected_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Decode the JSON-RPC reply to ``expected_id`` from a JSON body or an SSE stream.

    Returns None when the body carries no reply to that id, which is what a
    notification acknowledgement looks like. A message addressed to another id, such
    as a server notification arriving ahead of the reply, is skipped rather than
    mistaken for the answer.
    """
    text = data.decode("utf-8", "replace")

    if "text/event-stream" in content_type.lower():
        for message in _iter_sse_messages(text):
            if _answers(message, expected_id):
                return message
        return None

    try:
        message = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WebSearchError(f"Could not decode gateway response as JSON: {text[:200]!r}") from exc
    if not isinstance(message, dict):
        return None
    return message if _answers(message, expected_id) else None


def _answers(message: Any, expected_id: Optional[int]) -> bool:
    """Whether a decoded JSON-RPC message is the reply to ``expected_id``."""
    if not isinstance(message, dict) or "jsonrpc" not in message:
        return False
    if "error" in message:
        # A JSON-RPC error is allowed to carry a null id, so it is always surfaced
        # rather than filtered out for not matching.
        return True
    if expected_id is None:
        return True
    return message.get("id") == expected_id


def _iter_sse_messages(text: str) -> Iterator[Dict[str, Any]]:
    """Yield the JSON objects carried by the ``data`` field of each SSE event.

    Consecutive ``data:`` lines belonging to one event are joined with a newline, as
    the SSE specification requires, so a message split across lines is decoded rather
    than dropped. A blank line ends an event.
    """
    for data in iter_sse_data([text]):
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict):
            yield message


class WebSearchClient:
    """Client for AgentCore Web Search.

    Calls are authenticated with SigV4 using ordinary AWS credentials. There is no
    web search API key. With no gateway argument, the client calls the Web Search API, and
    the credentials it resolves need ``bedrock-agentcore:InvokeWebSearch`` on
    ``arn:<partition>:bedrock-agentcore:<region>:aws:tool/web-search.v1``. Through a gateway,
    those credentials need ``bedrock-agentcore:InvokeGateway`` on the gateway ARN, and the
    gateway's own service role needs ``bedrock-agentcore:InvokeWebSearch`` on the connector.

    Attributes:
        region (str): The region being used.
        backend (WebSearchBackend): The transport in use.

    Basic Usage:
        >>> from bedrock_agentcore.tools import WebSearchClient
        >>>
        >>> client = WebSearchClient(region="us-east-1")
        >>> response = client.search("latest boto3 release notes")
        >>> response.results[0].url

    Filters every search must carry:
        >>> client = WebSearchClient(region="us-east-1", exclude_domains=["example.com"],
        ...                          published_after="2026-11-01T00:00:00Z")
        >>> client.search("openssl advisory", published_after="2026-01-01T00:00:00Z")
        ... # still searches from 2026-11-01: a call can narrow these, never widen them

    Through a gateway:
        >>> client = WebSearchClient(region="us-east-1", gateway_id="my-gateway-abc123")

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
        web_search_id: str = DEFAULT_WEB_SEARCH_ID,
        include_domains: Optional[Sequence[str]] = None,
        exclude_domains: Optional[Sequence[str]] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
    ):
        """Initialize the client.

        At most one of ``gateway_id``, ``gateway_arn``, ``gateway_endpoint`` or
        ``backend`` says where the search goes. With none of them, the client calls the
        Web Search API directly.

        The filter arguments set rules that every search through this client carries. A
        call to :meth:`search` can narrow them but never widen them: its exclude list is
        added to this one, its include list is intersected with this one, and its date
        bounds only take effect when they are stricter. Use them for rules your
        application enforces, so that a model choosing search arguments can't lift them.

        Args:
            region: Region to call. Defaults to the session's region.
            gateway_id: ID of a gateway with a web search connector target.
            gateway_arn: ARN of that gateway. The ID and region are read from it.
            gateway_endpoint: A gateway MCP endpoint URL, if you already have one. The
                host has to be an AWS one, the same check applied to endpoints this SDK
                builds itself.
            target_name: Name of the connector target. Supplying it avoids a
                ``tools/list`` round trip on the first search.
            tool_name: Fully qualified tool name, if you already know it.
            backend: A backend to use as is. Overrides every gateway argument.
            boto3_session: Session to take credentials and region from.
            timeout: Per-request timeout in seconds.
            integration_source: Optional framework identifier for the User-Agent.
            web_search_id: Identifier of the web search tool the Web Search API invokes.
                Ignored when searching through a gateway or a supplied backend.
            include_domains: Every search returns only these domains. Up to 100.
            exclude_domains: Every search drops these domains. Up to 100.
            published_after: Every search returns only pages published at or after this
                time, ISO 8601 UTC.
            published_before: Every search returns only pages published before this time,
                ISO 8601 UTC.

        Raises:
            ValueError: If more than one gateway is identified, a filter list is longer
                than the documented maximum, or a date bound can't be parsed.
            InvalidGatewayIdentifierError: If ``gateway_id`` is not a gateway ID.
            InvalidRegionError: If the region is malformed, or a supplied
                ``gateway_endpoint`` does not point at an AWS host.
        """
        self._session = boto3_session
        self._owns_backend = backend is None
        self._fixed_include = _checked_domains("include_domains", include_domains) if include_domains else None
        self._fixed_exclude = _checked_domains("exclude_domains", exclude_domains) if exclude_domains else None
        self._fixed_after = _checked_bound("published_after", published_after)
        self._fixed_before = _checked_bound("published_before", published_before)

        if backend is not None:
            if any(value is not None for value in (gateway_id, gateway_arn, gateway_endpoint)):
                raise ValueError("Pass either backend or one of gateway_id/gateway_arn/gateway_endpoint, not both")
            # Only resolve a session if the region has to come from one, so supplying a
            # backend does not pay for the credential provider chain it will not use.
            self.region = region or self._resolve_session().region_name
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

        self.region = region or self._resolve_session().region_name
        if not self.region:
            raise ValueError("region could not be determined. Pass region= or configure a default region.")

        if not given:
            if self.region not in KNOWN_API_REGIONS:
                logger.warning(
                    "The Web Search API is offered in %s. Calling %s may fail.",
                    ", ".join(KNOWN_API_REGIONS),
                    self.region,
                )
            self.backend = WebSearchApiBackend(
                self.region,
                boto3_session=self._resolve_session(),
                web_search_id=web_search_id,
                integration_source=integration_source,
                timeout=timeout,
            )
            return

        if self.region not in KNOWN_REGIONS:
            logger.warning(
                "Web search is offered in %s. Calling %s may fail if the connector is not available there.",
                ", ".join(KNOWN_REGIONS),
                self.region,
            )

        if gateway_id:
            gateway_endpoint = get_gateway_mcp_endpoint(gateway_id, self.region)
        elif gateway_endpoint:
            # Signed requests carry the caller's credentials, including a session token,
            # so a supplied endpoint gets the same host check as one this SDK builds.
            gateway_endpoint = _validate_endpoint_url(gateway_endpoint)
        if not gateway_endpoint:
            raise ValueError("One of gateway_id, gateway_arn, gateway_endpoint or backend is required")

        self.backend = GatewayMcpBackend(
            endpoint=gateway_endpoint,
            region=self.region,
            boto3_session=self._resolve_session(),
            tool_name=tool_name,
            target_name=target_name,
            timeout=timeout,
            integration_source=integration_source,
        )

    def _resolve_session(self) -> Any:
        """Return the boto3 session, creating a default one only when first needed."""
        if self._session is None:
            import boto3

            self._session = boto3.Session()
        return self._session

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

        Request filters compose with the target's own domain rules and can never
        widen them. A domain is dropped if it appears on either exclude list. A
        domain is returned only if it appears on every include list that is set,
        so when the target already has an include list, passing ``include_domains``
        narrows to the intersection of the two. If the two share no domains the
        search returns nothing, which is a silent empty result rather than an
        error, so check the target's configuration when a filtered search comes
        back empty.

        Args:
            query: What to search for. 200 characters or fewer.
            max_results: How many results to return, 1 to 25. Service default is 10.
            include_domains: Restrict results to these domains. Up to 100. A root
                domain matches its subdomains.
            exclude_domains: Drop results from these domains. Up to 100.
            published_after: Earliest publication date, ISO-8601 UTC, inclusive.
                Applies to web results only.
            published_before: Latest publication date, ISO-8601 UTC. Applies to web
                results only. The Web Search API treats a date-time as exclusive, and a
                date-only value such as ``2026-11-01`` as through the end of that day.

        Returns:
            The search results.

        Raises:
            ValueError: If the query, max_results or either domain list is outside the
                documented limits, including an exclude list that goes over 100 once
                combined with the client's own.
            WebSearchError: If a gateway call fails or its response cannot be decoded.
            botocore.exceptions.ClientError: If the Web Search API rejects the call.
        """
        # Validate the caller's own arguments first, so bad input raises even when the
        # fixed filters would leave nothing to search.
        _build_arguments(
            query=query, max_results=max_results, include_domains=include_domains, exclude_domains=exclude_domains
        )
        nothing_allowed = False
        if self._fixed_include and include_domains:
            include_domains = [domain for domain in include_domains if _covered(domain, self._fixed_include)]
            nothing_allowed = not include_domains
        else:
            include_domains = include_domains or self._fixed_include
        if self._fixed_exclude:
            exclude_domains = _combined_excludes(self._fixed_exclude, exclude_domains or [])
        if nothing_allowed:
            # None of the requested domains is inside the client's include list.
            return WebSearchResponse()
        arguments = _build_arguments(
            query=query,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            published_after=_stricter(self._fixed_after, published_after, later=True),
            published_before=_stricter(self._fixed_before, published_before, later=False),
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


def _checked_bound(argument: str, value: Optional[str]) -> Optional[str]:
    """Return a fixed date bound, rejecting one that can't be parsed."""
    if value is not None and _parse_timestamp(value) is None:
        raise ValueError(f"{argument} must be an ISO 8601 date or date-time, got {value!r}")
    return value


def _covered(domain: str, allowed: Sequence[str]) -> bool:
    """Whether a domain equals or is a subdomain of one in ``allowed``, ignoring case.

    This matches the service, where a root domain also covers its subdomains.
    """
    name = domain.strip().lower().rstrip(".")
    for root in allowed:
        root = root.strip().lower().rstrip(".")
        if name == root or name.endswith("." + root):
            return True
    return False


def _combined_excludes(fixed: Sequence[str], requested: Sequence[str]) -> List[str]:
    """Combine the client's and the call's exclude lists, within the API's limit."""
    combined: List[str] = []
    seen = set()
    for domain in [*fixed, *requested]:
        if domain.lower() not in seen:
            seen.add(domain.lower())
            combined.append(domain)
    if len(combined) > MAX_DOMAIN_FILTER_ENTRIES:
        raise ValueError(
            f"exclude_domains accepts at most {MAX_DOMAIN_FILTER_ENTRIES} domains including the client's own: "
            f"{len(fixed)} set on the client plus {len(combined) - len(fixed)} from this call is {len(combined)}"
        )
    return combined


def _stricter(fixed: Optional[str], requested: Optional[str], *, later: bool) -> Optional[str]:
    """Return whichever date bound is stricter, keeping ``fixed`` when they can't be compared."""
    if not fixed or not requested:
        return fixed or requested
    fixed_at, requested_at = _parse_timestamp(fixed), _parse_timestamp(requested)
    if fixed_at is None or requested_at is None:
        return fixed
    if later:
        return requested if requested_at > fixed_at else fixed
    return requested if requested_at < fixed_at else fixed


def _parse_gateway_arn(arn: str) -> Tuple[str, str]:
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
