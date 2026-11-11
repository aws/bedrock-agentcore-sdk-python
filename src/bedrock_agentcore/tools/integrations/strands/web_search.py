"""AgentCore Web Search as a Strands Agents tool.

Web search reaches the service through an AgentCore Gateway connector target
today, and a direct API is expected later. Which transport is used is decided
inside ``WebSearchClient`` rather than here, so every gateway argument on this
class is keyword-only and optional, and only the arguments actually supplied are
forwarded. A later transport needing none of them works through the same call,
without a signature change here.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from strands.tools import tool

from ...web_search_client import WebSearchClient, WebSearchResponse

logger = logging.getLogger(__name__)

#: Web search is offered in a subset of regions, so this default is not the one
#: the browser and code interpreter tools use. It is applied only when no region
#: and no gateway ARN were given, because the client prefers an explicit region
#: over the one carried in the ARN.
DEFAULT_REGION = "us-east-1"


def _format_response(response: WebSearchResponse) -> str:
    """Render search results as text a model can cite from.

    Args:
        response: The results of one search.

    Returns:
        One numbered block per result, or a plain sentence when there were none.
    """
    if not response.results:
        return "No results. The query may be too narrow, or a domain or date filter may have excluded everything."

    blocks: List[str] = []
    for position, result in enumerate(response.results, start=1):
        lines = [f"{position}. {result.title or 'Untitled'}"]
        if result.url:
            lines.append(f"   URL: {result.url}")
        if result.published_date:
            lines.append(f"   Published: {result.published_date}")
        if result.text:
            lines.append(f"   {result.text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class AgentCoreWebSearch:
    """Exposes AgentCore Web Search as a Strands tool.

    The search runs over an AgentCore Gateway target that has the web search
    connector attached. Calls are authenticated with SigV4 from the ambient AWS
    credentials; there is no web search API key. Those credentials need
    ``bedrock-agentcore:InvokeGateway`` on the gateway, and the gateway's own
    service role needs ``bedrock-agentcore:InvokeWebSearch`` on the connector.

    Basic Usage:
        >>> from strands import Agent
        >>> from bedrock_agentcore.tools.integrations.strands import AgentCoreWebSearch
        >>>
        >>> search = AgentCoreWebSearch(region="us-east-1", gateway_id="my-gateway-abc123")
        >>> agent = Agent(tools=[search.web_search])
        >>> agent("What changed in the most recent boto3 release?")
        >>> search.close()

    Context Manager:
        >>> with AgentCoreWebSearch(gateway_id="my-gateway-abc123") as search:
        ...     agent = Agent(tools=[search.web_search])
        ...     agent("Who maintains urllib3?")
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
        boto3_session: Optional[Any] = None,
        client: Optional[WebSearchClient] = None,
    ):
        """Initialize the tool.

        Exactly one of ``gateway_id``, ``gateway_arn``, ``gateway_endpoint`` or
        ``client`` says where the search goes.

        Args:
            region: AWS region to call. Defaults to ``DEFAULT_REGION``, except when
                ``gateway_arn`` is given, in which case the ARN's region is used.
            gateway_id: ID of a gateway carrying a web search connector target.
            gateway_arn: ARN of that gateway. The ID and the region are read from it.
            gateway_endpoint: A gateway MCP endpoint URL, if one is already known.
            target_name: Name the connector target was created under. Supplying it
                avoids a tool discovery round trip on the first search.
            tool_name: Fully qualified name of the gateway tool, if already known.
            boto3_session: Session to take credentials from.
            client: A client to use as is. The caller keeps ownership of it, so
                ``close`` leaves it open.

        Raises:
            ValueError: If both ``client`` and a gateway argument are given.
        """
        if client is not None and any(value is not None for value in (gateway_id, gateway_arn, gateway_endpoint)):
            raise ValueError("Pass either client or one of gateway_id, gateway_arn or gateway_endpoint, not both.")

        # WebSearchClient prefers an explicit region over the one in a gateway
        # ARN, so filling the default in here would silently override the ARN's
        # region. Leaving it unset is what lets the ARN decide.
        if region is None and gateway_arn is None:
            region = DEFAULT_REGION

        self.region = region
        self._gateway_id = gateway_id
        self._gateway_arn = gateway_arn
        self._gateway_endpoint = gateway_endpoint
        self._target_name = target_name
        self._tool_name = tool_name
        self._boto3_session = boto3_session
        self._owns_client = client is None
        self._client = client if client is not None else self._create_client()
        self._closed = False
        self._lock = threading.Lock()

    def _create_client(self) -> WebSearchClient:
        """Build the client, forwarding only the arguments that were given.

        Returns:
            A client pointed at whichever destination was identified.
        """
        # Passing gateway_id=None explicitly would tie this module to the gateway
        # transport. Forwarding only what was supplied means a future transport
        # needing no gateway argument works through this same path.
        kwargs: Dict[str, Any] = {"integration_source": "strands"}
        optional = (
            ("region", self.region),
            ("gateway_id", self._gateway_id),
            ("gateway_arn", self._gateway_arn),
            ("gateway_endpoint", self._gateway_endpoint),
            ("target_name", self._target_name),
            ("tool_name", self._tool_name),
            ("boto3_session", self._boto3_session),
        )
        for name, value in optional:
            if value is not None:
                kwargs[name] = value
        return WebSearchClient(**kwargs)

    @tool
    def web_search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
    ) -> str:
        """Search the web for current information and return source-attributed results.

        Use this tool for questions about recent events, releases or prices, for
        facts that need a citable source, and for topics outside your training
        data. Each result carries a title, a URL, a publication date when the
        index reports one, and an extract. Cite the URLs you used in your answer.

        Prefer one broad query over several narrow ones. Use include_domains to
        restrict a search to sources you trust, such as official documentation.

        Args:
            query: What to search for, as a natural language query. 200 characters or fewer.
            max_results: How many results to return, between 1 and 25. Leave unset
                for the service default.
            include_domains: Only return results from these domains, e.g.
                ["docs.aws.amazon.com"]. A root domain also matches its subdomains.
                Can only narrow the search, never widen it.
            exclude_domains: Drop results from these domains.
            published_after: Only return pages published on or after this date, as
                ISO-8601 UTC, e.g. 2026-01-01T00:00:00Z.
            published_before: Only return pages published on or before this date,
                as ISO-8601 UTC.

        Returns:
            One numbered block per result, carrying the title, URL, publication
            date and an extract.

        Raises:
            ValueError: If the tool has been closed, or an argument is outside the
                documented limits.
            WebSearchError: If the search fails.
        """
        with self._lock:
            if self._closed:
                raise ValueError("This AgentCoreWebSearch has been closed.")
            client = self._client

        try:
            response = client.search(
                query,
                max_results=max_results,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                published_after=published_after,
                published_before=published_before,
            )
        except Exception as exc:
            logger.error("Web search failed: %s - %s", type(exc).__name__, exc)
            raise

        logger.debug("Web search returned %d results", len(response.results))
        return _format_response(response)

    def close(self) -> None:
        """Release the underlying client, if this object created it."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_client:
                self._client.close()

    def __enter__(self) -> "AgentCoreWebSearch":
        """Enter the context manager.

        Returns:
            This object.
        """
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close on exit."""
        self.close()
