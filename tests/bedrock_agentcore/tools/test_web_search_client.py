"""Tests for WebSearchClient."""

import json
from unittest.mock import MagicMock

import pytest

from bedrock_agentcore._utils.endpoints import InvalidGatewayIdentifierError, InvalidRegionError
from bedrock_agentcore.tools.web_search_client import (
    GatewayMcpBackend,
    WebSearchBackend,
    WebSearchClient,
    WebSearchError,
    WebSearchResponse,
    WebSearchResult,
    _build_arguments,
    _decode_jsonrpc,
    _parse_gateway_arn,
)

ENDPOINT = "https://gw-abc123.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"

SEARCH_PAYLOAD = {
    "id": "search-1",
    "results": [
        {
            "text": "urllib3 is a HTTP client for Python.",
            "url": "https://urllib3.readthedocs.io/",
            "title": "urllib3 docs",
            "publishedDate": "2026-01-05T00:00:00Z",
        },
        {"text": "Only text is required."},
    ],
}


def _http_response(status=200, body=b"", headers=None, content_type="application/json"):
    response = MagicMock()
    response.status = status
    response.data = body
    merged = {"Content-Type": content_type}
    merged.update(headers or {})
    response.headers = merged
    return response


def _json_rpc_response(result, request_id=1, **kwargs):
    body = json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}).encode()
    return _http_response(body=body, **kwargs)


def _initialize_response():
    return _json_rpc_response(
        {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "gateway"}},
        headers={"Mcp-Session-Id": "sess-1"},
    )


def _tools_call_response(payload=None):
    return _json_rpc_response(
        {"content": [{"type": "text", "text": json.dumps(payload if payload is not None else SEARCH_PAYLOAD)}]},
        request_id=2,
    )


def _make_backend(responses, **kwargs):
    """Build a backend whose HTTP layer replays the given responses in order."""
    session = MagicMock()
    session.get_credentials.return_value.get_frozen_credentials.return_value = _frozen_credentials()

    kwargs.setdefault("tool_name", "amazon-web-search___WebSearch")
    backend = GatewayMcpBackend(endpoint=ENDPOINT, region="us-east-1", boto3_session=session, **kwargs)
    backend._http = MagicMock()
    backend._http.request.side_effect = list(responses)
    return backend


def _frozen_credentials():
    from botocore.credentials import ReadOnlyCredentials

    return ReadOnlyCredentials("AKIAEXAMPLE", "secret", None)


class TestBuildArguments:
    """Tests for input validation and argument shaping."""

    def test_minimal(self):
        assert _build_arguments("hello") == {"query": "hello"}

    def test_all_options(self):
        arguments = _build_arguments(
            "hello",
            max_results=5,
            include_domains=["a.example"],
            exclude_domains=["b.example"],
            published_after="2026-01-01T00:00:00Z",
            published_before="2026-06-01T00:00:00Z",
        )
        assert arguments == {
            "query": "hello",
            "maxResults": 5,
            "filters": {
                "domainFilter": {"include": ["a.example"], "exclude": ["b.example"]},
                "publishedDateFilter": {"from": "2026-01-01T00:00:00Z", "to": "2026-06-01T00:00:00Z"},
            },
        }

    @pytest.mark.parametrize("query", ["", "   "])
    def test_empty_query_rejected(self, query):
        with pytest.raises(ValueError, match="non-empty"):
            _build_arguments(query)

    def test_query_length_limit(self):
        _build_arguments("x" * 200)
        with pytest.raises(ValueError, match="200 characters or fewer"):
            _build_arguments("x" * 201)

    @pytest.mark.parametrize("max_results", [0, 26, -1])
    def test_max_results_range(self, max_results):
        with pytest.raises(ValueError, match="between 1 and 25"):
            _build_arguments("hello", max_results=max_results)

    @pytest.mark.parametrize("max_results", [1, 25])
    def test_max_results_boundaries_allowed(self, max_results):
        assert _build_arguments("hello", max_results=max_results)["maxResults"] == max_results

    @pytest.mark.parametrize("max_results", ["5", 5.0, True])
    def test_max_results_must_be_int(self, max_results):
        with pytest.raises(ValueError, match="must be an integer"):
            _build_arguments("hello", max_results=max_results)

    def test_empty_filter_lists_omitted(self):
        assert "filters" not in _build_arguments("hello", include_domains=[], exclude_domains=[])

    def test_only_one_date_bound(self):
        arguments = _build_arguments("hello", published_after="2026-01-01T00:00:00Z")
        assert arguments["filters"] == {"publishedDateFilter": {"from": "2026-01-01T00:00:00Z"}}


class TestResponseParsing:
    """Tests for turning the tool payload into result objects."""

    def test_from_payload(self):
        response = WebSearchResponse.from_payload(SEARCH_PAYLOAD)
        assert response.search_id == "search-1"
        assert len(response) == 2
        first = response.results[0]
        assert first.title == "urllib3 docs"
        assert first.url == "https://urllib3.readthedocs.io/"
        assert first.published_date == "2026-01-05T00:00:00Z"

    def test_optional_fields_default_to_none(self):
        result = WebSearchResponse.from_payload(SEARCH_PAYLOAD).results[1]
        assert result.text == "Only text is required."
        assert result.url is None
        assert result.title is None
        assert result.published_date is None

    def test_empty_payload(self):
        response = WebSearchResponse.from_payload({})
        assert len(response) == 0
        assert response.search_id is None

    def test_non_dict_entries_skipped(self):
        response = WebSearchResponse.from_payload({"results": ["nope", {"text": "yes"}]})
        assert [r.text for r in response] == ["yes"]

    def test_missing_text_becomes_empty_string(self):
        assert WebSearchResult.from_payload({"url": "https://example.com"}).text == ""

    def test_iterable(self):
        assert [r.text for r in WebSearchResponse.from_payload(SEARCH_PAYLOAD)][0].startswith("urllib3")


class TestDecodeJsonRpc:
    """Tests for both response framings the transport allows."""

    def test_json_body(self):
        message = _decode_jsonrpc("application/json", b'{"jsonrpc":"2.0","id":1,"result":{}}')
        assert message["id"] == 1

    def test_event_stream_body(self):
        body = b'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'
        message = _decode_jsonrpc("text/event-stream", body)
        assert message["result"] == {"ok": True}

    def test_event_stream_skips_non_data_and_undecodable_lines(self):
        body = b': ping\nid: 7\ndata: not json\ndata: {"jsonrpc":"2.0","id":1,"result":{}}\n'
        assert _decode_jsonrpc("text/event-stream", body)["id"] == 1

    def test_event_stream_without_message_returns_none(self):
        assert _decode_jsonrpc("text/event-stream", b"event: ping\ndata: \n\n") is None

    def test_undecodable_json_raises(self):
        with pytest.raises(WebSearchError, match="Could not decode gateway response"):
            _decode_jsonrpc("application/json", b"<html>gateway error</html>")

    def test_non_object_json_returns_none(self):
        assert _decode_jsonrpc("application/json", b"[1, 2]") is None


class TestGatewayMcpBackendHandshake:
    """Tests for the MCP request sequence and its signed headers."""

    def test_initialize_then_notify_then_call(self):
        backend = _make_backend([_initialize_response(), _http_response(status=202, body=b""), _tools_call_response()])

        payload = backend.search({"query": "hello"})

        assert payload == SEARCH_PAYLOAD
        methods = [json.loads(call.kwargs["body"])["method"] for call in backend._http.request.call_args_list]
        assert methods == ["initialize", "notifications/initialized", "tools/call"]

    def test_session_reused_across_searches(self):
        backend = _make_backend(
            [
                _initialize_response(),
                _http_response(status=202, body=b""),
                _tools_call_response(),
                _tools_call_response(),
            ]
        )

        backend.search({"query": "one"})
        backend.search({"query": "two"})

        methods = [json.loads(call.kwargs["body"])["method"] for call in backend._http.request.call_args_list]
        assert methods == ["initialize", "notifications/initialized", "tools/call", "tools/call"]

    def test_notification_carries_no_id(self):
        backend = _make_backend([_initialize_response(), _http_response(status=202, body=b""), _tools_call_response()])
        backend.search({"query": "hello"})

        notification = json.loads(backend._http.request.call_args_list[1].kwargs["body"])
        assert "id" not in notification

    def test_session_id_and_protocol_version_sent_after_initialize(self):
        backend = _make_backend([_initialize_response(), _http_response(status=202, body=b""), _tools_call_response()])
        backend.search({"query": "hello"})

        initialize_headers = backend._http.request.call_args_list[0].kwargs["headers"]
        assert "Mcp-Session-Id" not in initialize_headers
        assert "MCP-Protocol-Version" not in initialize_headers

        call_headers = backend._http.request.call_args_list[2].kwargs["headers"]
        assert call_headers["Mcp-Session-Id"] == "sess-1"
        assert call_headers["MCP-Protocol-Version"] == "2025-06-18"

    def test_negotiated_protocol_version_is_echoed_back(self):
        negotiated = _json_rpc_response({"protocolVersion": "2025-03-26"}, headers={"Mcp-Session-Id": "sess-1"})
        backend = _make_backend([negotiated, _http_response(status=202, body=b""), _tools_call_response()])

        backend.search({"query": "hello"})

        call_headers = backend._http.request.call_args_list[2].kwargs["headers"]
        assert call_headers["MCP-Protocol-Version"] == "2025-03-26"

    def test_requests_are_sigv4_signed(self):
        backend = _make_backend([_initialize_response(), _http_response(status=202, body=b""), _tools_call_response()])
        backend.search({"query": "hello"})

        headers = backend._http.request.call_args_list[2].kwargs["headers"]
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/")
        assert "/us-east-1/bedrock-agentcore/aws4_request" in headers["Authorization"]
        assert "X-Amz-Date" in headers
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json, text/event-stream"
        assert headers["Content-Length"] == str(len(backend._http.request.call_args_list[2].kwargs["body"]))

    def test_connection_header_is_never_signed(self):
        backend = _make_backend([_initialize_response(), _http_response(status=202, body=b""), _tools_call_response()])
        backend.search({"query": "hello"})

        for call in backend._http.request.call_args_list:
            signed = call.kwargs["headers"]["Authorization"].split("SignedHeaders=")[1].split(",")[0]
            assert "connection" not in signed

    def test_user_agent_reports_the_sdk(self):
        backend = _make_backend(
            [_initialize_response(), _http_response(status=202, body=b""), _tools_call_response()],
            integration_source="langchain",
        )
        backend.search({"query": "hello"})

        user_agent = backend._http.request.call_args_list[0].kwargs["headers"]["User-Agent"]
        assert "bedrock-agentcore/" in user_agent
        assert "integration_source=langchain" in user_agent

    def test_no_credentials_raises(self):
        session = MagicMock()
        session.get_credentials.return_value = None
        backend = GatewayMcpBackend(endpoint=ENDPOINT, region="us-east-1", boto3_session=session, tool_name="t")
        backend._http = MagicMock()

        with pytest.raises(WebSearchError, match="No AWS credentials"):
            backend.search({"query": "hello"})

    def test_keepalive_only_notification_reply_is_tolerated(self):
        """A body carrying no JSON-RPC message is not an error, it is an ack."""
        keepalive = _http_response(status=202, body=b"event: ping\ndata: \n\n", content_type="text/event-stream")
        backend = _make_backend([_initialize_response(), keepalive, _tools_call_response()])

        assert backend.search({"query": "hello"}) == SEARCH_PAYLOAD

    def test_event_stream_tools_call_is_parsed(self):
        sse_body = (
            b"event: message\ndata: "
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {"content": [{"type": "text", "text": json.dumps(SEARCH_PAYLOAD)}]},
                }
            ).encode()
            + b"\n\n"
        )
        backend = _make_backend(
            [
                _initialize_response(),
                _http_response(status=202, body=b""),
                _http_response(body=sse_body, content_type="text/event-stream"),
            ]
        )

        assert backend.search({"query": "hello"}) == SEARCH_PAYLOAD

    def test_close_resets_the_session(self):
        backend = _make_backend([_initialize_response(), _http_response(status=202, body=b""), _tools_call_response()])
        backend.search({"query": "hello"})

        backend.close()

        assert backend._initialized is False
        assert backend._mcp_session_id is None
        backend._http.clear.assert_called_once()


class TestGatewayMcpBackendErrors:
    """Tests for the failure paths."""

    def test_http_error_is_surfaced(self):
        backend = _make_backend([_http_response(status=403, body=b"not authorized")])

        with pytest.raises(WebSearchError, match="HTTP 403"):
            backend.search({"query": "hello"})

    def test_json_rpc_error_is_surfaced(self):
        error_body = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "Unknown tool"}}
        ).encode()
        backend = _make_backend([_initialize_response(), _http_response(status=202), _http_response(body=error_body)])

        with pytest.raises(WebSearchError, match="Unknown tool"):
            backend.search({"query": "hello"})

    def test_tool_error_flag_is_surfaced(self):
        error_result = _json_rpc_response(
            {"isError": True, "content": [{"type": "text", "text": "query too long"}]}, request_id=2
        )
        backend = _make_backend([_initialize_response(), _http_response(status=202), error_result])

        with pytest.raises(WebSearchError, match="query too long"):
            backend.search({"query": "hello"})

    def test_missing_text_content_is_surfaced(self):
        backend = _make_backend(
            [_initialize_response(), _http_response(status=202), _json_rpc_response({"content": []}, request_id=2)]
        )

        with pytest.raises(WebSearchError, match="no text content"):
            backend.search({"query": "hello"})

    def test_undecodable_tool_payload_is_surfaced(self):
        bad = _json_rpc_response({"content": [{"type": "text", "text": "not json"}]}, request_id=2)
        backend = _make_backend([_initialize_response(), _http_response(status=202), bad])

        with pytest.raises(WebSearchError, match="Could not decode web search response"):
            backend.search({"query": "hello"})

    def test_non_object_tool_payload_is_surfaced(self):
        bad = _json_rpc_response({"content": [{"type": "text", "text": "[1,2]"}]}, request_id=2)
        backend = _make_backend([_initialize_response(), _http_response(status=202), bad])

        with pytest.raises(WebSearchError, match="Expected a JSON object"):
            backend.search({"query": "hello"})

    def test_empty_initialize_reply_is_surfaced(self):
        backend = _make_backend([_http_response(status=202, body=b"")])

        with pytest.raises(WebSearchError, match="did not answer the MCP initialize"):
            backend.search({"query": "hello"})

    def test_empty_tools_call_reply_is_surfaced(self):
        backend = _make_backend(
            [_initialize_response(), _http_response(status=202), _http_response(status=202, body=b"")]
        )

        with pytest.raises(WebSearchError, match="did not answer the web search tool call"):
            backend.search({"query": "hello"})


class TestToolNameResolution:
    """Tests for finding the fully qualified tool name."""

    def test_target_name_derives_the_prefixed_name(self):
        backend = _make_backend(
            [_initialize_response(), _http_response(status=202), _tools_call_response()],
            tool_name=None,
            target_name="amazon-web-search",
        )
        backend.search({"query": "hello"})

        params = json.loads(backend._http.request.call_args_list[2].kwargs["body"])["params"]
        assert params["name"] == "amazon-web-search___WebSearch"
        methods = [json.loads(c.kwargs["body"])["method"] for c in backend._http.request.call_args_list]
        assert "tools/list" not in methods

    def test_explicit_tool_name_skips_discovery(self):
        backend = _make_backend(
            [_initialize_response(), _http_response(status=202), _tools_call_response()],
            tool_name="custom___WebSearch",
        )
        backend.search({"query": "hello"})

        params = json.loads(backend._http.request.call_args_list[2].kwargs["body"])["params"]
        assert params["name"] == "custom___WebSearch"

    def test_discovery_picks_the_prefixed_tool(self):
        tools_list = _json_rpc_response(
            {"tools": [{"name": "other___Lookup"}, {"name": "amazon-web-search___WebSearch"}]}
        )
        backend = _make_backend(
            [_initialize_response(), _http_response(status=202), tools_list, _tools_call_response()],
            tool_name=None,
        )
        backend.search({"query": "hello"})

        params = json.loads(backend._http.request.call_args_list[3].kwargs["body"])["params"]
        assert params["name"] == "amazon-web-search___WebSearch"

    def test_discovery_accepts_an_unprefixed_tool(self):
        tools_list = _json_rpc_response({"tools": [{"name": "WebSearch"}]})
        backend = _make_backend(
            [_initialize_response(), _http_response(status=202), tools_list, _tools_call_response()],
            tool_name=None,
        )
        backend.search({"query": "hello"})

        params = json.loads(backend._http.request.call_args_list[3].kwargs["body"])["params"]
        assert params["name"] == "WebSearch"

    def test_discovery_follows_pagination(self):
        page_one = _json_rpc_response({"tools": [{"name": "other___Lookup"}], "nextCursor": "c1"})
        page_two = _json_rpc_response({"tools": [{"name": "amazon-web-search___WebSearch"}]})
        backend = _make_backend(
            [_initialize_response(), _http_response(status=202), page_one, page_two, _tools_call_response()],
            tool_name=None,
        )
        backend.search({"query": "hello"})

        second_page = json.loads(backend._http.request.call_args_list[3].kwargs["body"])
        assert second_page["params"] == {"cursor": "c1"}
        assert json.loads(backend._http.request.call_args_list[4].kwargs["body"])["params"]["name"] == (
            "amazon-web-search___WebSearch"
        )

    def test_discovery_resolves_once_and_is_cached(self):
        tools_list = _json_rpc_response({"tools": [{"name": "amazon-web-search___WebSearch"}]})
        backend = _make_backend(
            [
                _initialize_response(),
                _http_response(status=202),
                tools_list,
                _tools_call_response(),
                _tools_call_response(),
            ],
            tool_name=None,
        )

        backend.search({"query": "one"})
        backend.search({"query": "two"})

        methods = [json.loads(c.kwargs["body"])["method"] for c in backend._http.request.call_args_list]
        assert methods.count("tools/list") == 1

    def test_no_web_search_tool_raises(self):
        tools_list = _json_rpc_response({"tools": [{"name": "other___Lookup"}]})
        backend = _make_backend([_initialize_response(), _http_response(status=202), tools_list], tool_name=None)

        with pytest.raises(WebSearchError, match="No WebSearch tool found"):
            backend.search({"query": "hello"})

    def test_ambiguous_web_search_tools_raise(self):
        tools_list = _json_rpc_response({"tools": [{"name": "a___WebSearch"}, {"name": "b___WebSearch"}]})
        backend = _make_backend([_initialize_response(), _http_response(status=202), tools_list], tool_name=None)

        with pytest.raises(WebSearchError, match="more than one WebSearch tool"):
            backend.search({"query": "hello"})


class TestParseGatewayArn:
    """Tests for reading a gateway ID and region out of an ARN."""

    def test_valid_arn(self):
        gateway_id, region = _parse_gateway_arn("arn:aws:bedrock-agentcore:eu-west-1:123456789012:gateway/gw-abc123")
        assert (gateway_id, region) == ("gw-abc123", "eu-west-1")

    @pytest.mark.parametrize(
        "arn",
        [
            "gw-abc123",
            "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/r-1",
            "not:an:arn:at:all:gateway/gw-1",
        ],
    )
    def test_invalid_arn(self, arn):
        with pytest.raises(ValueError, match="gateway ARN"):
            _parse_gateway_arn(arn)

    def test_arn_without_id(self):
        with pytest.raises(ValueError, match="no gateway ID"):
            _parse_gateway_arn("arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/")


class _RecordingBackend(WebSearchBackend):
    """A backend that records the arguments it was asked to search with."""

    def __init__(self, payload=None):
        self.payload = payload if payload is not None else SEARCH_PAYLOAD
        self.arguments = None
        self.closed = False

    def search(self, arguments):
        self.arguments = arguments
        return self.payload

    def close(self):
        self.closed = True


class TestWebSearchClient:
    """Tests for the client surface."""

    def test_search_returns_results(self):
        backend = _RecordingBackend()
        client = WebSearchClient(region="us-east-1", backend=backend)

        response = client.search("who maintains urllib3", max_results=2)

        assert backend.arguments == {"query": "who maintains urllib3", "maxResults": 2}
        assert len(response) == 2
        assert response.results[0].title == "urllib3 docs"

    def test_search_passes_filters_through(self):
        backend = _RecordingBackend()
        client = WebSearchClient(region="us-east-1", backend=backend)

        client.search(
            "agentcore",
            include_domains=["docs.aws.amazon.com"],
            exclude_domains=["spam.example"],
            published_after="2026-01-01T00:00:00Z",
        )

        assert backend.arguments["filters"] == {
            "domainFilter": {"include": ["docs.aws.amazon.com"], "exclude": ["spam.example"]},
            "publishedDateFilter": {"from": "2026-01-01T00:00:00Z"},
        }

    def test_validation_happens_before_the_call(self):
        backend = _RecordingBackend()
        client = WebSearchClient(region="us-east-1", backend=backend)

        with pytest.raises(ValueError):
            client.search("x" * 201)

        assert backend.arguments is None

    def test_gateway_id_builds_the_endpoint(self):
        client = WebSearchClient(region="us-east-1", gateway_id="gw-abc123", boto3_session=MagicMock())

        assert client.backend._endpoint == ENDPOINT
        assert client.region == "us-east-1"

    def test_gateway_arn_supplies_the_region(self):
        client = WebSearchClient(
            gateway_arn="arn:aws:bedrock-agentcore:eu-west-1:123456789012:gateway/gw-abc123",
            boto3_session=MagicMock(),
        )

        assert client.region == "eu-west-1"
        assert client.backend._endpoint.startswith("https://gw-abc123.gateway.bedrock-agentcore.eu-west-1.")

    def test_explicit_region_wins_over_the_arn(self):
        client = WebSearchClient(
            region="us-east-1",
            gateway_arn="arn:aws:bedrock-agentcore:eu-west-1:123456789012:gateway/gw-abc123",
            boto3_session=MagicMock(),
        )

        assert client.region == "us-east-1"

    def test_gateway_endpoint_used_as_given(self):
        endpoint = "https://gw-abc123.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
        client = WebSearchClient(region="us-east-1", gateway_endpoint=endpoint, boto3_session=MagicMock())

        assert client.backend._endpoint == endpoint

    def test_region_from_the_session(self):
        session = MagicMock()
        session.region_name = "us-east-1"
        client = WebSearchClient(gateway_id="gw-abc123", boto3_session=session)

        assert client.region == "us-east-1"

    def test_missing_region_raises(self):
        session = MagicMock()
        session.region_name = None

        with pytest.raises(ValueError, match="region could not be determined"):
            WebSearchClient(gateway_id="gw-abc123", boto3_session=session)

    def test_unknown_region_warns_but_proceeds(self, caplog):
        with caplog.at_level("WARNING"):
            client = WebSearchClient(region="us-west-2", gateway_id="gw-abc123", boto3_session=MagicMock())

        assert client.region == "us-west-2"
        assert "us-east-1, eu-west-1, ap-northeast-1" in caplog.text

    def test_no_gateway_raises(self):
        with pytest.raises(ValueError, match="is required"):
            WebSearchClient(region="us-east-1", boto3_session=MagicMock())

    def test_two_gateways_raise(self):
        with pytest.raises(ValueError, match="only one of"):
            WebSearchClient(
                region="us-east-1",
                gateway_id="gw-abc123",
                gateway_endpoint=ENDPOINT,
                boto3_session=MagicMock(),
            )

    def test_backend_and_gateway_together_raise(self):
        with pytest.raises(ValueError, match="not both"):
            WebSearchClient(region="us-east-1", gateway_id="gw-abc123", backend=_RecordingBackend())

    def test_invalid_gateway_id_raises(self):
        with pytest.raises(InvalidGatewayIdentifierError):
            WebSearchClient(region="us-east-1", gateway_id="evil.example.com/", boto3_session=MagicMock())

    def test_invalid_region_raises(self):
        with pytest.raises(InvalidRegionError):
            WebSearchClient(region="not a region", gateway_id="gw-abc123", boto3_session=MagicMock())

    def test_close_only_closes_an_owned_backend(self):
        backend = _RecordingBackend()
        WebSearchClient(region="us-east-1", backend=backend).close()

        assert backend.closed is False

    def test_context_manager_closes_an_owned_backend(self):
        with WebSearchClient(region="us-east-1", gateway_id="gw-abc123", boto3_session=MagicMock()) as client:
            client.backend._http = MagicMock()
            http = client.backend._http

        http.clear.assert_called_once()

    def test_target_name_is_handed_to_the_backend(self):
        client = WebSearchClient(
            region="us-east-1",
            gateway_id="gw-abc123",
            target_name="amazon-web-search",
            boto3_session=MagicMock(),
        )

        assert client.backend._target_name == "amazon-web-search"


class TestBackendProtocol:
    """Tests for the extension point."""

    def test_base_search_is_not_implemented(self):
        with pytest.raises(NotImplementedError):
            WebSearchBackend().search({"query": "hello"})

    def test_base_close_is_a_no_op(self):
        assert WebSearchBackend().close() is None


def test_exported_from_the_tools_package():
    from bedrock_agentcore.tools import WebSearchClient as exported

    assert exported is WebSearchClient
