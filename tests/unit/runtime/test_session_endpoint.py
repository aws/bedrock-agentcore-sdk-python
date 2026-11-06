"""Tests for SessionEndpointClient, the signed client for a per-session endpoint URL."""

import datetime
import hashlib
import json
from unittest.mock import MagicMock, patch

import boto3
import pytest

from bedrock_agentcore.runtime.agent_core_runtime_client import AgentCoreRuntimeClient
from bedrock_agentcore.runtime.auth import OAuthAuth
from bedrock_agentcore.runtime.session_endpoint import (
    EMPTY_PAYLOAD_SHA256,
    PROXY_ERROR_HEADER,
    InvalidSessionEndpointError,
    SessionEndpointClient,
    SessionEndpointResponse,
    session_endpoint_url,
)

SESSION_HOST = "abcdefghijklmnopqrstuvwx.runtime.bedrock-agentcore.us-west-2.on.aws"


@pytest.fixture(autouse=True)
def _frozen_signing_clock():
    """Sign every request in these tests at the same instant, so signatures compare equal."""
    real = datetime.datetime
    fixed = real(2026, 10, 23, 12, 0, 0, tzinfo=datetime.timezone.utc)

    class _Fixed(real):
        @classmethod
        def now(cls, tz=None):
            return fixed

    with patch("bedrock_agentcore.runtime.agent_core_runtime_client.datetime.datetime", _Fixed):
        yield


SESSION_URL = f"https://{SESSION_HOST}"


def _session() -> boto3.Session:
    return boto3.Session(
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        aws_session_token="token",
        region_name="us-west-2",
    )


def _endpoint(http=None, auth="sigv4") -> SessionEndpointClient:
    return SessionEndpointClient(SESSION_URL, region="us-west-2", session=_session(), http=http, auth=auth)


def _fake_http(status=200, body=b"{}", headers=None) -> MagicMock:
    reply = MagicMock()
    reply.status = status
    reply.data = body
    reply.headers = headers or {}
    http = MagicMock()
    http.request.return_value = reply
    return http


# ── URL validation ────────────────────────────────────────────────────────────


class TestUrlValidation:
    def test_accepts_an_on_aws_host(self):
        assert _endpoint().host == SESSION_HOST

    def test_trailing_slash_is_dropped(self):
        endpoint = SessionEndpointClient(SESSION_URL + "/", region="us-west-2", session=_session())
        assert endpoint.url == SESSION_URL

    def test_rejects_a_non_aws_host(self):
        with pytest.raises(InvalidSessionEndpointError, match="non-AWS host"):
            SessionEndpointClient("https://example.com/", region="us-west-2", session=_session())

    def test_rejects_plain_http(self):
        with pytest.raises(InvalidSessionEndpointError, match="https"):
            SessionEndpointClient(f"http://{SESSION_HOST}", region="us-west-2", session=_session())


# ── sign ──────────────────────────────────────────────────────────────────────


class TestSign:
    def test_bodyless_request_gets_the_empty_payload_hash(self):
        _, headers = _endpoint().sign("GET", "/ping")
        assert headers["X-Amz-Content-SHA256"] == hashlib.sha256(b"").hexdigest()

    def test_a_body_is_still_signed_over_the_empty_payload_hash(self):
        # The canonical payload for session URL requests is the empty payload, whether
        # or not the forwarded request carries a body.
        _, headers = _endpoint().sign("POST", "/orders", body=b'{"customer": "Ada"}')
        assert headers["X-Amz-Content-SHA256"] == EMPTY_PAYLOAD_SHA256

    def test_the_body_does_not_change_the_signature(self):
        _, empty = _endpoint().sign("POST", "/orders", body=b"{}")
        _, full = _endpoint().sign("POST", "/orders", body=b'{"customer": "Ada", "items": ["widget"]}')
        assert empty["Authorization"] == full["Authorization"]

    def test_a_caller_supplied_empty_payload_hash_is_accepted(self):
        _, headers = _endpoint().sign(
            "POST", "/orders", body=b"{}", headers={"x-amz-content-sha256": EMPTY_PAYLOAD_SHA256}
        )
        assert headers["X-Amz-Content-SHA256"] == EMPTY_PAYLOAD_SHA256

    def test_a_caller_supplied_body_hash_is_rejected(self):
        body = b'{"customer": "Ada"}'
        with pytest.raises(ValueError, match="empty payload hash"):
            _endpoint().sign(
                "POST", "/orders", body=body, headers={"X-Amz-Content-SHA256": hashlib.sha256(body).hexdigest()}
            )

    def test_the_payload_hash_is_covered_by_the_signature(self):
        _, headers = _endpoint().sign("GET", "/ping")
        assert "x-amz-content-sha256" in headers["Authorization"]

    def test_signed_with_the_agentcore_signing_name_and_region(self):
        _, headers = _endpoint().sign("GET", "/ping")
        assert "/us-west-2/bedrock-agentcore/aws4_request" in headers["Authorization"]

    def test_security_token_is_sent_when_the_credentials_are_temporary(self):
        _, headers = _endpoint().sign("GET", "/ping")
        assert headers["X-Amz-Security-Token"] == "token"

    def test_a_body_defaults_to_json_content_type(self):
        _, headers = _endpoint().sign("PUT", "/artifacts/notes.md", body=b"{}")
        assert headers["Content-Type"] == "application/json"

    def test_a_caller_supplied_content_type_wins(self):
        _, headers = _endpoint().sign(
            "PUT", "/artifacts/notes.md", body=b"# Notes\n", headers={"Content-Type": "text/markdown"}
        )
        assert headers["Content-Type"] == "text/markdown"

    def test_the_default_port_is_not_prefixed(self):
        url, _ = _endpoint().sign("GET", "/ping")
        assert url == f"https://{SESSION_HOST}/ping"

    def test_another_port_travels_as_a_hostname_prefix(self):
        url, headers = _endpoint().sign("GET", "/status", port=8081)
        assert url == f"https://8081--{SESSION_HOST}/status"
        # The prefixed host is covered by the signature. botocore derives it from the
        # URL rather than setting a header, and the HTTP client sends the same value.
        assert "SignedHeaders=host;" in headers["Authorization"]

    def test_a_query_string_is_kept(self):
        url, _ = _endpoint().sign("GET", "/inspect?tenant=acme&page=2")
        assert url.endswith("/inspect?tenant=acme&page=2")

    def test_a_path_without_a_leading_slash_is_rejected(self):
        with pytest.raises(ValueError, match="must start with"):
            _endpoint().sign("GET", "ping")

    @pytest.mark.parametrize("port", [0, -1, 65536, 99999])
    def test_a_port_outside_the_valid_range_is_rejected(self, port):
        with pytest.raises(ValueError, match="between 1 and 65535"):
            _endpoint().sign("GET", "/status", port=port)

    def test_missing_credentials_raise(self):
        session = MagicMock()
        session.get_credentials.return_value = None
        session.region_name = "us-west-2"
        endpoint = SessionEndpointClient(SESSION_URL, region="us-west-2", session=session)
        with pytest.raises(RuntimeError, match="No AWS credentials"):
            endpoint.sign("GET", "/ping")


# ── sign_websocket ────────────────────────────────────────────────────────────


class TestSignWebsocket:
    def test_returns_a_wss_url_on_the_prefixed_host(self):
        url, _ = _endpoint().sign_websocket("/live", port=8090)
        assert url == f"wss://8090--{SESSION_HOST}/live"

    def test_the_handshake_is_signed_as_a_get(self):
        url, headers = _endpoint().sign_websocket("/live", port=8090)
        _, as_get = _endpoint().sign("GET", "/live", port=8090)
        assert headers["Authorization"] == as_get["Authorization"]
        assert headers["X-Amz-Content-SHA256"] == EMPTY_PAYLOAD_SHA256

    def test_no_host_header_for_the_websocket_library_to_duplicate(self):
        _, headers = _endpoint().sign_websocket("/live", port=8090)
        assert not any(name.lower() == "host" for name in headers)

    def test_a_bearer_token_handshake_carries_no_signature_or_host(self):
        _, headers = _endpoint(auth=OAuthAuth(bearer_token="jwt-value")).sign_websocket("/live")
        assert headers["Authorization"] == "Bearer jwt-value"
        assert "X-Amz-Content-SHA256" not in headers
        assert not any(name.lower() == "host" for name in headers)


# ── session_endpoint_url ──────────────────────────────────────────────────────


class TestSessionEndpointUrl:
    def test_reads_the_modeled_member(self):
        assert session_endpoint_url({"sessionEndpointUrl": SESSION_URL}) == SESSION_URL

    def test_falls_back_to_the_response_header(self):
        response = {
            "ResponseMetadata": {"HTTPHeaders": {"x-amzn-bedrock-agentcore-runtime-session-endpoint-url": SESSION_URL}}
        }
        assert session_endpoint_url(response) == SESSION_URL

    def test_the_header_is_matched_case_insensitively(self):
        response = {
            "ResponseMetadata": {"HTTPHeaders": {"X-Amzn-Bedrock-AgentCore-Runtime-Session-Endpoint-Url": SESSION_URL}}
        }
        assert session_endpoint_url(response) == SESSION_URL

    def test_none_when_remote_access_is_not_enabled(self):
        assert session_endpoint_url({"ResponseMetadata": {"HTTPHeaders": {}}}) is None


# ── request ───────────────────────────────────────────────────────────────────


class TestRequest:
    def test_returns_status_headers_and_body(self):
        http = _fake_http(status=201, body=b'{"stored":"notes.md"}')
        response = _endpoint(http).request("PUT", "/artifacts/notes.md", body="# Notes\n")
        assert response.status == 201
        assert response.json() == {"stored": "notes.md"}

    def test_a_string_body_is_encoded_and_sent_with_the_empty_payload_hash(self):
        http = _fake_http()
        _endpoint(http).request("PUT", "/artifacts/notes.md", body="# Notes\n")
        sent = http.request.call_args
        assert sent.kwargs["body"] == b"# Notes\n"
        assert sent.kwargs["headers"]["X-Amz-Content-SHA256"] == EMPTY_PAYLOAD_SHA256

    def test_a_bodyless_request_says_its_length_is_zero(self):
        http = _fake_http()
        _endpoint(http).request("GET", "/")
        assert http.request.call_args.kwargs["headers"]["Content-Length"] == "0"

    def test_the_method_is_upper_cased(self):
        http = _fake_http()
        _endpoint(http).request("delete", "/artifacts/notes.md")
        assert http.request.call_args.args[0] == "DELETE"

    def test_redirects_are_not_followed(self):
        http = _fake_http()
        _endpoint(http).request("GET", "/ping")
        assert http.request.call_args.kwargs["redirect"] is False


# ── SessionEndpointResponse ───────────────────────────────────────────────────


class TestSessionEndpointResponse:
    def test_a_container_response_has_no_proxy_error(self):
        response = SessionEndpointResponse(status=404, headers={"Content-Type": "text/plain"}, data=b"nope")
        assert response.proxy_error is None
        assert response.reached_container is True

    def test_a_proxy_refusal_is_read_from_the_header(self):
        response = SessionEndpointResponse(status=403, headers={PROXY_ERROR_HEADER: "AccessDenied"})
        assert response.proxy_error == "AccessDenied"
        assert response.reached_container is False

    def test_an_allowed_port_with_nothing_listening_is_a_proxy_error(self):
        response = SessionEndpointResponse(status=502, headers={PROXY_ERROR_HEADER: "PortNotReachable"})
        assert response.proxy_error == "PortNotReachable"
        assert response.reached_container is False

    def test_the_header_name_seen_on_the_beta_stack_is_also_read(self):
        response = SessionEndpointResponse(
            status=403, headers={"x-amzn-bedrock-agentcore-runtime-error": "AccessDenied"}
        )
        assert response.proxy_error == "AccessDenied"

    def test_the_proxy_error_header_is_matched_case_insensitively(self):
        response = SessionEndpointResponse(
            status=403, headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Proxy-Error": "AccessDenied"}
        )
        assert response.proxy_error == "AccessDenied"

    def test_json_parses_the_body(self):
        assert SessionEndpointResponse(status=200, data=json.dumps({"a": 1}).encode()).json() == {"a": 1}

    def test_text_decodes_utf8(self):
        assert SessionEndpointResponse(status=200, data="café".encode()).text == "café"

    def test_a_binary_body_is_kept_as_bytes(self):
        png = b"\x89PNG\r\n\x1a\n\x00\xff\xfe"
        http = _fake_http(body=png, headers={"Content-Type": "image/png"})
        assert _endpoint(http).request("GET", "/logo.png").data == png


# ── the client factory ────────────────────────────────────────────────────────


class TestClientFactory:
    def test_session_endpoint_inherits_the_client_region(self):
        client = AgentCoreRuntimeClient(region="us-west-2", session=_session())
        endpoint = client.session_endpoint_client(SESSION_URL)
        assert isinstance(endpoint, SessionEndpointClient)
        assert endpoint.region == "us-west-2"

    def test_session_endpoint_rejects_a_non_aws_host(self):
        client = AgentCoreRuntimeClient(region="us-west-2", session=_session())
        with pytest.raises(InvalidSessionEndpointError):
            client.session_endpoint_client("https://example.com/")


# ── transport ─────────────────────────────────────────────────────────────────


class TestTransport:
    def test_the_default_pool_does_not_retry_or_follow_redirects(self):
        retries = _endpoint()._http.connection_pool_kw.get("retries")
        assert retries.total == 0
        assert retries.redirect == 0

    def test_close_clears_the_pool_it_created(self):
        endpoint = _endpoint()
        endpoint._http = MagicMock()
        endpoint.close()
        endpoint._http.clear.assert_called_once()

    def test_close_leaves_a_pool_passed_in_by_the_caller(self):
        http = _fake_http()
        with _endpoint(http):
            pass
        http.clear.assert_not_called()


# ── URL validation ────────────────────────────────────────────────────────────


class TestUrlValidationDomains:
    def test_the_service_domains_are_accepted_too(self):
        url = "https://abc.bedrock-agentcore.us-west-2.amazonaws.com"
        assert SessionEndpointClient(url, region="us-west-2", session=_session()).host == url[len("https://") :]

    def test_a_lookalike_domain_is_refused(self):
        with pytest.raises(InvalidSessionEndpointError):
            SessionEndpointClient("https://abc.on.aws.example.com", region="us-west-2", session=_session())
