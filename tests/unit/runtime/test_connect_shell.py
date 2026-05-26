"""Tests for the connect_shell* auth helpers on AgentCoreRuntimeClient."""

from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, quote, urlparse

import pytest

from bedrock_agentcore.runtime.agent_core_runtime_client import AgentCoreRuntimeClient, validate_shell_id

FAKE_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime"
ENCODED_ARN = quote(FAKE_ARN, safe="")


def _client() -> AgentCoreRuntimeClient:
    return AgentCoreRuntimeClient(region="us-west-2")


def _mock_credentials(token: str = "") -> MagicMock:
    creds = MagicMock()
    frozen = MagicMock()
    frozen.token = token
    creds.get_frozen_credentials.return_value = frozen
    return creds


# ── _build_shell_url ──────────────────────────────────────────────────────────


class TestBuildShellUrl:
    def test_path_contains_ws_commands(self):
        client = _client()
        url = client._build_shell_url(FAKE_ARN)
        assert "/ws/commands" in url
        assert url.startswith("wss://")

    def test_arn_is_url_encoded(self):
        client = _client()
        url = client._build_shell_url(FAKE_ARN)
        assert ENCODED_ARN in url

    def test_qualifier_param(self):
        client = _client()
        url = client._build_shell_url(FAKE_ARN, endpoint_name="prod")
        qs = parse_qs(urlparse(url).query)
        assert qs["qualifier"] == ["prod"]

    def test_shell_id_maps_to_command_session_id_param(self):
        """shell_id must be sent as commandSessionId (wire format unchanged)."""
        client = _client()
        url = client._build_shell_url(FAKE_ARN, shell_id="my-shell")
        qs = parse_qs(urlparse(url).query)
        assert qs["commandSessionId"] == ["my-shell"]

    def test_no_params_no_query_string(self):
        client = _client()
        url = client._build_shell_url(FAKE_ARN)
        assert "?" not in url


# ── validate_shell_id ─────────────────────────────────────────────────────────


class TestValidateShellId:
    def test_valid_simple(self):
        validate_shell_id("my-shell")

    def test_valid_single_char(self):
        validate_shell_id("x")

    def test_valid_128_chars(self):
        validate_shell_id("a" * 128)

    def test_valid_with_underscore(self):
        validate_shell_id("debug_session_01")

    def test_valid_alphanumeric_start(self):
        validate_shell_id("abc123")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            validate_shell_id("")

    def test_too_long_raises(self):
        with pytest.raises(ValueError):
            validate_shell_id("a" * 129)

    def test_starts_with_hyphen_raises(self):
        with pytest.raises(ValueError):
            validate_shell_id("-shell")

    def test_starts_with_underscore_raises(self):
        with pytest.raises(ValueError):
            validate_shell_id("_shell")

    @pytest.mark.parametrize("char", ["?", "#", "&", ".", "/", " "])
    def test_forbidden_char_raises(self, char):
        with pytest.raises(ValueError):
            validate_shell_id(f"shell{char}bad")

    @pytest.mark.parametrize("bad", [0, False, True, b"foo", 1.5, [], {}])
    def test_non_string_raises_value_error(self, bad):
        with pytest.raises(ValueError, match="must be str"):
            validate_shell_id(bad)

    def test_trailing_newline_raises(self):
        with pytest.raises(ValueError):
            validate_shell_id("my-shell\n")


# ── connect_shell ─────────────────────────────────────────────────────────────


class TestConnectShell:
    def test_returns_wss_url_and_headers(self):
        client = _client()
        with patch.object(client.session, "get_credentials", return_value=_mock_credentials()):
            with patch("bedrock_agentcore.runtime.agent_core_runtime_client.SigV4Auth") as mock_auth:
                mock_auth.return_value.add_auth = MagicMock()
                url, headers = client.connect_shell(FAKE_ARN)

        assert url.startswith("wss://")
        assert "/ws/commands" in url
        assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id" in headers
        assert "Authorization" in headers

    def test_uses_provided_shell_id(self):
        client = _client()
        with patch.object(client.session, "get_credentials", return_value=_mock_credentials()):
            with patch("bedrock_agentcore.runtime.agent_core_runtime_client.SigV4Auth") as mock_auth:
                mock_auth.return_value.add_auth = MagicMock()
                url, _ = client.connect_shell(FAKE_ARN, shell_id="my-shell")

        qs = parse_qs(urlparse(url).query)
        assert qs["commandSessionId"] == ["my-shell"]

    def test_autogenerates_shell_id_when_absent(self):
        client = _client()
        with patch.object(client.session, "get_credentials", return_value=_mock_credentials()):
            with patch("bedrock_agentcore.runtime.agent_core_runtime_client.SigV4Auth") as mock_auth:
                mock_auth.return_value.add_auth = MagicMock()
                url, _ = client.connect_shell(FAKE_ARN)

        qs = parse_qs(urlparse(url).query)
        assert "commandSessionId" in qs

    def test_includes_security_token_when_present(self):
        client = _client()
        creds = _mock_credentials(token="SESSION-TOKEN")
        with patch.object(client.session, "get_credentials", return_value=creds):
            with patch("bedrock_agentcore.runtime.agent_core_runtime_client.SigV4Auth") as mock_auth:
                mock_auth.return_value.add_auth = MagicMock()
                _, headers = client.connect_shell(FAKE_ARN)

        assert headers.get("X-Amz-Security-Token") == "SESSION-TOKEN"

    def test_invalid_arn_raises(self):
        client = _client()
        with pytest.raises(ValueError):
            client.connect_shell("not-an-arn")

    def test_invalid_shell_id_raises(self):
        client = _client()
        with pytest.raises(ValueError):
            client.connect_shell(FAKE_ARN, shell_id="bad?id")

    def test_no_credentials_raises(self):
        client = _client()
        with patch.object(client.session, "get_credentials", return_value=None):
            with pytest.raises(RuntimeError, match="No AWS credentials"):
                client.connect_shell(FAKE_ARN)

    def test_user_agent_includes_integration_source(self):
        from bedrock_agentcore.runtime.agent_core_runtime_client import AgentCoreRuntimeClient

        client = AgentCoreRuntimeClient(region="us-west-2", integration_source="mycli")
        with patch.object(client.session, "get_credentials", return_value=_mock_credentials()):
            with patch("bedrock_agentcore.runtime.agent_core_runtime_client.SigV4Auth") as mock_auth:
                mock_auth.return_value.add_auth = MagicMock()
                _, headers = client.connect_shell(FAKE_ARN)

        assert "mycli" in headers.get("User-Agent", "")


# ── connect_shell_presigned ───────────────────────────────────────────────────


class TestConnectShellPresigned:
    def test_returns_presigned_wss_url(self):
        client = _client()
        with patch.object(client.session, "get_credentials", return_value=_mock_credentials()):
            with patch("bedrock_agentcore.runtime.agent_core_runtime_client.SigV4QueryAuth") as mock_auth:
                mock_request = MagicMock()
                mock_request.url = (
                    "https://bedrock-agentcore.us-west-2.amazonaws.com/runtimes/x/ws/commands?X-Amz-Algorithm=AWS4"
                )
                mock_auth.return_value.add_auth = MagicMock(side_effect=lambda r: setattr(r, "url", mock_request.url))
                url = client.connect_shell_presigned(FAKE_ARN)

        assert url.startswith("wss://")

    def test_session_id_embedded_before_signing(self):
        """X-Amzn-Bedrock-AgentCore-Runtime-Session-Id must be in the URL that
        SigV4QueryAuth signs, so it is covered by the signature."""
        from urllib.parse import parse_qs, urlparse

        client = _client()
        signed_url = None

        def capture_and_sign(request):
            nonlocal signed_url
            signed_url = request.url
            request.url = request.url + "&X-Amz-Signature=fakesig"

        with patch.object(client.session, "get_credentials", return_value=_mock_credentials()):
            with patch("bedrock_agentcore.runtime.agent_core_runtime_client.SigV4QueryAuth") as mock_auth:
                mock_auth.return_value.add_auth = MagicMock(side_effect=capture_and_sign)
                client.connect_shell_presigned(FAKE_ARN, session_id="test-session-99")

        assert signed_url is not None
        qs = parse_qs(urlparse(signed_url).query)
        assert qs.get("X-Amzn-Bedrock-AgentCore-Runtime-Session-Id") == ["test-session-99"]

    def test_session_id_autogenerated_and_embedded_before_signing(self):
        """When session_id is omitted an auto-generated one must still be
        embedded in the URL before signing."""
        from urllib.parse import parse_qs, urlparse

        client = _client()
        signed_url = None

        def capture_and_sign(request):
            nonlocal signed_url
            signed_url = request.url
            request.url = request.url + "&X-Amz-Signature=fakesig"

        with patch.object(client.session, "get_credentials", return_value=_mock_credentials()):
            with patch("bedrock_agentcore.runtime.agent_core_runtime_client.SigV4QueryAuth") as mock_auth:
                mock_auth.return_value.add_auth = MagicMock(side_effect=capture_and_sign)
                client.connect_shell_presigned(FAKE_ARN)

        assert signed_url is not None
        qs = parse_qs(urlparse(signed_url).query)
        assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id" in qs

    def test_expires_too_large_raises(self):
        client = _client()
        with pytest.raises(ValueError, match="Expiry timeout cannot exceed"):
            client.connect_shell_presigned(FAKE_ARN, expires=301)

    def test_invalid_arn_raises(self):
        client = _client()
        with pytest.raises(ValueError):
            client.connect_shell_presigned("bad-arn")

    def test_invalid_shell_id_raises(self):
        client = _client()
        with pytest.raises(ValueError):
            client.connect_shell_presigned(FAKE_ARN, shell_id="bad#id")

    def test_no_credentials_raises(self):
        client = _client()
        with patch.object(client.session, "get_credentials", return_value=None):
            with pytest.raises(RuntimeError, match="No AWS credentials"):
                client.connect_shell_presigned(FAKE_ARN)


# ── connect_shell_oauth ───────────────────────────────────────────────────────


class TestConnectShellOAuth:
    def test_returns_url_and_subprotocols(self):
        import base64

        client = _client()
        url, protos = client.connect_shell_oauth(FAKE_ARN, bearer_token="tok123")
        assert url.startswith("wss://")
        assert "/ws/commands" in url
        encoded = base64.urlsafe_b64encode(b"tok123").decode().rstrip("=")
        assert f"base64UrlBearerAuthorization.{encoded}" in protos
        assert "base64UrlBearerAuthorization" in protos

    def test_token_embedded_in_subprotocol(self):
        import base64

        client = _client()
        _, protos = client.connect_shell_oauth(FAKE_ARN, bearer_token="my-token")
        encoded = base64.urlsafe_b64encode(b"my-token").decode().rstrip("=")
        assert protos == [f"base64UrlBearerAuthorization.{encoded}", "base64UrlBearerAuthorization"]

    def test_session_id_in_query_string(self):
        client = _client()
        url, _ = client.connect_shell_oauth(FAKE_ARN, bearer_token="tok", session_id="sid-123")
        assert "sid-123" in url

    def test_empty_token_raises(self):
        client = _client()
        with pytest.raises(ValueError, match="bearer_token"):
            client.connect_shell_oauth(FAKE_ARN, bearer_token="")

    def test_invalid_arn_raises(self):
        client = _client()
        with pytest.raises(ValueError):
            client.connect_shell_oauth("bad-arn", bearer_token="tok")

    def test_invalid_shell_id_raises(self):
        client = _client()
        with pytest.raises(ValueError):
            client.connect_shell_oauth(FAKE_ARN, bearer_token="tok", shell_id="a&b")


# ── open_shell ────────────────────────────────────────────────────────────────


class TestOpenShell:
    def test_returns_shell_session(self):
        from bedrock_agentcore.runtime.shell import ShellSession

        client = _client()
        session = client.open_shell(FAKE_ARN)
        assert isinstance(session, ShellSession)

    def test_passes_shell_id(self):
        client = _client()
        session = client.open_shell(FAKE_ARN, shell_id="my-shell")
        assert session._shell_id == "my-shell"

    def test_default_auth_is_sigv4(self):
        client = _client()
        session = client.open_shell(FAKE_ARN)
        assert session._auth == "sigv4"

    def test_passes_presigned_auth(self):
        from bedrock_agentcore.runtime.shell import PresignedAuth

        client = _client()
        auth = PresignedAuth(expires=120)
        session = client.open_shell(FAKE_ARN, auth=auth)
        assert session._auth is auth

    def test_passes_oauth_auth(self):
        from bedrock_agentcore.runtime.shell import OAuthAuth

        client = _client()
        auth = OAuthAuth(bearer_token="tok")
        session = client.open_shell(FAKE_ARN, auth=auth)
        assert session._auth is auth

    def test_passes_reconnect_config(self):
        from bedrock_agentcore.runtime.shell import ReconnectConfig

        client = _client()
        config = ReconnectConfig(max_retries=3)
        session = client.open_shell(FAKE_ARN, reconnect_config=config)
        assert session._reconnect_config is config

    def test_no_reconnect_config_by_default(self):
        client = _client()
        session = client.open_shell(FAKE_ARN)
        assert session._reconnect_config is None

    def test_invalid_shell_id_raises(self):
        client = _client()
        with pytest.raises(ValueError):
            client.open_shell(FAKE_ARN, shell_id="bad?id")

    def test_invalid_arn_raises_eagerly(self):
        client = _client()
        with pytest.raises(ValueError):
            client.open_shell("not-an-arn")

    def test_region_mismatch_raises(self):
        client = AgentCoreRuntimeClient(region="us-east-1")
        east_arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-runtime"
        west_arn = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime"
        session = client.open_shell(east_arn)
        assert session is not None
        with pytest.raises(ValueError, match="does not match client region"):
            client.open_shell(west_arn)
