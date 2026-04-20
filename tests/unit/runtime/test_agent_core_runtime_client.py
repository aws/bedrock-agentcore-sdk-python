"""Tests for AgentCoreRuntimeClient."""

import json
from typing import Iterator, Optional
from unittest.mock import MagicMock, Mock, patch
from urllib.parse import quote

import pytest

from bedrock_agentcore.runtime.agent_core_runtime_client import (
    AgentCoreRuntimeClient,
    AgentRuntimeError,
)


class TestAgentCoreRuntimeClientInit:
    """Tests for AgentCoreRuntimeClient initialization."""

    def test_init_stores_region(self):
        """Test that initialization stores the region."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        assert client.region == "us-west-2"

    def test_init_creates_logger(self):
        """Test that initialization creates a logger."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        assert client.logger is not None


class TestParseRuntimeArn:
    """Tests for _parse_runtime_arn helper."""

    def test_parse_valid_arn(self):
        """Test parsing a valid runtime ARN."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        arn = (
            "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime-abc123"
        )

        result = client._parse_runtime_arn(arn)

        assert result["region"] == "us-west-2"
        assert result["account_id"] == "123456789012"
        assert result["runtime_id"] == "my-runtime-abc123"

    def test_parse_valid_gov_arn(self):
        """Test parsing a valid govcloud runtime ARN."""
        client = AgentCoreRuntimeClient(region="us-gov-west-1")
        arn = "arn:aws-us-gov:bedrock-agentcore:us-gov-west-1:123456789012:runtime/my-runtime-abc123"

        result = client._parse_runtime_arn(arn)

        assert result["region"] == "us-gov-west-1"
        assert result["account_id"] == "123456789012"
        assert result["runtime_id"] == "my-runtime-abc123"

    def test_parse_invalid_arn_partition(self):
        """Test parsing an invalid runtime ARN."""
        client = AgentCoreRuntimeClient(region="us-iso-east-1")
        invalid_arn = "arn:aws-iso:bedrock-agentcore:us-iso-east-1:123456789012:runtime/my-runtime-abc123"

        with pytest.raises(ValueError, match="Invalid runtime ARN format"):
            client._parse_runtime_arn(invalid_arn)

    def test_parse_invalid_arn_raises_error(self):
        """Test that invalid ARN format raises ValueError."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        invalid_arn = "not-a-valid-arn"

        with pytest.raises(ValueError, match="Invalid runtime ARN format"):
            client._parse_runtime_arn(invalid_arn)

    def test_parse_wrong_service_raises_error(self):
        """Test that wrong service in ARN raises ValueError."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        wrong_service = "arn:aws:s3:us-west-2:123456789012:bucket/my-bucket"

        with pytest.raises(ValueError, match="Invalid runtime ARN format"):
            client._parse_runtime_arn(wrong_service)

    def test_parse_empty_region_raises_error(self):
        """Test that empty region in ARN raises ValueError."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        empty_region = "arn:aws:bedrock-agentcore::123456789012:runtime/my-runtime"

        with pytest.raises(ValueError, match="ARN components cannot be empty"):
            client._parse_runtime_arn(empty_region)

    def test_parse_empty_account_id_raises_error(self):
        """Test that empty account_id in ARN raises ValueError."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        empty_account = "arn:aws:bedrock-agentcore:us-west-2::runtime/my-runtime"

        with pytest.raises(ValueError, match="ARN components cannot be empty"):
            client._parse_runtime_arn(empty_account)

    def test_parse_empty_runtime_id_raises_error(self):
        """Test that empty runtime_id in ARN raises ValueError."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        empty_runtime = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/"

        with pytest.raises(ValueError, match="ARN components cannot be empty"):
            client._parse_runtime_arn(empty_runtime)


class TestBuildWebsocketUrl:
    """Tests for _build_websocket_url helper."""

    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_build_basic_url(self, mock_endpoint):
        """Test building basic WebSocket URL without query params."""
        mock_endpoint.return_value = "https://example.aws.dev"
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        result = client._build_websocket_url(runtime_arn)

        # ARN should be URL encoded
        encoded_arn = quote(runtime_arn, safe="")
        assert result == f"wss://example.aws.dev/runtimes/{encoded_arn}/ws"

    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_build_url_with_endpoint_name(self, mock_endpoint):
        """Test building URL with endpoint name (qualifier param)."""
        mock_endpoint.return_value = "https://example.aws.dev"
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        result = client._build_websocket_url(runtime_arn, endpoint_name="DEFAULT")

        encoded_arn = quote(runtime_arn, safe="")
        assert (
            result
            == f"wss://example.aws.dev/runtimes/{encoded_arn}/ws?qualifier=DEFAULT"
        )

    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_build_url_with_custom_headers(self, mock_endpoint):
        """Test building URL with custom headers as query params."""
        mock_endpoint.return_value = "https://example.aws.dev"
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        result = client._build_websocket_url(
            runtime_arn, custom_headers={"abc": "pqr", "foo": "bar"}
        )

        encoded_arn = quote(runtime_arn, safe="")
        assert f"wss://example.aws.dev/runtimes/{encoded_arn}/ws?" in result
        assert "abc=pqr" in result
        assert "foo=bar" in result

    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_build_url_with_all_params(self, mock_endpoint):
        """Test building URL with endpoint name and custom headers."""
        mock_endpoint.return_value = "https://example.aws.dev"
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        result = client._build_websocket_url(
            runtime_arn, endpoint_name="DEFAULT", custom_headers={"abc": "pqr"}
        )

        encoded_arn = quote(runtime_arn, safe="")
        assert f"wss://example.aws.dev/runtimes/{encoded_arn}/ws?" in result
        assert "qualifier=DEFAULT" in result
        assert "abc=pqr" in result


class TestGenerateWsConnection:
    """Tests for generate_ws_connection method."""

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_generate_basic_connection(self, mock_endpoint, mock_session):
        """Test generating basic WebSocket connection."""
        # Setup mocks
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(
            access_key="AKIATEST", secret_key="secret", token=None
        )
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        ws_url, headers = client.generate_ws_connection(runtime_arn)

        # Verify URL structure
        assert ws_url.startswith("wss://example.aws.dev/runtimes/")
        assert "/ws" in ws_url

        # Verify required headers
        assert "Host" in headers
        assert "X-Amz-Date" in headers
        assert "Authorization" in headers
        assert "Upgrade" in headers
        assert "Connection" in headers
        assert "Sec-WebSocket-Version" in headers
        assert "Sec-WebSocket-Key" in headers

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_generate_connection_with_session_id(self, mock_endpoint, mock_session):
        """Test generating connection with explicit session ID."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(
            access_key="AKIATEST", secret_key="secret", token=None
        )
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        ws_url, headers = client.generate_ws_connection(
            runtime_arn, session_id="test-session-123"
        )

        assert ws_url is not None
        assert headers is not None
        # Verify session ID is in headers
        assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id" in headers
        assert (
            headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == "test-session-123"
        )

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_generate_connection_user_agent(self, mock_endpoint, mock_session):
        """Test that User-Agent header is set correctly."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(
            access_key="AKIATEST", secret_key="secret", token=None
        )
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        ws_url, headers = client.generate_ws_connection(runtime_arn)

        assert "User-Agent" in headers
        assert headers["User-Agent"] == "AgentCoreRuntimeClient/1.0"

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_generate_connection_with_endpoint_name(self, mock_endpoint, mock_session):
        """Test generating connection with endpoint name."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(
            access_key="AKIATEST", secret_key="secret", token=None
        )
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        ws_url, headers = client.generate_ws_connection(
            runtime_arn, endpoint_name="DEFAULT"
        )

        assert "qualifier=DEFAULT" in ws_url

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    def test_generate_connection_no_credentials_raises_error(self, mock_session):
        """Test that missing credentials raises RuntimeError."""
        mock_session.return_value.get_credentials.return_value = None

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        with pytest.raises(RuntimeError, match="No AWS credentials found"):
            client.generate_ws_connection(runtime_arn)


class TestGeneratePresignedUrl:
    """Tests for generate_presigned_url method."""

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_generate_basic_presigned_url(self, mock_endpoint, mock_session):
        """Test generating basic presigned URL."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(
            access_key="AKIATEST", secret_key="secret", token=None
        )
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        presigned_url = client.generate_presigned_url(runtime_arn)

        # Verify URL structure
        assert presigned_url.startswith("wss://example.aws.dev/runtimes/")
        assert "/ws?" in presigned_url

        # Verify SigV4 query parameters
        assert "X-Amz-Algorithm" in presigned_url
        assert "X-Amz-Credential" in presigned_url
        assert "X-Amz-Date" in presigned_url
        assert "X-Amz-Expires" in presigned_url
        assert "X-Amz-SignedHeaders" in presigned_url
        assert "X-Amz-Signature" in presigned_url

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_generate_presigned_url_with_endpoint_name(
        self, mock_endpoint, mock_session
    ):
        """Test generating presigned URL with endpoint name."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(
            access_key="AKIATEST", secret_key="secret", token=None
        )
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        presigned_url = client.generate_presigned_url(
            runtime_arn, endpoint_name="DEFAULT"
        )

        assert "qualifier=DEFAULT" in presigned_url

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_generate_presigned_url_with_custom_headers(
        self, mock_endpoint, mock_session
    ):
        """Test generating presigned URL with custom headers."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(
            access_key="AKIATEST", secret_key="secret", token=None
        )
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        presigned_url = client.generate_presigned_url(
            runtime_arn, custom_headers={"abc": "pqr"}
        )

        assert "abc=pqr" in presigned_url

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_generate_presigned_url_with_session_id(self, mock_endpoint, mock_session):
        """Test generating presigned URL with explicit session ID."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(
            access_key="AKIATEST", secret_key="secret", token=None
        )
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        presigned_url = client.generate_presigned_url(
            runtime_arn, session_id="test-session-456"
        )

        # Verify session ID is in query params
        assert (
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id=test-session-456"
            in presigned_url
        )

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_generate_presigned_url_with_custom_expires(
        self, mock_endpoint, mock_session
    ):
        """Test generating presigned URL with custom expiration."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(
            access_key="AKIATEST", secret_key="secret", token=None
        )
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        presigned_url = client.generate_presigned_url(runtime_arn, expires=60)

        assert "X-Amz-Expires=60" in presigned_url
        # Verify auto-generated session ID is present
        assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id=" in presigned_url

    def test_generate_presigned_url_exceeds_max_expires_raises_error(self):
        """Test that exceeding max expiration raises ValueError."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        with pytest.raises(ValueError, match="Expiry timeout cannot exceed"):
            client.generate_presigned_url(runtime_arn, expires=400)

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    def test_generate_presigned_url_no_credentials_raises_error(self, mock_session):
        """Test that missing credentials raises RuntimeError."""
        mock_session.return_value.get_credentials.return_value = None

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        with pytest.raises(RuntimeError, match="No AWS credentials found"):
            client.generate_presigned_url(runtime_arn)


class TestAgentCoreRuntimeClientSession:
    """Tests for AgentCoreRuntimeClient with custom boto3 session."""

    def test_init_with_custom_session(self):
        """Test initialization with custom boto3 session."""
        custom_session = Mock()
        client = AgentCoreRuntimeClient(region="us-west-2", session=custom_session)

        assert client.region == "us-west-2"
        assert client.session == custom_session

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    def test_init_without_session_creates_default(self, mock_session_class):
        """Test that default session is created when not provided."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        client = AgentCoreRuntimeClient(region="us-west-2")

        assert client.session == mock_session
        mock_session_class.assert_called_once()

    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_generate_ws_connection_uses_custom_session(self, mock_endpoint):
        """Test that generate_ws_connection uses the custom session."""
        mock_endpoint.return_value = "https://example.aws.dev"

        # Create custom session with credentials
        custom_session = Mock()
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(
            access_key="AKIATEST", secret_key="secret", token=None
        )
        custom_session.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2", session=custom_session)
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        ws_url, headers = client.generate_ws_connection(runtime_arn)

        # Verify custom session was used
        custom_session.get_credentials.assert_called_once()
        assert ws_url.startswith("wss://")
        assert "Authorization" in headers


class TestGenerateWsConnectionOAuth:
    """Tests for generate_ws_connection_oauth method."""

    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_generate_oauth_connection_basic(self, mock_endpoint):
        """Test generating basic OAuth WebSocket connection."""
        mock_endpoint.return_value = "https://example.aws.dev"
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"
        bearer_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test.token"

        ws_url, headers = client.generate_ws_connection_oauth(runtime_arn, bearer_token)

        # Verify URL structure
        assert ws_url.startswith("wss://example.aws.dev/runtimes/")
        assert "/ws" in ws_url

        # Verify OAuth headers
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {bearer_token}"
        assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id" in headers
        assert "Sec-WebSocket-Key" in headers
        assert "Sec-WebSocket-Version" in headers
        assert headers["Sec-WebSocket-Version"] == "13"

    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_generate_oauth_connection_with_session_id(self, mock_endpoint):
        """Test generating OAuth connection with explicit session ID."""
        mock_endpoint.return_value = "https://example.aws.dev"
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"
        bearer_token = "test-token"
        custom_session_id = "custom-oauth-session-123"

        ws_url, headers = client.generate_ws_connection_oauth(
            runtime_arn, bearer_token, session_id=custom_session_id
        )

        assert (
            headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == custom_session_id
        )

    @patch(
        "bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint"
    )
    def test_generate_oauth_connection_with_endpoint_name(self, mock_endpoint):
        """Test generating OAuth connection with endpoint name."""
        mock_endpoint.return_value = "https://example.aws.dev"
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"
        bearer_token = "test-token"

        ws_url, headers = client.generate_ws_connection_oauth(
            runtime_arn, bearer_token, endpoint_name="DEFAULT"
        )

        assert "qualifier=DEFAULT" in ws_url

    def test_generate_oauth_connection_empty_token_raises_error(self):
        """Test that empty bearer token raises ValueError."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        with pytest.raises(ValueError, match="Bearer token cannot be empty"):
            client.generate_ws_connection_oauth(runtime_arn, "")

    def test_generate_oauth_connection_invalid_arn_raises_error(self):
        """Test that invalid ARN raises ValueError."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        invalid_arn = "invalid-arn"
        bearer_token = "test-token"

        with pytest.raises(ValueError, match="Invalid runtime ARN format"):
            client.generate_ws_connection_oauth(invalid_arn, bearer_token)


# ===================================================================== #
# HTTP invocation tests (bearer-token auth)
# ===================================================================== #

ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime-abc"
BEARER = "test-token"
SESSION = "a" * 36


class _FakeResponse:
    """Minimal stand-in for urllib3's response used by the HTTP methods.

    Exposes the exact surface the client touches: ``status``, ``reason``,
    ``headers``, ``data``, ``read``, ``stream``, and ``release_conn``.
    """

    def __init__(
        self,
        status: int = 200,
        headers: Optional[dict] = None,
        body: bytes = b"",
        chunks: Optional[list] = None,
        reason: str = "",
    ) -> None:
        self.status = status
        self.reason = reason
        self.headers = headers or {}
        self._body = body
        self._chunks = chunks
        self._consumed = False
        self.release_calls = 0

    @property
    def data(self) -> bytes:
        return self._body

    def read(self) -> bytes:
        if self._consumed:
            return b""
        self._consumed = True
        return self._body

    def stream(self, amt: int = 1024) -> Iterator[bytes]:
        if self._chunks is not None:
            for chunk in self._chunks:
                yield chunk
            return
        if self._body:
            yield self._body

    def release_conn(self) -> None:
        self.release_calls += 1


def _make_client(response: _FakeResponse) -> "tuple[AgentCoreRuntimeClient, MagicMock]":
    """Build a client whose lazy PoolManager is bypassed with a mock."""
    pool = MagicMock()
    pool.request.return_value = response
    client = AgentCoreRuntimeClient(region="us-west-2")
    client._pool_manager = pool  # bypass the lazy property
    return client, pool


# --------------------------------------------------------------------- #
# Lazy PoolManager
# --------------------------------------------------------------------- #


class TestLazyPoolManager:
    """Tests for the @property _http lazy PoolManager."""

    def test_pool_not_created_at_init(self) -> None:
        """Constructor does not create a PoolManager."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        assert client._pool_manager is None

    def test_pool_created_on_first_access(self) -> None:
        """First access to _http creates the PoolManager."""
        import urllib3

        client = AgentCoreRuntimeClient(region="us-west-2")
        pool = client._http
        assert isinstance(pool, urllib3.PoolManager)
        assert client._pool_manager is pool

    def test_pool_reused_across_accesses(self) -> None:
        """Repeated _http access returns the same instance."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        first = client._http
        second = client._http
        assert first is second


# --------------------------------------------------------------------- #
# _build_http_url / _build_bearer_headers / _serialize_body
# --------------------------------------------------------------------- #


class TestBuildHttpUrl:
    """Tests for _build_http_url."""

    def test_invocations_url(self) -> None:
        """URL embeds region (via endpoint helper), URL-encoded ARN, path, and qualifier."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        url = client._build_http_url(ARN, "invocations")
        assert url.startswith(
            "https://bedrock-agentcore.us-west-2.amazonaws.com/runtimes/"
        )
        assert "/invocations?qualifier=DEFAULT" in url
        assert "arn%3Aaws%3Abedrock-agentcore" in url

    def test_commands_url(self) -> None:
        """Different path suffix yields the commands endpoint."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        assert "/commands?qualifier=DEFAULT" in client._build_http_url(ARN, "commands")

    def test_non_default_qualifier(self) -> None:
        """Qualifier reflects the endpoint_name argument."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        assert "qualifier=DEV" in client._build_http_url(
            ARN, "invocations", endpoint_name="DEV"
        )

    def test_region_derived_from_arn(self) -> None:
        """URL uses the ARN region, not the client's init region."""
        client = AgentCoreRuntimeClient(region="us-east-1")
        arn = "arn:aws:bedrock-agentcore:eu-west-2:123456789012:runtime/other"
        assert (
            "https://bedrock-agentcore.eu-west-2.amazonaws.com/"
            in client._build_http_url(arn, "invocations")
        )

    def test_invalid_arn_raises(self) -> None:
        """Invalid ARN propagates from _parse_runtime_arn."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        with pytest.raises(ValueError, match="Invalid runtime ARN format"):
            client._build_http_url("not-an-arn", "invocations")


class TestBuildBearerHeaders:
    """Tests for _build_bearer_headers."""

    def test_all_headers_populated(self) -> None:
        """All four bearer-auth headers are set."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        headers = client._build_bearer_headers(
            BEARER, SESSION, accept="application/json", content_type="application/json"
        )
        assert headers["Authorization"] == f"Bearer {BEARER}"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"
        assert headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == SESSION

    def test_eventstream_accept(self) -> None:
        """Accept value is passed through verbatim."""
        client = AgentCoreRuntimeClient(region="us-west-2")
        headers = client._build_bearer_headers(
            BEARER,
            SESSION,
            accept="application/vnd.amazon.eventstream",
            content_type="application/json",
        )
        assert headers["Accept"] == "application/vnd.amazon.eventstream"


class TestSerializeBody:
    """Tests for _serialize_body."""

    def test_json_dict(self) -> None:
        """Dict body is JSON-serialized when content type is JSON."""
        assert (
            AgentCoreRuntimeClient._serialize_body({"a": 1}, "application/json")
            == b'{"a": 1}'
        )

    def test_bytes_passthrough_non_json(self) -> None:
        """Bytes body is sent verbatim for non-JSON content types."""
        assert (
            AgentCoreRuntimeClient._serialize_body(b"raw", "application/octet-stream")
            == b"raw"
        )

    def test_str_utf8_encoded_non_json(self) -> None:
        """String body is UTF-8 encoded for non-JSON content types."""
        assert AgentCoreRuntimeClient._serialize_body(
            "héllo", "text/plain"
        ) == "héllo".encode("utf-8")

    def test_fallback_to_json(self) -> None:
        """Non-str, non-bytes body falls back to JSON for non-JSON content types."""
        assert (
            AgentCoreRuntimeClient._serialize_body({"x": 1}, "application/cbor")
            == b'{"x": 1}'
        )


# --------------------------------------------------------------------- #
# invoke (non-streaming JSON, non-streaming plain, SSE)
# --------------------------------------------------------------------- #


class TestInvoke:
    """Tests for invoke."""

    def test_non_streaming_json_string(self) -> None:
        """JSON string response is unwrapped."""
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=b'"hello"',
        )
        client, _ = _make_client(resp)
        assert (
            client.invoke(ARN, BEARER, {"prompt": "hi"}, session_id=SESSION) == "hello"
        )
        assert resp.release_calls == 1

    def test_non_streaming_json_object_returns_text(self) -> None:
        """Non-string JSON response returns the raw body text."""
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=b'{"answer": 42}',
        )
        client, _ = _make_client(resp)
        assert (
            client.invoke(ARN, BEARER, {"prompt": "hi"}, session_id=SESSION)
            == '{"answer": 42}'
        )

    def test_non_streaming_invalid_json_returns_text(self) -> None:
        """If the body isn't valid JSON, the raw text is returned."""
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=b"plain-text-not-json",
        )
        client, _ = _make_client(resp)
        assert (
            client.invoke(ARN, BEARER, {"prompt": "hi"}, session_id=SESSION)
            == "plain-text-not-json"
        )

    def test_sse_streaming_concatenates(self) -> None:
        """Multiple SSE data lines are concatenated in order."""
        body = b"data: hello\ndata: world\n"
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "text/event-stream"},
            chunks=[body],
        )
        client, _ = _make_client(resp)
        assert (
            client.invoke(ARN, BEARER, {"prompt": "hi"}, session_id=SESSION)
            == "helloworld"
        )

    def test_custom_headers_merged(self) -> None:
        """Caller-supplied headers are merged into the request."""
        resp = _FakeResponse(
            status=200, headers={"content-type": "application/json"}, body=b'"ok"'
        )
        client, pool = _make_client(resp)
        client.invoke(ARN, BEARER, {}, session_id=SESSION, headers={"X-Test": "1"})
        sent_headers = pool.request.call_args.kwargs["headers"]
        assert sent_headers["X-Test"] == "1"
        assert sent_headers["Authorization"] == f"Bearer {BEARER}"

    def test_non_ok_raises(self) -> None:
        """4xx and 5xx responses become AgentRuntimeError."""
        resp = _FakeResponse(
            status=500,
            body=b'{"error": "oops", "message": "boom"}',
            reason="Server Error",
        )
        client, _ = _make_client(resp)
        with pytest.raises(AgentRuntimeError) as excinfo:
            client.invoke(ARN, BEARER, {}, session_id=SESSION)
        assert excinfo.value.error == "oops"
        assert "boom" in str(excinfo.value)

    def test_session_id_auto_generated(self) -> None:
        """Missing session_id is auto-generated as a UUID."""
        resp = _FakeResponse(
            status=200, headers={"content-type": "application/json"}, body=b'"ok"'
        )
        client, pool = _make_client(resp)
        client.invoke(ARN, BEARER, {})
        sent_headers = pool.request.call_args.kwargs["headers"]
        # UUID4 string is 36 chars.
        assert len(sent_headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"]) == 36


# --------------------------------------------------------------------- #
# invoke_streaming
# --------------------------------------------------------------------- #


class TestInvokeStreaming:
    """Tests for invoke_streaming."""

    def test_yields_chunks(self) -> None:
        """Yields each decoded SSE payload in order."""
        body = b"data: first\ndata: second\n"
        resp = _FakeResponse(
            status=200, headers={"content-type": "text/event-stream"}, chunks=[body]
        )
        client, _ = _make_client(resp)
        assert list(client.invoke_streaming(ARN, BEARER, {}, session_id=SESSION)) == [
            "first",
            "second",
        ]

    def test_non_ok_raises(self) -> None:
        """Errors are surfaced before the generator yields."""
        resp = _FakeResponse(status=404, body=b"not found", reason="Not Found")
        client, _ = _make_client(resp)
        with pytest.raises(AgentRuntimeError):
            list(client.invoke_streaming(ARN, BEARER, {}, session_id=SESSION))

    def test_custom_headers_merged(self) -> None:
        """Custom headers are merged into the request."""
        resp = _FakeResponse(
            status=200, headers={"content-type": "text/event-stream"}, chunks=[b""]
        )
        client, pool = _make_client(resp)
        list(
            client.invoke_streaming(
                ARN, BEARER, {}, session_id=SESSION, headers={"X-Custom": "v"}
            )
        )
        assert pool.request.call_args.kwargs["headers"]["X-Custom"] == "v"


# --------------------------------------------------------------------- #
# invoke_streaming_async
# --------------------------------------------------------------------- #


class TestInvokeStreamingAsync:
    """Tests for invoke_streaming_async."""

    @pytest.mark.asyncio
    async def test_yields_async_chunks(self) -> None:
        """Async generator yields the same chunks as the sync version."""
        body = b"data: a\ndata: b\n"
        resp = _FakeResponse(
            status=200, headers={"content-type": "text/event-stream"}, chunks=[body]
        )
        client, _ = _make_client(resp)
        out = []
        async for chunk in client.invoke_streaming_async(
            ARN, BEARER, {}, session_id=SESSION
        ):
            out.append(chunk)
        assert out == ["a", "b"]

    @pytest.mark.asyncio
    async def test_propagates_exception(self) -> None:
        """Exceptions from the background thread surface to the caller."""
        resp = _FakeResponse(status=500, body=b"err", reason="Server Error")
        client, _ = _make_client(resp)
        with pytest.raises(AgentRuntimeError):
            async for _ in client.invoke_streaming_async(
                ARN, BEARER, {}, session_id=SESSION
            ):
                pass


# --------------------------------------------------------------------- #
# execute_command / execute_command_streaming
# --------------------------------------------------------------------- #


def _encode_eventstream_frame(payload: dict) -> bytes:
    """Encode a JSON payload using the EventStream binary framing.

    Builds a valid frame the client's EventStreamBuffer can parse. Format:
    total_length(4) + headers_length(4) + prelude_crc(4) + headers + payload + message_crc(4).
    """
    import binascii
    import struct

    body = json.dumps(payload).encode("utf-8")
    headers = b""
    headers_length = len(headers)
    total_length = 4 + 4 + 4 + headers_length + len(body) + 4
    prelude = struct.pack(">II", total_length, headers_length)
    prelude_crc = struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message_bytes = prelude + prelude_crc + headers + body
    message_crc = struct.pack(">I", binascii.crc32(message_bytes) & 0xFFFFFFFF)
    return message_bytes + message_crc


class TestExecuteCommand:
    """Tests for execute_command (blocking)."""

    def test_aggregates_output(self) -> None:
        """stdout/stderr/exitCode/status are aggregated across events."""
        events = [
            _encode_eventstream_frame({"chunk": {"contentStart": {}}}),
            _encode_eventstream_frame({"chunk": {"contentDelta": {"stdout": "hi "}}}),
            _encode_eventstream_frame({"chunk": {"contentDelta": {"stdout": "there"}}}),
            _encode_eventstream_frame({"chunk": {"contentDelta": {"stderr": "warn"}}}),
            _encode_eventstream_frame(
                {"chunk": {"contentStop": {"exitCode": 0, "status": "COMPLETED"}}}
            ),
        ]
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=events,
        )
        client, _ = _make_client(resp)
        result = client.execute_command(
            ARN, BEARER, "echo hi there", session_id=SESSION
        )
        assert result == {
            "stdout": "hi there",
            "stderr": "warn",
            "exitCode": 0,
            "status": "COMPLETED",
        }

    def test_missing_stop_has_unknown_status(self) -> None:
        """Without a contentStop event, defaults are UNKNOWN / -1."""
        events = [
            _encode_eventstream_frame({"chunk": {"contentDelta": {"stdout": "x"}}}),
        ]
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=events,
        )
        client, _ = _make_client(resp)
        result = client.execute_command(ARN, BEARER, "echo x", session_id=SESSION)
        assert result == {
            "stdout": "x",
            "stderr": "",
            "exitCode": -1,
            "status": "UNKNOWN",
        }


class TestExecuteCommandStreaming:
    """Tests for execute_command_streaming."""

    def test_sends_command_body(self) -> None:
        """Request body contains command and timeout."""
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=[],
        )
        client, pool = _make_client(resp)
        list(
            client.execute_command_streaming(
                ARN, BEARER, "ls", session_id=SESSION, command_timeout=42
            )
        )
        sent = pool.request.call_args
        assert sent.kwargs["body"] == b'{"command": "ls", "timeout": 42}'
        assert sent.kwargs["timeout"] == 72  # command_timeout + 30

    def test_defaults_to_constant(self) -> None:
        """Omitting command_timeout uses DEFAULT_COMMAND_TIMEOUT."""
        from bedrock_agentcore.runtime.agent_core_runtime_client import (
            DEFAULT_COMMAND_TIMEOUT,
        )

        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=[],
        )
        client, pool = _make_client(resp)
        list(client.execute_command_streaming(ARN, BEARER, "ls", session_id=SESSION))
        sent = pool.request.call_args
        body = json.loads(sent.kwargs["body"])
        assert body["timeout"] == DEFAULT_COMMAND_TIMEOUT
        assert sent.kwargs["timeout"] == DEFAULT_COMMAND_TIMEOUT + 30

    def test_yields_parsed_events(self) -> None:
        """Each EventStream frame is yielded as a dict."""
        events = [
            _encode_eventstream_frame({"chunk": {"contentStart": {}}}),
            _encode_eventstream_frame(
                {"chunk": {"contentStop": {"exitCode": 2, "status": "COMPLETED"}}}
            ),
        ]
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=events,
        )
        client, _ = _make_client(resp)
        parsed = list(
            client.execute_command_streaming(ARN, BEARER, "echo hi", session_id=SESSION)
        )
        assert parsed[0] == {"contentStart": {}}
        assert parsed[1] == {"contentStop": {"exitCode": 2, "status": "COMPLETED"}}

    def test_skips_event_without_payload(self) -> None:
        """Events with empty payload are skipped silently."""
        import binascii
        import struct

        headers = b""
        body = b""
        headers_length = 0
        total_length = 4 + 4 + 4 + headers_length + len(body) + 4
        prelude = struct.pack(">II", total_length, headers_length)
        prelude_crc = struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
        message_bytes = prelude + prelude_crc + headers + body
        message_crc = struct.pack(">I", binascii.crc32(message_bytes) & 0xFFFFFFFF)
        empty_frame = message_bytes + message_crc

        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=[empty_frame],
        )
        client, _ = _make_client(resp)
        assert (
            list(
                client.execute_command_streaming(ARN, BEARER, "ls", session_id=SESSION)
            )
            == []
        )

    def test_skips_bad_json_payload(self) -> None:
        """Events whose payload is not valid JSON are skipped."""
        import binascii
        import struct

        headers = b""
        body = b"not-json"
        headers_length = 0
        total_length = 4 + 4 + 4 + headers_length + len(body) + 4
        prelude = struct.pack(">II", total_length, headers_length)
        prelude_crc = struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
        message_bytes = prelude + prelude_crc + headers + body
        message_crc = struct.pack(">I", binascii.crc32(message_bytes) & 0xFFFFFFFF)
        bad_frame = message_bytes + message_crc

        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=[bad_frame],
        )
        client, _ = _make_client(resp)
        assert (
            list(
                client.execute_command_streaming(ARN, BEARER, "ls", session_id=SESSION)
            )
            == []
        )

    def test_non_ok_raises(self) -> None:
        """Non-2xx responses raise AgentRuntimeError."""
        resp = _FakeResponse(status=403, body=b"forbidden", reason="Forbidden")
        client, _ = _make_client(resp)
        with pytest.raises(AgentRuntimeError):
            list(
                client.execute_command_streaming(ARN, BEARER, "ls", session_id=SESSION)
            )


# --------------------------------------------------------------------- #
# stop_runtime_session
# --------------------------------------------------------------------- #


class TestStopRuntimeSession:
    """Tests for stop_runtime_session."""

    def test_sends_client_token(self) -> None:
        """Request body contains the provided client token."""
        resp = _FakeResponse(status=200, body=b"{}")
        client, pool = _make_client(resp)
        client.stop_runtime_session(ARN, BEARER, session_id=SESSION, client_token="abc")
        sent = json.loads(pool.request.call_args.kwargs["body"])
        assert sent == {"clientToken": "abc"}

    def test_autogenerates_client_token(self) -> None:
        """Missing client_token is replaced with a UUID4."""
        resp = _FakeResponse(status=200, body=b"{}")
        client, pool = _make_client(resp)
        client.stop_runtime_session(ARN, BEARER, session_id=SESSION)
        sent = json.loads(pool.request.call_args.kwargs["body"])
        assert len(sent["clientToken"]) == 36

    def test_returns_empty_dict_on_blank_body(self) -> None:
        """Empty response body yields an empty dict."""
        resp = _FakeResponse(status=200, body=b"")
        client, _ = _make_client(resp)
        assert client.stop_runtime_session(ARN, BEARER, session_id=SESSION) == {}

    def test_returns_parsed_dict(self) -> None:
        """Dict body is returned as-is."""
        resp = _FakeResponse(status=200, body=b'{"sessionId": "x"}')
        client, _ = _make_client(resp)
        assert client.stop_runtime_session(ARN, BEARER, session_id=SESSION) == {
            "sessionId": "x"
        }

    def test_non_dict_body_wrapped(self) -> None:
        """A JSON body that isn't a dict is wrapped under 'response'."""
        resp = _FakeResponse(status=200, body=b'["a", "b"]')
        client, _ = _make_client(resp)
        assert client.stop_runtime_session(ARN, BEARER, session_id=SESSION) == {
            "response": ["a", "b"]
        }

    def test_invalid_json_body_returns_empty(self) -> None:
        """Invalid JSON body returns empty dict rather than raising."""
        resp = _FakeResponse(status=200, body=b"not json")
        client, _ = _make_client(resp)
        assert client.stop_runtime_session(ARN, BEARER, session_id=SESSION) == {}

    def test_404_raises(self) -> None:
        """Unknown-session HTTP 404 becomes AgentRuntimeError."""
        resp = _FakeResponse(status=404, body=b"{}", reason="Not Found")
        client, _ = _make_client(resp)
        with pytest.raises(AgentRuntimeError) as excinfo:
            client.stop_runtime_session(ARN, BEARER, session_id=SESSION)
        assert excinfo.value.error_type == "HTTP 404"


# --------------------------------------------------------------------- #
# _check_response
# --------------------------------------------------------------------- #


class TestCheckResponse:
    """Tests for _check_response."""

    def test_noop_on_2xx(self) -> None:
        """2xx responses do not raise."""
        AgentCoreRuntimeClient._check_response(_FakeResponse(status=200))

    def test_parses_json_error_body(self) -> None:
        """JSON error bodies populate error/error_type/message."""
        resp = _FakeResponse(
            status=400,
            body=b'{"error": "bad", "error_type": "Validation", "message": "details"}',
            reason="Bad Request",
        )
        with pytest.raises(AgentRuntimeError) as excinfo:
            AgentCoreRuntimeClient._check_response(resp)
        assert excinfo.value.error == "bad"
        assert excinfo.value.error_type == "Validation"

    def test_falls_back_to_text_body(self) -> None:
        """Non-JSON bodies become the error string."""
        resp = _FakeResponse(status=500, body=b"internal boom", reason="Server Error")
        with pytest.raises(AgentRuntimeError) as excinfo:
            AgentCoreRuntimeClient._check_response(resp)
        assert "internal boom" in excinfo.value.error
        assert excinfo.value.error_type == "HTTP 500"

    def test_defaults_error_type_for_dict_body(self) -> None:
        """Missing error_type in JSON body defaults to 'HTTP <status>'."""
        resp = _FakeResponse(status=502, body=b'{"message": "bad gateway"}')
        with pytest.raises(AgentRuntimeError) as excinfo:
            AgentCoreRuntimeClient._check_response(resp)
        assert excinfo.value.error_type == "HTTP 502"

    def test_empty_body_uses_reason(self) -> None:
        """Empty body falls back to the response reason string."""
        resp = _FakeResponse(status=503, body=b"", reason="Service Unavailable")
        with pytest.raises(AgentRuntimeError) as excinfo:
            AgentCoreRuntimeClient._check_response(resp)
        assert excinfo.value.error == "Service Unavailable"


# --------------------------------------------------------------------- #
# _decode_sse_line — all branches
# --------------------------------------------------------------------- #


class TestDecodeSseLine:
    """Tests for _decode_sse_line."""

    def test_empty_line(self) -> None:
        """Empty / whitespace-only lines return None."""
        assert AgentCoreRuntimeClient._decode_sse_line("") is None
        assert AgentCoreRuntimeClient._decode_sse_line("   ") is None

    def test_comment_line(self) -> None:
        """SSE comment lines (starting with ':') return None."""
        assert AgentCoreRuntimeClient._decode_sse_line(": keepalive") is None

    def test_data_with_json_encoded_string(self) -> None:
        """data: "hello" -> hello (unwrapped)."""
        assert AgentCoreRuntimeClient._decode_sse_line('data: "hello"') == "hello"

    def test_data_with_plain_text(self) -> None:
        """data: plain -> plain."""
        assert AgentCoreRuntimeClient._decode_sse_line("data: plain") == "plain"

    def test_data_with_malformed_json_string(self) -> None:
        """data: "bad" missing closing quote: strip what we can."""
        assert AgentCoreRuntimeClient._decode_sse_line('data: "oops') == '"oops'

    def test_data_with_malformed_quoted_string(self) -> None:
        """data: \"bad\\escape\" with invalid escape yields trimmed string."""
        assert (
            AgentCoreRuntimeClient._decode_sse_line('data: "bad\\escape"')
            == "bad\\escape"
        )

    def test_data_with_json_error_raises(self) -> None:
        """data: {"error": ...} raises AgentRuntimeError."""
        with pytest.raises(AgentRuntimeError) as excinfo:
            AgentCoreRuntimeClient._decode_sse_line(
                'data: {"error": "boom", "error_type": "T"}'
            )
        assert excinfo.value.error == "boom"

    def test_data_with_json_object_no_error(self) -> None:
        """data: {...} without an error key passes through as raw content."""
        assert (
            AgentCoreRuntimeClient._decode_sse_line('data: {"foo": "bar"}')
            == '{"foo": "bar"}'
        )

    def test_data_with_broken_json_passes_through(self) -> None:
        """data: { broken passes through as the content after 'data:'."""
        assert AgentCoreRuntimeClient._decode_sse_line("data: {not json") == "{not json"

    def test_plain_json_text_key(self) -> None:
        """Plain JSON line with 'text' key returns that value."""
        assert AgentCoreRuntimeClient._decode_sse_line('{"text": "hi"}') == "hi"

    def test_plain_json_content_key(self) -> None:
        """Plain JSON line with 'content' key returns that value."""
        assert AgentCoreRuntimeClient._decode_sse_line('{"content": "c"}') == "c"

    def test_plain_json_data_key(self) -> None:
        """Plain JSON line with 'data' key returns str(that value)."""
        assert AgentCoreRuntimeClient._decode_sse_line('{"data": 42}') == "42"

    def test_plain_json_error_raises(self) -> None:
        """Plain JSON line with 'error' raises."""
        with pytest.raises(AgentRuntimeError):
            AgentCoreRuntimeClient._decode_sse_line('{"error": "bad"}')

    def test_plain_json_unknown_shape_returns_none(self) -> None:
        """Plain JSON dict without known keys returns None."""
        assert AgentCoreRuntimeClient._decode_sse_line('{"something": "else"}') is None

    def test_plain_text_passes_through(self) -> None:
        """Non-JSON plain text is returned verbatim."""
        assert AgentCoreRuntimeClient._decode_sse_line("hello world") == "hello world"


# --------------------------------------------------------------------- #
# _iter_lines
# --------------------------------------------------------------------- #


class TestIterLines:
    """Tests for _iter_lines chunk splitting."""

    def test_splits_on_lf(self) -> None:
        """Splits on newline and yields each line."""
        resp = _FakeResponse(chunks=[b"one\ntwo\nthree\n"])
        lines = list(AgentCoreRuntimeClient._iter_lines(resp))
        assert lines == [b"one", b"two", b"three"]

    def test_strips_crlf(self) -> None:
        r"""Trailing \r is stripped."""
        resp = _FakeResponse(chunks=[b"a\r\nb\r\n"])
        lines = list(AgentCoreRuntimeClient._iter_lines(resp))
        assert lines == [b"a", b"b"]

    def test_trailing_without_newline(self) -> None:
        """A trailing fragment without newline is still yielded."""
        resp = _FakeResponse(chunks=[b"done"])
        lines = list(AgentCoreRuntimeClient._iter_lines(resp))
        assert lines == [b"done"]

    def test_trailing_cr_stripped(self) -> None:
        """A trailing CR without LF is stripped."""
        resp = _FakeResponse(chunks=[b"done\r"])
        lines = list(AgentCoreRuntimeClient._iter_lines(resp))
        assert lines == [b"done"]

    def test_split_across_chunks(self) -> None:
        """Lines spanning chunk boundaries are reassembled."""
        resp = _FakeResponse(chunks=[b"hel", b"lo\nwo", b"rld\n"])
        lines = list(AgentCoreRuntimeClient._iter_lines(resp))
        assert lines == [b"hello", b"world"]

    def test_empty_chunk_skipped(self) -> None:
        """Empty chunks in the stream are ignored."""
        resp = _FakeResponse(chunks=[b"a\n", b"", b"b\n"])
        lines = list(AgentCoreRuntimeClient._iter_lines(resp))
        assert lines == [b"a", b"b"]


# --------------------------------------------------------------------- #
# AgentRuntimeError
# --------------------------------------------------------------------- #


class TestAgentRuntimeError:
    """Tests for AgentRuntimeError construction and rendering."""

    def test_stores_error_and_type(self) -> None:
        """error and error_type are accessible as attributes."""
        exc = AgentRuntimeError(error="boom", error_type="T")
        assert exc.error == "boom"
        assert exc.error_type == "T"

    def test_default_error_type_empty(self) -> None:
        """error_type defaults to the empty string."""
        exc = AgentRuntimeError(error="boom")
        assert exc.error_type == ""

    def test_str_uses_message(self) -> None:
        """str() prefers the message when provided."""
        exc = AgentRuntimeError(error="e", message="human-readable")
        assert str(exc) == "human-readable"

    def test_str_falls_back_to_error(self) -> None:
        """str() falls back to the error token when no message."""
        exc = AgentRuntimeError(error="only-error")
        assert str(exc) == "only-error"

    def test_is_exception(self) -> None:
        """Subclasses Exception so it can be raised."""
        try:
            raise AgentRuntimeError(error="x")
        except Exception as exc:
            assert isinstance(exc, AgentRuntimeError)
