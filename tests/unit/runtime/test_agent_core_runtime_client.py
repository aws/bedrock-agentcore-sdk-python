"""Tests for AgentCoreRuntimeClient."""

from unittest.mock import Mock, patch
from urllib.parse import quote

import pytest

from bedrock_agentcore.runtime.agent_core_runtime_client import AgentCoreRuntimeClient


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
        arn = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime-abc123"

        result = client._parse_runtime_arn(arn)

        assert result["region"] == "us-west-2"
        assert result["account_id"] == "123456789012"
        assert result["runtime_id"] == "my-runtime-abc123"

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

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_build_basic_url(self, mock_endpoint):
        """Test building basic WebSocket URL without query params."""
        mock_endpoint.return_value = "https://example.aws.dev"
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        result = client._build_websocket_url(runtime_arn)

        # ARN should be URL encoded
        encoded_arn = quote(runtime_arn, safe="")
        assert result == f"wss://example.aws.dev/runtimes/{encoded_arn}/ws"

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_build_url_with_endpoint_name(self, mock_endpoint):
        """Test building URL with endpoint name (qualifier param)."""
        mock_endpoint.return_value = "https://example.aws.dev"
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        result = client._build_websocket_url(runtime_arn, endpoint_name="DEFAULT")

        encoded_arn = quote(runtime_arn, safe="")
        assert result == f"wss://example.aws.dev/runtimes/{encoded_arn}/ws?qualifier=DEFAULT"

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_build_url_with_custom_headers(self, mock_endpoint):
        """Test building URL with custom headers as query params."""
        mock_endpoint.return_value = "https://example.aws.dev"
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        result = client._build_websocket_url(runtime_arn, custom_headers={"abc": "pqr", "foo": "bar"})

        encoded_arn = quote(runtime_arn, safe="")
        assert f"wss://example.aws.dev/runtimes/{encoded_arn}/ws?" in result
        assert "abc=pqr" in result
        assert "foo=bar" in result

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_build_url_with_all_params(self, mock_endpoint):
        """Test building URL with endpoint name and custom headers."""
        mock_endpoint.return_value = "https://example.aws.dev"
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        result = client._build_websocket_url(runtime_arn, endpoint_name="DEFAULT", custom_headers={"abc": "pqr"})

        encoded_arn = quote(runtime_arn, safe="")
        assert f"wss://example.aws.dev/runtimes/{encoded_arn}/ws?" in result
        assert "qualifier=DEFAULT" in result
        assert "abc=pqr" in result


class TestGenerateWsConnection:
    """Tests for generate_ws_connection method."""

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_generate_basic_connection(self, mock_endpoint, mock_session):
        """Test generating basic WebSocket connection."""
        # Setup mocks
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(access_key="AKIATEST", secret_key="secret", token=None)
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
    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_generate_connection_with_session_id(self, mock_endpoint, mock_session):
        """Test generating connection with explicit session ID."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(access_key="AKIATEST", secret_key="secret", token=None)
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        ws_url, headers = client.generate_ws_connection(runtime_arn, session_id="test-session-123")

        assert ws_url is not None
        assert headers is not None

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_generate_connection_with_endpoint_name(self, mock_endpoint, mock_session):
        """Test generating connection with endpoint name."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(access_key="AKIATEST", secret_key="secret", token=None)
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        ws_url, headers = client.generate_ws_connection(runtime_arn, endpoint_name="DEFAULT")

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
    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_generate_basic_presigned_url(self, mock_endpoint, mock_session):
        """Test generating basic presigned URL."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(access_key="AKIATEST", secret_key="secret", token=None)
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
    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_generate_presigned_url_with_endpoint_name(self, mock_endpoint, mock_session):
        """Test generating presigned URL with endpoint name."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(access_key="AKIATEST", secret_key="secret", token=None)
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        presigned_url = client.generate_presigned_url(runtime_arn, endpoint_name="DEFAULT")

        assert "qualifier=DEFAULT" in presigned_url

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_generate_presigned_url_with_custom_headers(self, mock_endpoint, mock_session):
        """Test generating presigned URL with custom headers."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(access_key="AKIATEST", secret_key="secret", token=None)
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        presigned_url = client.generate_presigned_url(runtime_arn, custom_headers={"abc": "pqr"})

        assert "abc=pqr" in presigned_url

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.boto3.Session")
    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_generate_presigned_url_with_custom_expires(self, mock_endpoint, mock_session):
        """Test generating presigned URL with custom expiration."""
        mock_endpoint.return_value = "https://example.aws.dev"
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(access_key="AKIATEST", secret_key="secret", token=None)
        mock_session.return_value.get_credentials.return_value = mock_creds

        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"

        presigned_url = client.generate_presigned_url(runtime_arn, expires=60)

        assert "X-Amz-Expires=60" in presigned_url

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

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_generate_ws_connection_uses_custom_session(self, mock_endpoint):
        """Test that generate_ws_connection uses the custom session."""
        mock_endpoint.return_value = "https://example.aws.dev"

        # Create custom session with credentials
        custom_session = Mock()
        mock_creds = Mock()
        mock_creds.get_frozen_credentials.return_value = Mock(access_key="AKIATEST", secret_key="secret", token=None)
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

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
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

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_generate_oauth_connection_with_session_id(self, mock_endpoint):
        """Test generating OAuth connection with explicit session ID."""
        mock_endpoint.return_value = "https://example.aws.dev"
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"
        bearer_token = "test-token"
        custom_session_id = "custom-oauth-session-123"

        ws_url, headers = client.generate_ws_connection_oauth(runtime_arn, bearer_token, session_id=custom_session_id)

        assert headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == custom_session_id

    @patch("bedrock_agentcore.runtime.agent_core_runtime_client.get_data_plane_endpoint")
    def test_generate_oauth_connection_with_endpoint_name(self, mock_endpoint):
        """Test generating OAuth connection with endpoint name."""
        mock_endpoint.return_value = "https://example.aws.dev"
        client = AgentCoreRuntimeClient(region="us-west-2")
        runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime"
        bearer_token = "test-token"

        ws_url, headers = client.generate_ws_connection_oauth(runtime_arn, bearer_token, endpoint_name="DEFAULT")

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
