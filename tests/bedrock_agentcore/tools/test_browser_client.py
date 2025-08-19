import datetime
from unittest.mock import MagicMock, patch

from bedrock_agentcore.tools.browser_client import (
    MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT,
    BrowserClient,
    browser_session,
)


class TestBrowserClient:
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    def test_init(self, mock_get_endpoint, mock_boto3):
        # Arrange
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_get_endpoint.return_value = "https://mock-endpoint.com"
        region = "us-west-2"

        # Act
        client = BrowserClient(region)

        # Assert
        mock_boto3.client.assert_called_once_with(
            "bedrock-agentcore", region_name=region, endpoint_url="https://mock-endpoint.com"
        )
        assert client.client == mock_client
        assert client.region == region
        assert client.identifier is None
        assert client.session_id is None

    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_property_getters_setters(self, mock_boto3):
        # Arrange
        client = BrowserClient("us-west-2")
        test_identifier = "test.identifier"
        test_session_id = "test-session-id"

        # Act & Assert - identifier
        client.identifier = test_identifier
        assert client.identifier == test_identifier

        # Act & Assert - session_id
        client.session_id = test_session_id
        assert client.session_id == test_session_id

    @patch("bedrock_agentcore.tools.browser_client.boto3")
    @patch("bedrock_agentcore.tools.browser_client.uuid.uuid4")
    def test_start_with_defaults(self, mock_uuid4, mock_boto3):
        # Arrange
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_uuid4.return_value.hex = "12345678abcdef"

        client = BrowserClient("us-west-2")
        mock_response = {"browserIdentifier": "aws.browser.v1", "sessionId": "session-123"}
        mock_client.start_browser_session.return_value = mock_response

        # Act
        session_id = client.start()

        # Assert
        mock_client.start_browser_session.assert_called_once_with(
            browserIdentifier="aws.browser.v1",
            name="browser-session-12345678",
            sessionTimeoutSeconds=3600,
        )
        assert session_id == "session-123"
        assert client.identifier == "aws.browser.v1"
        assert client.session_id == "session-123"

    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_start_with_custom_params(self, mock_boto3):
        # Arrange
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        client = BrowserClient("us-west-2")
        mock_response = {"browserIdentifier": "custom.browser", "sessionId": "custom-session-123"}
        mock_client.start_browser_session.return_value = mock_response

        # Act
        session_id = client.start(identifier="custom.browser", name="custom-session", session_timeout_seconds=600)

        # Assert
        mock_client.start_browser_session.assert_called_once_with(
            browserIdentifier="custom.browser",
            name="custom-session",
            sessionTimeoutSeconds=600,
        )
        assert session_id == "custom-session-123"
        assert client.identifier == "custom.browser"
        assert client.session_id == "custom-session-123"

    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_stop_when_session_exists(self, mock_boto3):
        # Arrange
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        client = BrowserClient("us-west-2")
        client.identifier = "test.identifier"
        client.session_id = "test-session-id"

        # Act
        client.stop()

        # Assert
        mock_client.stop_browser_session.assert_called_once_with(
            browserIdentifier="test.identifier", sessionId="test-session-id"
        )
        assert client.identifier is None
        assert client.session_id is None

    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_stop_when_no_session(self, mock_boto3):
        # Arrange
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        client = BrowserClient("us-west-2")
        client.identifier = None
        client.session_id = None

        # Act
        result = client.stop()

        # Assert
        mock_client.stop_browser_session.assert_not_called()
        assert result is True

    @patch("bedrock_agentcore.tools.browser_client.boto3")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.datetime")
    @patch("bedrock_agentcore.tools.browser_client.base64")
    @patch("bedrock_agentcore.tools.browser_client.secrets")
    def test_get_ws_headers(self, mock_secrets, mock_base64, mock_datetime, mock_get_host, mock_boto3):
        # Arrange
        mock_boto_session = MagicMock()
        mock_credentials = MagicMock()
        mock_frozen_creds = MagicMock()
        mock_frozen_creds.token = "mock-token"
        mock_frozen_creds.access_key = "mock-access-key"
        mock_frozen_creds.secret_key = "mock-secret-key"
        mock_credentials.get_frozen_credentials.return_value = mock_frozen_creds
        mock_boto_session.get_credentials.return_value = mock_credentials
        mock_boto3.Session.return_value = mock_boto_session

        mock_get_host.return_value = "https://api.example.com"
        mock_datetime.datetime.now.return_value = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        mock_secrets.token_bytes.return_value = b"secrettoken"
        mock_base64.b64encode.return_value.decode.return_value = "c2VjcmV0dG9rZW4="

        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"
        client.session_id = "test-session-id"

        # Mock the SigV4Auth
        with patch("bedrock_agentcore.tools.browser_client.SigV4Auth") as mock_sigv4:
            mock_auth = MagicMock()
            mock_sigv4.return_value = mock_auth

            # Mock the request headers after auth
            auth_value = "AWS4-HMAC-SHA256 Credential=mock-access-key/20250101/us-west-2/bedrock-agentcore/aws4_request"
            mock_auth.add_auth.side_effect = lambda req: setattr(
                req,
                "headers",
                {
                    "x-amz-date": "20250101T120000Z",
                    "Authorization": auth_value,
                },
            )

            # Act
            url, headers = client.generate_ws_headers()

            # Assert
            assert url == "wss://api.example.com/browser-streams/test-browser-id/sessions/test-session-id/automation"
            assert headers["Host"] == "api.example.com"
            assert headers["X-Amz-Date"] == "20250101T120000Z"
            assert headers["Authorization"] == auth_value
            assert headers["Upgrade"] == "websocket"
            assert headers["Connection"] == "Upgrade"
            assert headers["Sec-WebSocket-Version"] == "13"
            assert headers["Sec-WebSocket-Key"] == "c2VjcmV0dG9rZW4="
            assert headers["User-Agent"] == "BrowserSandbox-Client/1.0 (Session: test-session-id)"
            assert headers["X-Amz-Security-Token"] == "mock-token"

    @patch("bedrock_agentcore.tools.browser_client.BrowserClient")
    def test_browser_session_context_manager(self, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Act
        with browser_session("us-west-2"):
            pass

        # Assert
        mock_client_class.assert_called_once_with("us-west-2")
        mock_client.start.assert_called_once()
        mock_client.stop.assert_called_once()

    @patch("bedrock_agentcore.tools.browser_client.boto3")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    def test_get_ws_headers_no_credentials(self, mock_get_endpoint, mock_boto3):
        # Arrange
        mock_boto_session = MagicMock()
        mock_boto_session.get_credentials.return_value = None  # No credentials
        mock_boto3.Session.return_value = mock_boto_session
        mock_get_endpoint.return_value = "https://api.example.com"

        client = BrowserClient("us-west-2")

        # Act & Assert
        try:
            client.generate_ws_headers()
            raise AssertionError("Expected RuntimeError")
        except RuntimeError as e:
            assert "No AWS credentials found" in str(e)

    @patch("bedrock_agentcore.tools.browser_client.boto3")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    def test_generate_live_view_url(self, mock_get_endpoint, mock_boto3):
        # Arrange
        mock_boto_session = MagicMock()
        mock_credentials = MagicMock()
        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "mock-access-key"
        mock_frozen_creds.secret_key = "mock-secret-key"
        mock_frozen_creds.token = "mock-token"
        mock_credentials.get_frozen_credentials.return_value = mock_frozen_creds
        mock_boto_session.get_credentials.return_value = mock_credentials
        mock_boto3.Session.return_value = mock_boto_session

        mock_get_endpoint.return_value = "https://api.example.com"

        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"
        client.session_id = "test-session-id"

        # Mock the SigV4QueryAuth
        with patch("bedrock_agentcore.tools.browser_client.SigV4QueryAuth") as mock_sigv4_query:
            mock_signer = MagicMock()
            mock_sigv4_query.return_value = mock_signer

            # Mock the request with signed URL
            mock_request = MagicMock()
            mock_request.url = "https://api.example.com/browser-sandbox-streams/test-browser-id/sessions/test-session-id/live-view?X-Amz-Signature=test-signature"

            with patch("bedrock_agentcore.tools.browser_client.AWSRequest", return_value=mock_request):
                mock_signer.add_auth.return_value = None

                # Act
                result_url = client.generate_live_view_url(expires=MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT)

                # Assert
                assert (
                    result_url
                    == "https://api.example.com/browser-sandbox-streams/test-browser-id/sessions/test-session-id/live-view?X-Amz-Signature=test-signature"
                )
                mock_sigv4_query.assert_called_once_with(
                    credentials=mock_frozen_creds,
                    service_name="bedrock-agentcore",
                    region_name="us-west-2",
                    expires=MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT,
                )

    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_generate_live_view_url_expires_validation_valid(self, mock_boto3):
        # Arrange
        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"
        client.session_id = "test-session-id"

        # Mock the dependencies for URL generation
        with (
            patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint") as mock_get_endpoint,
            patch("bedrock_agentcore.tools.browser_client.SigV4QueryAuth") as mock_sigv4_query,
            patch("bedrock_agentcore.tools.browser_client.AWSRequest") as mock_aws_request,
        ):
            mock_get_endpoint.return_value = "https://api.example.com"

            # Mock boto3 session and credentials
            mock_boto_session = MagicMock()
            mock_credentials = MagicMock()
            mock_frozen_creds = MagicMock()
            mock_credentials.get_frozen_credentials.return_value = mock_frozen_creds
            mock_boto_session.get_credentials.return_value = mock_credentials
            mock_boto3.Session.return_value = mock_boto_session

            # Mock the signer and request
            mock_signer = MagicMock()
            mock_sigv4_query.return_value = mock_signer

            mock_request = MagicMock()
            mock_request.url = "https://api.example.com/signed-url"
            mock_aws_request.return_value = mock_request

            # Act - test valid expires values
            for valid_expires in [1, 150, MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT]:
                result = client.generate_live_view_url(expires=valid_expires)
                # Assert
                assert result == "https://api.example.com/signed-url"

    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_generate_live_view_url_expires_validation_invalid(self, mock_boto3):
        # Arrange
        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"
        client.session_id = "test-session-id"

        # Act & Assert - test invalid expires values
        for invalid_expires in [MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT + 1, 500, 1000]:
            try:
                client.generate_live_view_url(expires=invalid_expires)
                raise AssertionError(f"Expected ValueError for expires={invalid_expires}")
            except ValueError as e:
                expected_msg = (
                    f"Expiry timeout cannot exceed {MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT} seconds, got {invalid_expires}"
                )
                assert expected_msg in str(e)

    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_take_control(self, mock_boto3):
        # Arrange
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"
        client.session_id = "test-session-id"

        # Act
        client.take_control()

        # Assert
        mock_client.update_browser_stream.assert_called_once_with(
            browserIdentifier="test-browser-id",
            sessionId="test-session-id",
            streamUpdate={"automationStreamUpdate": {"streamStatus": "DISABLED"}},
        )

    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_release_control(self, mock_boto3):
        # Arrange
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"
        client.session_id = "test-session-id"

        # Act
        client.release_control()

        # Assert
        mock_client.update_browser_stream.assert_called_once_with(
            browserIdentifier="test-browser-id",
            sessionId="test-session-id",
            streamUpdate={"automationStreamUpdate": {"streamStatus": "ENABLED"}},
        )
