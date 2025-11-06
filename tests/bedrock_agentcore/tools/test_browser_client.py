import datetime
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.tools.browser_client import (
    MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT,
    BrowserClient,
    browser_session,
)


class TestBrowserClient:
    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_init(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_control_client = MagicMock()
        mock_data_client = MagicMock()
        mock_boto3.client.side_effect = [mock_control_client, mock_data_client]
        mock_get_control_endpoint.return_value = "https://mock-control-endpoint.com"
        mock_get_data_endpoint.return_value = "https://mock-data-endpoint.com"
        region = "us-west-2"

        # Act
        client = BrowserClient(region)

        # Assert
        assert mock_boto3.client.call_count == 2
        mock_boto3.client.assert_any_call(
            "bedrock-agentcore-control",
            region_name=region,
            endpoint_url="https://mock-control-endpoint.com",
        )
        mock_boto3.client.assert_any_call(
            "bedrock-agentcore", region_name=region, endpoint_url="https://mock-data-endpoint.com"
        )
        assert client.control_plane_client == mock_control_client
        assert client.data_plane_client == mock_data_client
        assert client.region == region
        assert client.identifier is None
        assert client.session_id is None

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_property_getters_setters(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")
        test_identifier = "test.identifier"
        test_session_id = "test-session-id"

        # Act & Assert - identifier
        client.identifier = test_identifier
        assert client.identifier == test_identifier

        # Act & Assert - session_id
        client.session_id = test_session_id
        assert client.session_id == test_session_id

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_create_browser_minimal(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_control_client = MagicMock()
        mock_data_client = MagicMock()
        mock_boto3.client.side_effect = [mock_control_client, mock_data_client]
        client = BrowserClient("us-west-2")

        mock_response = {
            "browserArn": "arn:aws:bedrock-agentcore:us-west-2:123456789012:browser/test-browser",
            "browserId": "test-browser-123",
            "createdAt": datetime.datetime.now(),
            "status": "CREATING",
        }
        client.control_plane_client.create_browser.return_value = mock_response

        # Act
        result = client.create_browser(
            name="test_browser",
            execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
        )

        # Assert
        client.control_plane_client.create_browser.assert_called_once_with(
            name="test_browser",
            executionRoleArn="arn:aws:iam::123456789012:role/BrowserRole",
            networkConfiguration={"networkMode": "PUBLIC"},
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_create_browser_with_all_options(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_control_client = MagicMock()
        mock_data_client = MagicMock()
        mock_boto3.client.side_effect = [mock_control_client, mock_data_client]
        client = BrowserClient("us-west-2")

        mock_response = {
            "browserArn": "arn:aws:bedrock-agentcore:us-west-2:123456789012:browser/test-browser",
            "browserId": "test-browser-123",
            "createdAt": datetime.datetime.now(),
            "status": "CREATING",
        }
        client.control_plane_client.create_browser.return_value = mock_response

        network_config = {
            "networkMode": "VPC",
            "vpcConfig": {"securityGroups": ["sg-123"], "subnets": ["subnet-123"]},
        }
        recording_config = {"enabled": True, "s3Location": {"bucket": "test-bucket", "keyPrefix": "recordings/"}}
        browser_signing_config = {"enabled": True}
        tags = {"Environment": "Test"}

        # Act
        result = client.create_browser(
            name="test_browser",
            execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
            network_configuration=network_config,
            description="Test browser",
            recording=recording_config,
            browser_signing=browser_signing_config,
            tags=tags,
            client_token="test-token",
        )

        # Assert
        client.control_plane_client.create_browser.assert_called_once_with(
            name="test_browser",
            executionRoleArn="arn:aws:iam::123456789012:role/BrowserRole",
            networkConfiguration=network_config,
            description="Test browser",
            recording=recording_config,
            browserSigning=browser_signing_config,
            tags=tags,
            clientToken="test-token",
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_delete_browser(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")

        mock_response = {
            "browserId": "test-browser-123",
            "lastUpdatedAt": datetime.datetime.now(),
            "status": "DELETING",
        }
        client.control_plane_client.delete_browser.return_value = mock_response

        # Act
        result = client.delete_browser("test-browser-123")

        # Assert
        client.control_plane_client.delete_browser.assert_called_once_with(browserId="test-browser-123")
        assert result == mock_response

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_delete_browser_with_client_token(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")

        mock_response = {
            "browserId": "test-browser-123",
            "lastUpdatedAt": datetime.datetime.now(),
            "status": "DELETING",
        }
        client.control_plane_client.delete_browser.return_value = mock_response

        # Act
        result = client.delete_browser("test-browser-123", client_token="test-token")

        # Assert
        client.control_plane_client.delete_browser.assert_called_once_with(
            browserId="test-browser-123", clientToken="test-token"
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_get_browser(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")

        mock_response = {
            "browserArn": "arn:aws:bedrock-agentcore:us-west-2:123456789012:browser/test-browser",
            "browserId": "test-browser-123",
            "name": "test_browser",
            "status": "READY",
            "browserSigning": {"enabled": True},
        }
        client.control_plane_client.get_browser.return_value = mock_response

        # Act
        result = client.get_browser("test-browser-123")

        # Assert
        client.control_plane_client.get_browser.assert_called_once_with(browserId="test-browser-123")
        assert result == mock_response

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_list_browsers_default(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")

        mock_response = {
            "browserSummaries": [
                {"browserId": "browser-1", "name": "browser_1", "status": "READY"},
                {"browserId": "browser-2", "name": "browser_2", "status": "CREATING"},
            ]
        }
        client.control_plane_client.list_browsers.return_value = mock_response

        # Act
        result = client.list_browsers()

        # Assert
        client.control_plane_client.list_browsers.assert_called_once_with(maxResults=10)
        assert result == mock_response

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_list_browsers_with_filters(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")

        mock_response = {
            "browserSummaries": [{"browserId": "browser-1", "name": "browser_1", "status": "READY"}],
            "nextToken": "next-page-token",
        }
        client.control_plane_client.list_browsers.return_value = mock_response

        # Act
        result = client.list_browsers(browser_type="CUSTOM", max_results=50, next_token="token-123")

        # Assert
        client.control_plane_client.list_browsers.assert_called_once_with(
            maxResults=50, type="CUSTOM", nextToken="token-123"
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    @patch("bedrock_agentcore.tools.browser_client.uuid.uuid4")
    def test_start_with_defaults(self, mock_uuid4, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        mock_uuid4.return_value.hex = "12345678abcdef"

        client = BrowserClient("us-west-2")
        mock_response = {"browserIdentifier": "aws.browser.v1", "sessionId": "session-123"}
        client.data_plane_client.start_browser_session.return_value = mock_response

        # Act
        session_id = client.start()

        # Assert
        client.data_plane_client.start_browser_session.assert_called_once_with(
            browserIdentifier="aws.browser.v1",
            name="browser-session-12345678",
            sessionTimeoutSeconds=3600,
        )
        assert session_id == "session-123"
        assert client.identifier == "aws.browser.v1"
        assert client.session_id == "session-123"

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    @patch("bedrock_agentcore.tools.browser_client.uuid.uuid4")
    def test_start_with_custom_params(self, mock_uuid4, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        mock_uuid4.return_value.hex = "12345678abcdef"

        client = BrowserClient("us-west-2")
        mock_response = {"browserIdentifier": "custom.browser", "sessionId": "custom-session-123"}
        client.data_plane_client.start_browser_session.return_value = mock_response

        # Act
        session_id = client.start(identifier="custom.browser", name="custom-session", session_timeout_seconds=600)

        # Assert
        client.data_plane_client.start_browser_session.assert_called_once_with(
            browserIdentifier="custom.browser",
            name="custom-session",
            sessionTimeoutSeconds=600,
        )
        assert session_id == "custom-session-123"
        assert client.identifier == "custom.browser"
        assert client.session_id == "custom-session-123"

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_stop_when_session_exists(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")
        client.identifier = "test.identifier"
        client.session_id = "test-session-id"

        # Act
        client.stop()

        # Assert
        client.data_plane_client.stop_browser_session.assert_called_once_with(
            browserIdentifier="test.identifier", sessionId="test-session-id"
        )
        assert client.identifier is None
        assert client.session_id is None

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_stop_when_no_session(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")
        client.identifier = None
        client.session_id = None

        # Act
        result = client.stop()

        # Assert
        client.data_plane_client.stop_browser_session.assert_not_called()
        assert result is True

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_get_session(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"
        client.session_id = "test-session-id"

        mock_response = {
            "sessionId": "test-session-id",
            "browserIdentifier": "test-browser-id",
            "status": "READY",
        }
        client.data_plane_client.get_browser_session.return_value = mock_response

        # Act
        result = client.get_session()

        # Assert
        client.data_plane_client.get_browser_session.assert_called_once_with(
            browserIdentifier="test-browser-id", sessionId="test-session-id"
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_get_session_with_params(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")

        mock_response = {
            "sessionId": "other-session-id",
            "browserIdentifier": "other-browser-id",
            "status": "READY",
        }
        client.data_plane_client.get_browser_session.return_value = mock_response

        # Act
        result = client.get_session(browser_id="other-browser-id", session_id="other-session-id")

        # Assert
        client.data_plane_client.get_browser_session.assert_called_once_with(
            browserIdentifier="other-browser-id", sessionId="other-session-id"
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_get_session_missing_ids(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")

        # Act & Assert
        with pytest.raises(ValueError, match="Browser ID and Session ID must be provided"):
            client.get_session()

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_list_sessions(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"

        mock_response = {
            "items": [
                {"sessionId": "session-1", "status": "READY"},
                {"sessionId": "session-2", "status": "TERMINATED"},
            ]
        }
        client.data_plane_client.list_browser_sessions.return_value = mock_response

        # Act
        result = client.list_sessions()

        # Assert
        client.data_plane_client.list_browser_sessions.assert_called_once_with(
            browserIdentifier="test-browser-id", maxResults=10
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_list_sessions_with_filters(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")

        mock_response = {
            "items": [{"sessionId": "session-1", "status": "READY"}],
            "nextToken": "next-token",
        }
        client.data_plane_client.list_browser_sessions.return_value = mock_response

        # Act
        result = client.list_sessions(
            browser_id="custom-browser",
            status="READY",
            max_results=50,
            next_token="token-123",
        )

        # Assert
        client.data_plane_client.list_browser_sessions.assert_called_once_with(
            browserIdentifier="custom-browser",
            maxResults=50,
            status="READY",
            nextToken="token-123",
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_list_sessions_missing_browser_id(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")

        # Act & Assert
        with pytest.raises(ValueError, match="Browser ID must be provided"):
            client.list_sessions()

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_update_stream(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"
        client.session_id = "test-session-id"

        # Act
        client.update_stream("DISABLED")

        # Assert
        client.data_plane_client.update_browser_stream.assert_called_once_with(
            browserIdentifier="test-browser-id",
            sessionId="test-session-id",
            streamUpdate={"automationStreamUpdate": {"streamStatus": "DISABLED"}},
        )

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_update_stream_with_params(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")

        # Act
        client.update_stream("ENABLED", browser_id="custom-browser", session_id="custom-session")

        # Assert
        client.data_plane_client.update_browser_stream.assert_called_once_with(
            browserIdentifier="custom-browser",
            sessionId="custom-session",
            streamUpdate={"automationStreamUpdate": {"streamStatus": "ENABLED"}},
        )

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_update_stream_missing_ids(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")

        # Act & Assert
        with pytest.raises(ValueError, match="Browser ID and Session ID must be provided"):
            client.update_stream("DISABLED")

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    @patch("bedrock_agentcore.tools.browser_client.datetime")
    @patch("bedrock_agentcore.tools.browser_client.base64")
    @patch("bedrock_agentcore.tools.browser_client.secrets")
    def test_get_ws_headers(
        self,
        mock_secrets,
        mock_base64,
        mock_datetime,
        mock_boto3,
        mock_get_data_endpoint,
        mock_get_control_endpoint,
    ):
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

        mock_get_data_endpoint.return_value = "https://api.example.com"
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
        mock_client.start.assert_called_once_with()
        mock_client.stop.assert_called_once()

    @patch("bedrock_agentcore.tools.browser_client.BrowserClient")
    def test_browser_session_context_manager_with_identifier(self, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Act
        with browser_session("us-west-2", identifier="custom-browser-123"):
            pass

        # Assert
        mock_client_class.assert_called_once_with("us-west-2")
        mock_client.start.assert_called_once_with(identifier="custom-browser-123")  # ✅ CORRECT
        mock_client.stop.assert_called_once()

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_get_ws_headers_no_credentials(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto_session = MagicMock()
        mock_boto_session.get_credentials.return_value = None  # No credentials
        mock_boto3.Session.return_value = mock_boto_session
        mock_get_data_endpoint.return_value = "https://api.example.com"

        client = BrowserClient("us-west-2")

        # Act & Assert
        with pytest.raises(RuntimeError, match="No AWS credentials found"):
            client.generate_ws_headers()

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_generate_live_view_url(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
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

        mock_get_data_endpoint.return_value = "https://api.example.com"

        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"
        client.session_id = "test-session-id"

        # Mock the SigV4QueryAuth
        with patch("bedrock_agentcore.tools.browser_client.SigV4QueryAuth") as mock_sigv4_query:
            mock_signer = MagicMock()
            mock_sigv4_query.return_value = mock_signer

            # Mock the request with signed URL
            mock_request = MagicMock()
            mock_request.url = "https://api.example.com/browser-streams/test-browser-id/sessions/test-session-id/live-view?X-Amz-Signature=test-signature"

            with patch("bedrock_agentcore.tools.browser_client.AWSRequest", return_value=mock_request):
                mock_signer.add_auth.return_value = None

                # Act
                result_url = client.generate_live_view_url(expires=MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT)

                # Assert
                assert (
                    result_url
                    == "https://api.example.com/browser-streams/test-browser-id/sessions/test-session-id/live-view?X-Amz-Signature=test-signature"
                )
                mock_sigv4_query.assert_called_once_with(
                    credentials=mock_frozen_creds,
                    service_name="bedrock-agentcore",
                    region_name="us-west-2",
                    expires=MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT,
                )

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_generate_live_view_url_expires_validation_valid(
        self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint
    ):
        # Arrange
        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"
        client.session_id = "test-session-id"

        # Mock boto3 session and credentials
        mock_boto_session = MagicMock()
        mock_credentials = MagicMock()
        mock_frozen_creds = MagicMock()
        mock_credentials.get_frozen_credentials.return_value = mock_frozen_creds
        mock_boto_session.get_credentials.return_value = mock_credentials
        mock_boto3.Session.return_value = mock_boto_session

        mock_get_data_endpoint.return_value = "https://api.example.com"

        # Mock the signer and request
        with (
            patch("bedrock_agentcore.tools.browser_client.SigV4QueryAuth") as mock_sigv4_query,
            patch("bedrock_agentcore.tools.browser_client.AWSRequest") as mock_aws_request,
        ):
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

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_generate_live_view_url_expires_validation_invalid(
        self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint
    ):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"
        client.session_id = "test-session-id"

        # Act & Assert - test invalid expires values
        for invalid_expires in [MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT + 1, 500, 1000]:
            expected_msg = (
                f"Expiry timeout cannot exceed {MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT} seconds, got {invalid_expires}"
            )
            with pytest.raises(ValueError, match=expected_msg):
                client.generate_live_view_url(expires=invalid_expires)

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_take_control(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"
        client.session_id = "test-session-id"

        # Act
        client.take_control()

        # Assert
        client.data_plane_client.update_browser_stream.assert_called_once_with(
            browserIdentifier="test-browser-id",
            sessionId="test-session-id",
            streamUpdate={"automationStreamUpdate": {"streamStatus": "DISABLED"}},
        )

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    def test_release_control(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        client = BrowserClient("us-west-2")
        client.identifier = "test-browser-id"
        client.session_id = "test-session-id"

        # Act
        client.release_control()

        # Assert
        client.data_plane_client.update_browser_stream.assert_called_once_with(
            browserIdentifier="test-browser-id",
            sessionId="test-session-id",
            streamUpdate={"automationStreamUpdate": {"streamStatus": "ENABLED"}},
        )

    @patch("bedrock_agentcore.tools.browser_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.browser_client.boto3")
    @patch("bedrock_agentcore.tools.browser_client.uuid.uuid4")
    def test_start_with_viewport(self, mock_uuid4, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_boto3.client.return_value = MagicMock()
        mock_uuid4.return_value.hex = "12345678abcdef"

        client = BrowserClient("us-west-2")
        mock_response = {"browserIdentifier": "aws.browser.v1", "sessionId": "session-123"}
        client.data_plane_client.start_browser_session.return_value = mock_response
        viewport = {"width": 1920, "height": 1080}

        # Act
        session_id = client.start(viewport=viewport)

        # Assert
        client.data_plane_client.start_browser_session.assert_called_once_with(
            browserIdentifier="aws.browser.v1",
            name="browser-session-12345678",
            sessionTimeoutSeconds=3600,
            viewPort=viewport,
        )
        assert session_id == "session-123"
        assert client.identifier == "aws.browser.v1"
        assert client.session_id == "session-123"

    @patch("bedrock_agentcore.tools.browser_client.BrowserClient")
    def test_browser_session_context_manager_with_viewport(self, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        viewport = {"width": 1280, "height": 720}

        # Act
        with browser_session("us-west-2", viewport=viewport):
            pass

        # Assert
        mock_client_class.assert_called_once_with("us-west-2")
        mock_client.start.assert_called_once_with(viewport=viewport)
        mock_client.stop.assert_called_once()

    @patch("bedrock_agentcore.tools.browser_client.BrowserClient")
    def test_browser_session_context_manager_with_all_params(self, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        viewport = {"width": 1280, "height": 720}

        # Act
        with browser_session("us-west-2", viewport=viewport, identifier="custom-browser"):
            pass

        # Assert
        mock_client_class.assert_called_once_with("us-west-2")
        mock_client.start.assert_called_once_with(viewport=viewport, identifier="custom-browser")
        mock_client.stop.assert_called_once()
