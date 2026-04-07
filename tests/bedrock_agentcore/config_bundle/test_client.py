"""Tests for ConfigBundleClient."""

from unittest.mock import MagicMock

from bedrock_agentcore.config_bundle.client import ConfigBundleClient


class TestConfigBundleClient:
    def test_boto_client_created_lazily_on_first_access(self):
        mock_session = MagicMock()
        mock_boto_client = MagicMock()
        mock_session.client.return_value = mock_boto_client

        client = ConfigBundleClient(region_name="us-east-1", boto3_session=mock_session)

        # No boto3 client created yet
        mock_session.client.assert_not_called()

        # Trigger lazy init via __getattr__
        _ = client.some_method

        mock_session.client.assert_called_once_with(
            "bedrock-agentcore-control",
            region_name="us-east-1",
            endpoint_url="https://bedrock-agentcore-control.us-east-1.amazonaws.com",
        )

    def test_boto_client_reused_across_calls(self):
        mock_session = MagicMock()
        mock_boto_client = MagicMock()
        mock_session.client.return_value = mock_boto_client

        client = ConfigBundleClient(region_name="us-east-1", boto3_session=mock_session)
        _ = client.some_method
        _ = client.some_method

        mock_session.client.assert_called_once()

    def test_getattr_forwards_to_boto_client(self):
        mock_session = MagicMock()
        mock_boto_client = MagicMock()
        mock_session.client.return_value = mock_boto_client

        client = ConfigBundleClient(region_name="us-east-1", boto3_session=mock_session)
        client.list_configuration_bundles(maxResults=10)

        mock_boto_client.list_configuration_bundles.assert_called_once_with(maxResults=10)
