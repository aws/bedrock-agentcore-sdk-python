"""Tests for ConfigBundleClient."""

from unittest.mock import MagicMock

import pytest

from bedrock_agentcore.config_bundle.client import ConfigBundleClient


class TestConfigBundleClient:
    def test_boto_client_created_lazily_on_first_access(self, monkeypatch):
        # No override: boto3 resolves the endpoint natively (partition-correct).
        monkeypatch.setattr("bedrock_agentcore.config_bundle.client.CP_ENDPOINT_OVERRIDE", None)
        mock_session = MagicMock()
        mock_boto_client = MagicMock()
        mock_session.client.return_value = mock_boto_client

        client = ConfigBundleClient(region_name="us-east-1", boto3_session=mock_session)

        # No boto3 client created yet
        mock_session.client.assert_not_called()

        # Trigger lazy init via __getattr__ with an allowed operation
        _ = client.list_configuration_bundles

        mock_session.client.assert_called_once_with(
            "bedrock-agentcore-control",
            region_name="us-east-1",
        )

    def test_boto_client_honours_endpoint_override(self, monkeypatch):
        # With BEDROCK_AGENTCORE_CP_ENDPOINT set, the override must reach boto3.
        override = "https://bedrock-agentcore-control.gamma.example.com"
        monkeypatch.setattr("bedrock_agentcore.config_bundle.client.CP_ENDPOINT_OVERRIDE", override)
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()

        client = ConfigBundleClient(region_name="us-east-1", boto3_session=mock_session)
        _ = client.list_configuration_bundles

        mock_session.client.assert_called_once_with(
            "bedrock-agentcore-control",
            region_name="us-east-1",
            endpoint_url=override,
        )

    def test_boto_client_reused_across_calls(self):
        mock_session = MagicMock()
        mock_boto_client = MagicMock()
        mock_session.client.return_value = mock_boto_client

        client = ConfigBundleClient(region_name="us-east-1", boto3_session=mock_session)
        _ = client.list_configuration_bundles
        _ = client.list_configuration_bundles

        mock_session.client.assert_called_once()

    def test_getattr_forwards_to_boto_client(self):
        mock_session = MagicMock()
        mock_boto_client = MagicMock()
        mock_session.client.return_value = mock_boto_client

        client = ConfigBundleClient(region_name="us-east-1", boto3_session=mock_session)
        client.list_configuration_bundles(maxResults=10)

        mock_boto_client.list_configuration_bundles.assert_called_once_with(maxResults=10)

    def test_disallowed_operation_raises_attribute_error(self):
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()

        client = ConfigBundleClient(region_name="us-east-1", boto3_session=mock_session)

        with pytest.raises(AttributeError, match="does not expose operation 'create_evaluator'"):
            _ = client.create_evaluator
