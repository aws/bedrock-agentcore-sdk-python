"""Tests for ConfigBundleClient."""

import pytest
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

        # Trigger lazy init via __getattr__ with an allowed operation
        _ = client.list_configuration_bundles

        mock_session.client.assert_called_once_with(
            "agentcore-evaluation-controlplane",
            region_name="us-east-1",
            endpoint_url="https://bedrock-agentcore-control.us-east-1.amazonaws.com",
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
