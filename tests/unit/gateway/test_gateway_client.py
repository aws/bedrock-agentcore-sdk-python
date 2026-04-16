"""Tests for GatewayClient."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.gateway.client import GatewayClient


class TestGatewayClientInit:
    def test_init_with_region(self):
        mock_session = MagicMock()
        mock_session.region_name = "eu-west-1"
        client = GatewayClient(region_name="us-west-2", boto3_session=mock_session)
        assert client.region_name == "us-west-2"

    def test_init_default_region_fallback(self):
        mock_session = MagicMock()
        mock_session.region_name = None
        client = GatewayClient(boto3_session=mock_session)
        assert client.region_name == "us-west-2"


class TestGatewayClientPassthrough:
    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = GatewayClient(boto3_session=mock_session)
        client.cp_client = Mock()
        return client

    def test_cp_method_forwarded(self):
        client = self._make_client()
        client.cp_client.create_gateway.return_value = {"gatewayId": "gw-123"}
        result = client.create_gateway(name="test")
        client.cp_client.create_gateway.assert_called_once_with(name="test")
        assert result["gatewayId"] == "gw-123"

    def test_snake_case_kwargs_converted(self):
        client = self._make_client()
        client.cp_client.get_gateway.return_value = {"gatewayId": "gw-123"}
        client.get_gateway(gateway_identifier="gw-123")
        client.cp_client.get_gateway.assert_called_once_with(gatewayIdentifier="gw-123")

    def test_non_allowlisted_method_raises(self):
        client = self._make_client()
        with pytest.raises(AttributeError):
            client.not_a_real_method()


class TestGatewayListAll:
    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = GatewayClient(boto3_session=mock_session)
        client.cp_client = Mock()
        return client

    def test_list_all_gateways(self):
        client = self._make_client()
        client.cp_client.list_gateways.return_value = {
            "items": [{"gatewayId": "gw-1"}],
        }
        result = client.list_all_gateways()
        assert len(result) == 1
        assert result[0]["gatewayId"] == "gw-1"

    def test_list_all_gateway_targets(self):
        client = self._make_client()
        client.cp_client.list_gateway_targets.return_value = {
            "items": [{"targetId": "t-1"}],
        }
        result = client.list_all_gateway_targets(gatewayIdentifier="gw-1")
        assert len(result) == 1


class TestGatewayAndWait:
    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = GatewayClient(boto3_session=mock_session)
        client.cp_client = Mock()
        return client

    def test_create_gateway_and_wait(self):
        client = self._make_client()
        client.cp_client.create_gateway.return_value = {"gatewayId": "gw-123"}
        client.cp_client.get_gateway.return_value = {"status": "READY", "gatewayId": "gw-123"}

        result = client.create_gateway_and_wait(name="test")
        assert result["status"] == "READY"

    def test_create_gateway_and_wait_failed(self):
        client = self._make_client()
        client.cp_client.create_gateway.return_value = {"gatewayId": "gw-123"}
        client.cp_client.get_gateway.return_value = {
            "status": "FAILED",
            "statusReasons": ["bad config"],
        }

        with pytest.raises(RuntimeError, match="FAILED"):
            client.create_gateway_and_wait(name="test")

    def test_create_gateway_target_and_wait(self):
        client = self._make_client()
        client.cp_client.create_gateway_target.return_value = {
            "gatewayArn": "arn:gw",
            "targetId": "t-123",
        }
        client.cp_client.get_gateway_target.return_value = {
            "status": "READY",
            "targetId": "t-123",
        }

        result = client.create_gateway_target_and_wait(gatewayIdentifier="gw-1")
        assert result["status"] == "READY"

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    def test_delete_gateway_and_wait(self, _mock_sleep):
        client = self._make_client()
        client.cp_client.delete_gateway.return_value = {"gatewayId": "gw-123"}
        client.cp_client.get_gateway.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "gone"}},
            "GetGateway",
        )

        client.delete_gateway_and_wait(gatewayIdentifier="gw-123")
        client.cp_client.delete_gateway.assert_called_once()


class TestGatewayNameLookup:
    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = GatewayClient(boto3_session=mock_session)
        client.cp_client = Mock()
        return client

    def test_get_gateway_by_name_found(self):
        client = self._make_client()
        client.cp_client.list_gateways.return_value = {
            "items": [{"name": "my-gw", "gatewayId": "gw-123"}],
        }
        client.cp_client.get_gateway.return_value = {
            "gatewayId": "gw-123",
            "name": "my-gw",
        }

        result = client.get_gateway_by_name(name="my-gw")
        assert result["gatewayId"] == "gw-123"

    def test_get_gateway_by_name_not_found(self):
        client = self._make_client()
        client.cp_client.list_gateways.return_value = {"items": []}

        result = client.get_gateway_by_name(name="nonexistent")
        assert result is None
