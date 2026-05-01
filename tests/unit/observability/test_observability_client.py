"""Unit tests for ObservabilityClient."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.observability.client import ObservabilityClient


def _client_error(code, message="error"):
    return ClientError({"Error": {"Code": code, "Message": message}}, "op")


def _make_client():
    with patch("boto3.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        sts = MagicMock()
        sts.get_caller_identity.return_value = {"Account": "123456789012"}
        logs = MagicMock()
        xray = MagicMock()

        def client_factory(service, **kwargs):
            return {"sts": sts, "logs": logs, "xray": xray}[service]

        mock_session.client.side_effect = client_factory
        mock_session_class.return_value = mock_session
        client = ObservabilityClient(region_name="us-east-1", session=mock_session)
    return client, logs, xray


class TestInit:
    def test_init_with_region(self):
        client, _, _ = _make_client()
        assert client.region == "us-east-1"
        assert client._account_id == "123456789012"

    def test_init_without_region_raises(self):
        with patch("boto3.Session") as mock_cls:
            mock_session = MagicMock()
            mock_session.region_name = None
            mock_cls.return_value = mock_session
            with pytest.raises(ValueError, match="AWS region must be specified"):
                ObservabilityClient(session=mock_session)


class TestEnableObservability:
    def test_memory_success(self):
        client, logs, _ = _make_client()
        logs.put_delivery_source.return_value = {"deliverySource": {"name": "src"}}
        logs.put_delivery_destination.return_value = {
            "deliveryDestination": {
                "name": "dst",
                "arn": "arn:aws:logs:us-east-1:123456789012:delivery-destination:dst",
            }
        }
        logs.create_delivery.return_value = {"id": "d-123"}

        result = client.enable_observability_for_resource(
            resource_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/mem-1",
            resource_id="mem-1",
            resource_type="memory",
        )
        assert result["status"] == "success"
        assert result["logs_enabled"] is True
        assert result["traces_enabled"] is True
        assert result["log_group"] == "/aws/vendedlogs/bedrock-agentcore/memory/APPLICATION_LOGS/mem-1"

    def test_runtime_skips_log_creation(self):
        client, logs, _ = _make_client()
        logs.put_delivery_source.return_value = {"deliverySource": {"name": "src"}}
        logs.put_delivery_destination.return_value = {
            "deliveryDestination": {
                "name": "dst",
                "arn": "arn:aws:logs:us-east-1:123456789012:delivery-destination:dst",
            }
        }
        logs.create_delivery.return_value = {"id": "d-123"}

        result = client.enable_observability_for_resource(
            resource_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/rt-1",
            resource_id="rt-1",
            resource_type="runtime",
        )
        assert result["status"] == "success"
        assert result["logs_enabled"] is True
        assert result["deliveries"]["logs"] == {"status": "auto-created by AWS"}
        logs.create_log_group.assert_not_called()

    def test_logs_only(self):
        client, logs, _ = _make_client()
        logs.put_delivery_source.return_value = {"deliverySource": {"name": "src"}}
        logs.put_delivery_destination.return_value = {
            "deliveryDestination": {
                "name": "dst",
                "arn": "arn:aws:logs:us-east-1:123456789012:delivery-destination:dst",
            }
        }
        logs.create_delivery.return_value = {"id": "d-123"}

        result = client.enable_observability_for_resource(
            resource_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/mem-1",
            resource_id="mem-1",
            resource_type="memory",
            enable_traces=False,
        )
        assert result["status"] == "success"
        assert result["logs_enabled"] is True
        assert result["traces_enabled"] is False

    def test_traces_only(self):
        client, logs, _ = _make_client()
        logs.put_delivery_source.return_value = {"deliverySource": {"name": "src"}}
        logs.put_delivery_destination.return_value = {
            "deliveryDestination": {
                "name": "dst",
                "arn": "arn:aws:logs:us-east-1:123456789012:delivery-destination:dst",
            }
        }
        logs.create_delivery.return_value = {"id": "d-123"}

        result = client.enable_observability_for_resource(
            resource_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/mem-1",
            resource_id="mem-1",
            resource_type="memory",
            enable_logs=False,
        )
        assert result["status"] == "success"
        assert result["logs_enabled"] is False
        assert result["traces_enabled"] is True

    def test_custom_log_group(self):
        client, logs, _ = _make_client()
        logs.put_delivery_source.return_value = {"deliverySource": {"name": "src"}}
        logs.put_delivery_destination.return_value = {
            "deliveryDestination": {
                "name": "dst",
                "arn": "arn:aws:logs:us-east-1:123456789012:delivery-destination:dst",
            }
        }
        logs.create_delivery.return_value = {"id": "d-123"}

        result = client.enable_observability_for_resource(
            resource_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/mem-1",
            resource_id="mem-1",
            resource_type="memory",
            custom_log_group="/my/custom/group",
        )
        assert result["log_group"] == "/my/custom/group"

    def test_invalid_resource_type(self):
        client, _, _ = _make_client()
        with pytest.raises(ValueError, match="Unsupported resource_type"):
            client.enable_observability_for_resource(
                resource_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:invalid/x",
                resource_id="x",
                resource_type="invalid",
            )

    def test_parses_arn_when_ids_not_provided(self):
        client, logs, _ = _make_client()
        logs.put_delivery_source.return_value = {"deliverySource": {"name": "src"}}
        logs.put_delivery_destination.return_value = {
            "deliveryDestination": {
                "name": "dst",
                "arn": "arn:aws:logs:us-east-1:123456789012:delivery-destination:dst",
            }
        }
        logs.create_delivery.return_value = {"id": "d-123"}

        result = client.enable_observability_for_resource(
            resource_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/gw-99",
        )
        assert result["resource_type"] == "gateway"
        assert result["resource_id"] == "gw-99"

    def test_invalid_arn_raises(self):
        client, _, _ = _make_client()
        with pytest.raises(ValueError, match="Could not parse"):
            client.enable_observability_for_resource(resource_arn="bad-arn")

    def test_log_group_already_exists(self):
        client, logs, _ = _make_client()
        logs.create_log_group.side_effect = _client_error("ResourceAlreadyExistsException")
        logs.put_delivery_source.return_value = {"deliverySource": {"name": "src"}}
        logs.put_delivery_destination.return_value = {
            "deliveryDestination": {
                "name": "dst",
                "arn": "arn:aws:logs:us-east-1:123456789012:delivery-destination:dst",
            }
        }
        logs.create_delivery.return_value = {"id": "d-123"}

        result = client.enable_observability_for_resource(
            resource_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/mem-1",
            resource_id="mem-1",
            resource_type="memory",
        )
        assert result["status"] == "success"

    def test_delivery_already_exists(self):
        client, logs, _ = _make_client()
        logs.put_delivery_source.return_value = {"deliverySource": {"name": "src"}}
        logs.put_delivery_destination.return_value = {
            "deliveryDestination": {
                "name": "dst",
                "arn": "arn:aws:logs:us-east-1:123456789012:delivery-destination:dst",
            }
        }
        logs.create_delivery.side_effect = _client_error("ConflictException")

        result = client.enable_observability_for_resource(
            resource_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/mem-1",
            resource_id="mem-1",
            resource_type="memory",
        )
        assert result["status"] == "success"
        assert result["deliveries"]["logs"]["delivery_id"] == "existing"

    def test_api_error_returns_error_status(self):
        client, logs, _ = _make_client()
        logs.create_log_group.side_effect = _client_error("AccessDeniedException")

        result = client.enable_observability_for_resource(
            resource_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/mem-1",
            resource_id="mem-1",
            resource_type="memory",
        )
        assert result["status"] == "error"
        assert "AccessDeniedException" in result["error"]


class TestDisableObservability:
    def test_success(self):
        client, logs, _ = _make_client()
        result = client.disable_observability_for_resource(resource_id="mem-1")
        assert result["status"] == "success"
        assert len(result["deleted"]) == 4

    def test_resource_not_found_is_ok(self):
        client, logs, _ = _make_client()
        logs.delete_delivery_source.side_effect = _client_error("ResourceNotFoundException")
        logs.delete_delivery_destination.side_effect = _client_error("ResourceNotFoundException")

        result = client.disable_observability_for_resource(resource_id="nonexistent")
        assert result["status"] == "success"
        assert len(result["deleted"]) == 0

    def test_with_log_group_deletion(self):
        client, logs, _ = _make_client()
        result = client.disable_observability_for_resource(resource_id="mem-1", delete_log_group=True)
        assert result["status"] == "success"
        assert logs.delete_log_group.called


class TestEnableTransactionSearch:
    def test_all_steps_needed(self):
        client, logs, xray = _make_client()
        logs.describe_resource_policies.return_value = {"resourcePolicies": []}
        xray.get_trace_segment_destination.return_value = {"Destination": "XRay"}
        xray.get_indexing_rules.return_value = {"IndexingRules": []}

        assert client.enable_transaction_search() is True
        logs.put_resource_policy.assert_called_once()
        xray.update_trace_segment_destination.assert_called_once_with(Destination="CloudWatchLogs")
        xray.update_indexing_rule.assert_called_once()

    def test_all_already_configured(self):
        client, logs, xray = _make_client()
        logs.describe_resource_policies.return_value = {
            "resourcePolicies": [{"policyName": "TransactionSearchXRayAccess"}]
        }
        xray.get_trace_segment_destination.return_value = {"Destination": "CloudWatchLogs"}
        xray.get_indexing_rules.return_value = {"IndexingRules": [{"Name": "Default"}]}

        assert client.enable_transaction_search() is True
        logs.put_resource_policy.assert_not_called()
        xray.update_trace_segment_destination.assert_not_called()
        xray.update_indexing_rule.assert_not_called()

    def test_idempotent_on_invalid_request(self):
        client, logs, xray = _make_client()
        logs.describe_resource_policies.return_value = {"resourcePolicies": []}
        xray.get_trace_segment_destination.return_value = {"Destination": "XRay"}
        xray.update_trace_segment_destination.side_effect = _client_error("InvalidRequestException")
        xray.get_indexing_rules.return_value = {"IndexingRules": [{"Name": "Default"}]}

        assert client.enable_transaction_search() is True

    def test_failure_returns_false(self):
        client, logs, xray = _make_client()
        logs.describe_resource_policies.return_value = {"resourcePolicies": []}
        logs.put_resource_policy.side_effect = _client_error("AccessDeniedException")

        assert client.enable_transaction_search() is False


class TestGetObservabilityStatus:
    def test_both_configured(self):
        client, logs, _ = _make_client()
        logs.get_delivery_source.return_value = {"deliverySource": {"name": "src"}}

        result = client.get_observability_status(resource_id="mem-1")
        assert result["logs"]["configured"] is True
        assert result["traces"]["configured"] is True

    def test_none_configured(self):
        client, logs, _ = _make_client()
        logs.get_delivery_source.side_effect = _client_error("ResourceNotFoundException")

        result = client.get_observability_status(resource_id="mem-1")
        assert result["logs"]["configured"] is False
        assert result["traces"]["configured"] is False
