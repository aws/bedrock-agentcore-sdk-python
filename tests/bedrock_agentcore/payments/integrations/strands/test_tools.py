"""Tests for payment tools utilities."""

import json
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.payments.integrations.strands.tools import (
    format_error_response,
    format_success_response,
    validate_required_params,
)


class TestValidateRequiredParams:
    """Test validate_required_params function."""

    def test_valid_required_params(self):
        """Test validation passes with all required parameters."""
        params = {"user_id": "test-user", "payment_id": "test-payment"}
        result = validate_required_params(params, required=["user_id", "payment_id"])
        assert result is None

    def test_missing_required_param(self):
        """Test validation fails when required parameter is missing."""
        params = {"user_id": "test-user"}
        result = validate_required_params(params, required=["user_id", "payment_id"])
        assert result is not None
        assert result["error"] == "ValidationError"
        assert "Missing required parameter: payment_id" in result["message"]

    def test_empty_string_required_param(self):
        """Test validation fails when required parameter is empty string."""
        params = {"user_id": "", "payment_id": "test-payment"}
        result = validate_required_params(params, required=["user_id", "payment_id"])
        assert result is not None
        assert result["error"] == "ValidationError"
        assert "Parameter cannot be empty: user_id" in result["message"]

    def test_whitespace_only_required_param(self):
        """Test validation fails when required parameter is whitespace only."""
        params = {"user_id": "   ", "payment_id": "test-payment"}
        result = validate_required_params(params, required=["user_id", "payment_id"])
        assert result is not None
        assert result["error"] == "ValidationError"
        assert "Parameter cannot be empty: user_id" in result["message"]

    def test_optional_params_not_provided(self):
        """Test validation passes when optional parameters are not provided."""
        params = {"user_id": "test-user"}
        result = validate_required_params(params, required=["user_id"], optional=["payment_id", "connector_id"])
        assert result is None

    def test_optional_params_provided_valid(self):
        """Test validation passes when optional parameters are provided and valid."""
        params = {"user_id": "test-user", "payment_id": "test-payment", "connector_id": "test-connector"}
        result = validate_required_params(params, required=["user_id"], optional=["payment_id", "connector_id"])
        assert result is None

    def test_optional_param_empty_string(self):
        """Test validation fails when optional parameter is empty string."""
        params = {"user_id": "test-user", "payment_id": ""}
        result = validate_required_params(params, required=["user_id"], optional=["payment_id"])
        assert result is not None
        assert result["error"] == "ValidationError"
        assert "Parameter cannot be empty: payment_id" in result["message"]

    def test_optional_param_whitespace_only(self):
        """Test validation fails when optional parameter is whitespace only."""
        params = {"user_id": "test-user", "payment_id": "   "}
        result = validate_required_params(params, required=["user_id"], optional=["payment_id"])
        assert result is not None
        assert result["error"] == "ValidationError"
        assert "Parameter cannot be empty: payment_id" in result["message"]

    def test_non_string_optional_param_not_validated(self):
        """Test validation passes for non-string optional parameters."""
        params = {"user_id": "test-user", "max_results": 100}
        result = validate_required_params(params, required=["user_id"], optional=["max_results"])
        assert result is None

    def test_non_string_required_param_not_validated(self):
        """Test validation passes for non-string required parameters."""
        params = {"user_id": "test-user", "count": 5}
        result = validate_required_params(params, required=["user_id", "count"])
        assert result is None

    def test_multiple_missing_params_reports_first(self):
        """Test validation reports first missing parameter."""
        params = {}
        result = validate_required_params(params, required=["user_id", "payment_id"])
        assert result is not None
        assert "Missing required parameter: user_id" in result["message"]

    def test_empty_required_list(self):
        """Test validation passes with empty required list."""
        params = {"user_id": "test-user"}
        result = validate_required_params(params, required=[])
        assert result is None

    def test_none_optional_list(self):
        """Test validation passes with None optional list."""
        params = {"user_id": "test-user"}
        result = validate_required_params(params, required=["user_id"], optional=None)
        assert result is None


class TestFormatErrorResponse:
    """Test format_error_response function."""

    def test_format_value_error(self):
        """Test formatting ValueError."""
        exception = ValueError("Invalid value provided")
        result = format_error_response("tool-123", exception)
        assert result["toolUseId"] == "tool-123"
        assert result["status"] == "error"
        assert len(result["content"]) == 1
        assert "text" in result["content"][0]
        # Parse the JSON text
        error_data = json.loads(result["content"][0]["text"])
        assert error_data["error"] == "ValueError"
        assert error_data["message"] == "Invalid value provided"

    def test_format_key_error(self):
        """Test formatting KeyError."""
        exception = KeyError("missing_key")
        result = format_error_response("tool-456", exception)
        assert result["toolUseId"] == "tool-456"
        assert result["status"] == "error"
        error_data = json.loads(result["content"][0]["text"])
        assert error_data["error"] == "KeyError"

    def test_format_runtime_error(self):
        """Test formatting RuntimeError."""
        exception = RuntimeError("Runtime error occurred")
        result = format_error_response("tool-789", exception)
        assert result["toolUseId"] == "tool-789"
        assert result["status"] == "error"
        error_data = json.loads(result["content"][0]["text"])
        assert error_data["error"] == "RuntimeError"
        assert error_data["message"] == "Runtime error occurred"

    def test_format_exception_with_empty_message(self):
        """Test formatting exception with empty message."""
        exception = Exception()
        result = format_error_response("tool-000", exception)
        assert result["toolUseId"] == "tool-000"
        assert result["status"] == "error"
        error_data = json.loads(result["content"][0]["text"])
        assert error_data["error"] == "Exception"

    def test_format_exception_with_special_characters(self):
        """Test formatting exception with special characters in message."""
        exception = ValueError('Error with "quotes" and \\backslashes\\')
        result = format_error_response("tool-special", exception)
        assert result["toolUseId"] == "tool-special"
        assert result["status"] == "error"
        error_data = json.loads(result["content"][0]["text"])
        assert error_data["error"] == "ValueError"
        assert "quotes" in error_data["message"]

    def test_format_exception_with_multiline_message(self):
        """Test formatting exception with multiline message."""
        exception = ValueError("Line 1\nLine 2\nLine 3")
        result = format_error_response("tool-multi", exception)
        assert result["toolUseId"] == "tool-multi"
        assert result["status"] == "error"
        error_data = json.loads(result["content"][0]["text"])
        assert "Line 1" in error_data["message"]
        assert "Line 2" in error_data["message"]


class TestFormatSuccessResponse:
    """Test format_success_response function."""

    def test_format_simple_dict(self):
        """Test formatting simple dictionary."""
        data = {"id": "123", "name": "test"}
        result = format_success_response("tool-123", data)
        assert result["toolUseId"] == "tool-123"
        assert result["status"] == "success"
        assert len(result["content"]) == 1
        assert "text" in result["content"][0]
        response_data = json.loads(result["content"][0]["text"])
        assert response_data["id"] == "123"
        assert response_data["name"] == "test"

    def test_format_nested_dict(self):
        """Test formatting nested dictionary."""
        data = {"instrument": {"id": "instr-123", "details": {"type": "crypto", "address": "0x123"}}}
        result = format_success_response("tool-456", data)
        assert result["toolUseId"] == "tool-456"
        assert result["status"] == "success"
        response_data = json.loads(result["content"][0]["text"])
        assert response_data["instrument"]["id"] == "instr-123"
        assert response_data["instrument"]["details"]["type"] == "crypto"

    def test_format_list_data(self):
        """Test formatting list data."""
        data = {"items": [{"id": "1", "name": "item1"}, {"id": "2", "name": "item2"}]}
        result = format_success_response("tool-789", data)
        assert result["toolUseId"] == "tool-789"
        assert result["status"] == "success"
        response_data = json.loads(result["content"][0]["text"])
        assert len(response_data["items"]) == 2
        assert response_data["items"][0]["id"] == "1"

    def test_format_empty_dict(self):
        """Test formatting empty dictionary."""
        data = {}
        result = format_success_response("tool-empty", data)
        assert result["toolUseId"] == "tool-empty"
        assert result["status"] == "success"
        response_data = json.loads(result["content"][0]["text"])
        assert response_data == {}

    def test_format_dict_with_special_characters(self):
        """Test formatting dictionary with special characters."""
        data = {"message": 'Contains "quotes" and \\backslashes\\', "unicode": "Contains émojis 🎉"}
        result = format_success_response("tool-special", data)
        assert result["toolUseId"] == "tool-special"
        assert result["status"] == "success"
        response_data = json.loads(result["content"][0]["text"])
        assert "quotes" in response_data["message"]
        assert "émojis" in response_data["unicode"]

    def test_format_dict_with_none_values(self):
        """Test formatting dictionary with None values."""
        data = {"id": "123", "optional_field": None, "name": "test"}
        result = format_success_response("tool-none", data)
        assert result["toolUseId"] == "tool-none"
        assert result["status"] == "success"
        response_data = json.loads(result["content"][0]["text"])
        assert response_data["optional_field"] is None

    def test_format_dict_with_numeric_values(self):
        """Test formatting dictionary with numeric values."""
        data = {"count": 42, "amount": 99.99, "negative": -10, "zero": 0}
        result = format_success_response("tool-numeric", data)
        assert result["toolUseId"] == "tool-numeric"
        assert result["status"] == "success"
        response_data = json.loads(result["content"][0]["text"])
        assert response_data["count"] == 42
        assert response_data["amount"] == 99.99
        assert response_data["negative"] == -10
        assert response_data["zero"] == 0

    def test_format_dict_with_boolean_values(self):
        """Test formatting dictionary with boolean values."""
        data = {"active": True, "deleted": False}
        result = format_success_response("tool-bool", data)
        assert result["toolUseId"] == "tool-bool"
        assert result["status"] == "success"
        response_data = json.loads(result["content"][0]["text"])
        assert response_data["active"] is True
        assert response_data["deleted"] is False


class TestGetPaymentInstrumentTool:
    """Test getPaymentInstrument tool implementation."""

    def test_get_payment_instrument_success(self):
        """Test getPaymentInstrument returns instrument details on success."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        # Mock PaymentManager
        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_instrument.return_value = {
            "paymentInstrumentId": "instr-123",
            "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
            "paymentInstrumentDetails": {"address": "0x123"},
            "status": "ACTIVE",
        }

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        # Call tool
        result = plugin.get_payment_instrument(
            user_id="test-user",
            payment_instrument_id="instr-123",
        )

        # Verify result
        assert result["paymentInstrumentId"] == "instr-123"
        assert result["paymentInstrumentType"] == "EMBEDDED_CRYPTO_WALLET"
        assert result["status"] == "ACTIVE"
        mock_payment_manager.get_payment_instrument.assert_called_once_with(
            user_id="test-user",
            payment_instrument_id="instr-123",
            payment_connector_id=None,
        )

    def test_get_payment_instrument_with_connector_id(self):
        """Test getPaymentInstrument with optional connector_id parameter."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_instrument.return_value = {
            "paymentInstrumentId": "instr-123",
            "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
        }

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        result = plugin.get_payment_instrument(
            user_id="test-user",
            payment_instrument_id="instr-123",
            payment_connector_id="connector-456",
        )

        assert result["paymentInstrumentId"] == "instr-123"
        mock_payment_manager.get_payment_instrument.assert_called_once_with(
            user_id="test-user",
            payment_instrument_id="instr-123",
            payment_connector_id="connector-456",
        )

    def test_get_payment_instrument_missing_user_id_falls_back_to_config(self):
        """Test getPaymentInstrument falls back to config user_id when not provided."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()
        plugin.payment_manager.get_payment_instrument.return_value = {"paymentInstrumentId": "instr-123"}

        plugin.get_payment_instrument(
            user_id="",
            payment_instrument_id="instr-123",
        )
        plugin.payment_manager.get_payment_instrument.assert_called_once_with(
            user_id="test-user", payment_instrument_id="instr-123", payment_connector_id=None
        )

    def test_get_payment_instrument_empty_id_falls_back_to_config(self):
        """Test getPaymentInstrument falls back to config when payment_instrument_id is empty."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()
        plugin.payment_manager.get_payment_instrument.return_value = {"paymentInstrumentId": "test-instrument"}

        plugin.get_payment_instrument(
            user_id="test-user",
            payment_instrument_id="",
        )
        plugin.payment_manager.get_payment_instrument.assert_called_once_with(
            user_id="test-user", payment_instrument_id="test-instrument", payment_connector_id=None
        )

    def test_get_payment_instrument_no_id_anywhere_raises(self):
        """Test getPaymentInstrument raises error when no instrument_id in param or config."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin
        from bedrock_agentcore.payments.manager import PaymentError

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            # payment_instrument_id not set
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()

        with pytest.raises(PaymentError, match="payment_instrument_id is not set"):
            plugin.get_payment_instrument(
                user_id="test-user",
            )

    def test_get_payment_instrument_whitespace_user_id_falls_back_to_config(self):
        """Test getPaymentInstrument falls back to config user_id when whitespace provided."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()
        plugin.payment_manager.get_payment_instrument.return_value = {"paymentInstrumentId": "instr-123"}

        plugin.get_payment_instrument(
            user_id="   ",
            payment_instrument_id="instr-123",
        )
        plugin.payment_manager.get_payment_instrument.assert_called_once_with(
            user_id="test-user", payment_instrument_id="instr-123", payment_connector_id=None
        )

    def test_get_payment_instrument_payment_manager_not_initialized(self):
        """Test getPaymentInstrument raises error when PaymentManager is not initialized."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = None

        with pytest.raises(Exception, match="PaymentManager not initialized"):
            plugin.get_payment_instrument(
                user_id="test-user",
                payment_instrument_id="instr-123",
            )

    def test_get_payment_instrument_not_found_error(self):
        """Test getPaymentInstrument handles PaymentInstrumentNotFound exception."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin
        from bedrock_agentcore.payments.manager import PaymentInstrumentNotFound

        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_instrument.side_effect = PaymentInstrumentNotFound(
            "Instrument not found: instr-123"
        )

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        with pytest.raises(PaymentInstrumentNotFound):
            plugin.get_payment_instrument(
                user_id="test-user",
                payment_instrument_id="instr-123",
            )

    def test_get_payment_instrument_payment_error(self):
        """Test getPaymentInstrument handles PaymentError exception."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin
        from bedrock_agentcore.payments.manager import PaymentError

        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_instrument.side_effect = PaymentError("API call failed")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        with pytest.raises(PaymentError):
            plugin.get_payment_instrument(
                user_id="test-user",
                payment_instrument_id="instr-123",
            )

    def test_get_payment_instrument_unexpected_exception(self):
        """Test getPaymentInstrument handles unexpected exceptions."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_instrument.side_effect = RuntimeError("Unexpected error")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        with pytest.raises(RuntimeError):
            plugin.get_payment_instrument(
                user_id="test-user",
                payment_instrument_id="instr-123",
            )

    def test_get_payment_instrument_complex_response(self):
        """Test getPaymentInstrument with complex nested response."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_instrument.return_value = {
            "paymentInstrumentId": "instr-123",
            "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
            "paymentInstrumentDetails": {
                "address": "0x123abc",
                "network": "ethereum",
                "balance": "100.50",
            },
            "status": "ACTIVE",
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-15T12:30:00Z",
            "metadata": {
                "verified": True,
                "riskLevel": "LOW",
            },
        }

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        result = plugin.get_payment_instrument(
            user_id="test-user",
            payment_instrument_id="instr-123",
        )

        assert result["paymentInstrumentId"] == "instr-123"
        assert result["paymentInstrumentDetails"]["network"] == "ethereum"
        assert result["metadata"]["verified"] is True


class TestListPaymentInstrumentsTool:
    """Test listPaymentInstruments tool implementation."""

    def test_list_payment_instruments_success(self):
        """Test listPaymentInstruments returns instruments list on success."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        mock_payment_manager = MagicMock()
        mock_payment_manager.list_payment_instruments.return_value = {
            "paymentInstruments": [
                {
                    "paymentInstrumentId": "instr-1",
                    "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                    "status": "ACTIVE",
                },
                {
                    "paymentInstrumentId": "instr-2",
                    "paymentInstrumentType": "CREDIT_CARD",
                    "status": "ACTIVE",
                },
            ]
        }

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        result = plugin.list_payment_instruments(user_id="test-user")

        assert len(result["paymentInstruments"]) == 2
        assert result["paymentInstruments"][0]["paymentInstrumentId"] == "instr-1"
        assert result["paymentInstruments"][1]["paymentInstrumentId"] == "instr-2"
        mock_payment_manager.list_payment_instruments.assert_called_once_with(
            user_id="test-user",
            payment_connector_id=None,
            max_results=100,
            next_token=None,
        )

    def test_list_payment_instruments_with_optional_params(self):
        """Test listPaymentInstruments with optional parameters."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        mock_payment_manager = MagicMock()
        mock_payment_manager.list_payment_instruments.return_value = {
            "paymentInstruments": [
                {"paymentInstrumentId": "instr-1", "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET"}
            ],
            "nextToken": "token-123",
        }

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        result = plugin.list_payment_instruments(
            user_id="test-user",
            payment_connector_id="connector-456",
            max_results=50,
            next_token="prev-token",
        )

        assert len(result["paymentInstruments"]) == 1
        assert result["nextToken"] == "token-123"
        mock_payment_manager.list_payment_instruments.assert_called_once_with(
            user_id="test-user",
            payment_connector_id="connector-456",
            max_results=50,
            next_token="prev-token",
        )

    def test_list_payment_instruments_empty_list(self):
        """Test listPaymentInstruments returns empty list when no instruments found."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        mock_payment_manager = MagicMock()
        mock_payment_manager.list_payment_instruments.return_value = {"paymentInstruments": []}

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        result = plugin.list_payment_instruments(user_id="test-user")

        assert result["paymentInstruments"] == []
        assert "nextToken" not in result

    def test_list_payment_instruments_missing_user_id_falls_back_to_config(self):
        """Test listPaymentInstruments falls back to config user_id when not provided."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()
        plugin.payment_manager.list_payment_instruments.return_value = {"paymentInstruments": []}

        plugin.list_payment_instruments(user_id="")
        plugin.payment_manager.list_payment_instruments.assert_called_once_with(
            user_id="test-user", payment_connector_id=None, max_results=100, next_token=None
        )

    def test_list_payment_instruments_empty_user_id_falls_back_to_config(self):
        """Test listPaymentInstruments falls back to config user_id when whitespace provided."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()
        plugin.payment_manager.list_payment_instruments.return_value = {"paymentInstruments": []}

        plugin.list_payment_instruments(user_id="   ")
        plugin.payment_manager.list_payment_instruments.assert_called_once_with(
            user_id="test-user", payment_connector_id=None, max_results=100, next_token=None
        )

    def test_list_payment_instruments_payment_manager_not_initialized(self):
        """Test listPaymentInstruments raises error when PaymentManager is not initialized."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = None

        with pytest.raises(Exception, match="PaymentManager not initialized"):
            plugin.list_payment_instruments(user_id="test-user")

    def test_list_payment_instruments_payment_error(self):
        """Test listPaymentInstruments handles PaymentError exception."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin
        from bedrock_agentcore.payments.manager import PaymentError

        mock_payment_manager = MagicMock()
        mock_payment_manager.list_payment_instruments.side_effect = PaymentError("API call failed")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        with pytest.raises(PaymentError):
            plugin.list_payment_instruments(user_id="test-user")

    def test_list_payment_instruments_unexpected_exception(self):
        """Test listPaymentInstruments handles unexpected exceptions."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        mock_payment_manager = MagicMock()
        mock_payment_manager.list_payment_instruments.side_effect = RuntimeError("Unexpected error")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        with pytest.raises(RuntimeError):
            plugin.list_payment_instruments(user_id="test-user")

    def test_list_payment_instruments_with_pagination(self):
        """Test listPaymentInstruments with pagination tokens."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        mock_payment_manager = MagicMock()
        mock_payment_manager.list_payment_instruments.return_value = {
            "paymentInstruments": [
                {"paymentInstrumentId": f"instr-{i}", "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET"}
                for i in range(50)
            ],
            "nextToken": "next-page-token",
        }

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        result = plugin.list_payment_instruments(
            user_id="test-user",
            max_results=50,
            next_token="current-page-token",
        )

        assert len(result["paymentInstruments"]) == 50
        assert result["nextToken"] == "next-page-token"
        mock_payment_manager.list_payment_instruments.assert_called_once_with(
            user_id="test-user",
            payment_connector_id=None,
            max_results=50,
            next_token="current-page-token",
        )


class TestGetPaymentSessionTool:
    """Test getPaymentSession tool implementation."""

    def test_get_payment_session_success(self):
        """Test getPaymentSession returns session details on success."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_session.return_value = {
            "paymentSessionId": "session-123",
            "userId": "test-user",
            "remainingAmount": {"value": "500.00", "currency": "USD"},
            "spentAmount": {"value": "100.00", "currency": "USD"},
            "limits": {"maxSpendAmount": {"value": "600.00", "currency": "USD"}},
            "expiryTime": "2024-12-31T23:59:59Z",
            "createdAt": "2024-01-01T00:00:00Z",
            "status": "ACTIVE",
        }

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        result = plugin.get_payment_session(
            user_id="test-user",
            payment_session_id="session-123",
        )

        assert result["paymentSessionId"] == "session-123"
        assert result["userId"] == "test-user"
        assert result["remainingAmount"]["value"] == "500.00"
        assert result["status"] == "ACTIVE"
        mock_payment_manager.get_payment_session.assert_called_once_with(
            user_id="test-user",
            payment_session_id="session-123",
        )

    def test_get_payment_session_missing_user_id_falls_back_to_config(self):
        """Test getPaymentSession falls back to config user_id when not provided."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()
        plugin.payment_manager.get_payment_session.return_value = {"paymentSessionId": "session-123"}

        plugin.get_payment_session(
            user_id="",
            payment_session_id="session-123",
        )
        plugin.payment_manager.get_payment_session.assert_called_once_with(
            user_id="test-user", payment_session_id="session-123"
        )

    def test_get_payment_session_empty_session_id_falls_back_to_config(self):
        """Test getPaymentSession falls back to config when payment_session_id is empty."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()
        plugin.payment_manager.get_payment_session.return_value = {"paymentSessionId": "test-session"}

        plugin.get_payment_session(
            user_id="test-user",
            payment_session_id="",
        )
        plugin.payment_manager.get_payment_session.assert_called_once_with(
            user_id="test-user", payment_session_id="test-session"
        )

    def test_get_payment_session_no_session_id_anywhere_raises(self):
        """Test getPaymentSession raises error when no session_id in param or config."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin
        from bedrock_agentcore.payments.manager import PaymentError

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            # payment_session_id not set
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()

        with pytest.raises(PaymentError, match="payment_session_id is not set"):
            plugin.get_payment_session(
                user_id="test-user",
            )

    def test_get_payment_session_whitespace_user_id_falls_back_to_config(self):
        """Test getPaymentSession falls back to config user_id when whitespace provided."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()
        plugin.payment_manager.get_payment_session.return_value = {"paymentSessionId": "session-123"}

        plugin.get_payment_session(
            user_id="   ",
            payment_session_id="session-123",
        )
        plugin.payment_manager.get_payment_session.assert_called_once_with(
            user_id="test-user", payment_session_id="session-123"
        )

    def test_get_payment_session_payment_manager_not_initialized(self):
        """Test getPaymentSession raises error when PaymentManager is not initialized."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = None

        with pytest.raises(Exception, match="PaymentManager not initialized"):
            plugin.get_payment_session(
                user_id="test-user",
                payment_session_id="session-123",
            )

    def test_get_payment_session_not_found_error(self):
        """Test getPaymentSession handles PaymentSessionNotFound exception."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin
        from bedrock_agentcore.payments.manager import PaymentSessionNotFound

        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_session.side_effect = PaymentSessionNotFound("Session not found: session-123")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        with pytest.raises(PaymentSessionNotFound):
            plugin.get_payment_session(
                user_id="test-user",
                payment_session_id="session-123",
            )

    def test_get_payment_session_payment_error(self):
        """Test getPaymentSession handles PaymentError exception."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin
        from bedrock_agentcore.payments.manager import PaymentError

        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_session.side_effect = PaymentError("API call failed")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        with pytest.raises(PaymentError):
            plugin.get_payment_session(
                user_id="test-user",
                payment_session_id="session-123",
            )

    def test_get_payment_session_unexpected_exception(self):
        """Test getPaymentSession handles unexpected exceptions."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_session.side_effect = RuntimeError("Unexpected error")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        with pytest.raises(RuntimeError):
            plugin.get_payment_session(
                user_id="test-user",
                payment_session_id="session-123",
            )

    def test_get_payment_session_complex_response(self):
        """Test getPaymentSession with complex nested response."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_session.return_value = {
            "paymentSessionId": "session-123",
            "userId": "test-user",
            "remainingAmount": {"value": "500.00", "currency": "USD"},
            "spentAmount": {"value": "100.00", "currency": "USD"},
            "limits": {
                "maxSpendAmount": {"value": "600.00", "currency": "USD"},
                "dailyLimit": {"value": "200.00", "currency": "USD"},
            },
            "expiryTime": "2024-12-31T23:59:59Z",
            "createdAt": "2024-01-01T00:00:00Z",
            "status": "ACTIVE",
            "metadata": {
                "region": "US",
                "riskLevel": "LOW",
            },
        }

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        result = plugin.get_payment_session(
            user_id="test-user",
            payment_session_id="session-123",
        )

        assert result["paymentSessionId"] == "session-123"
        assert result["limits"]["dailyLimit"]["value"] == "200.00"
        assert result["metadata"]["riskLevel"] == "LOW"


class TestAfterToolCallHookWithAutoPayment:
    """Test after_tool_call hook behavior with auto_payment flag."""

    def test_after_tool_call_auto_payment_disabled_skips_processing(self):
        """Test after_tool_call skips payment processing when auto_payment=False."""
        from strands.hooks import AfterToolCallEvent

        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            auto_payment=False,
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()

        # Create a mock event with 402 response
        event = MagicMock(spec=AfterToolCallEvent)
        event.tool_use = {"name": "test_tool", "input": {}}
        event.result = {"statusCode": 402, "headers": {}, "body": {}}

        # Call after_tool_call
        plugin.after_tool_call(event)

        # Verify retry was NOT set (payment processing was skipped)
        assert not hasattr(event, "retry") or event.retry is not True

    def test_after_tool_call_auto_payment_enabled_processes_402(self):
        """Test after_tool_call processes 402 when auto_payment=True."""
        from strands.hooks import AfterToolCallEvent

        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            auto_payment=True,
        )
        plugin = AgentCorePaymentsPlugin(config)
        mock_payment_manager = MagicMock()
        plugin.payment_manager = mock_payment_manager

        # Set up mock agent for interrupt retry tracking
        mock_agent = MagicMock()
        mock_agent.state.get.return_value = 0

        # Create a mock event with 402 response
        event = MagicMock(spec=AfterToolCallEvent)
        event.agent = mock_agent
        event.tool_use = {"name": "test_tool", "input": {}}
        event.result = {"statusCode": 402, "headers": {}, "body": {}}
        event.invocation_state = {}
        with patch("bedrock_agentcore.payments.integrations.strands.plugin.get_payment_handler") as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.extract_status_code.return_value = 402
            mock_handler.extract_headers.return_value = {}
            mock_handler.extract_body.return_value = {}
            mock_handler.apply_payment_header.return_value = True
            mock_get_handler.return_value = mock_handler

            # Mock PaymentManager.generate_payment_header
            mock_payment_manager.generate_payment_header.return_value = {"Authorization": "Bearer token"}

            # Call after_tool_call
            plugin.after_tool_call(event)

            # Verify handler was called to extract status code
            mock_handler.extract_status_code.assert_called_once()

    def test_after_tool_call_auto_payment_true_non_402_response(self):
        """Test after_tool_call with auto_payment=True and non-402 response."""
        from strands.hooks import AfterToolCallEvent

        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            auto_payment=True,
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()

        # Set up mock agent for interrupt retry tracking
        mock_agent = MagicMock()
        mock_agent.state.get.return_value = 0

        # Create a mock event with 200 response
        event = MagicMock(spec=AfterToolCallEvent)
        event.agent = mock_agent
        event.tool_use = {"name": "test_tool", "input": {}}
        event.result = {"statusCode": 200, "headers": {}, "body": {"data": "success"}}
        event.invocation_state = {}

        with patch("bedrock_agentcore.payments.integrations.strands.plugin.get_payment_handler") as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.extract_status_code.return_value = 200
            mock_get_handler.return_value = mock_handler

            # Call after_tool_call
            plugin.after_tool_call(event)

            # Verify handler was called but no retry was set
            mock_handler.extract_status_code.assert_called_once()
            assert not hasattr(event, "retry") or event.retry is not True

    def test_after_tool_call_auto_payment_false_non_402_response(self):
        """Test after_tool_call with auto_payment=False and non-402 response."""
        from strands.hooks import AfterToolCallEvent

        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            auto_payment=False,
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()

        # Create a mock event with 200 response
        event = MagicMock(spec=AfterToolCallEvent)
        event.tool_use = {"name": "test_tool", "input": {}}
        event.result = {"statusCode": 200, "headers": {}, "body": {"data": "success"}}

        # Call after_tool_call
        plugin.after_tool_call(event)

        # Verify no processing occurred
        assert not hasattr(event, "retry") or event.retry is not True

    def test_after_tool_call_auto_payment_default_true(self):
        """Test after_tool_call with default auto_payment=True."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig

        # Create config without specifying auto_payment (should default to True)
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )

        # Verify auto_payment defaults to True
        assert config.auto_payment is True

    def test_after_tool_call_no_result_skips_processing(self):
        """Test after_tool_call skips processing when result is None."""
        from strands.hooks import AfterToolCallEvent

        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            auto_payment=True,
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = MagicMock()

        # Create a mock event with no result
        mock_agent = MagicMock()
        mock_agent.state.get.return_value = 0
        event = MagicMock(spec=AfterToolCallEvent)
        event.agent = mock_agent
        event.tool_use = {"name": "test_tool", "toolUseId": "tool-123", "input": {}}
        event.result = None
        event.invocation_state = {}

        # Call after_tool_call
        plugin.after_tool_call(event)

        # Verify no processing occurred
        assert not hasattr(event, "retry") or event.retry is not True


class TestToolDiscoverability:
    """Test tool discoverability and descriptions."""

    def test_get_payment_instrument_has_description(self):
        """Test that getPaymentInstrument tool has a description attribute."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)

        # Verify the tool method has a docstring
        assert hasattr(plugin.get_payment_instrument, "__doc__")
        assert plugin.get_payment_instrument.__doc__ is not None
        assert len(plugin.get_payment_instrument.__doc__.strip()) > 0

    def test_list_payment_instruments_has_description(self):
        """Test that listPaymentInstruments tool has a description attribute."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)

        # Verify the tool method has a docstring
        assert hasattr(plugin.list_payment_instruments, "__doc__")
        assert plugin.list_payment_instruments.__doc__ is not None
        assert len(plugin.list_payment_instruments.__doc__.strip()) > 0

    def test_get_payment_session_has_description(self):
        """Test that getPaymentSession tool has a description attribute."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)

        # Verify the tool method has a docstring
        assert hasattr(plugin.get_payment_session, "__doc__")
        assert plugin.get_payment_session.__doc__ is not None
        assert len(plugin.get_payment_session.__doc__.strip()) > 0

    def test_tool_descriptions_are_non_empty_strings(self):
        """Test that all tool descriptions are non-empty strings."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)

        # Verify all tool descriptions are non-empty strings
        tools = [
            ("get_payment_instrument", plugin.get_payment_instrument),
            ("list_payment_instruments", plugin.list_payment_instruments),
            ("get_payment_session", plugin.get_payment_session),
        ]

        for tool_name, tool_method in tools:
            assert isinstance(tool_method.__doc__, str), f"{tool_name} docstring is not a string"
            assert len(tool_method.__doc__.strip()) > 0, f"{tool_name} docstring is empty"
            # Verify docstring contains meaningful content (not just whitespace)
            assert tool_method.__doc__.strip().startswith("Retrieve") or tool_method.__doc__.strip().startswith(
                "List"
            ), f"{tool_name} docstring does not start with expected verb"


class TestToolAvailability:
    """Test that tools are always available regardless of auto_payment setting."""

    def test_tools_available_with_auto_payment_true(self):
        """Test all tools are callable when auto_payment=True."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            auto_payment=True,
        )
        plugin = AgentCorePaymentsPlugin(config)

        # Verify all tools are callable
        assert callable(plugin.get_payment_instrument)
        assert callable(plugin.list_payment_instruments)
        assert callable(plugin.get_payment_session)

    def test_tools_available_with_auto_payment_false(self):
        """Test all tools are callable when auto_payment=False."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            auto_payment=False,
        )
        plugin = AgentCorePaymentsPlugin(config)

        # Verify all tools are callable
        assert callable(plugin.get_payment_instrument)
        assert callable(plugin.list_payment_instruments)
        assert callable(plugin.get_payment_session)

    def test_tools_function_identically_regardless_of_auto_payment(self):
        """Test tools function identically regardless of auto_payment setting."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        # Create two plugins with different auto_payment settings
        config_true = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            auto_payment=True,
        )
        plugin_true = AgentCorePaymentsPlugin(config_true)

        config_false = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            auto_payment=False,
        )
        plugin_false = AgentCorePaymentsPlugin(config_false)

        # Mock PaymentManager for both
        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_instrument.return_value = {
            "paymentInstrumentId": "instr-123",
            "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
            "status": "ACTIVE",
        }

        plugin_true.payment_manager = mock_payment_manager
        plugin_false.payment_manager = mock_payment_manager

        # Call get_payment_instrument on both plugins
        result_true = plugin_true.get_payment_instrument(
            user_id="test-user",
            payment_instrument_id="instr-123",
        )
        result_false = plugin_false.get_payment_instrument(
            user_id="test-user",
            payment_instrument_id="instr-123",
        )

        # Verify results are identical
        assert result_true == result_false
        assert result_true["paymentInstrumentId"] == "instr-123"
        assert result_false["paymentInstrumentId"] == "instr-123"

        # Verify PaymentManager was called identically for both
        assert mock_payment_manager.get_payment_instrument.call_count == 2
        call_args_list = mock_payment_manager.get_payment_instrument.call_args_list
        assert call_args_list[0] == call_args_list[1]


class TestErrorHandlingConsistency:
    """Test comprehensive error handling across all tools."""

    def test_error_response_format_consistency_across_tools(self):
        """Test all error responses have consistent format across tools."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin
        from bedrock_agentcore.payments.manager import PaymentError

        # Test get_payment_instrument error (no instrument_id in param or config)
        config_no_instrument = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            # payment_instrument_id not set
            payment_session_id="test-session",
        )
        plugin_no_instrument = AgentCorePaymentsPlugin(config_no_instrument)
        plugin_no_instrument.payment_manager = MagicMock()

        with pytest.raises(PaymentError, match="payment_instrument_id is not set"):
            plugin_no_instrument.get_payment_instrument(payment_instrument_id="")

        # Test getPaymentSession error (no session_id in param or config)
        config_no_session = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            # payment_session_id not set
        )
        plugin_no_session = AgentCorePaymentsPlugin(config_no_session)
        plugin_no_session.payment_manager = MagicMock()

        with pytest.raises(PaymentError, match="payment_session_id is not set"):
            plugin_no_session.get_payment_session(payment_session_id="")

    def test_error_responses_include_error_and_message_fields(self):
        """Test error responses include 'error' and 'message' fields."""
        error_response = format_error_response("tool-123", ValueError("Test error"))

        assert "content" in error_response
        assert len(error_response["content"]) > 0
        assert "text" in error_response["content"][0]

        error_data = json.loads(error_response["content"][0]["text"])
        assert "error" in error_data
        assert "message" in error_data
        assert error_data["error"] == "ValueError"
        assert error_data["message"] == "Test error"

    def test_error_responses_are_json_serializable(self):
        """Test error responses are JSON serializable."""

        error_response = format_error_response("tool-123", RuntimeError("Serialization test"))

        # Verify the response can be JSON serialized
        try:
            json_str = json.dumps(error_response)
            assert json_str is not None
            # Verify it can be deserialized
            deserialized = json.loads(json_str)
            assert deserialized["status"] == "error"
        except (TypeError, ValueError) as e:
            pytest.fail(f"Error response is not JSON serializable: {e}")

    def test_error_logging_with_appropriate_log_levels(self):
        """Test error logging uses appropriate log levels."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin
        from bedrock_agentcore.payments.manager import PaymentError

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)

        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_instrument.side_effect = PaymentError("API error")
        plugin.payment_manager = mock_payment_manager

        # Capture logs
        with patch("bedrock_agentcore.payments.integrations.strands.plugin.logger") as mock_logger:
            try:
                plugin.get_payment_instrument(
                    user_id="test-user",
                    payment_instrument_id="instr-123",
                )
            except PaymentError:
                pass

            # Verify error was logged
            mock_logger.error.assert_called()

    def test_validation_error_responses_consistent_format(self):
        """Test validation error responses have consistent format."""
        error_response = validate_required_params(
            {"user_id": ""},
            required=["user_id"],
        )

        assert error_response is not None
        assert "error" in error_response
        assert "message" in error_response
        assert error_response["error"] == "ValidationError"
        assert "Parameter cannot be empty" in error_response["message"]

    def test_payment_manager_error_responses_consistent_format(self):
        """Test PaymentManager error responses have consistent format."""
        from bedrock_agentcore.payments.manager import PaymentError

        error_response = format_error_response("tool-123", PaymentError("Payment API failed"))

        assert "content" in error_response
        assert error_response["status"] == "error"

        error_data = json.loads(error_response["content"][0]["text"])
        assert error_data["error"] == "PaymentError"
        assert "Payment API failed" in error_data["message"]

    def test_unexpected_exception_responses_consistent_format(self):
        """Test unexpected exception responses have consistent format."""
        error_response = format_error_response("tool-123", RuntimeError("Unexpected error"))

        assert "content" in error_response
        assert error_response["status"] == "error"

        error_data = json.loads(error_response["content"][0]["text"])
        assert error_data["error"] == "RuntimeError"
        assert error_data["message"] == "Unexpected error"

    def test_error_response_includes_tool_use_id(self):
        """Test error responses include toolUseId."""
        error_response = format_error_response("tool-abc-123", ValueError("Test"))

        assert "toolUseId" in error_response
        assert error_response["toolUseId"] == "tool-abc-123"

    def test_error_response_status_is_error(self):
        """Test error responses have status='error'."""
        error_response = format_error_response("tool-123", ValueError("Test"))

        assert "status" in error_response
        assert error_response["status"] == "error"

    def test_success_response_status_is_success(self):
        """Test success responses have status='success'."""
        success_response = format_success_response("tool-123", {"data": "test"})

        assert "status" in success_response
        assert success_response["status"] == "success"

    def test_all_error_types_handled_consistently(self):
        """Test various error types are handled consistently."""
        error_types = [
            ValueError("Value error"),
            KeyError("Key error"),
            RuntimeError("Runtime error"),
            Exception("Generic exception"),
            TypeError("Type error"),
        ]

        for error in error_types:
            error_response = format_error_response("tool-123", error)

            # Verify consistent structure
            assert "toolUseId" in error_response
            assert "status" in error_response
            assert error_response["status"] == "error"
            assert "content" in error_response
            assert len(error_response["content"]) > 0
            assert "text" in error_response["content"][0]

            # Verify JSON serializable
            error_data = json.loads(error_response["content"][0]["text"])
            assert "error" in error_data
            assert "message" in error_data


class TestUserIdPrecedence:
    """Test user_id resolution: explicit > config > None."""

    def test_explicit_user_id_takes_precedence_over_config(self):
        """Test that explicit user_id passed to tool takes precedence over config.user_id."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_instrument.return_value = {
            "paymentInstrumentId": "instr-123",
            "status": "ACTIVE",
        }

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="config-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        plugin.get_payment_instrument(
            user_id="override-user",
            payment_instrument_id="instr-123",
        )

        mock_payment_manager.get_payment_instrument.assert_called_once_with(
            user_id="override-user",
            payment_instrument_id="instr-123",
            payment_connector_id=None,
        )

    def test_bearer_auth_mode_user_id_none_passes_none_to_manager(self):
        """Test that bearer auth with no user_id passes None to manager."""
        from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
        from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin

        mock_payment_manager = MagicMock()
        mock_payment_manager.get_payment_instrument.return_value = {
            "paymentInstrumentId": "instr-123",
            "status": "ACTIVE",
        }

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            bearer_token="my-jwt-token",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        plugin = AgentCorePaymentsPlugin(config)
        plugin.payment_manager = mock_payment_manager

        plugin.get_payment_instrument(
            payment_instrument_id="instr-123",
        )

        mock_payment_manager.get_payment_instrument.assert_called_once_with(
            user_id=None,
            payment_instrument_id="instr-123",
            payment_connector_id=None,
        )
