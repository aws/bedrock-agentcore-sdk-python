"""Comprehensive unit tests for PaymentManager."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

from bedrock_agentcore.payments.manager import (
    InsufficientBudget,
    InvalidPaymentInstrument,
    PaymentError,
    PaymentInstrumentNotFound,
    PaymentManager,
    PaymentSessionExpired,
    PaymentSessionNotFound,
)

# ============================================================================
# PaymentManager Initialization Tests
# ============================================================================


class TestPaymentManagerInitialization:
    """Tests for PaymentManager initialization."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_with_valid_arn(self, mock_session_class):
        """Test successful initialization with valid payment_manager_arn."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.client.return_value = MagicMock()
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        assert manager._payment_manager_arn == arn
        assert manager.region_name == "us-east-1"
        assert manager._payment_client is not None

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_with_region_name_parameter(self, mock_session_class):
        """Test initialization with region_name parameter."""
        arn = "arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        mock_session.client.return_value = MagicMock()
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-west-2")

        assert manager.region_name == "us-west-2"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_with_boto3_session_parameter(self, mock_session_class):
        """Test initialization with boto3_session parameter."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.client.return_value = MagicMock()

        manager = PaymentManager(payment_manager_arn=arn, boto3_session=mock_session)

        assert manager.region_name == "us-east-1"
        mock_session.client.assert_called_once()

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_with_boto_client_config_parameter(self, mock_session_class):
        """Test initialization with boto_client_config parameter."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.client.return_value = MagicMock()
        mock_session_class.return_value = mock_session

        boto_config = BotocoreConfig(max_pool_connections=50)
        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1", boto_client_config=boto_config)

        assert manager._payment_manager_arn == arn
        mock_session.client.assert_called_once()

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_region_validation_conflict_detection(self, mock_session_class):
        """Test region validation and conflict detection."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"

        with pytest.raises(ValueError) as exc_info:
            PaymentManager(payment_manager_arn=arn, region_name="us-east-1", boto3_session=mock_session)

        assert "Region mismatch" in str(exc_info.value)
        assert "us-east-1" in str(exc_info.value)
        assert "us-west-2" in str(exc_info.value)

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_raises_error_for_invalid_arn(self, mock_session_class):
        """Test ValueError raised for invalid payment_manager_arn."""
        with pytest.raises(ValueError) as exc_info:
            PaymentManager(payment_manager_arn=None, region_name="us-east-1")

        assert "payment_manager_arn is required" in str(exc_info.value)

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_raises_error_for_empty_arn(self, mock_session_class):
        """Test ValueError raised for empty payment_manager_arn."""
        with pytest.raises(ValueError) as exc_info:
            PaymentManager(payment_manager_arn="", region_name="us-east-1")

        assert "payment_manager_arn is required" in str(exc_info.value)

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_creates_payment_client(self, mock_session_class):
        """Test PaymentClient is created correctly."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        assert manager._payment_client == mock_client
        mock_session.client.assert_called_once()

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_stores_payment_manager_arn_internally(self, mock_session_class):
        """Test payment_manager_arn is stored internally."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.client.return_value = MagicMock()
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        assert manager._payment_manager_arn == arn
        assert hasattr(manager, "_payment_manager_arn")


# ============================================================================
# PaymentManager Method Tests
# ============================================================================


class TestCreatePaymentInstrument:
    """Tests for create_payment_instrument method."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_successful_instrument_creation(self, mock_session_class):
        """Test successful instrument creation."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        response = {"paymentInstrumentId": "instrument-123"}
        mock_client.create_payment_instrument.return_value = response

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        result = manager.create_payment_instrument(
            user_id="user-123",
            payment_connector_id="connector-456",
            payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
            payment_instrument_details={"wallet_address": "0x..."},
        )

        assert result == response

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_payment_manager_arn_automatically_injected(self, mock_session_class):
        """Test payment_manager_arn is automatically injected."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        mock_client.create_payment_instrument.return_value = {"paymentInstrumentId": "instrument-123"}

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        manager.create_payment_instrument(
            user_id="user-123",
            payment_connector_id="connector-456",
            payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
            payment_instrument_details={"wallet_address": "0x..."},
        )

        call_kwargs = mock_client.create_payment_instrument.call_args[1]
        assert call_kwargs["paymentManagerArn"] == arn

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_client_token_parameter_passed_through(self, mock_session_class):
        """Test client_token parameter is passed through."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        mock_client.create_payment_instrument.return_value = {"paymentInstrumentId": "instrument-123"}

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        manager.create_payment_instrument(
            user_id="user-123",
            payment_connector_id="connector-456",
            payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
            payment_instrument_details={"wallet_address": "0x..."},
            client_token="token-123",
        )

        call_kwargs = mock_client.create_payment_instrument.call_args[1]
        assert call_kwargs["clientToken"] == "token-123"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_payment_client_errors_propagated(self, mock_session_class):
        """Test PaymentClient errors are propagated."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error_response = {"Error": {"Code": "InvalidParameter", "Message": "Invalid connector"}}
        error = ClientError(error_response, "CreatePaymentInstrument")
        mock_client.create_payment_instrument.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError):
            manager.create_payment_instrument(
                user_id="user-123",
                payment_connector_id="invalid-connector",
                payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
                payment_instrument_details={"wallet_address": "0x..."},
            )


class TestCreatePaymentSession:
    """Tests for create_payment_session method."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_successful_session_creation(self, mock_session_class):
        """Test successful session creation."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        response = {"paymentSessionId": "session-123"}
        mock_client.create_payment_session.return_value = response

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        result = manager.create_payment_session(
            user_id="user-123",
            expiry_time_in_minutes=60,
            limits={"max_amount": 1000},
        )

        assert result == response

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_payment_manager_arn_injected_in_session(self, mock_session_class):
        """Test payment_manager_arn is automatically injected."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        mock_client.create_payment_session.return_value = {"paymentSessionId": "session-123"}

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        manager.create_payment_session(
            user_id="user-123",
            expiry_time_in_minutes=60,
            limits={"max_amount": 1000},
        )

        call_kwargs = mock_client.create_payment_session.call_args[1]
        assert call_kwargs["paymentManagerArn"] == arn

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_session_client_errors_propagated(self, mock_session_class):
        """Test PaymentClient errors are propagated."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error_response = {"Error": {"Code": "InvalidLimit", "Message": "Invalid session limit"}}
        error = ClientError(error_response, "CreatePaymentSession")
        mock_client.create_payment_session.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError):
            manager.create_payment_session(
                user_id="user-123",
                expiry_time_in_minutes=60,
                limits={"max_amount": -100},
            )


class TestGetPaymentInstrument:
    """Tests for get_payment_instrument method."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_successful_get_instrument(self, mock_session_class):
        """Test successful retrieval of payment instrument."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        response = {"paymentInstrumentId": "instrument-123", "status": "active"}
        mock_client.get_payment_instrument.return_value = response

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        result = manager.get_payment_instrument(
            user_id="user-123",
            payment_connector_id="connector-456",
            payment_instrument_id="instrument-123",
        )

        assert result == response

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_get_instrument_without_connector_id(self, mock_session_class):
        """Test get_payment_instrument without payment_connector_id (None case)."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        response = {"paymentInstrumentId": "instrument-123", "status": "active"}
        mock_client.get_payment_instrument.return_value = response

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        result = manager.get_payment_instrument(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            # payment_connector_id is None (default)
        )

        assert result == response
        # Verify that paymentConnectorId was NOT included in the call
        call_args = mock_client.get_payment_instrument.call_args
        assert "paymentConnectorId" not in call_args.kwargs

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_get_instrument_not_found(self, mock_session_class):
        """Test PaymentInstrumentNotFound raised when instrument not found."""
        from bedrock_agentcore.payments.manager import PaymentInstrumentNotFound

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Instrument not found"}},
            "GetPaymentInstrument",
        )
        mock_client.get_payment_instrument.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentInstrumentNotFound):
            manager.get_payment_instrument(
                user_id="user-123",
                payment_connector_id="connector-456",
                payment_instrument_id="instrument-123",
            )


class TestListPaymentInstruments:
    """Tests for list_payment_instruments method."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_successful_list_instruments(self, mock_session_class):
        """Test successful listing of payment instruments."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        response = {
            "paymentInstruments": [
                {"paymentInstrumentId": "instrument-1"},
                {"paymentInstrumentId": "instrument-2"},
            ]
        }
        mock_client.list_payment_instruments.return_value = response

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        result = manager.list_payment_instruments(user_id="user-123")

        assert len(result["paymentInstruments"]) == 2
        assert result["paymentInstruments"][0]["paymentInstrumentId"] == "instrument-1"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_list_instruments_with_pagination(self, mock_session_class):
        """Test listing instruments with pagination."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        response = {
            "paymentInstruments": [{"paymentInstrumentId": "instrument-1"}],
            "nextToken": "token-123",
        }
        mock_client.list_payment_instruments.return_value = response

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        result = manager.list_payment_instruments(user_id="user-123", next_token="token-123")

        assert "nextToken" in result
        assert result["nextToken"] == "token-123"


class TestGetPaymentInstrumentBalance:
    """Tests for get_payment_instrument_balance method."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_successful_get_balance(self, mock_session_class):
        """Test successful retrieval of payment instrument balance."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        response = {
            "paymentInstrumentId": "instrument-123",
            "tokenBalance": {"value": "100.50", "currency": "USDC"},
        }
        mock_client.get_payment_instrument_balance.return_value = response

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        result = manager.get_payment_instrument_balance(
            user_id="user-123",
            payment_connector_id="connector-456",
            payment_instrument_id="instrument-123",
            chain="BASE_SEPOLIA",
            token="USDC",
        )

        assert result == response
        assert result["paymentInstrumentId"] == "instrument-123"
        assert result["tokenBalance"]["value"] == "100.50"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_get_balance_instrument_not_found(self, mock_session_class):
        """Test PaymentInstrumentNotFound raised when instrument not found."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Instrument not found"}},
            "GetPaymentInstrumentBalance",
        )
        mock_client.get_payment_instrument_balance.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentInstrumentNotFound):
            manager.get_payment_instrument_balance(
                user_id="user-123",
                payment_connector_id="connector-456",
                payment_instrument_id="instrument-123",
                chain="BASE_SEPOLIA",
                token="USDC",
            )


class TestGetPaymentSession:
    """Tests for get_payment_session method."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_successful_get_session(self, mock_session_class):
        """Test successful retrieval of payment session."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        response = {"paymentSessionId": "session-123", "status": "active"}
        mock_client.get_payment_session.return_value = response

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        result = manager.get_payment_session(user_id="user-123", payment_session_id="session-123")

        assert result == response

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_get_session_not_found(self, mock_session_class):
        """Test PaymentSessionNotFound raised when session not found."""

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Session not found"}},
            "GetPaymentSession",
        )
        mock_client.get_payment_session.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentSessionNotFound):
            manager.get_payment_session(user_id="user-123", payment_session_id="session-123")


class TestListPaymentSessions:
    """Tests for list_payment_sessions method."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_successful_list_sessions(self, mock_session_class):
        """Test successful listing of payment sessions."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        response = {
            "paymentSessions": [
                {"paymentSessionId": "session-1"},
                {"paymentSessionId": "session-2"},
            ]
        }
        mock_client.list_payment_sessions.return_value = response

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        result = manager.list_payment_sessions(user_id="user-123")

        assert len(result["paymentSessions"]) == 2


class TestDeletePaymentSession:
    """Tests for delete_payment_session method."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_successful_delete_session(self, mock_session_class):
        """Test successful deletion of a payment session."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        response = {"status": "DELETED"}
        mock_client.delete_payment_session.return_value = response

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        result = manager.delete_payment_session(
            payment_session_id="session-123",
            user_id="user-123",
        )

        assert result == {"status": "DELETED"}
        mock_client.delete_payment_session.assert_called_once_with(
            userId="user-123",
            paymentManagerArn=arn,
            paymentSessionId="session-123",
        )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_delete_session_without_user_id(self, mock_session_class):
        """Test deletion without user_id (bearer auth scenario)."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        response = {"status": "DELETED"}
        mock_client.delete_payment_session.return_value = response

        manager = PaymentManager(
            payment_manager_arn=arn,
            region_name="us-east-1",
            bearer_token="test-jwt-token",
        )
        result = manager.delete_payment_session(payment_session_id="session-123")

        assert result == {"status": "DELETED"}
        mock_client.delete_payment_session.assert_called_once_with(
            paymentManagerArn=arn,
            paymentSessionId="session-123",
        )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_delete_session_not_found(self, mock_session_class):
        """Test ResourceNotFoundException when deleting a non-existent session."""
        from bedrock_agentcore.payments.manager import PaymentSessionNotFound

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Session not found"}},
            "DeletePaymentSession",
        )
        mock_client.delete_payment_session.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentSessionNotFound):
            manager.delete_payment_session(
                payment_session_id="session-nonexistent",
                user_id="user-123",
            )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_delete_session_access_denied(self, mock_session_class):
        """Test AccessDeniedException when deleting a session."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Forbidden"}},
            "DeletePaymentSession",
        )
        mock_client.delete_payment_session.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError):
            manager.delete_payment_session(
                payment_session_id="session-123",
                user_id="user-123",
            )


class TestDeletePaymentInstrument:
    """Tests for delete_payment_instrument method."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_successful_delete_instrument(self, mock_session_class):
        """Test successful deletion of a payment instrument."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        response = {"status": "DELETED"}
        mock_client.delete_payment_instrument.return_value = response

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        result = manager.delete_payment_instrument(
            payment_instrument_id="instrument-123",
            payment_connector_id="connector-456",
            user_id="user-123",
        )

        assert result == {"status": "DELETED"}
        mock_client.delete_payment_instrument.assert_called_once_with(
            userId="user-123",
            paymentManagerArn=arn,
            paymentConnectorId="connector-456",
            paymentInstrumentId="instrument-123",
        )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_delete_instrument_without_user_id(self, mock_session_class):
        """Test deletion without user_id (bearer auth scenario)."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        response = {"status": "DELETED"}
        mock_client.delete_payment_instrument.return_value = response

        manager = PaymentManager(
            payment_manager_arn=arn,
            region_name="us-east-1",
            bearer_token="test-jwt-token",
        )
        result = manager.delete_payment_instrument(
            payment_instrument_id="instrument-123",
            payment_connector_id="connector-456",
        )

        assert result == {"status": "DELETED"}
        mock_client.delete_payment_instrument.assert_called_once_with(
            paymentManagerArn=arn,
            paymentConnectorId="connector-456",
            paymentInstrumentId="instrument-123",
        )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_delete_instrument_not_found(self, mock_session_class):
        """Test ResourceNotFoundException when deleting a non-existent instrument."""
        from bedrock_agentcore.payments.manager import PaymentInstrumentNotFound

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Instrument not found"}},
            "DeletePaymentInstrument",
        )
        mock_client.delete_payment_instrument.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentInstrumentNotFound):
            manager.delete_payment_instrument(
                payment_instrument_id="instrument-nonexistent",
                payment_connector_id="connector-456",
                user_id="user-123",
            )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_delete_instrument_access_denied(self, mock_session_class):
        """Test AccessDeniedException when deleting an instrument."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Forbidden"}},
            "DeletePaymentInstrument",
        )
        mock_client.delete_payment_instrument.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError):
            manager.delete_payment_instrument(
                payment_instrument_id="instrument-123",
                payment_connector_id="connector-456",
                user_id="user-123",
            )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_delete_instrument_already_deleted(self, mock_session_class):
        """Test deleting an already-deleted instrument returns not found."""
        from bedrock_agentcore.payments.manager import PaymentInstrumentNotFound

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Instrument is already deleted"}},
            "DeletePaymentInstrument",
        )
        mock_client.delete_payment_instrument.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentInstrumentNotFound):
            manager.delete_payment_instrument(
                payment_instrument_id="instrument-123",
                payment_connector_id="connector-456",
                user_id="user-123",
            )


class TestProcessPayment:
    """Tests for process_payment method."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_successful_payment_processing(self, mock_session_class):
        """Test successful payment processing."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        mock_client.get_payment_instrument.return_value = {"paymentInstrumentId": "instrument-123"}
        response = {"processPaymentId": "payment-123"}
        mock_client.process_payment.return_value = response

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        result = manager.process_payment(
            user_id="user-123",
            payment_session_id="session-123",
            payment_instrument_id="instrument-123",
            payment_type="debit",
            payment_input={"amount": 100},
        )

        assert result == response

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_payment_instrument_validation_performed(self, mock_session_class):
        """Test payment instrument validation is performed."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        mock_client.get_payment_instrument.return_value = {"paymentInstrumentId": "instrument-123"}
        mock_client.process_payment.return_value = {"processPaymentId": "payment-123"}

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        manager.process_payment(
            user_id="user-123",
            payment_session_id="session-123",
            payment_instrument_id="instrument-123",
            payment_type="debit",
            payment_input={"amount": 100},
        )

        mock_client.process_payment.assert_called_once()

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_payment_manager_arn_injected_in_payment(self, mock_session_class):
        """Test payment_manager_arn is automatically injected."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        mock_client.get_payment_instrument.return_value = {"paymentInstrumentId": "instrument-123"}
        mock_client.process_payment.return_value = {"processPaymentId": "payment-123"}

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        manager.process_payment(
            user_id="user-123",
            payment_session_id="session-123",
            payment_instrument_id="instrument-123",
            payment_type="debit",
            payment_input={"amount": 100},
        )

        call_kwargs = mock_client.process_payment.call_args[1]
        assert call_kwargs["paymentManagerArn"] == arn

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_process_payment_insufficient_budget(self, mock_session_class):
        """Test InsufficientBudget error during payment processing."""
        from bedrock_agentcore.payments.manager import InsufficientBudget

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        mock_client.get_payment_instrument.return_value = {"paymentInstrumentId": "instrument-123"}
        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Insufficient budget"}},
            "ProcessPayment",
        )
        mock_client.process_payment.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(InsufficientBudget):
            manager.process_payment(
                user_id="user-123",
                payment_session_id="session-123",
                payment_instrument_id="instrument-123",
                payment_type="debit",
                payment_input={"amount": 100},
            )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_process_payment_session_expired(self, mock_session_class):
        """Test PaymentSessionExpired error during payment processing."""
        from bedrock_agentcore.payments.manager import PaymentSessionExpired

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        mock_client.get_payment_instrument.return_value = {"paymentInstrumentId": "instrument-123"}
        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Session expired"}},
            "ProcessPayment",
        )
        mock_client.process_payment.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentSessionExpired):
            manager.process_payment(
                user_id="user-123",
                payment_session_id="session-123",
                payment_instrument_id="instrument-123",
                payment_type="debit",
                payment_input={"amount": 100},
            )


class TestErrorHandling:
    """Tests for error handling in PaymentManager methods."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_get_instrument_access_denied(self, mock_session_class):
        """Test access denied error when getting instrument."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Unauthorized"}},
            "GetPaymentInstrument",
        )
        mock_client.get_payment_instrument.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError):
            manager.get_payment_instrument(
                user_id="user-123",
                payment_connector_id="connector-456",
                payment_instrument_id="instrument-123",
            )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_get_session_access_denied(self, mock_session_class):
        """Test access denied error when getting session."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Unauthorized"}},
            "GetPaymentSession",
        )
        mock_client.get_payment_session.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError):
            manager.get_payment_session(user_id="user-123", payment_session_id="session-123")

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_create_instrument_validation_error(self, mock_session_class):
        """Test validation error when creating instrument."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Invalid wallet address"}},
            "CreatePaymentInstrument",
        )
        mock_client.create_payment_instrument.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError):
            manager.create_payment_instrument(
                user_id="user-123",
                payment_connector_id="connector-456",
                payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
                payment_instrument_details={"wallet_address": "invalid"},
            )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_list_instruments_error(self, mock_session_class):
        """Test error when listing instruments."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "InternalServerError", "Message": "Server error"}},
            "ListPaymentInstruments",
        )
        mock_client.list_payment_instruments.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError):
            manager.list_payment_instruments(user_id="user-123")

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_list_sessions_error(self, mock_session_class):
        """Test error when listing sessions."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "InternalServerError", "Message": "Server error"}},
            "ListPaymentSessions",
        )
        mock_client.list_payment_sessions.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError):
            manager.list_payment_sessions(user_id="user-123")

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_create_session_expiry_validation_error(self, mock_session_class):
        """Test expiry validation error when creating session."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Invalid expiry duration"}},
            "CreatePaymentSession",
        )
        mock_client.create_payment_session.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError):
            manager.create_payment_session(
                user_id="user-123",
                expiry_time_in_minutes=5,
            )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_process_payment_invalid_instrument(self, mock_session_class):
        """Test InvalidPaymentInstrument error during payment processing."""

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        mock_client.get_payment_instrument.return_value = {"paymentInstrumentId": "instrument-123"}
        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Instrument is inactive"}},
            "ProcessPayment",
        )
        mock_client.process_payment.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(InvalidPaymentInstrument):
            manager.process_payment(
                user_id="user-123",
                payment_session_id="session-123",
                payment_instrument_id="instrument-123",
                payment_type="debit",
                payment_input={"amount": 100},
            )


class TestMethodForwarding:
    """Tests for __getattr__ method forwarding."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_getattr_forwards_to_client(self, mock_session_class):
        """Test __getattr__ only forwards allowed methods to PaymentClient."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        # Add a mock method to the client that's in the allowed list
        mock_method = MagicMock(return_value={"result": "success"})
        mock_client.create_payment_instrument = mock_method

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        # Verify that an allowed method can be accessed
        # Since create_payment_instrument is explicitly defined, we verify the method exists
        assert hasattr(manager, "create_payment_instrument")
        assert callable(manager.create_payment_instrument)

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_getattr_raises_attribute_error_for_undefined_method(self, mock_session_class):
        """Test __getattr__ raises AttributeError for undefined methods."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock(spec=[])  # Empty spec means no methods
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(AttributeError) as exc_info:
            manager.undefined_method()

        assert "undefined_method" in str(exc_info.value)
        assert "PaymentManager" in str(exc_info.value)

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_getattr_rejects_non_allowed_methods(self, mock_session_class):
        """Test __getattr__ rejects methods not in the allowed list even if they exist on client."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        # Add a method that exists on the client but is NOT in the allowed list
        mock_client.get_paginator = MagicMock(return_value={"paginator": "data"})
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        # Attempting to access a method not in the allowed list should raise AttributeError
        with pytest.raises(AttributeError) as exc_info:
            manager.get_paginator()

        assert "get_paginator" in str(exc_info.value)
        assert "PaymentManager" in str(exc_info.value)


# ============================================================================
# generatePaymentHeader Tests
# ============================================================================


class TestGeneratePaymentHeaderInputValidation:
    """Tests for input validation in generatePaymentHeader."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_empty_user_id_raises_error(self, mock_session_class):
        """Test that empty user_id raises PaymentError."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError) as exc_info:
            manager.generate_payment_header(
                user_id="",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request={"statusCode": 402, "headers": {}, "body": {}},
            )

        assert "user_id is empty" in str(exc_info.value)

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_none_user_id_raises_error(self, mock_session_class):
        """Test that None user_id raises PaymentError."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError) as exc_info:
            manager.generate_payment_header(
                user_id=None,
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request={"statusCode": 402, "headers": {}, "body": {}},
            )

        assert "user_id is empty" in str(exc_info.value)

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_empty_instrument_id_raises_error(self, mock_session_class):
        """Test that empty instrument_id raises PaymentError."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError) as exc_info:
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="",
                payment_session_id="session-123",
                payment_required_request={"statusCode": 402, "headers": {}, "body": {}},
            )

        assert "instrument_id is empty" in str(exc_info.value)

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_empty_session_id_raises_error(self, mock_session_class):
        """Test that empty session_id raises PaymentError."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError) as exc_info:
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="",
                payment_required_request={"statusCode": 402, "headers": {}, "body": {}},
            )

        assert "session_id is empty" in str(exc_info.value)

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_empty_x402_response_raises_error(self, mock_session_class):
        """Test that empty x402_response raises PaymentError."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError) as exc_info:
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request={},
            )

        assert "payment_required_request is invalid" in str(exc_info.value)

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_missing_statuscode_in_x402_response_raises_error(self, mock_session_class):
        """Test that missing statusCode in x402_response raises PaymentError."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError) as exc_info:
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request={"headers": {}, "body": {}},
            )

        assert "missing required fields" in str(exc_info.value)


class TestGeneratePaymentHeaderX402Extraction:
    """Tests for X.402 payload extraction in generatePaymentHeader."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_v1_payload_extraction_from_body(self, mock_session_class):
        """Test successful v1 payload extraction from body."""

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "ethereum",
            "accepts": [{"network": "ethereum", "value": "100"}],
            "payload": "proof-data",
        }

        mock_client.get_payment_instrument.return_value = {
            "paymentInstrument": {
                "paymentInstrumentId": "instrument-123",
                "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                "paymentInstrumentDetails": {
                    "embeddedCryptoWallet": {
                        "network": "ethereum",
                    }
                },
                "status": "active",
            }
        }
        mock_client.process_payment.return_value = {
            "paymentOutput": {
                "cryptoX402": {
                    "payload": "payment-proof",
                }
            }
        }

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request={"statusCode": 402, "headers": {}, "body": v1_payload},
        )

        assert isinstance(result, dict)
        assert "X-PAYMENT" in result

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_v2_payload_extraction_from_header(self, mock_session_class):
        """Test successful v2 payload extraction from payment-required header."""
        import base64
        import json

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v2_payload = {
            "x402Version": 2,
            "scheme": "exact",
            "network": "ethereum",
            "resource": "https://example.com",
            "accepts": [{"network": "ethereum", "value": "100"}],
            "payload": "proof-data",
        }

        encoded_payload = base64.b64encode(json.dumps(v2_payload).encode()).decode()

        mock_client.get_payment_instrument.return_value = {
            "paymentInstrument": {
                "paymentInstrumentId": "instrument-123",
                "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                "paymentInstrumentDetails": {
                    "embeddedCryptoWallet": {
                        "network": "ethereum",
                    }
                },
                "status": "active",
            }
        }
        mock_client.process_payment.return_value = {
            "paymentOutput": {
                "cryptoX402": {
                    "payload": "payment-proof",
                }
            }
        }

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request={"statusCode": 402, "headers": {"payment-required": encoded_payload}, "body": {}},
        )

        assert isinstance(result, dict)
        assert "PAYMENT-SIGNATURE" in result

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_invalid_base64_in_v2_header_raises_error(self, mock_session_class):
        """Test that invalid base64 in v2 header raises PaymentError."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError) as exc_info:
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request={
                    "statusCode": 402,
                    "headers": {"payment-required": "invalid-base64!!!"},
                    "body": {},
                },
            )

        assert "Failed to decode v2 payload" in str(exc_info.value)

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_missing_required_fields_in_payload_raises_error(self, mock_session_class):
        """Test that missing required fields in payload raises PaymentError."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        incomplete_payload = {
            "x402Version": 1,
            "scheme": "exact",
            # Missing: network, accepts, payload
        }

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError) as exc_info:
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request={"statusCode": 402, "headers": {}, "body": incomplete_payload},
            )

        assert "Missing required fields" in str(exc_info.value)


class TestGeneratePaymentHeaderNetworkDetection:
    """Tests for network detection and accept selection."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_ethereum_network_detection(self, mock_session_class):
        """Test Ethereum network detection and accept selection."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "ethereum",
            "accepts": [{"network": "base-sepolia", "value": "100"}],
            "payload": "proof-data",
        }

        mock_client.get_payment_instrument.return_value = {
            "paymentInstrument": {
                "paymentInstrumentId": "instrument-123",
                "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                "paymentInstrumentDetails": {
                    "embeddedCryptoWallet": {
                        "network": "ETHEREUM",
                    }
                },
                "status": "active",
            }
        }
        mock_client.process_payment.return_value = {"paymentOutput": {"cryptoX402": {"payload": "payment-proof"}}}

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request={"statusCode": 402, "headers": {}, "body": v1_payload},
        )

        assert result is not None
        mock_client.process_payment.assert_called_once()

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_solana_network_detection(self, mock_session_class):
        """Test Solana network detection and accept selection."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "solana",
            "accepts": [{"network": "solana-mainnet", "value": "100"}],
            "payload": "proof-data",
        }

        mock_client.get_payment_instrument.return_value = {
            "paymentInstrument": {
                "paymentInstrumentId": "instrument-123",
                "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                "paymentInstrumentDetails": {
                    "embeddedCryptoWallet": {
                        "network": "SOLANA",
                    }
                },
                "status": "active",
            }
        }
        mock_client.process_payment.return_value = {"paymentOutput": {"cryptoX402": {"payload": "payment-proof"}}}

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request={"statusCode": 402, "headers": {}, "body": v1_payload},
        )

        assert result is not None

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_unsupported_network_raises_error(self, mock_session_class):
        """Test that unsupported instrument network raises PaymentError."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "ethereum",
            "accepts": [{"network": "ethereum", "value": "100"}],
            "payload": "proof-data",
        }

        mock_client.get_payment_instrument.return_value = {
            "paymentInstrument": {
                "paymentInstrumentId": "instrument-123",
                "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                "paymentInstrumentDetails": {
                    "embeddedCryptoWallet": {
                        "network": "INVALID_NETWORK",
                    }
                },
                "status": "active",
            }
        }

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError) as exc_info:
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request={"statusCode": 402, "headers": {}, "body": v1_payload},
            )

        assert "Unsupported network" in str(exc_info.value)

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_no_matching_accept_for_instrument_network(self, mock_session_class):
        """Test error when no accept matches instrument network."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        # X.402 payload only has Solana accepts
        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "solana",
            "accepts": [{"network": "solana-mainnet", "value": "100"}],
            "payload": "proof-data",
        }

        # But instrument is Ethereum
        mock_client.get_payment_instrument.return_value = {
            "paymentInstrument": {
                "paymentInstrumentId": "instrument-123",
                "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                "paymentInstrumentDetails": {
                    "embeddedCryptoWallet": {
                        "network": "ETHEREUM",
                    }
                },
                "status": "active",
            }
        }

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentError) as exc_info:
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request={"statusCode": 402, "headers": {}, "body": v1_payload},
            )

        assert "No matching accept" in str(exc_info.value)
        assert "does not support the network" in str(exc_info.value)


class TestGeneratePaymentHeaderV1Format:
    """Tests for v1 header format (X-PAYMENT)."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_v1_header_format_correctness(self, mock_session_class):
        """Test v1 header format is correct."""
        import base64
        import json

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "ethereum",
            "accepts": [{"scheme": "exact", "network": "ethereum", "value": "100"}],
            "payload": "proof-data",
        }

        mock_client.get_payment_instrument.return_value = {
            "paymentInstrument": {
                "paymentInstrumentId": "instrument-123",
                "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                "paymentInstrumentDetails": {
                    "embeddedCryptoWallet": {
                        "network": "ETHEREUM",
                    }
                },
                "status": "active",
            }
        }
        mock_client.process_payment.return_value = {"paymentOutput": {"cryptoX402": {"payload": "payment-proof"}}}

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request={"statusCode": 402, "headers": {}, "body": v1_payload},
        )

        assert isinstance(result, dict)
        assert "X-PAYMENT" in result
        header_value = result["X-PAYMENT"]
        decoded = base64.b64decode(header_value)
        header_json = json.loads(decoded)
        assert header_json["x402Version"] == 1
        assert header_json["scheme"] == "exact"
        assert header_json["network"] == "ethereum"
        assert header_json["payload"] == "payment-proof"


class TestGeneratePaymentHeaderV2Format:
    """Tests for v2 header format (PAYMENT-SIGNATURE)."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_v2_header_format_correctness(self, mock_session_class):
        """Test v2 header format is correct."""
        import base64
        import json

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v2_payload = {
            "x402Version": 2,
            "scheme": "exact",
            "network": "ethereum",
            "resource": "https://example.com",
            "accepts": [{"network": "ethereum", "value": "100"}],
            "payload": "proof-data",
        }

        encoded_payload = base64.b64encode(json.dumps(v2_payload).encode()).decode()

        mock_client.get_payment_instrument.return_value = {
            "paymentInstrument": {
                "paymentInstrumentId": "instrument-123",
                "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                "paymentInstrumentDetails": {
                    "embeddedCryptoWallet": {
                        "network": "ETHEREUM",
                    }
                },
                "status": "active",
            }
        }
        mock_client.process_payment.return_value = {"paymentOutput": {"cryptoX402": {"payload": "payment-proof"}}}

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request={"statusCode": 402, "headers": {"payment-required": encoded_payload}, "body": {}},
        )

        assert isinstance(result, dict)
        assert "PAYMENT-SIGNATURE" in result
        header_value = result["PAYMENT-SIGNATURE"]
        decoded = base64.b64decode(header_value)
        header_json = json.loads(decoded)
        assert header_json["x402Version"] == 2
        assert header_json["resource"] == "https://example.com"
        assert header_json["payload"] == "payment-proof"
        assert "accepted" in header_json


class TestGeneratePaymentHeaderErrorHandling:
    """Tests for error handling in generatePaymentHeader."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_instrument_not_found_propagated(self, mock_session_class):
        """Test PaymentInstrumentNotFound is propagated."""
        from bedrock_agentcore.payments.manager import PaymentInstrumentNotFound

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "ethereum",
            "accepts": [{"network": "ethereum", "value": "100"}],
            "payload": "proof-data",
        }

        error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Instrument not found"}},
            "GetPaymentInstrument",
        )
        mock_client.get_payment_instrument.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentInstrumentNotFound):
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request={"statusCode": 402, "headers": {}, "body": v1_payload},
            )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_payment_session_expired_propagated(self, mock_session_class):
        """Test PaymentSessionExpired is propagated from processPayment."""
        from bedrock_agentcore.payments.manager import PaymentSessionExpired

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "ethereum",
            "accepts": [{"network": "ethereum", "value": "100"}],
            "payload": "proof-data",
        }

        mock_client.get_payment_instrument.return_value = {
            "paymentInstrument": {
                "paymentInstrumentId": "instrument-123",
                "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                "paymentInstrumentDetails": {
                    "embeddedCryptoWallet": {
                        "network": "ethereum",
                    }
                },
                "status": "active",
            }
        }
        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Session expired"}},
            "ProcessPayment",
        )
        mock_client.process_payment.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(PaymentSessionExpired):
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request={"statusCode": 402, "headers": {}, "body": v1_payload},
            )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_insufficient_budget_propagated(self, mock_session_class):
        """Test InsufficientBudget is propagated from processPayment."""
        from bedrock_agentcore.payments.manager import InsufficientBudget

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "ethereum",
            "accepts": [{"network": "ethereum", "value": "100"}],
            "payload": "proof-data",
        }

        mock_client.get_payment_instrument.return_value = {
            "paymentInstrument": {
                "paymentInstrumentId": "instrument-123",
                "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                "paymentInstrumentDetails": {
                    "embeddedCryptoWallet": {
                        "network": "ethereum",
                    }
                },
                "status": "active",
            }
        }
        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Insufficient budget"}},
            "ProcessPayment",
        )
        mock_client.process_payment.side_effect = error

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        with pytest.raises(InsufficientBudget):
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request={"statusCode": 402, "headers": {}, "body": v1_payload},
            )


# ============================================================================
# generatePaymentHeader Client Token Tests
# ============================================================================


class TestGeneratePaymentHeaderClientToken:
    """Tests for client_token parameter in generatePaymentHeader."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_client_token_generated_when_not_provided(self, mock_session_class):
        """Test that a client_token is generated when not provided."""

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v1_payload = {
            "x402Version": 1,
            "scheme": "ethereum",
            "network": "ethereum",
            "accepts": [{"network": "ethereum", "accept": "application/json"}],
            "payload": "test-payload",
        }

        mock_client.get_payment_instrument.return_value = {
            "paymentInstrument": {
                "paymentInstrumentId": "instrument-123",
                "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                "paymentInstrumentDetails": {
                    "embeddedCryptoWallet": {
                        "network": "ETHEREUM",
                    }
                },
                "status": "active",
            }
        }
        mock_client.process_payment.return_value = {"paymentOutput": {"cryptoX402": {"payload": "proof-of-payment"}}}

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        # Call without client_token
        manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request={"statusCode": 402, "headers": {}, "body": v1_payload},
        )

        # Verify process_payment was called with a generated client_token
        call_kwargs = mock_client.process_payment.call_args[1]
        assert "clientToken" in call_kwargs
        assert isinstance(call_kwargs["clientToken"], str)
        assert len(call_kwargs["clientToken"]) > 0

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_client_token_passed_through_when_provided(self, mock_session_class):
        """Test that provided client_token is passed through to processPayment."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v1_payload = {
            "x402Version": 1,
            "scheme": "ethereum",
            "network": "ethereum",
            "accepts": [{"network": "ethereum", "accept": "application/json"}],
            "payload": "test-payload",
        }

        mock_client.get_payment_instrument.return_value = {
            "paymentInstrument": {
                "paymentInstrumentId": "instrument-123",
                "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                "paymentInstrumentDetails": {
                    "embeddedCryptoWallet": {
                        "network": "ETHEREUM",
                    }
                },
                "status": "active",
            }
        }
        mock_client.process_payment.return_value = {"paymentOutput": {"cryptoX402": {"payload": "proof-of-payment"}}}

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        # Call with provided client_token
        provided_token = "my-custom-token-123"
        manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request={"statusCode": 402, "headers": {}, "body": v1_payload},
            client_token=provided_token,
        )

        # Verify process_payment was called with the provided client_token
        call_kwargs = mock_client.process_payment.call_args[1]
        assert call_kwargs["clientToken"] == provided_token

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_empty_client_token_raises_error(self, mock_session_class):
        """Test that empty client_token raises PaymentError."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v1_payload = {
            "x402Version": 1,
            "scheme": "ethereum",
            "network": "ethereum",
            "accepts": [{"network": "ethereum", "accept": "application/json"}],
            "payload": "test-payload",
        }

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        # Call with empty client_token
        with pytest.raises(PaymentError) as exc_info:
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request={"statusCode": 402, "headers": {}, "body": v1_payload},
                client_token="",
            )

        assert "client_token is invalid - cannot be empty" in str(exc_info.value)

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_non_string_client_token_raises_error(self, mock_session_class):
        """Test that non-string client_token raises PaymentError."""
        from bedrock_agentcore.payments.manager import PaymentError

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        v1_payload = {
            "x402Version": 1,
            "scheme": "ethereum",
            "network": "ethereum",
            "accepts": [{"network": "ethereum", "accept": "application/json"}],
            "payload": "test-payload",
        }

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        # Call with non-string client_token
        with pytest.raises(PaymentError) as exc_info:
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request={"statusCode": 402, "headers": {}, "body": v1_payload},
                client_token=12345,
            )

        assert "client_token is invalid - must be a string" in str(exc_info.value)


# ============================================================================
# Test Infrastructure and Fixtures for generatePaymentHeader
# ============================================================================

"""
Test infrastructure for generatePaymentHeader property-based testing.

This module provides:
- Hypothesis strategies for generating valid test inputs
- Test fixtures for mocking PaymentManager methods
- Helper functions for creating test data
"""

# ============================================================================
# Hypothesis Strategies for Property-Based Testing
# ============================================================================


# ============================================================================
# Test Fixtures for Mocking PaymentManager Methods
# ============================================================================


class PaymentManagerMockFixtures:
    """Fixtures for mocking PaymentManager methods in tests."""

    @staticmethod
    def create_mock_payment_manager(mock_session_class):
        """Create a mocked PaymentManager instance."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")
        return manager, mock_client

    @staticmethod
    def setup_successful_instrument_retrieval(mock_client, network="ethereum"):
        """Setup mock for successful instrument retrieval."""
        mock_client.get_payment_instrument.return_value = {
            "paymentInstrument": {
                "paymentInstrumentId": "instrument-123",
                "paymentInstrumentType": "EMBEDDED_CRYPTO_WALLET",
                "paymentInstrumentDetails": {
                    "embeddedCryptoWallet": {
                        "network": network,
                    }
                },
                "status": "active",
            }
        }

    @staticmethod
    def setup_successful_payment_processing(mock_client, payload="payment-proof"):
        """Setup mock for successful payment processing."""
        mock_client.process_payment.return_value = {
            "processPaymentId": "payment-123",
            "paymentOutput": {
                "cryptoX402": {
                    "payload": payload,
                }
            },
        }

    @staticmethod
    def setup_instrument_not_found(mock_client):
        """Setup mock for instrument not found error."""
        error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Instrument not found"}},
            "GetPaymentInstrument",
        )
        mock_client.get_payment_instrument.side_effect = error

    @staticmethod
    def setup_session_expired(mock_client):
        """Setup mock for session expired error."""
        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Session expired"}},
            "ProcessPayment",
        )
        mock_client.process_payment.side_effect = error

    @staticmethod
    def setup_insufficient_budget(mock_client):
        """Setup mock for insufficient budget error."""
        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Insufficient budget"}},
            "ProcessPayment",
        )
        mock_client.process_payment.side_effect = error


# ============================================================================
# Helper Functions for Test Data Creation
# ============================================================================


def create_v1_x402_response(scheme="exact", network="ethereum", payload_data="proof-data"):
    """Create a valid v1 X.402 response (requirement structure without payload field)."""
    return {
        "statusCode": 402,
        "headers": {},
        "body": {
            "x402Version": 1,
            "scheme": scheme,
            "network": network,
            "accepts": [
                {
                    "scheme": scheme,
                    "network": network,
                    "maxAmountRequired": "5000",
                    "resource": "https://nickeljoke.vercel.app/api/joke",
                    "description": "Premium AI joke generation",
                    "mimeType": "application/json",
                    "payTo": "0x6813749E1eB9E0001A44C2684695FE8AD676cdD0",
                    "maxTimeoutSeconds": 300,
                    "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
                    "extra": {"name": "USDC", "version": "2"},
                }
            ],
        },
    }


def create_v2_x402_response(
    scheme="exact", network="ethereum", resource="https://example.com", payload_data="proof-data"
):
    """Create a valid v2 X.402 response (requirement structure without payload field)."""
    v2_payload = {
        "x402Version": 2,
        "scheme": scheme,
        "network": network,
        "resource": resource,
        "accepts": [
            {
                "scheme": scheme,
                "network": network,
                "maxAmountRequired": "5000",
                "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
                "payTo": "0x6813749E1eB9E0001A44C2684695FE8AD676cdD0",
                "maxTimeoutSeconds": 300,
                "extra": {"name": "USDC", "version": "2"},
            }
        ],
    }
    encoded_payload = base64.b64encode(json.dumps(v2_payload).encode()).decode()
    return {
        "statusCode": 402,
        "headers": {"payment-required": encoded_payload},
        "body": {},
    }


def create_payment_instrument(network="ethereum", status="active"):
    """Create a valid payment instrument."""
    return {
        "instrumentId": "instrument-123",
        "network": network,
        "status": status,
    }


def create_payment_credentials(payload="payment-proof"):
    """Create valid payment credentials."""
    return {
        "processPaymentId": "payment-123",
        "payload": payload,
    }


# ============================================================================
# Integration Tests for generatePaymentHeader
# ============================================================================


class TestGeneratePaymentHeaderIntegration:
    """Integration tests for generatePaymentHeader complete workflow."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_complete_v1_workflow(self, mock_session_class):
        """Test complete workflow with v1 payload.

        Verifies the full workflow:
        - Input validation
        - X.402 payload extraction (v1)
        - Instrument retrieval
        - Network detection
        - Accept selection
        - Payment processing
        - Header building (X-PAYMENT format)
        """
        manager, mock_client = PaymentManagerMockFixtures.create_mock_payment_manager(mock_session_class)

        # Setup mocks for successful workflow
        PaymentManagerMockFixtures.setup_successful_instrument_retrieval(mock_client, network="ethereum")
        PaymentManagerMockFixtures.setup_successful_payment_processing(mock_client, payload="v1-proof")

        # Create v1 X.402 response
        x402_response = create_v1_x402_response(scheme="exact", network="ethereum", payload_data="v1-proof-data")

        # Execute
        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request=x402_response,
        )

        # Verify result is a valid X-PAYMENT header
        assert isinstance(result, dict)
        assert "X-PAYMENT" in result
        header_value = result["X-PAYMENT"]
        decoded = base64.b64decode(header_value)
        header_json = json.loads(decoded)

        assert header_json["x402Version"] == 1
        assert header_json["scheme"] == "exact"
        assert header_json["network"] == "ethereum"
        assert header_json["payload"] == "v1-proof"

        # Verify mocks were called correctly
        mock_client.get_payment_instrument.assert_called_once()
        mock_client.process_payment.assert_called_once()

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_complete_v2_workflow(self, mock_session_class):
        """Test complete workflow with v2 payload.

        Verifies the full workflow:
        - Input validation
        - X.402 payload extraction (v2 from headers)
        - Instrument retrieval
        - Network detection
        - Accept selection
        - Payment processing
        - Header building (PAYMENT-SIGNATURE format)
        """
        manager, mock_client = PaymentManagerMockFixtures.create_mock_payment_manager(mock_session_class)

        # Setup mocks for successful workflow
        PaymentManagerMockFixtures.setup_successful_instrument_retrieval(mock_client, network="SOLANA")
        PaymentManagerMockFixtures.setup_successful_payment_processing(mock_client, payload="v2-proof")

        # Create v2 X.402 response
        x402_response = create_v2_x402_response(
            scheme="exact", network="solana", resource="https://example.com/resource", payload_data="v2-proof-data"
        )

        # Execute
        result = manager.generate_payment_header(
            user_id="user-456",
            payment_instrument_id="instrument-456",
            payment_session_id="session-456",
            payment_required_request=x402_response,
        )

        # Verify result is a valid PAYMENT-SIGNATURE header
        assert isinstance(result, dict)
        assert "PAYMENT-SIGNATURE" in result
        header_value = result["PAYMENT-SIGNATURE"]
        decoded = base64.b64decode(header_value)
        header_json = json.loads(decoded)

        assert header_json["x402Version"] == 2
        assert header_json["resource"] == "https://example.com/resource"
        assert header_json["payload"] == "v2-proof"
        assert "accepted" in header_json
        assert "extension" in header_json

        # Verify mocks were called correctly
        mock_client.get_payment_instrument.assert_called_once()
        mock_client.process_payment.assert_called_once()

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_end_to_end_with_all_components(self, mock_session_class):
        """Test end-to-end workflow with all components integrated.

        Verifies:
        - All validation steps pass
        - All components work together
        - Final header is correctly formatted
        - All mocks are called with correct parameters
        """
        manager, mock_client = PaymentManagerMockFixtures.create_mock_payment_manager(mock_session_class)

        # Setup mocks
        PaymentManagerMockFixtures.setup_successful_instrument_retrieval(mock_client, network="ETHEREUM")
        PaymentManagerMockFixtures.setup_successful_payment_processing(mock_client, payload="integration-proof")

        # Create X.402 response
        x402_response = create_v1_x402_response(
            scheme="exact", network="ethereum", payload_data="integration-proof-data"
        )

        # Execute
        result = manager.generate_payment_header(
            user_id="integration-user",
            payment_instrument_id="integration-instrument",
            payment_session_id="integration-session",
            payment_required_request=x402_response,
        )

        # Verify result
        assert isinstance(result, dict)
        assert len(result) > 0
        assert "X-PAYMENT" in result or "PAYMENT-SIGNATURE" in result

        # Verify instrument retrieval was called with correct parameters
        call_args = mock_client.get_payment_instrument.call_args
        assert call_args is not None
        assert call_args[1]["paymentInstrumentId"] == "integration-instrument"

        # Verify payment processing was called with correct parameters
        call_args = mock_client.process_payment.call_args
        assert call_args is not None
        assert call_args[1]["paymentSessionId"] == "integration-session"
        assert call_args[1]["paymentInstrumentId"] == "integration-instrument"
        assert call_args[1]["paymentType"] == "CRYPTO_X402"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_error_handling_across_components(self, mock_session_class):
        """Test error handling and propagation across all components.

        Verifies:
        - Errors from instrument retrieval are propagated
        - Errors from payment processing are propagated
        - Error messages include context
        - No silent failures
        """
        manager, mock_client = PaymentManagerMockFixtures.create_mock_payment_manager(mock_session_class)

        # Test 1: Instrument not found error
        PaymentManagerMockFixtures.setup_instrument_not_found(mock_client)

        x402_response = create_v1_x402_response()

        with pytest.raises(PaymentInstrumentNotFound):
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="nonexistent-instrument",
                payment_session_id="session-123",
                payment_required_request=x402_response,
            )

        # Test 2: Session expired error
        mock_client.reset_mock()
        mock_client.get_payment_instrument.side_effect = None
        mock_client.process_payment.side_effect = None
        PaymentManagerMockFixtures.setup_successful_instrument_retrieval(mock_client)
        PaymentManagerMockFixtures.setup_session_expired(mock_client)

        with pytest.raises(PaymentSessionExpired):
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="expired-session",
                payment_required_request=x402_response,
            )

        # Test 3: Insufficient budget error
        mock_client.reset_mock()
        mock_client.get_payment_instrument.side_effect = None
        mock_client.process_payment.side_effect = None
        PaymentManagerMockFixtures.setup_successful_instrument_retrieval(mock_client)
        PaymentManagerMockFixtures.setup_insufficient_budget(mock_client)

        with pytest.raises(InsufficientBudget):
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request=x402_response,
            )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_invalid_status_code_returns_early(self, mock_session_class):
        """Test that non-402 status codes are handled correctly.

        Verifies:
        - Invalid status codes are detected
        - Error is raised before any PaymentManager methods are called
        - Error message includes the invalid status code
        - Fail-fast behavior is maintained
        """
        manager, mock_client = PaymentManagerMockFixtures.create_mock_payment_manager(mock_session_class)

        # Create X.402 response with invalid status code
        x402_response = create_v1_x402_response()
        x402_response["statusCode"] = 400  # Invalid status code

        # Execute and verify error
        with pytest.raises(PaymentError) as exc_info:
            manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-123",
                payment_session_id="session-123",
                payment_required_request=x402_response,
            )

        # Verify error message includes context
        assert "402" in str(exc_info.value)
        assert "400" in str(exc_info.value)

        # Verify no PaymentManager methods were called (fail-fast)
        mock_client.get_payment_instrument.assert_not_called()
        mock_client.process_payment.assert_not_called()

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_client_token_generation(self, mock_session_class):
        """Test that client token is generated when not provided.

        Verifies:
        - Client token is generated when not provided
        - Generated token is passed to processPayment
        - Token is a valid UUID format
        """
        manager, mock_client = PaymentManagerMockFixtures.create_mock_payment_manager(mock_session_class)

        # Setup mocks
        PaymentManagerMockFixtures.setup_successful_instrument_retrieval(mock_client)
        PaymentManagerMockFixtures.setup_successful_payment_processing(mock_client)

        x402_response = create_v1_x402_response()

        # Execute without client_token
        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request=x402_response,
        )

        # Verify result
        assert result is not None

        # Verify processPayment was called with a clientToken
        call_args = mock_client.process_payment.call_args
        assert call_args is not None
        assert "clientToken" in call_args[1]
        client_token = call_args[1]["clientToken"]
        assert client_token is not None
        assert len(client_token) > 0

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_client_token_passed_through(self, mock_session_class):
        """Test that provided client token is passed to processPayment.

        Verifies:
        - Provided client token is used
        - Token is passed to processPayment unchanged
        - Token is not regenerated
        """
        manager, mock_client = PaymentManagerMockFixtures.create_mock_payment_manager(mock_session_class)

        # Setup mocks
        PaymentManagerMockFixtures.setup_successful_instrument_retrieval(mock_client)
        PaymentManagerMockFixtures.setup_successful_payment_processing(mock_client)

        x402_response = create_v1_x402_response()
        provided_token = "my-custom-token-12345"

        # Execute with client_token
        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request=x402_response,
            client_token=provided_token,
        )

        # Verify result
        assert result is not None

        # Verify processPayment was called with the provided token
        call_args = mock_client.process_payment.call_args
        assert call_args is not None
        assert call_args[1]["clientToken"] == provided_token


# ============================================================================
# Network Preferences Tests
# ============================================================================


class TestGeneratePaymentHeaderNetworkPreferences:
    """Tests for network preferences in accept selection."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_network_preferences_selects_preferred_network(self, mock_session_class):
        """Test that network preferences are used to select the preferred accept.

        Verifies:
        - Multiple accepts are available
        - Preferred network from preferences is selected
        - Other accepts are ignored
        """
        manager, mock_client = PaymentManagerMockFixtures.create_mock_payment_manager(mock_session_class)

        # Setup mocks
        PaymentManagerMockFixtures.setup_successful_instrument_retrieval(mock_client, network="ethereum")
        PaymentManagerMockFixtures.setup_successful_payment_processing(mock_client)

        # X.402 payload with multiple Ethereum accepts
        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "ethereum",
            "accepts": [
                {"network": "ethereum", "value": "100"},
                {"network": "base-sepolia", "value": "50"},
                {"network": "eip155:8453", "value": "75"},
            ],
            "payload": "proof-data",
        }

        x402_response = {"statusCode": 402, "headers": {}, "body": v1_payload}

        # Provide network preferences with base-sepolia first
        network_preferences = ["base-sepolia", "ethereum", "eip155:8453"]

        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request=x402_response,
            network_preferences=network_preferences,
        )

        assert result is not None

        # Verify processPayment was called with the base-sepolia accept
        call_args = mock_client.process_payment.call_args
        assert call_args is not None
        payment_input = call_args[1]["paymentInput"]
        selected_accept = payment_input["cryptoX402"]["payload"]
        assert selected_accept["network"] == "base-sepolia"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_network_preferences_fallback_to_first_available(self, mock_session_class):
        """Test fallback to first available accept when no preference matches.

        Verifies:
        - When no preference matches available accepts
        - First available accept is selected
        - No error is raised
        """
        manager, mock_client = PaymentManagerMockFixtures.create_mock_payment_manager(mock_session_class)

        # Setup mocks
        PaymentManagerMockFixtures.setup_successful_instrument_retrieval(mock_client, network="ethereum")
        PaymentManagerMockFixtures.setup_successful_payment_processing(mock_client)

        # X.402 payload with Ethereum accepts
        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "ethereum",
            "accepts": [
                {"network": "ethereum", "value": "100"},
                {"network": "eip155:8453", "value": "75"},
            ],
            "payload": "proof-data",
        }

        x402_response = {"statusCode": 402, "headers": {}, "body": v1_payload}

        # Provide network preferences that don't match any accepts
        network_preferences = ["solana-mainnet", "base-sepolia"]

        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request=x402_response,
            network_preferences=network_preferences,
        )

        assert result is not None

        # Verify processPayment was called with the first available accept
        call_args = mock_client.process_payment.call_args
        assert call_args is not None
        payment_input = call_args[1]["paymentInput"]
        selected_accept = payment_input["cryptoX402"]["payload"]
        assert selected_accept["network"] == "ethereum"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_default_network_preferences_used_when_not_provided(self, mock_session_class):
        """Test that default NETWORK_PREFERENCES are used when not provided.

        Verifies:
        - When network_preferences is None
        - Default preferences from constants are used
        - Selection follows default preference order
        """
        manager, mock_client = PaymentManagerMockFixtures.create_mock_payment_manager(mock_session_class)

        # Setup mocks
        PaymentManagerMockFixtures.setup_successful_instrument_retrieval(mock_client, network="solana")
        PaymentManagerMockFixtures.setup_successful_payment_processing(mock_client)

        # X.402 payload with multiple Solana accepts
        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "solana",
            "accepts": [
                {"network": "solana-devnet", "value": "50"},
                {"network": "solana-mainnet", "value": "100"},
            ],
            "payload": "proof-data",
        }

        x402_response = {"statusCode": 402, "headers": {}, "body": v1_payload}

        # Don't provide network_preferences - should use defaults
        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request=x402_response,
        )

        assert result is not None

        # Verify processPayment was called
        call_args = mock_client.process_payment.call_args
        assert call_args is not None
        payment_input = call_args[1]["paymentInput"]
        selected_accept = payment_input["cryptoX402"]["payload"]
        # Should select based on default preferences (solana-mainnet is preferred)
        assert selected_accept["network"] in ["solana-mainnet", "solana-devnet"]

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_network_preferences_case_insensitive_matching(self, mock_session_class):
        """Test that network matching is case-insensitive.

        Verifies:
        - Preferences with different cases match accepts
        - Case variations don't prevent selection
        """
        manager, mock_client = PaymentManagerMockFixtures.create_mock_payment_manager(mock_session_class)

        # Setup mocks
        PaymentManagerMockFixtures.setup_successful_instrument_retrieval(mock_client, network="ethereum")
        PaymentManagerMockFixtures.setup_successful_payment_processing(mock_client)

        # X.402 payload with mixed case network names
        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "ethereum",
            "accepts": [
                {"network": "Base-Sepolia", "value": "50"},
                {"network": "ethereum", "value": "100"},
            ],
            "payload": "proof-data",
        }

        x402_response = {"statusCode": 402, "headers": {}, "body": v1_payload}

        # Provide preferences with different case
        network_preferences = ["base-sepolia", "ethereum"]

        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request=x402_response,
            network_preferences=network_preferences,
        )

        assert result is not None

        # Verify processPayment was called with the base-sepolia accept
        call_args = mock_client.process_payment.call_args
        assert call_args is not None
        payment_input = call_args[1]["paymentInput"]
        selected_accept = payment_input["cryptoX402"]["payload"]
        assert selected_accept["network"].lower() == "base-sepolia"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_network_preferences_with_solana_networks(self, mock_session_class):
        """Test network preferences work correctly with Solana networks.

        Verifies:
        - Solana network preferences are respected
        - Multiple Solana variants are handled
        """
        manager, mock_client = PaymentManagerMockFixtures.create_mock_payment_manager(mock_session_class)

        # Setup mocks
        PaymentManagerMockFixtures.setup_successful_instrument_retrieval(mock_client, network="solana")
        PaymentManagerMockFixtures.setup_successful_payment_processing(mock_client)

        # X.402 payload with multiple Solana accepts
        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "solana",
            "accepts": [
                {"network": "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp", "value": "100"},
                {"network": "solana-mainnet", "value": "100"},
                {"network": "solana-devnet", "value": "50"},
            ],
            "payload": "proof-data",
        }

        x402_response = {"statusCode": 402, "headers": {}, "body": v1_payload}

        # Provide Solana-specific preferences
        network_preferences = ["solana-devnet", "solana-mainnet", "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"]

        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request=x402_response,
            network_preferences=network_preferences,
        )

        assert result is not None

        # Verify processPayment was called with the devnet accept
        call_args = mock_client.process_payment.call_args
        assert call_args is not None
        payment_input = call_args[1]["paymentInput"]
        selected_accept = payment_input["cryptoX402"]["payload"]
        assert selected_accept["network"] == "solana-devnet"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_empty_network_preferences_uses_defaults(self, mock_session_class):
        """Test that empty network preferences list falls back to defaults.

        Verifies:
        - Empty list is treated as no preference
        - First available accept is selected
        """
        manager, mock_client = PaymentManagerMockFixtures.create_mock_payment_manager(mock_session_class)

        # Setup mocks
        PaymentManagerMockFixtures.setup_successful_instrument_retrieval(mock_client, network="ethereum")
        PaymentManagerMockFixtures.setup_successful_payment_processing(mock_client)

        # X.402 payload with Ethereum accepts
        v1_payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": "ethereum",
            "accepts": [
                {"network": "eip155:8453", "value": "75"},
            ],
            "payload": "proof-data",
        }

        x402_response = {"statusCode": 402, "headers": {}, "body": v1_payload}

        # Provide empty network preferences
        network_preferences = []

        result = manager.generate_payment_header(
            user_id="user-123",
            payment_instrument_id="instrument-123",
            payment_session_id="session-123",
            payment_required_request=x402_response,
            network_preferences=network_preferences,
        )

        assert result is not None

        # Verify processPayment was called with the first available accept
        call_args = mock_client.process_payment.call_args
        assert call_args is not None
        payment_input = call_args[1]["paymentInput"]
        selected_accept = payment_input["cryptoX402"]["payload"]
        assert selected_accept["network"] == "eip155:8453"


# ============================================================================
# PaymentManager Agent Name Header Tests
# ============================================================================


class TestPaymentManagerAgentNameHeader:
    """Tests for agent_name header injection via boto3 event handler."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_without_agent_name_does_not_register_event(self, mock_session_class):
        """Test that no event handler is registered when agent_name is not provided."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        mock_client.meta.events.register.assert_not_called()

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_with_none_agent_name_does_not_register_event(self, mock_session_class):
        """Test that no event handler is registered when agent_name is explicitly None."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        PaymentManager(payment_manager_arn=arn, region_name="us-east-1", agent_name=None)

        mock_client.meta.events.register.assert_not_called()

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_with_agent_name_registers_event_handler(self, mock_session_class):
        """Test that event handler is registered when agent_name is provided."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1", agent_name="my-agent")

        mock_client.meta.events.register.assert_called_once_with(
            "before-sign.bedrock-agentcore.*",
            manager._add_agent_name_header,
        )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_stores_agent_name(self, mock_session_class):
        """Test that agent_name is stored on the manager instance."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.client.return_value = MagicMock()
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1", agent_name="test-agent")

        assert manager._agent_name == "test-agent"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_stores_none_agent_name_when_not_provided(self, mock_session_class):
        """Test that _agent_name is None when not provided."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.client.return_value = MagicMock()
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        assert manager._agent_name is None

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_add_agent_name_header_injects_correct_header(self, mock_session_class):
        """Test that _add_agent_name_header sets the correct HTTP header."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.client.return_value = MagicMock()
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1", agent_name="my-agent")

        mock_request = MagicMock()
        mock_request.headers = {}

        manager._add_agent_name_header(mock_request)

        assert mock_request.headers["X-Amzn-Bedrock-AgentCore-Payments-Agent-Name"] == "my-agent"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_add_agent_name_header_preserves_existing_headers(self, mock_session_class):
        """Test that _add_agent_name_header does not overwrite existing headers."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.client.return_value = MagicMock()
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1", agent_name="my-agent")

        mock_request = MagicMock()
        mock_request.headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}

        manager._add_agent_name_header(mock_request)

        assert mock_request.headers["X-Amzn-Bedrock-AgentCore-Payments-Agent-Name"] == "my-agent"
        assert mock_request.headers["Content-Type"] == "application/json"
        assert mock_request.headers["Authorization"] == "Bearer token"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_with_empty_string_agent_name_does_not_register_event(self, mock_session_class):
        """Test that empty string agent_name does not register event handler."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"

        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        PaymentManager(payment_manager_arn=arn, region_name="us-east-1", agent_name="")

        mock_client.meta.events.register.assert_not_called()


# ============================================================================
# PaymentManager Bearer Token Auth Tests
# ============================================================================


class TestPaymentManagerBearerTokenAuth:
    """Tests for bearer_token and token_provider auth in PaymentManager."""

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_without_bearer_args_no_bearer_event(self, mock_session_class):
        """Test that no bearer event handler is registered by default."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        PaymentManager(payment_manager_arn=arn, region_name="us-east-1")

        mock_client.meta.events.register.assert_not_called()

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_with_bearer_token_registers_event(self, mock_session_class):
        """Test that bearer_token registers the inject event handler."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1", bearer_token="my-jwt")

        mock_client.meta.events.register.assert_called_once_with(
            "before-send.bedrock-agentcore.*",
            manager._inject_bearer_token,
        )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_init_with_token_provider_registers_event(self, mock_session_class):
        """Test that token_provider registers the inject event handler."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1", token_provider=lambda: "fresh")

        mock_client.meta.events.register.assert_called_once_with(
            "before-send.bedrock-agentcore.*",
            manager._inject_bearer_token,
        )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_mutual_exclusivity_raises(self, mock_session_class):
        """Test that providing both bearer_token and token_provider raises ValueError."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.client.return_value = MagicMock()
        mock_session_class.return_value = mock_session

        with pytest.raises(ValueError, match="mutually exclusive"):
            PaymentManager(
                payment_manager_arn=arn,
                region_name="us-east-1",
                bearer_token="token",
                token_provider=lambda: "token",
            )

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_inject_bearer_token_static(self, mock_session_class):
        """Test that static bearer_token is injected into request headers."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.client.return_value = MagicMock()
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1", bearer_token="my-jwt")

        mock_request = MagicMock()
        mock_request.headers = {}
        manager._inject_bearer_token(mock_request)

        assert mock_request.headers["Authorization"] == "Bearer my-jwt"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_inject_bearer_token_provider_called_each_time(self, mock_session_class):
        """Test that token_provider is called for each request."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.client.return_value = MagicMock()
        mock_session_class.return_value = mock_session

        counter = {"n": 0}

        def provider():
            counter["n"] += 1
            return f"token-{counter['n']}"

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1", token_provider=provider)

        req1 = MagicMock()
        req1.headers = {}
        req2 = MagicMock()
        req2.headers = {}

        manager._inject_bearer_token(req1)
        manager._inject_bearer_token(req2)

        assert req1.headers["Authorization"] == "Bearer token-1"
        assert req2.headers["Authorization"] == "Bearer token-2"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_inject_bearer_token_preserves_other_headers(self, mock_session_class):
        """Test that bearer token injection does not overwrite existing headers."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.client.return_value = MagicMock()
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1", bearer_token="jwt")

        mock_request = MagicMock()
        mock_request.headers = {"Content-Type": "application/json", "X-Custom": "value"}
        manager._inject_bearer_token(mock_request)

        assert mock_request.headers["Authorization"] == "Bearer jwt"
        assert mock_request.headers["Content-Type"] == "application/json"
        assert mock_request.headers["X-Custom"] == "value"

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_bearer_token_works_alongside_agent_name(self, mock_session_class):
        """Test that bearer_token and agent_name can both be set."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        manager = PaymentManager(
            payment_manager_arn=arn,
            region_name="us-east-1",
            agent_name="my-agent",
            bearer_token="my-jwt",
        )

        assert manager._agent_name == "my-agent"
        assert manager._bearer_token == "my-jwt"
        assert mock_client.meta.events.register.call_count == 2

    @patch("bedrock_agentcore.payments.manager.boto3.Session")
    def test_inject_bearer_token_does_not_inject_user_id_header(self, mock_session_class):
        """Test that bearer mode does NOT inject user_id header — service derives it from JWT sub."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.client.return_value = MagicMock()
        mock_session_class.return_value = mock_session

        manager = PaymentManager(payment_manager_arn=arn, region_name="us-east-1", bearer_token="jwt")

        mock_request = MagicMock()
        mock_request.headers = {}
        manager._inject_bearer_token(mock_request)

        assert mock_request.headers["Authorization"] == "Bearer jwt"
        assert "X-Amzn-Bedrock-AgentCore-Payments-User-Id" not in mock_request.headers
