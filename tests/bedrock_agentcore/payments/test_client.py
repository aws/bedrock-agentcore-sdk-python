"""Unit tests for PaymentClient control plane operations."""

import os
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.payments import PaymentClient
from bedrock_agentcore.payments.client import PaymentConnectorConfig

# Get role ARN from environment variable, with fallback for testing
TEST_ROLE_ARN = os.getenv("TEST_PAYMENT_ROLE_ARN", "arn:aws:iam::123456789012:role/bedrock-payment-role")


class TestPaymentClientInitialization:
    """Tests for PaymentClient initialization."""

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_init_with_valid_region(self, mock_session, mock_boto3_client):
        """Test initialization with valid region_name."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        client = PaymentClient(region_name="us-east-1")

        assert client.region_name == "us-east-1"
        assert client.payments_cp_client == mock_cp_client

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_init_with_no_region_uses_session_region(self, mock_session, mock_boto3_client):
        """Test initialization with no region_name uses boto3 session region."""
        mock_session.return_value.region_name = "eu-west-1"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        client = PaymentClient()

        assert client.region_name == "eu-west-1"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_init_with_no_region_uses_fallback(self, mock_session, mock_boto3_client):
        """Test initialization with no region_name uses us-west-2 fallback."""
        mock_session.return_value.region_name = None
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        client = PaymentClient()

        assert client.region_name == "us-west-2"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_init_with_integration_source(self, mock_session, mock_boto3_client):
        """Test initialization with integration_source."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        client = PaymentClient(integration_source="my-app")

        assert client.integration_source == "my-app"

    @patch("bedrock_agentcore.payments.client.IdentityClient")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_init_creates_control_plane_client(self, mock_session, mock_boto3_client, mock_identity_client):
        """Test initialization creates control plane client."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        PaymentClient(region_name="us-west-2")

        # Verify control plane client was created (first call should be for bedrock-agentcore-control)
        assert mock_boto3_client.call_count >= 1
        first_call_args = mock_boto3_client.call_args_list[0]
        assert first_call_args[0][0] == "bedrock-agentcore-control"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_init_failure_raises_payment_error(self, mock_session, mock_boto3_client):
        """Test initialization failure raises Exception."""
        mock_session.return_value.region_name = "us-west-2"
        mock_boto3_client.side_effect = RuntimeError("Connection failed")

        with pytest.raises(RuntimeError, match="Connection failed"):
            PaymentClient(region_name="us-west-2")


class TestPaymentManagerOperations:
    """Tests for PaymentClient payment manager operations."""

    role_arn = "arn:aws:iam::123456789012:role/bedrock-payment-role"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_create_payment_manager_success(self, mock_session, mock_boto3_client):
        """Test successful payment manager creation."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        mock_cp_client.create_payment_manager.return_value = {
            "paymentManagerArn": "arn:aws:bedrock:us-west-2:123456789012:payment-manager/pm-123",
            "paymentManagerId": "pm-123",
            "status": "ACTIVE",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.create_payment_manager(
            name="test-manager",
            role_arn=self.role_arn,
        )

        assert result["paymentManagerId"] == "pm-123"
        assert result["status"] == "ACTIVE"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_get_payment_manager_success(self, mock_session, mock_boto3_client):
        """Test successful payment manager retrieval."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        mock_cp_client.get_payment_manager.return_value = {
            "paymentManagerId": "pm-123",
            "paymentManagerArn": "arn:aws:bedrock:us-west-2:123456789012:payment-manager/pm-123",
            "name": "test-manager",
            "status": "ACTIVE",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.get_payment_manager(payment_manager_id="pm-123")

        assert result["paymentManagerId"] == "pm-123"
        assert result["name"] == "test-manager"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_list_payment_managers_success(self, mock_session, mock_boto3_client):
        """Test successful payment managers listing."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        mock_cp_client.list_payment_managers.return_value = {
            "paymentManagers": [
                {
                    "paymentManagerId": "pm-123",
                    "paymentManagerArn": "arn:aws:bedrock:us-west-2:123456789012:payment-manager/pm-123",
                    "name": "test-manager",
                    "status": "ACTIVE",
                }
            ]
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.list_payment_managers()

        assert len(result["paymentManagers"]) == 1
        assert result["paymentManagers"][0]["paymentManagerId"] == "pm-123"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_update_payment_manager_success(self, mock_session, mock_boto3_client):
        """Test successful payment manager update."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        mock_cp_client.update_payment_manager.return_value = {
            "paymentManagerId": "pm-123",
            "paymentManagerArn": "arn:aws:bedrock:us-west-2:123456789012:payment-manager/pm-123",
            "name": "test-manager",
            "description": "Updated description",
            "status": "ACTIVE",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.update_payment_manager(
            payment_manager_id="pm-123",
            description="Updated description",
        )

        assert result["paymentManagerId"] == "pm-123"
        assert result["description"] == "Updated description"
        assert result["status"] == "ACTIVE"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_delete_payment_manager_success(self, mock_session, mock_boto3_client):
        """Test successful payment manager deletion."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        mock_cp_client.delete_payment_manager.return_value = {
            "status": "DELETED",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.delete_payment_manager(payment_manager_id="pm-123")

        assert result["paymentManagerId"] == "pm-123"


class TestPaymentConnectorOperations:
    """Tests for PaymentClient payment connector operations."""

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_create_payment_connector_success(self, mock_session, mock_boto3_client):
        """Test successful payment connector creation."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        mock_cp_client.create_payment_connector.return_value = {
            "paymentConnectorId": "pc-123",
            "status": "ACTIVE",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.create_payment_connector(
            payment_manager_id="pm-123",
            name="test-connector",
            connector_type="CoinbaseCDP",
            credential_provider_configurations=[
                {"coinbaseCDP": {"credentialProviderArn": "arn:aws:secretsmanager:us-west-2:123456789012:secret:test"}}
            ],
        )

        assert result["paymentConnectorId"] == "pc-123"
        assert result["status"] == "ACTIVE"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_get_payment_connector_success(self, mock_session, mock_boto3_client):
        """Test successful payment connector retrieval."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        mock_cp_client.get_payment_connector.return_value = {
            "paymentConnectorId": "pc-123",
            "paymentManagerId": "pm-123",
            "name": "test-connector",
            "type": "CoinbaseCDP",
            "status": "ACTIVE",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.get_payment_connector(
            payment_manager_id="pm-123",
            payment_connector_id="pc-123",
        )

        assert result["paymentConnectorId"] == "pc-123"
        assert result["name"] == "test-connector"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_list_payment_connectors_success(self, mock_session, mock_boto3_client):
        """Test successful payment connectors listing."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        mock_cp_client.list_payment_connectors.return_value = {
            "paymentConnectors": [
                {
                    "paymentConnectorId": "pc-123",
                    "paymentManagerId": "pm-123",
                    "name": "test-connector",
                    "type": "CoinbaseCDP",
                    "status": "ACTIVE",
                }
            ]
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.list_payment_connectors(payment_manager_id="pm-123")

        assert len(result["paymentConnectors"]) == 1
        assert result["paymentConnectors"][0]["paymentConnectorId"] == "pc-123"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_update_payment_connector_success(self, mock_session, mock_boto3_client):
        """Test successful payment connector update."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        mock_cp_client.update_payment_connector.return_value = {
            "paymentConnectorId": "pc-123",
            "paymentManagerId": "pm-123",
            "name": "updated-connector",
            "type": "CoinbaseCDP",
            "status": "ACTIVE",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.update_payment_connector(
            payment_manager_id="pm-123",
            payment_connector_id="pc-123",
            description="updated description",
        )

        assert result["name"] == "updated-connector"

    @patch("bedrock_agentcore.payments.client.boto3.Session")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    def test_delete_payment_connector_success(self, mock_boto3_client, mock_session):
        """Test successful payment connector deletion."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        mock_cp_client.delete_payment_connector.return_value = {
            "paymentConnectorId": "pc-123",
            "status": "DELETED",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.delete_payment_connector(
            payment_manager_id="pm-123",
            payment_connector_id="pc-123",
        )

        assert result["paymentConnectorId"] == "pc-123"


class TestErrorHandling:
    """Tests for PaymentClient error handling."""

    role_arn = "arn:aws:iam::123456789012:role/bedrock-payment-role"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_create_payment_manager_client_error(self, mock_session, mock_boto3_client):
        """Test error handling for payment manager creation."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Invalid role ARN"}},
            "CreatePaymentManager",
        )
        mock_cp_client.create_payment_manager.side_effect = error

        client = PaymentClient(region_name="us-west-2")

        with pytest.raises(ClientError):
            client.create_payment_manager(
                name="test-manager",
                role_arn=self.role_arn,
            )

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_getattr_forwards_to_cp_client(self, mock_session, mock_boto3_client):
        """Test __getattr__ forwards allowed methods to control plane client."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        # Mock the create_payment_manager method on the boto3 client
        mock_cp_client.create_payment_manager.return_value = {
            "paymentManagerArn": "arn:aws:bedrock:us-west-2:123456789012:payment-manager/pm-123",
            "paymentManagerId": "pm-123",
            "status": "ACTIVE",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.create_payment_manager(name="test", role_arn="arn")

        # Verify the method was called on the boto3 client
        mock_cp_client.create_payment_manager.assert_called_once()
        assert result["paymentManagerId"] == "pm-123"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_getattr_raises_for_disallowed_methods(self, mock_session, mock_boto3_client):
        """Test __getattr__ raises AttributeError for disallowed methods."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        client = PaymentClient(region_name="us-west-2")

        with pytest.raises(AttributeError):
            client.create_payment_instrument(user_id="test")


class TestCreatePaymentManagerWithConnector:
    """Tests for create_payment_manager_with_connector method."""

    @patch("bedrock_agentcore.payments.client.IdentityClient")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    def test_successful_creation_with_all_resources(self, mock_boto_client, mock_identity_class):
        """Test successful creation of payment manager, connector, and credential provider."""
        # Setup mocks
        mock_cp_client = MagicMock()
        mock_boto_client.return_value = mock_cp_client

        mock_identity_client = MagicMock()
        mock_identity_class.return_value = mock_identity_client

        provider_arn = (
            "arn:aws:acps:us-east-1:123456789012:token-vault/default/"
            "paymentcredentialprovider/ansraju-coinbase-provider-kdi94"
        )
        mock_identity_client.create_payment_credential_provider.return_value = {
            "credentialProviderArn": provider_arn,
            "name": "ansraju-coinbase-provider-kdi94",
            "credentialProviderVendor": "CoinbaseCDP",
        }

        mock_cp_client.create_payment_manager.return_value = {
            "paymentManagerArn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123",
            "paymentManagerId": "pm-123",
            "status": "READY",
        }

        mock_cp_client.create_payment_connector.return_value = {
            "paymentConnectorId": "connector-456",
            "status": "READY",
        }

        # Create client
        client = PaymentClient(region_name="us-east-1")

        payment_connector_config: PaymentConnectorConfig = {
            "name": "coinbase-connector",
            "description": "Coinbase CDP Connector",
            "payment_credential_provider_config": {
                "name": "ansraju-coinbase-provider-kdi94",
                "credential_provider_vendor": "CoinbaseCDP",
                "credentials": {
                    "api_key_id": "test-api-key-id",
                    "api_key_secret": "test-api-key-secret",
                    "wallet_secret": "test-wallet-secret",
                },
            },
        }

        response = client.create_payment_manager_with_connector(
            payment_manager_name="CDPPaymentManager",
            payment_manager_description="Coinbase Payment Manager",
            authorizer_type="AWS_IAM",
            role_arn="arn:aws:iam::123456789012:role/BedrockAgentCoreFullAccess",
            payment_connector_config=payment_connector_config,
        )

        # Verify response structure
        assert "paymentManager" in response
        assert "paymentConnector" in response
        assert "credentialProvider" in response

        assert (
            response["paymentManager"]["paymentManagerArn"]
            == "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123"
        )
        assert response["paymentManager"]["paymentManagerId"] == "pm-123"
        assert response["paymentManager"]["name"] == "CDPPaymentManager"
        assert response["paymentManager"]["status"] == "READY"

        assert response["paymentConnector"]["paymentConnectorId"] == "connector-456"
        assert response["paymentConnector"]["name"] == "coinbase-connector"
        assert response["paymentConnector"]["status"] == "READY"

        expected_provider_arn = (
            "arn:aws:acps:us-east-1:123456789012:token-vault/default/"
            "paymentcredentialprovider/ansraju-coinbase-provider-kdi94"
        )
        assert response["credentialProvider"]["credentialProviderArn"] == expected_provider_arn
        assert response["credentialProvider"]["name"] == "ansraju-coinbase-provider-kdi94"

        # Verify method calls
        mock_identity_client.create_payment_credential_provider.assert_called_once()
        mock_cp_client.create_payment_manager.assert_called_once()
        mock_cp_client.create_payment_connector.assert_called_once()

    @patch("bedrock_agentcore.payments.client.IdentityClient")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    def test_rollback_on_connector_creation_failure(self, mock_boto_client, mock_identity_class):
        """Test rollback when payment connector creation fails."""
        # Setup mocks
        mock_cp_client = MagicMock()
        mock_boto_client.return_value = mock_cp_client

        mock_identity_client = MagicMock()
        mock_identity_class.return_value = mock_identity_client

        test_provider_arn = (
            "arn:aws:acps:us-east-1:123456789012:token-vault/default/paymentcredentialprovider/test-provider"
        )
        mock_identity_client.create_payment_credential_provider.return_value = {
            "credentialProviderArn": test_provider_arn,
            "name": "test-provider",
        }

        mock_cp_client.create_payment_manager.return_value = {
            "paymentManagerArn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123",
            "paymentManagerId": "pm-123",
            "status": "READY",
        }

        # Simulate connector creation failure
        mock_cp_client.create_payment_connector.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Invalid connector configuration"}},
            "CreatePaymentConnector",
        )

        mock_cp_client.delete_payment_manager.return_value = {"status": "DELETED"}
        mock_identity_client.delete_payment_credential_provider.return_value = {}

        # Create client
        client = PaymentClient(region_name="us-east-1")

        connector_config: PaymentConnectorConfig = {
            "name": "coinbase-connector",
            "description": "Coinbase CDP Connector",
            "payment_credential_provider_config": {
                "name": "test-provider",
                "credential_provider_vendor": "CoinbaseCDP",
                "credentials": {
                    "api_key_id": "test-api-key-id",
                    "api_key_secret": "test-api-key-secret",
                    "wallet_secret": "test-wallet-secret",
                },
            },
        }

        # Verify exception is raised
        with pytest.raises(ClientError) as exc_info:
            client.create_payment_manager_with_connector(
                payment_manager_name="CDPPaymentManager",
                payment_manager_description="Coinbase Payment Manager",
                authorizer_type="AWS_IAM",
                role_arn="arn:aws:iam::123456789012:role/BedrockAgentCoreFullAccess",
                payment_connector_config=connector_config,
            )

        assert exc_info.value.response["Error"]["Code"] == "PaymentManagerCreationFailed"

        # Verify rollback was called
        mock_cp_client.delete_payment_manager.assert_called_once_with(paymentManagerId="pm-123")
        mock_identity_client.delete_payment_credential_provider.assert_called_once_with(name="test-provider")

    @patch("bedrock_agentcore.payments.client.IdentityClient")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    def test_rollback_on_manager_creation_failure(self, mock_boto_client, mock_identity_class):
        """Test rollback when payment manager creation fails."""
        # Setup mocks
        mock_cp_client = MagicMock()
        mock_boto_client.return_value = mock_cp_client

        mock_identity_client = MagicMock()
        mock_identity_class.return_value = mock_identity_client

        test_provider_arn = (
            "arn:aws:acps:us-east-1:123456789012:token-vault/default/paymentcredentialprovider/test-provider"
        )
        mock_identity_client.create_payment_credential_provider.return_value = {
            "credentialProviderArn": test_provider_arn,
            "name": "test-provider",
        }

        # Simulate manager creation failure
        mock_cp_client.create_payment_manager.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Invalid role ARN"}},
            "CreatePaymentManager",
        )

        mock_identity_client.delete_payment_credential_provider.return_value = {}

        # Create client
        client = PaymentClient(region_name="us-east-1")

        connector_config: PaymentConnectorConfig = {
            "name": "coinbase-connector",
            "description": "Coinbase CDP Connector",
            "payment_credential_provider_config": {
                "name": "test-provider",
                "credential_provider_vendor": "CoinbaseCDP",
                "credentials": {
                    "api_key_id": "test-api-key-id",
                    "api_key_secret": "test-api-key-secret",
                    "wallet_secret": "test-wallet-secret",
                },
            },
        }

        # Verify exception is raised
        with pytest.raises(ClientError) as exc_info:
            client.create_payment_manager_with_connector(
                payment_manager_name="CDPPaymentManager",
                payment_manager_description="Coinbase Payment Manager",
                authorizer_type="AWS_IAM",
                role_arn="arn:aws:iam::123456789012:role/BedrockAgentCoreFullAccess",
                payment_connector_config=connector_config,
            )

        # Verify exception details
        assert exc_info.value.response["Error"]["Code"] == "PaymentManagerCreationFailed"
        assert "Failed to create payment manager with connector" in exc_info.value.response["Error"]["Message"]
        assert "Invalid role ARN" in exc_info.value.response["Error"]["Message"]

        # Verify rollback was called (only credential provider should be deleted)
        mock_cp_client.delete_payment_manager.assert_not_called()
        mock_identity_client.delete_payment_credential_provider.assert_called_once_with(name="test-provider")

    @patch("bedrock_agentcore.payments.client.IdentityClient")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    def test_rollback_on_credential_provider_creation_failure(self, mock_boto_client, mock_identity_class):
        """Test rollback when credential provider creation fails."""
        # Setup mocks
        mock_cp_client = MagicMock()
        mock_boto_client.return_value = mock_cp_client

        mock_identity_client = MagicMock()
        mock_identity_class.return_value = mock_identity_client

        # Mock credential provider creation to fail with "already exists" error
        error_msg = "Credential provider with name: test-provider already exists"
        mock_identity_client.create_payment_credential_provider.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ConflictException",
                    "Message": error_msg,
                }
            },
            "CreatePaymentCredentialProvider",
        )

        # Create client
        client = PaymentClient(region_name="us-east-1")

        connector_config: PaymentConnectorConfig = {
            "name": "coinbase-connector",
            "description": "Coinbase CDP Connector",
            "payment_credential_provider_config": {
                "name": "test-provider",
                "credential_provider_vendor": "CoinbaseCDP",
                "credentials": {
                    "api_key_id": "test-api-key-id",
                    "api_key_secret": "test-api-key-secret",
                    "wallet_secret": "test-wallet-secret",
                },
            },
        }

        # Verify exception is raised
        with pytest.raises(ClientError) as exc_info:
            client.create_payment_manager_with_connector(
                payment_manager_name="CDPPaymentManager",
                payment_manager_description="Coinbase Payment Manager",
                authorizer_type="AWS_IAM",
                role_arn="arn:aws:iam::123456789012:role/BedrockAgentCoreFullAccess",
                payment_connector_config=connector_config,
            )

        # Verify exception details
        assert exc_info.value.response["Error"]["Code"] == "PaymentManagerCreationFailed"
        assert "Failed to create payment manager with connector" in exc_info.value.response["Error"]["Message"]
        assert "ConflictException" in exc_info.value.response["Error"]["Message"]

    @patch("bedrock_agentcore.payments.client.IdentityClient")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    def test_client_token_generation(self, mock_boto_client, mock_identity_class):
        """Test that client_token is generated if not provided."""
        # Setup mocks
        mock_cp_client = MagicMock()
        mock_boto_client.return_value = mock_cp_client

        mock_identity_client = MagicMock()
        mock_identity_class.return_value = mock_identity_client

        test_provider_arn = (
            "arn:aws:acps:us-east-1:123456789012:token-vault/default/paymentcredentialprovider/test-provider"
        )
        mock_identity_client.create_payment_credential_provider.return_value = {
            "credentialProviderArn": test_provider_arn,
            "name": "test-provider",
        }

        mock_cp_client.create_payment_manager.return_value = {
            "paymentManagerArn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123",
            "paymentManagerId": "pm-123",
            "status": "READY",
        }

        mock_cp_client.create_payment_connector.return_value = {
            "paymentConnectorId": "connector-456",
            "status": "READY",
        }

        # Create client
        client = PaymentClient(region_name="us-east-1")

        connector_config: PaymentConnectorConfig = {
            "name": "coinbase-connector",
            "description": "Coinbase CDP Connector",
            "payment_credential_provider_config": {
                "name": "test-provider",
                "credential_provider_vendor": "CoinbaseCDP",
                "credentials": {
                    "api_key_id": "test-api-key-id",
                    "api_key_secret": "test-api-key-secret",
                    "wallet_secret": "test-wallet-secret",
                },
            },
        }

        response = client.create_payment_manager_with_connector(
            payment_manager_name="CDPPaymentManager",
            payment_manager_description="Coinbase Payment Manager",
            authorizer_type="AWS_IAM",
            role_arn="arn:aws:iam::123456789012:role/BedrockAgentCoreFullAccess",
            payment_connector_config=connector_config,
        )

        # Verify response is successful
        assert response["paymentManager"]["paymentManagerId"] == "pm-123"

        # Verify client_token was passed to create methods
        manager_call_kwargs = mock_cp_client.create_payment_manager.call_args[1]
        assert "clientToken" in manager_call_kwargs
        assert manager_call_kwargs["clientToken"] is not None

        connector_call_kwargs = mock_cp_client.create_payment_connector.call_args[1]
        assert "clientToken" in connector_call_kwargs
        assert connector_call_kwargs["clientToken"] is not None

    @patch("bedrock_agentcore.payments.client.IdentityClient")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    def test_credential_provider_config_structure(self, mock_boto_client, mock_identity_class):
        """Test that credential provider config is correctly structured."""
        # Setup mocks
        mock_cp_client = MagicMock()
        mock_boto_client.return_value = mock_cp_client

        mock_identity_client = MagicMock()
        mock_identity_class.return_value = mock_identity_client

        test_provider_arn = (
            "arn:aws:acps:us-east-1:123456789012:token-vault/default/paymentcredentialprovider/test-provider"
        )
        mock_identity_client.create_payment_credential_provider.return_value = {
            "credentialProviderArn": test_provider_arn,
            "name": "test-provider",
        }

        mock_cp_client.create_payment_manager.return_value = {
            "paymentManagerArn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123",
            "paymentManagerId": "pm-123",
            "status": "READY",
        }

        mock_cp_client.create_payment_connector.return_value = {
            "paymentConnectorId": "connector-456",
            "status": "READY",
        }

        # Create client
        client = PaymentClient(region_name="us-east-1")

        payment_connector_config: PaymentConnectorConfig = {
            "name": "coinbase-connector",
            "description": "Coinbase CDP Connector",
            "payment_credential_provider_config": {
                "name": "test-provider",
                "credential_provider_vendor": "CoinbaseCDP",
                "credentials": {
                    "api_key_id": "test-api-key-id",
                    "api_key_secret": "test-api-key-secret",
                    "wallet_secret": "test-wallet-secret",
                },
            },
        }

        client.create_payment_manager_with_connector(
            payment_manager_name="CDPPaymentManager",
            payment_manager_description="Coinbase Payment Manager",
            authorizer_type="AWS_IAM",
            role_arn="arn:aws:iam::123456789012:role/BedrockAgentCoreFullAccess",
            payment_connector_config=payment_connector_config,
        )

        # Verify credential provider was called with correct structure
        call_args = mock_identity_client.create_payment_credential_provider.call_args
        assert call_args[1]["name"] == "test-provider"
        assert call_args[1]["credential_provider_vendor"] == "CoinbaseCDP"

        provider_config_input = call_args[1]["provider_configuration_input"]
        assert "coinbaseCdpConfiguration" in provider_config_input
        assert provider_config_input["coinbaseCdpConfiguration"]["apiKeyId"] == "test-api-key-id"
        assert provider_config_input["coinbaseCdpConfiguration"]["apiKeySecret"] == "test-api-key-secret"
        assert provider_config_input["coinbaseCdpConfiguration"]["walletSecret"] == "test-wallet-secret"

    @patch("bedrock_agentcore.payments.client.IdentityClient")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    def test_unsupported_vendor_raises_error(self, mock_boto_client, mock_identity_class):
        """Test that unsupported vendor type raises ValueError wrapped in ClientError."""
        # Setup mocks
        mock_cp_client = MagicMock()
        mock_boto_client.return_value = mock_cp_client

        mock_identity_client = MagicMock()
        mock_identity_class.return_value = mock_identity_client

        # Create client
        client = PaymentClient(region_name="us-east-1")

        payment_connector_config: PaymentConnectorConfig = {
            "name": "generic-connector",
            "description": "Generic Vendor Connector",
            "payment_credential_provider_config": {
                "name": "test-provider",
                "credential_provider_vendor": "GenericVendor",
                "credentials": {
                    "api_key_id": "test-api-key-id",
                    "api_key_secret": "test-api-key-secret",
                },
            },
        }

        with pytest.raises(ClientError) as exc_info:
            client.create_payment_manager_with_connector(
                payment_manager_name="GenericPaymentManager",
                payment_manager_description="Generic Payment Manager",
                authorizer_type="AWS_IAM",
                role_arn="arn:aws:iam::123456789012:role/BedrockAgentCoreFullAccess",
                payment_connector_config=payment_connector_config,
            )

        assert exc_info.value.response["Error"]["Code"] == "PaymentManagerCreationFailed"
        assert "Unsupported credential_provider_vendor" in exc_info.value.response["Error"]["Message"]
        assert "GenericVendor" in exc_info.value.response["Error"]["Message"]

    @patch("bedrock_agentcore.payments.client.IdentityClient")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    def test_stripe_privy_vendor_config_structure(self, mock_boto_client, mock_identity_class):
        """Test that StripePrivy vendor config uses stripePrivyConfiguration."""
        # Setup mocks
        mock_cp_client = MagicMock()
        mock_boto_client.return_value = mock_cp_client

        mock_identity_client = MagicMock()
        mock_identity_class.return_value = mock_identity_client

        test_provider_arn = (
            "arn:aws:acps:us-east-1:123456789012:token-vault/default/paymentcredentialprovider/test-provider"
        )
        mock_identity_client.create_payment_credential_provider.return_value = {
            "credentialProviderArn": test_provider_arn,
            "name": "test-provider",
            "credentialProviderVendor": "StripePrivy",
        }

        mock_cp_client.create_payment_manager.return_value = {
            "paymentManagerArn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123",
            "paymentManagerId": "pm-123",
            "status": "READY",
        }

        mock_cp_client.create_payment_connector.return_value = {
            "paymentConnectorId": "connector-456",
            "status": "READY",
        }

        # Create client
        client = PaymentClient(region_name="us-east-1")

        payment_connector_config: PaymentConnectorConfig = {
            "name": "stripe-privy-connector",
            "description": "Stripe + Privy Connector",
            "payment_credential_provider_config": {
                "name": "test-provider",
                "credential_provider_vendor": "StripePrivy",
                "credentials": {
                    "app_id": "test-app-id",
                    "app_secret": "test-app-secret",
                    "authorization_private_key": "test-auth-private-key",
                    "authorization_id": "test-auth-id",
                },
            },
        }

        response = client.create_payment_manager_with_connector(
            payment_manager_name="StripePaymentManager",
            payment_manager_description="Stripe + Privy Payment Manager",
            authorizer_type="AWS_IAM",
            role_arn="arn:aws:iam::123456789012:role/BedrockAgentCoreFullAccess",
            payment_connector_config=payment_connector_config,
        )

        # Verify response structure
        assert response["paymentManager"]["paymentManagerId"] == "pm-123"
        assert response["paymentConnector"]["paymentConnectorId"] == "connector-456"
        assert response["credentialProvider"]["credentialProviderArn"] == test_provider_arn
        assert response["credentialProvider"]["credentialProviderVendor"] == "StripePrivy"

        # Verify credential provider was called with correct structure
        call_args = mock_identity_client.create_payment_credential_provider.call_args
        assert call_args[1]["name"] == "test-provider"
        assert call_args[1]["credential_provider_vendor"] == "StripePrivy"

        provider_config_input = call_args[1]["provider_configuration_input"]
        assert "stripePrivyConfiguration" in provider_config_input
        assert provider_config_input["stripePrivyConfiguration"]["appId"] == "test-app-id"
        assert provider_config_input["stripePrivyConfiguration"]["appSecret"] == "test-app-secret"
        assert provider_config_input["stripePrivyConfiguration"]["authorizationPrivateKey"] == "test-auth-private-key"
        assert provider_config_input["stripePrivyConfiguration"]["authorizationId"] == "test-auth-id"

        # Verify connector was created with stripePrivy config key
        connector_call_args = mock_cp_client.create_payment_connector.call_args[1]
        assert connector_call_args["credentialProviderConfigurations"] == [
            {"stripePrivy": {"credentialProviderArn": test_provider_arn}}
        ]


class TestSafeErrorMessage:
    """Tests for PaymentClient._safe_error_message static method."""

    def test_client_error_message(self):
        """Test _safe_error_message with ClientError."""
        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Invalid input"}},
            "CreatePaymentManager",
        )
        result = PaymentClient._safe_error_message(error)
        assert result == "ValidationException: Invalid input"

    def test_value_error_message(self):
        """Test _safe_error_message with ValueError."""
        error = ValueError("Missing required field")
        result = PaymentClient._safe_error_message(error)
        assert result == "ValueError: Missing required field"

    def test_generic_error_message_redacted(self):
        """Test _safe_error_message redacts details for generic exceptions."""
        error = RuntimeError("sensitive internal details")
        result = PaymentClient._safe_error_message(error)
        assert result == "RuntimeError: (details redacted for security)"
        assert "sensitive" not in result


class TestBuildProviderConfigInput:
    """Tests for _build_provider_config_input static method."""

    def test_coinbase_cdp_config(self):
        """Test CoinbaseCDP vendor produces correct configuration."""
        config = {
            "credential_provider_vendor": "CoinbaseCDP",
            "credentials": {
                "api_key_id": "key-id",
                "api_key_secret": "key-secret",
                "wallet_secret": "wallet-secret",
            },
        }
        result = PaymentClient._build_provider_config_input(config)
        assert result == {
            "coinbaseCdpConfiguration": {
                "apiKeyId": "key-id",
                "apiKeySecret": "key-secret",
                "walletSecret": "wallet-secret",
            }
        }

    def test_stripe_privy_config(self):
        """Test StripePrivy vendor produces correct configuration."""
        config = {
            "credential_provider_vendor": "StripePrivy",
            "credentials": {
                "app_id": "app-id",
                "app_secret": "app-secret",
                "authorization_private_key": "auth-key",
                "authorization_id": "auth-id",
            },
        }
        result = PaymentClient._build_provider_config_input(config)
        assert result == {
            "stripePrivyConfiguration": {
                "appId": "app-id",
                "appSecret": "app-secret",
                "authorizationPrivateKey": "auth-key",
                "authorizationId": "auth-id",
            }
        }

    def test_unsupported_vendor_raises_error(self):
        """Test unsupported vendor raises ValueError."""
        config = {
            "credential_provider_vendor": "UnknownVendor",
            "credentials": {"some_key": "some_value"},
        }
        with pytest.raises(ValueError, match="Unsupported credential_provider_vendor"):
            PaymentClient._build_provider_config_input(config)

    def test_missing_vendor_raises_error(self):
        """Test missing vendor raises ValueError."""
        config = {"credentials": {"api_key_id": "key"}}
        with pytest.raises(ValueError, match="credential_provider_vendor is required"):
            PaymentClient._build_provider_config_input(config)

    def test_coinbase_missing_fields_raises_error(self):
        """Test CoinbaseCDP with missing fields raises ValueError."""
        config = {
            "credential_provider_vendor": "CoinbaseCDP",
            "credentials": {"api_key_id": "key-id"},
        }
        with pytest.raises(ValueError, match="Missing required CoinbaseCDP credential fields"):
            PaymentClient._build_provider_config_input(config)

    def test_stripe_privy_missing_fields_raises_error(self):
        """Test StripePrivy with missing fields raises ValueError."""
        config = {
            "credential_provider_vendor": "StripePrivy",
            "credentials": {"app_id": "app-id"},
        }
        with pytest.raises(ValueError, match="Missing required StripePrivy credential fields"):
            PaymentClient._build_provider_config_input(config)


class TestWaitForStatus:
    """Tests for PaymentClient._wait_for_status method."""

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_wait_for_status_immediate_success(self, mock_session, mock_boto3_client):
        """Test _wait_for_status when resource is already in target status."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        client = PaymentClient(region_name="us-west-2")

        mock_get = MagicMock(return_value={"status": "ACTIVE", "id": "resource-123"})
        result = client._wait_for_status(mock_get, "resource-123", "ACTIVE")
        assert result["status"] == "ACTIVE"
        mock_get.assert_called_once_with("resource-123")

    @patch("bedrock_agentcore.payments.client.time.sleep")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_wait_for_status_polls_until_ready(self, mock_session, mock_boto3_client, mock_sleep):
        """Test _wait_for_status polls until target status is reached."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        client = PaymentClient(region_name="us-west-2")

        mock_get = MagicMock(
            side_effect=[
                {"status": "CREATING", "id": "resource-123"},
                {"status": "CREATING", "id": "resource-123"},
                {"status": "ACTIVE", "id": "resource-123"},
            ]
        )
        result = client._wait_for_status(mock_get, "resource-123", "ACTIVE", poll_interval=1)
        assert result["status"] == "ACTIVE"
        assert mock_get.call_count == 3

    @patch("bedrock_agentcore.payments.client.time.sleep")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_wait_for_status_timeout(self, mock_session, mock_boto3_client, mock_sleep):
        """Test _wait_for_status raises TimeoutError when max_wait exceeded."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        client = PaymentClient(region_name="us-west-2")

        mock_get = MagicMock(return_value={"status": "CREATING"})
        with pytest.raises(TimeoutError, match="Timeout waiting for resource"):
            client._wait_for_status(mock_get, "resource-123", "ACTIVE", max_wait=0)

    @patch("bedrock_agentcore.payments.client.time.sleep")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_wait_for_status_failed_status(self, mock_session, mock_boto3_client, mock_sleep):
        """Test _wait_for_status raises ClientError on failed status."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        client = PaymentClient(region_name="us-west-2")

        mock_get = MagicMock(return_value={"status": "CREATE_FAILED"})
        with pytest.raises(ClientError):
            client._wait_for_status(mock_get, "resource-123", "ACTIVE")

    @patch("bedrock_agentcore.payments.client.time.sleep")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_wait_for_status_resource_not_found_retries(self, mock_session, mock_boto3_client, mock_sleep):
        """Test _wait_for_status retries on ResourceNotFoundException."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        client = PaymentClient(region_name="us-west-2")

        not_found_error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
            "GetResource",
        )
        mock_get = MagicMock(
            side_effect=[
                not_found_error,
                {"status": "ACTIVE", "id": "resource-123"},
            ]
        )
        result = client._wait_for_status(mock_get, "resource-123", "ACTIVE", poll_interval=1)
        assert result["status"] == "ACTIVE"
        assert mock_get.call_count == 2

    @patch("bedrock_agentcore.payments.client.time.sleep")
    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_wait_for_status_non_retryable_client_error(self, mock_session, mock_boto3_client, mock_sleep):
        """Test _wait_for_status raises non-retryable ClientError."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        client = PaymentClient(region_name="us-west-2")

        access_denied = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "GetResource",
        )
        mock_get = MagicMock(side_effect=access_denied)
        with pytest.raises(ClientError):
            client._wait_for_status(mock_get, "resource-123", "ACTIVE")

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_wait_for_status_with_extra_kwargs(self, mock_session, mock_boto3_client):
        """Test _wait_for_status passes extra kwargs to get_method."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client

        client = PaymentClient(region_name="us-west-2")

        mock_get = MagicMock(return_value={"status": "READY"})
        result = client._wait_for_status(mock_get, "resource-123", "READY", payment_manager_id="pm-456")
        assert result["status"] == "READY"
        mock_get.assert_called_once_with("resource-123", payment_manager_id="pm-456")


class TestPaymentManagerCRUDPaths:
    """Tests for PaymentClient CRUD method success and error paths."""

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_get_payment_manager_returns_formatted_response(self, mock_session, mock_boto3_client):
        """Test get_payment_manager formats the response correctly."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.get_payment_manager.return_value = {
            "paymentManagerId": "pm-123",
            "paymentManagerArn": "arn:aws:bedrock:us-west-2:123:payment-manager/pm-123",
            "name": "test-manager",
            "description": "desc",
            "status": "READY",
            "createdAt": "2025-01-01",
            "updatedAt": "2025-01-02",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.get_payment_manager("pm-123")

        assert result["paymentManagerId"] == "pm-123"
        assert result["name"] == "test-manager"
        assert result["status"] == "READY"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_get_payment_manager_error_raises(self, mock_session, mock_boto3_client):
        """Test get_payment_manager raises ClientError on failure."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.get_payment_manager.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
            "GetPaymentManager",
        )

        client = PaymentClient(region_name="us-west-2")
        with pytest.raises(ClientError):
            client.get_payment_manager("pm-nonexistent")

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_list_payment_managers_returns_formatted_response(self, mock_session, mock_boto3_client):
        """Test list_payment_managers formats the response correctly."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.list_payment_managers.return_value = {
            "paymentManagers": [
                {"paymentManagerId": "pm-1", "name": "mgr-1", "status": "READY"},
                {"paymentManagerId": "pm-2", "name": "mgr-2", "status": "READY"},
            ],
            "nextToken": "token-abc",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.list_payment_managers(max_results=10)

        assert len(result["paymentManagers"]) == 2
        assert result.get("nextToken") == "token-abc"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_list_payment_managers_with_next_token(self, mock_session, mock_boto3_client):
        """Test list_payment_managers with pagination token."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.list_payment_managers.return_value = {
            "paymentManagers": [],
        }

        client = PaymentClient(region_name="us-west-2")
        client.list_payment_managers(next_token="page-2")

        mock_cp_client.list_payment_managers.assert_called_once()
        call_kwargs = mock_cp_client.list_payment_managers.call_args[1]
        assert call_kwargs["nextToken"] == "page-2"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_list_payment_managers_error_raises(self, mock_session, mock_boto3_client):
        """Test list_payment_managers raises ClientError on failure."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.list_payment_managers.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Denied"}},
            "ListPaymentManagers",
        )

        client = PaymentClient(region_name="us-west-2")
        with pytest.raises(ClientError):
            client.list_payment_managers()

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_update_payment_manager_success(self, mock_session, mock_boto3_client):
        """Test update_payment_manager formats the response correctly."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.update_payment_manager.return_value = {
            "paymentManagerId": "pm-123",
            "paymentManagerArn": "arn:aws:bedrock:us-west-2:123:payment-manager/pm-123",
            "status": "READY",
            "lastUpdatedAt": "2025-01-02",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.update_payment_manager(
            payment_manager_id="pm-123",
            description="updated desc",
        )

        assert result["paymentManagerId"] == "pm-123"
        assert result["status"] == "READY"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_update_payment_manager_error_raises(self, mock_session, mock_boto3_client):
        """Test update_payment_manager raises ClientError on failure."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.update_payment_manager.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Invalid"}},
            "UpdatePaymentManager",
        )

        client = PaymentClient(region_name="us-west-2")
        with pytest.raises(ClientError):
            client.update_payment_manager(payment_manager_id="pm-123")

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_delete_payment_manager_success(self, mock_session, mock_boto3_client):
        """Test delete_payment_manager returns formatted response."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.delete_payment_manager.return_value = {
            "paymentManagerId": "pm-123",
            "status": "DELETING",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.delete_payment_manager("pm-123")

        assert result["paymentManagerId"] == "pm-123"
        assert result["status"] == "DELETING"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_delete_payment_manager_error_raises(self, mock_session, mock_boto3_client):
        """Test delete_payment_manager raises ClientError on failure."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.delete_payment_manager.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
            "DeletePaymentManager",
        )

        client = PaymentClient(region_name="us-west-2")
        with pytest.raises(ClientError):
            client.delete_payment_manager("pm-nonexistent")


class TestPaymentConnectorCRUDPaths:
    """Tests for PaymentClient connector CRUD method paths."""

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_create_payment_connector_returns_formatted_response(self, mock_session, mock_boto3_client):
        """Test create_payment_connector formats the response correctly."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.create_payment_connector.return_value = {
            "paymentConnectorId": "pc-123",
            "paymentManagerId": "pm-123",
            "name": "test-connector",
            "status": "CREATING",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.create_payment_connector(
            payment_manager_id="pm-123",
            name="test-connector",
            connector_type="BLOCKCHAIN",
            credential_provider_configurations=[{"coinbaseCDP": {"credentialProviderArn": "arn:test"}}],
        )

        assert result["paymentConnectorId"] == "pc-123"
        assert result["status"] == "CREATING"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_create_payment_connector_error_raises(self, mock_session, mock_boto3_client):
        """Test create_payment_connector raises ClientError on failure."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.create_payment_connector.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Invalid"}},
            "CreatePaymentConnector",
        )

        client = PaymentClient(region_name="us-west-2")
        with pytest.raises(ClientError):
            client.create_payment_connector(
                payment_manager_id="pm-123",
                name="test",
                connector_type="BLOCKCHAIN",
                credential_provider_configurations=[{"coinbaseCDP": {"credentialProviderArn": "arn:test"}}],
            )

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_get_payment_connector_returns_formatted_response(self, mock_session, mock_boto3_client):
        """Test get_payment_connector formats the response correctly."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.get_payment_connector.return_value = {
            "paymentConnectorId": "pc-123",
            "paymentManagerId": "pm-123",
            "name": "test-connector",
            "status": "READY",
            "type": "BLOCKCHAIN",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.get_payment_connector("pm-123", "pc-123")

        assert result["paymentConnectorId"] == "pc-123"
        assert result["status"] == "READY"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_get_payment_connector_error_raises(self, mock_session, mock_boto3_client):
        """Test get_payment_connector raises ClientError on failure."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.get_payment_connector.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
            "GetPaymentConnector",
        )

        client = PaymentClient(region_name="us-west-2")
        with pytest.raises(ClientError):
            client.get_payment_connector("pm-123", "pc-nonexistent")

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_list_payment_connectors_returns_formatted_response(self, mock_session, mock_boto3_client):
        """Test list_payment_connectors formats the response correctly."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.list_payment_connectors.return_value = {
            "paymentConnectors": [
                {"paymentConnectorId": "pc-1", "name": "conn-1"},
            ],
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.list_payment_connectors("pm-123")

        assert len(result["paymentConnectors"]) == 1

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_list_payment_connectors_with_pagination(self, mock_session, mock_boto3_client):
        """Test list_payment_connectors with pagination parameters."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.list_payment_connectors.return_value = {
            "paymentConnectors": [],
            "nextToken": "page-2",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.list_payment_connectors("pm-123", max_results=5, next_token="page-1")

        assert result.get("nextToken") == "page-2"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_list_payment_connectors_error_raises(self, mock_session, mock_boto3_client):
        """Test list_payment_connectors raises ClientError on failure."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.list_payment_connectors.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Denied"}},
            "ListPaymentConnectors",
        )

        client = PaymentClient(region_name="us-west-2")
        with pytest.raises(ClientError):
            client.list_payment_connectors("pm-123")

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_delete_payment_connector_success(self, mock_session, mock_boto3_client):
        """Test delete_payment_connector returns formatted response."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.delete_payment_connector.return_value = {
            "paymentConnectorId": "pc-123",
            "status": "DELETING",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.delete_payment_connector("pm-123", "pc-123")

        assert result["paymentConnectorId"] == "pc-123"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_delete_payment_connector_error_raises(self, mock_session, mock_boto3_client):
        """Test delete_payment_connector raises ClientError on failure."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.delete_payment_connector.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
            "DeletePaymentConnector",
        )

        client = PaymentClient(region_name="us-west-2")
        with pytest.raises(ClientError):
            client.delete_payment_connector("pm-123", "pc-nonexistent")

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_update_payment_connector_success(self, mock_session, mock_boto3_client):
        """Test update_payment_connector returns formatted response."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.update_payment_connector.return_value = {
            "paymentConnectorId": "pc-123",
            "paymentManagerId": "pm-123",
            "name": "updated-connector",
            "status": "READY",
        }

        client = PaymentClient(region_name="us-west-2")
        result = client.update_payment_connector(
            payment_manager_id="pm-123",
            payment_connector_id="pc-123",
            description="updated",
        )

        assert result["paymentConnectorId"] == "pc-123"

    @patch("bedrock_agentcore.payments.client.boto3.client")
    @patch("bedrock_agentcore.payments.client.boto3.Session")
    def test_update_payment_connector_error_raises(self, mock_session, mock_boto3_client):
        """Test update_payment_connector raises ClientError on failure."""
        mock_session.return_value.region_name = "us-west-2"
        mock_cp_client = MagicMock()
        mock_boto3_client.return_value = mock_cp_client
        mock_cp_client.update_payment_connector.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Invalid"}},
            "UpdatePaymentConnector",
        )

        client = PaymentClient(region_name="us-west-2")
        with pytest.raises(ClientError):
            client.update_payment_connector(
                payment_manager_id="pm-123",
                payment_connector_id="pc-123",
            )
