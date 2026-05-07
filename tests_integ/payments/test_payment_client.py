"""Integration tests for Payment Control Plane Client.

This module contains tests for the Bedrock AgentCore Payment control plane operations.

SETUP INSTRUCTIONS:
===================

1. Set the following environment variables before running tests:

   # Required: AWS region
   export BEDROCK_TEST_REGION="us-west-2"

   # For create_payment_manager_with_connector tests (optional):
   # Base64-encoded Ed25519 private key for API key secret
   export PAYMENT_TEST_API_KEY_SECRET="<base64-encoded-ed25519-key>"
   # Wallet secret (can be any valid format expected by the service)
   export PAYMENT_TEST_WALLET_SECRET="<wallet-secret>"

2. Ensure AWS credentials are configured:
   - Via ~/.aws/credentials
   - Via environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
   - Via IAM role (if running on EC2/ECS)

3. Run the tests:
   pytest tests_integ/payment/test_payment_client.py -v

4. To run specific test class:
   pytest tests_integ/payment/test_payment_client.py::TestPaymentClientControlPlane -v
   pytest tests_integ/payment/test_payment_client.py::TestCreatePaymentManagerWithConnectorIntegration -v

5. To run with detailed output:
   pytest tests_integ/payment/test_payment_client.py -vv -s

SERVICE SIDE VERIFICATION:
==========================

Monitor service logs to verify:
- Payment manager creation/retrieval events
- Payment connector creation/retrieval events
- Error handling and validation
"""

import os
import time
import uuid

import pytest

from bedrock_agentcore.payments import PaymentClient


@pytest.fixture(scope="function", autouse=True)
def cleanup_resources():
    """Fixture to ensure resources are cleaned up after each test."""
    yield
    # Cleanup happens after test execution


@pytest.mark.integration
class TestPaymentClientControlPlane:
    """Integration tests for PaymentClient control plane operations."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.role_arn = os.environ.get("RUNTIME_ROLE_ARN")
        if not cls.role_arn:
            pytest.skip("RUNTIME_ROLE_ARN must be set")
        cls.client = PaymentClient(region_name=cls.region)
        cls.test_prefix = f"t{int(time.time() * 1000000)}"
        cls.created_managers = []
        cls.created_connectors = []

    @classmethod
    def teardown_class(cls):
        """Clean up test resources."""
        # Clean up created connectors first
        for payment_manager_id, payment_connector_id in cls.created_connectors:
            try:
                cls.client.delete_payment_connector(
                    payment_manager_id=payment_manager_id,
                    payment_connector_id=payment_connector_id,
                )
            except Exception:
                pass

        # Clean up created managers
        for payment_manager_id in cls.created_managers:
            try:
                cls.client.delete_payment_manager(payment_manager_id=payment_manager_id)
            except Exception:
                pass

    def test_create_and_get_payment_manager(self):
        """Test creating and retrieving a payment manager."""
        manager_name = f"testManager{int(time.time())}"

        # Create a payment manager
        result = self.client.create_payment_manager(
            name=manager_name,
            role_arn=self.role_arn,
            description="Test payment manager",
        )

        payment_manager_id = result.get("paymentManagerId")
        assert payment_manager_id is not None
        self.__class__.created_managers.append(payment_manager_id)

        # Retrieve the manager
        retrieved = self.client.get_payment_manager(payment_manager_id=payment_manager_id)

        assert retrieved.get("paymentManagerId") == payment_manager_id
        assert retrieved.get("name") == manager_name

    def test_list_payment_managers(self):
        """Test listing payment managers."""
        result = self.client.list_payment_managers(max_results=10)

        assert isinstance(result, dict)
        assert "paymentManagers" in result
        managers = result.get("paymentManagers", [])
        assert isinstance(managers, list)

        # Verify structure of returned managers
        for manager in managers:
            assert "paymentManagerId" in manager
            assert "paymentManagerArn" in manager
            assert "name" in manager
            assert "status" in manager

    def test_update_payment_manager(self):
        """Test updating a payment manager."""
        manager_name = f"testManagerUpdate{int(time.time())}"

        # Create a payment manager
        create_result = self.client.create_payment_manager(
            name=manager_name,
            role_arn=self.role_arn,
        )

        payment_manager_id = create_result.get("paymentManagerId")
        self.__class__.created_managers.append(payment_manager_id)

        # Update the manager (only description can be updated)
        update_result = self.client.update_payment_manager(
            payment_manager_id=payment_manager_id,
            description="Updated description",
        )

        assert update_result.get("paymentManagerId") == payment_manager_id

    def test_create_and_get_payment_connector(self):
        """Test creating and retrieving a payment connector."""
        # First create a payment manager
        manager_name = f"testManagerConnector{int(time.time())}"

        manager_result = self.client.create_payment_manager(
            name=manager_name,
            role_arn=self.role_arn,
        )

        payment_manager_id = manager_result.get("paymentManagerId")
        self.__class__.created_managers.append(payment_manager_id)

        # Create a payment connector
        connector_name = f"testConnector{int(time.time())}"
        credential_provider_arn = "arn:aws:secretsmanager:us-west-2:123456789012:secret:test"

        connector_result = self.client.create_payment_connector(
            payment_manager_id=payment_manager_id,
            name=connector_name,
            connector_type="CoinbaseCDP",
            credential_provider_configurations=[{"coinbaseCDP": {"credentialProviderArn": credential_provider_arn}}],
            description="Test connector",
        )

        payment_connector_id = connector_result.get("paymentConnectorId")
        assert payment_connector_id is not None
        self.__class__.created_connectors.append((payment_manager_id, payment_connector_id))

        # Retrieve the connector
        retrieved = self.client.get_payment_connector(
            payment_manager_id=payment_manager_id,
            payment_connector_id=payment_connector_id,
        )

        assert retrieved.get("paymentConnectorId") == payment_connector_id
        assert retrieved.get("name") == connector_name

    def test_list_payment_connectors(self):
        """Test listing payment connectors."""
        # First create a payment manager
        manager_name = f"testManagerListConnectors{int(time.time())}"

        manager_result = self.client.create_payment_manager(
            name=manager_name,
            role_arn=self.role_arn,
        )

        payment_manager_id = manager_result.get("paymentManagerId")
        self.__class__.created_managers.append(payment_manager_id)

        # List connectors
        result = self.client.list_payment_connectors(
            payment_manager_id=payment_manager_id,
            max_results=10,
        )

        assert isinstance(result, dict)
        assert "paymentConnectors" in result
        connectors = result.get("paymentConnectors", [])
        assert isinstance(connectors, list)

        # Verify structure of returned connectors
        for connector in connectors:
            assert "paymentConnectorId" in connector
            assert "paymentManagerId" in connector
            assert "name" in connector
            assert "status" in connector

    def test_update_payment_connector(self):
        """Test updating a payment connector."""
        # First create a payment manager
        manager_name = f"testManagerUpdateConnector{int(time.time())}"

        manager_result = self.client.create_payment_manager(
            name=manager_name,
            role_arn=self.role_arn,
        )

        payment_manager_id = manager_result.get("paymentManagerId")
        self.__class__.created_managers.append(payment_manager_id)

        # Create a connector
        connector_name = f"testConnectorUpdate{int(time.time())}"
        credential_provider_arn = "arn:aws:secretsmanager:us-west-2:123456789012:secret:test"

        connector_result = self.client.create_payment_connector(
            payment_manager_id=payment_manager_id,
            name=connector_name,
            connector_type="CoinbaseCDP",
            credential_provider_configurations=[{"coinbaseCDP": {"credentialProviderArn": credential_provider_arn}}],
        )

        payment_connector_id = connector_result.get("paymentConnectorId")
        self.__class__.created_connectors.append((payment_manager_id, payment_connector_id))

        # Update the connector
        updated_description = "Updated connector description"
        update_result = self.client.update_payment_connector(
            payment_manager_id=payment_manager_id,
            payment_connector_id=payment_connector_id,
            description=updated_description,
        )

        assert update_result.get("paymentConnectorId") == payment_connector_id
        # Note: The API may not return the description in the update response
        # so we just verify the update succeeded


@pytest.mark.integration
class TestCreatePaymentManagerWithConnectorIntegration:
    """Integration tests for create_payment_manager_with_connector method."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.client = PaymentClient(region_name=cls.region)
        # Use timestamp with microseconds for uniqueness
        cls.test_prefix = f"t{int(time.time() * 1000000)}"
        cls.created_managers = []
        cls.created_connectors = []  # List of (payment_manager_id, payment_connector_id) tuples
        cls.created_providers = []  # List of provider names

        # Read credentials from environment variables
        cls.api_key_secret = os.environ.get("PAYMENT_TEST_API_KEY_SECRET")
        cls.wallet_secret = os.environ.get("PAYMENT_TEST_WALLET_SECRET")
        cls.role_arn = os.environ.get("RUNTIME_ROLE_ARN")
        cls.skip_tests = not (cls.api_key_secret and cls.wallet_secret and cls.role_arn)

    @classmethod
    def teardown_class(cls):
        """Clean up test resources."""
        # Clean up connectors first
        for manager_id, connector_id in cls.created_connectors:
            try:
                cls.client.delete_payment_connector(
                    payment_manager_id=manager_id,
                    payment_connector_id=connector_id,
                )
            except Exception as e:
                print(f"Failed to delete connector {connector_id}: {e}")

        # Clean up credential providers
        for provider_name in cls.created_providers:
            try:
                cls.client.identity_client.delete_payment_credential_provider(name=provider_name)
            except Exception as e:
                print(f"Failed to delete provider {provider_name}: {e}")

        # Clean up payment managers
        for manager_id in cls.created_managers:
            try:
                cls.client.delete_payment_manager(payment_manager_id=manager_id)
            except Exception as e:
                print(f"Failed to delete manager {manager_id}: {e}")

    def test_create_payment_manager_with_connector_success(self):
        """Test successful creation of payment manager with connector and credential provider."""
        if self.skip_tests:
            pytest.skip("PAYMENT_TEST_API_KEY_SECRET and PAYMENT_TEST_WALLET_SECRET environment variables not set")

        manager_name = f"{self.test_prefix}MgrWC"
        connector_name = f"{self.test_prefix}ConnWC"
        # Use UUID for provider name to ensure uniqueness - include full UUID
        provider_name = f"prov-{uuid.uuid4()}"

        payment_connector_config = {
            "name": connector_name,
            "description": "Test connector for integration",
            "payment_credential_provider_config": {
                "name": provider_name,
                "credential_provider_vendor": "CoinbaseCDP",
                "credentials": {
                    "api_key_id": "test-api-key-id",
                    "api_key_secret": self.api_key_secret,
                    "wallet_secret": self.wallet_secret,
                },
            },
        }

        # Create payment manager with connector
        response = self.client.create_payment_manager_with_connector(
            payment_manager_name=manager_name,
            payment_manager_description="Test payment manager with connector",
            authorizer_type="AWS_IAM",
            role_arn=self.role_arn,
            payment_connector_config=payment_connector_config,
        )

        # Verify response structure
        assert "paymentManager" in response
        assert "paymentConnector" in response
        assert "credentialProvider" in response

        # Verify payment manager details
        payment_manager = response["paymentManager"]
        assert payment_manager["name"] == manager_name
        assert "paymentManagerId" in payment_manager
        assert "paymentManagerArn" in payment_manager
        assert "status" in payment_manager

        manager_id = payment_manager["paymentManagerId"]
        self.__class__.created_managers.append(manager_id)

        # Verify payment connector details
        payment_connector = response["paymentConnector"]
        assert payment_connector["name"] == connector_name
        assert "paymentConnectorId" in payment_connector
        assert payment_connector["paymentManagerId"] == manager_id
        assert "status" in payment_connector

        connector_id = payment_connector["paymentConnectorId"]
        self.__class__.created_connectors.append((manager_id, connector_id))

        # Verify credential provider details
        credential_provider = response["credentialProvider"]
        assert credential_provider["name"] == provider_name
        assert credential_provider["credentialProviderVendor"] == "CoinbaseCDP"
        assert "credentialProviderArn" in credential_provider

        self.__class__.created_providers.append(provider_name)

    def test_create_payment_manager_with_connector_with_wait_for_ready(self):
        """Test creation with wait_for_ready to ensure resources reach READY status."""
        if self.skip_tests:
            pytest.skip("PAYMENT_TEST_API_KEY_SECRET and PAYMENT_TEST_WALLET_SECRET environment variables not set")

        manager_name = f"{self.test_prefix}MgrWR"
        connector_name = f"{self.test_prefix}ConnWR"
        # Use UUID for provider name to ensure uniqueness - include full UUID
        provider_name = f"prov-{uuid.uuid4()}"

        payment_connector_config = {
            "name": connector_name,
            "description": "Test connector with wait for ready",
            "payment_credential_provider_config": {
                "name": provider_name,
                "credential_provider_vendor": "CoinbaseCDP",
                "credentials": {
                    "api_key_id": "test-api-key-id",
                    "api_key_secret": self.api_key_secret,
                    "wallet_secret": self.wallet_secret,
                },
            },
        }

        # Create payment manager with connector and wait for ready
        response = self.client.create_payment_manager_with_connector(
            payment_manager_name=manager_name,
            payment_manager_description="Test payment manager with wait for ready",
            authorizer_type="AWS_IAM",
            role_arn=self.role_arn,
            payment_connector_config=payment_connector_config,
            wait_for_ready=True,
            max_wait=300,
            poll_interval=10,
        )

        # Verify resources reached READY status
        payment_manager = response["paymentManager"]
        assert payment_manager["status"] == "READY"

        payment_connector = response["paymentConnector"]
        assert payment_connector["status"] == "READY"

        manager_id = payment_manager["paymentManagerId"]
        connector_id = payment_connector["paymentConnectorId"]
        self.__class__.created_managers.append(manager_id)
        self.__class__.created_connectors.append((manager_id, connector_id))
        self.__class__.created_providers.append(provider_name)

    def test_create_payment_manager_with_connector_minimal_description(self):
        """Test creation with minimal/empty descriptions."""
        if self.skip_tests:
            pytest.skip("PAYMENT_TEST_API_KEY_SECRET and PAYMENT_TEST_WALLET_SECRET environment variables not set")

        manager_name = f"{self.test_prefix}MgrMin"
        connector_name = f"{self.test_prefix}ConnMin"
        # Use UUID for provider name to ensure uniqueness - include full UUID
        provider_name = f"prov-{uuid.uuid4()}"

        connector_config = {
            "name": connector_name,
            "description": "",  # Empty description
            "payment_credential_provider_config": {
                "name": provider_name,
                "credential_provider_vendor": "CoinbaseCDP",
                "credentials": {
                    "api_key_id": "test-api-key-id",
                    "api_key_secret": self.api_key_secret,
                    "wallet_secret": self.wallet_secret,
                },
            },
        }

        # Create with minimal descriptions
        response = self.client.create_payment_manager_with_connector(
            payment_manager_name=manager_name,
            payment_manager_description="",  # Empty description
            authorizer_type="AWS_IAM",
            role_arn=self.role_arn,
            payment_connector_config=connector_config,
        )

        # Verify creation succeeded
        assert "paymentManager" in response
        assert "paymentConnector" in response
        assert "credentialProvider" in response

        manager_id = response["paymentManager"]["paymentManagerId"]
        connector_id = response["paymentConnector"]["paymentConnectorId"]
        self.__class__.created_managers.append(manager_id)
        self.__class__.created_connectors.append((manager_id, connector_id))
        self.__class__.created_providers.append(provider_name)

        # Verify we can retrieve the created resources
        manager = self.client.get_payment_manager(payment_manager_id=manager_id)
        assert manager["paymentManagerId"] == manager_id
        assert manager["name"] == manager_name

    def test_create_payment_manager_with_connector_retrieve_resources(self):
        """Test that created resources can be retrieved individually."""
        if self.skip_tests:
            pytest.skip("PAYMENT_TEST_API_KEY_SECRET and PAYMENT_TEST_WALLET_SECRET environment variables not set")

        manager_name = f"{self.test_prefix}MgrRet"
        connector_name = f"{self.test_prefix}ConnRet"
        # Use UUID for provider name to ensure uniqueness - include full UUID
        provider_name = f"prov-{uuid.uuid4()}"

        connector_config = {
            "name": connector_name,
            "description": "Test connector for retrieval",
            "payment_credential_provider_config": {
                "name": provider_name,
                "credential_provider_vendor": "CoinbaseCDP",
                "credentials": {
                    "api_key_id": "test-api-key-id",
                    "api_key_secret": self.api_key_secret,
                    "wallet_secret": self.wallet_secret,
                },
            },
        }

        # Create payment manager with connector
        response = self.client.create_payment_manager_with_connector(
            payment_manager_name=manager_name,
            payment_manager_description="Test retrieval",
            authorizer_type="AWS_IAM",
            role_arn=self.role_arn,
            payment_connector_config=connector_config,
        )

        manager_id = response["paymentManager"]["paymentManagerId"]
        connector_id = response["paymentConnector"]["paymentConnectorId"]
        self.__class__.created_managers.append(manager_id)
        self.__class__.created_connectors.append((manager_id, connector_id))
        self.__class__.created_providers.append(provider_name)

        # Retrieve payment manager
        retrieved_manager = self.client.get_payment_manager(payment_manager_id=manager_id)
        assert retrieved_manager["paymentManagerId"] == manager_id
        assert retrieved_manager["name"] == manager_name

        # Retrieve payment connector
        retrieved_connector = self.client.get_payment_connector(
            payment_manager_id=manager_id,
            payment_connector_id=connector_id,
        )
        assert retrieved_connector["paymentConnectorId"] == connector_id
        assert retrieved_connector["name"] == connector_name

        # List connectors for the manager
        connectors_list = self.client.list_payment_connectors(payment_manager_id=manager_id)
        assert "paymentConnectors" in connectors_list
        connector_ids = [c["paymentConnectorId"] for c in connectors_list["paymentConnectors"]]
        assert connector_id in connector_ids
