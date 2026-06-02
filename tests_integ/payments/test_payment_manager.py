"""Integration tests for PaymentManager.

This module contains integration tests for the PaymentManager class, which provides
a high-level wrapper around PaymentClient for simplified payment operations.

SETUP INSTRUCTIONS:
===================

1. Set the following environment variables before running tests:

   # Required: AWS region
   export BEDROCK_TEST_REGION="us-west-2"

   # Required: Payment manager ARN (created via control plane)
   export TEST_PAYMENT_MANAGER_ARN="arn:aws:bedrock:us-west-2:123456789012:payment-manager/pm-123"

   # Required: Payment connector ID (created via control plane)
   export TEST_PAYMENT_CONNECTOR_ID="pc-123"

   # Optional: User ID for testing (default: test-user)
   export TEST_USER_ID="test-user"

2. Ensure AWS credentials are configured:
   - Via ~/.aws/credentials
   - Via environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
   - Via IAM role (if running on EC2/ECS)

3. Run the tests:
   pytest tests_integ/payment/test_payment_manager.py -v

4. To run specific test class:
   pytest tests_integ/payment/test_payment_manager.py::TestPaymentManagerWorkflow -v

5. To run with detailed output:
   pytest tests_integ/payment/test_payment_manager.py -vv -s

SERVICE SIDE VERIFICATION:
==========================

Monitor service logs to verify:
- Payment manager initialization
- Payment instrument creation/retrieval events
- Payment session creation/update events
- Payment processing events
- Error handling and validation
"""

import os
import uuid

import boto3
import pytest

from bedrock_agentcore.payments.manager import PaymentManager


@pytest.mark.integration
class TestPaymentManagerInitialization:
    """Tests for PaymentManager initialization with real AWS client."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        default_arn = "arn:aws:bedrock:us-west-2:123456789012:payment-manager/pm-test"
        cls.payment_manager_arn = os.environ.get("TEST_PAYMENT_MANAGER_ARN", default_arn)
        cls.payment_connector_id = os.environ.get("TEST_PAYMENT_CONNECTOR_ID", "pc-test")
        cls.user_id = os.environ.get("TEST_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")

    def test_manager_initialization_with_config(self):
        """Test PaymentManager initialization with payment_manager_arn."""
        manager = PaymentManager(payment_manager_arn=self.payment_manager_arn, region_name=self.region)

        assert manager._payment_manager_arn == self.payment_manager_arn
        assert manager.region_name == self.region
        assert manager._payment_client is not None

    def test_manager_initialization_with_region(self):
        """Test PaymentManager initialization with region_name parameter."""
        manager = PaymentManager(payment_manager_arn=self.payment_manager_arn, region_name=self.region)

        assert manager.region_name == self.region

    def test_manager_initialization_with_session(self):
        """Test PaymentManager initialization with boto3_session parameter."""
        session = boto3.Session(region_name=self.region)
        manager = PaymentManager(payment_manager_arn=self.payment_manager_arn, boto3_session=session)

        assert manager.region_name == self.region


@pytest.mark.integration
class TestPaymentManagerWorkflow:
    """Tests for complete PaymentManager workflows with real AWS service."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.payment_manager_arn = os.environ.get("TEST_PAYMENT_MANAGER_ARN")
        cls.payment_connector_id = os.environ.get("TEST_PAYMENT_CONNECTOR_ID")
        cls.user_id = os.environ.get("TEST_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")

        # Initialize PaymentManager if env vars are set
        if cls.payment_manager_arn:
            manager = PaymentManager(payment_manager_arn=cls.payment_manager_arn, region_name=cls.region)
            cls.manager = manager
        else:
            cls.manager = None

        # Store created resource IDs for cleanup
        cls.instrument_ids = []
        cls.session_ids = []

    @classmethod
    def teardown_class(cls):
        """Clean up test resources.

        Payment instruments and sessions do not currently support delete operations.
        Sessions expire naturally via expiryDuration. Instruments persist per-user.
        Once delete APIs are GA, add cleanup here using cls.session_ids and cls.instrument_ids.
        """
        pass

    @pytest.mark.skipif(
        not os.environ.get("TEST_PAYMENT_MANAGER_ARN"),
        reason="TEST_PAYMENT_MANAGER_ARN environment variable not set",
    )
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow: create instrument → create session → process payment."""
        # Step 1: Create payment instrument
        instrument_response = self.manager.create_payment_instrument(
            user_id=self.user_id,
            payment_connector_id=self.payment_connector_id,
            payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
            payment_instrument_details={"embeddedCryptoWallet": {"network": "ETHEREUM"}},
            client_token=str(uuid.uuid4()),
        )
        # Response is already unwrapped by PaymentManager
        assert "paymentInstrumentId" in instrument_response
        instrument_id = instrument_response["paymentInstrumentId"]
        self.__class__.instrument_ids.append(instrument_id)

        # Step 2: Create payment session
        session_response = self.manager.create_payment_session(
            user_id=self.user_id,
            expiry_time_in_minutes=60,
            limits={"maxSpendAmount": {"value": "100.00", "currency": "USD"}},
            client_token=str(uuid.uuid4()),
        )
        # Response is already unwrapped by PaymentManager
        assert "paymentSessionId" in session_response
        session_id = session_response["paymentSessionId"]
        self.__class__.session_ids.append(session_id)

        # Step 3: Process payment
        payment_input = {
            "cryptoX402": {
                "version": "1",
                "payload": {
                    "scheme": "exact",
                    "network": "base-sepolia",
                    "maxAmountRequired": "5000",
                    "resource": "https://nickeljoke.vercel.app/api/joke",
                    "description": "Premium AI joke generation",
                    "mimeType": "application/json",
                    "payTo": "0x6813749E1eB9E0001A44C2684695FE8AD676cdD0",
                    "maxTimeoutSeconds": 300,
                    "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
                    "outputSchema": {"input": {"type": "http", "method": "GET", "discoverable": True}},
                    "extra": {"name": "USDC", "version": "2"},
                },
            }
        }
        payment_response = self.manager.process_payment(
            user_id=self.user_id,
            payment_session_id=session_id,
            payment_instrument_id=instrument_id,
            payment_type="CRYPTO_X402",
            payment_input=payment_input,
            client_token=str(uuid.uuid4()),
        )
        # Verify response contains payment processing result
        assert payment_response is not None
        assert isinstance(payment_response, dict)

    @pytest.mark.skipif(
        not os.environ.get("TEST_PAYMENT_MANAGER_ARN"),
        reason="TEST_PAYMENT_MANAGER_ARN environment variable not set",
    )
    def test_idempotency_with_client_token(self):
        """Test idempotency with client_token."""
        client_token = str(uuid.uuid4())

        # First call with client_token
        response1 = self.manager.create_payment_instrument(
            user_id=self.user_id,
            payment_connector_id=self.payment_connector_id,
            payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
            payment_instrument_details={"embeddedCryptoWallet": {"network": "ETHEREUM"}},
            client_token=client_token,
        )
        # Response is already unwrapped by PaymentManager
        assert "paymentInstrumentId" in response1
        instrument_id_1 = response1["paymentInstrumentId"]
        self.__class__.instrument_ids.append(instrument_id_1)

        # Second call with same client_token should return same result (idempotent)
        response2 = self.manager.create_payment_instrument(
            user_id=self.user_id,
            payment_connector_id=self.payment_connector_id,
            payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
            payment_instrument_details={"embeddedCryptoWallet": {"network": "ETHEREUM"}},
            client_token=client_token,
        )
        instrument_id_2 = response2["paymentInstrumentId"]
        assert instrument_id_1 == instrument_id_2

    @pytest.mark.skipif(
        not os.environ.get("TEST_PAYMENT_MANAGER_ARN"),
        reason="TEST_PAYMENT_MANAGER_ARN environment variable not set",
    )
    def test_region_and_session_configuration(self):
        """Test region and session configuration."""
        session = boto3.Session(region_name=self.region)
        manager = PaymentManager(payment_manager_arn=self.payment_manager_arn, boto3_session=session)

        assert manager.region_name == self.region


@pytest.mark.integration
class TestPaymentManagerMethodForwarding:
    """Tests for PaymentManager method forwarding to real PaymentClient."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.payment_manager_arn = os.environ.get("TEST_PAYMENT_MANAGER_ARN")
        cls.payment_connector_id = os.environ.get("TEST_PAYMENT_CONNECTOR_ID")
        cls.user_id = os.environ.get("TEST_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")

        # Initialize PaymentManager if env vars are set
        if cls.payment_manager_arn:
            manager = PaymentManager(payment_manager_arn=cls.payment_manager_arn, region_name=cls.region)
            cls.manager = manager
        else:
            cls.manager = None

        # Store created resource IDs for cleanup
        cls.instrument_ids = []

    @pytest.mark.skipif(
        not os.environ.get("TEST_PAYMENT_MANAGER_ARN"),
        reason="TEST_PAYMENT_MANAGER_ARN environment variable not set",
    )
    def test_forwarding_to_payment_client(self):
        """Test method forwarding to PaymentClient."""
        # This tests that unmapped methods are properly forwarded
        # list_payment_instruments is wrapped by PaymentManager, so use it directly
        result = self.manager.list_payment_instruments(
            user_id=self.user_id,
            payment_connector_id=self.payment_connector_id,
        )

        # Should return a valid response from PaymentClient
        assert isinstance(result, dict)
        assert "paymentInstruments" in result

    @pytest.mark.skipif(
        not os.environ.get("TEST_PAYMENT_MANAGER_ARN"),
        reason="TEST_PAYMENT_MANAGER_ARN environment variable not set",
    )
    def test_forwarding_with_arguments(self):
        """Test method forwarding with arguments."""
        # Create an instrument first
        instrument_response = self.manager.create_payment_instrument(
            user_id=self.user_id,
            payment_connector_id=self.payment_connector_id,
            payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
            payment_instrument_details={"embeddedCryptoWallet": {"network": "ETHEREUM"}},
            client_token=str(uuid.uuid4()),
        )
        # Response is already unwrapped by PaymentManager
        instrument_id = instrument_response["paymentInstrumentId"]
        self.__class__.instrument_ids.append(instrument_id)

        # Now retrieve it using the wrapped method
        result = self.manager.get_payment_instrument(
            user_id=self.user_id,
            payment_connector_id=self.payment_connector_id,
            payment_instrument_id=instrument_id,
        )

        assert result["paymentInstrumentId"] == instrument_id


@pytest.mark.integration
class TestGeneratePaymentHeaderWorkflow:
    """Tests for generatePaymentHeader method with real AWS service."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.payment_manager_arn = os.environ.get("TEST_PAYMENT_MANAGER_ARN")
        cls.payment_connector_id = os.environ.get("TEST_PAYMENT_CONNECTOR_ID")
        cls.user_id = os.environ.get("TEST_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")

        # Initialize PaymentManager if env vars are set
        if cls.payment_manager_arn:
            manager = PaymentManager(payment_manager_arn=cls.payment_manager_arn, region_name=cls.region)
            cls.manager = manager
        else:
            cls.manager = None

        # Store created resource IDs for cleanup
        cls.instrument_ids = []
        cls.session_ids = []

    @classmethod
    def teardown_class(cls):
        """Clean up test resources."""
        # Note: In a real scenario, you might want to clean up created resources
        # However, payment instruments and sessions may have retention policies
        pass

    @staticmethod
    def create_v1_x402_response(scheme="exact", network="eip155:84532"):
        """Create a v1 X.402 response for testing (requirement structure without payload field)."""
        return {
            "statusCode": 402,
            "headers": {},
            "body": {
                "x402Version": 1,
                "error": "X-PAYMENT header is required",
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

    @staticmethod
    def create_v2_x402_response(scheme="exact", network="ethereum"):
        """Create a v2 X.402 response for testing (requirement structure without payload field)."""
        import base64
        import json

        v2_payload = {
            "x402Version": 2,
            "resource": {
                "url": "https://api.example.com/premium-data",
                "description": "Access to premium market data",
                "mimeType": "application/json",
            },
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
            "headers": {
                "payment-required": encoded_payload,
            },
            "body": {},
        }

    @pytest.mark.skipif(
        not os.environ.get("TEST_PAYMENT_MANAGER_ARN"),
        reason="TEST_PAYMENT_MANAGER_ARN environment variable not set",
    )
    def test_generate_payment_header_v1_workflow(self):
        """Test complete v1 workflow for generatePaymentHeader.

        Verifies:
        - Payment instrument created with Ethereum network
        - Payment session created successfully
        - generatePaymentHeader called with v1 X.402 response
        - Header starts with "X-PAYMENT:"
        - Header is base64 encoded
        - Decoded header contains required fields
        """
        # Step 1: Create payment instrument with Ethereum network
        instrument_response = self.manager.create_payment_instrument(
            user_id=self.user_id,
            payment_connector_id=self.payment_connector_id,
            payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
            payment_instrument_details={"embeddedCryptoWallet": {"network": "ETHEREUM"}},
            client_token=str(uuid.uuid4()),
        )
        instrument_id = instrument_response["paymentInstrumentId"]
        self.__class__.instrument_ids.append(instrument_id)

        # Step 2: Create payment session
        session_response = self.manager.create_payment_session(
            user_id=self.user_id,
            expiry_time_in_minutes=60,
            limits={"maxSpendAmount": {"value": "100.00", "currency": "USD"}},
            client_token=str(uuid.uuid4()),
        )
        session_id = session_response["paymentSessionId"]
        self.__class__.session_ids.append(session_id)

        # Step 3: Create v1 X.402 response with Base Mainnet network
        x402_response = self.create_v1_x402_response(
            scheme="exact",
            network="eip155:8453",
        )

        # Step 4: Call generatePaymentHeader
        header = self.manager.generate_payment_header(
            user_id=self.user_id,
            payment_instrument_id=instrument_id,
            payment_session_id=session_id,
            payment_required_request=x402_response,
        )

        # Step 5: Verify header format
        assert header is not None
        assert isinstance(header, dict)
        assert "X-PAYMENT" in header
        assert header["X-PAYMENT"] is not None
        assert header["X-PAYMENT"] != ""

        # Step 6: Verify header is base64 encoded
        import base64
        import json

        header_value = header["X-PAYMENT"]
        decoded_header = base64.b64decode(header_value).decode()
        header_json = json.loads(decoded_header)

        # Step 7: Verify decoded header contains required fields
        assert "x402Version" in header_json
        assert header_json["x402Version"] == 1
        assert "scheme" in header_json
        assert header_json["scheme"] is not None
        assert "network" in header_json
        assert header_json["network"] is not None
        assert "payload" in header_json
        assert header_json["payload"] is not None

        # Step 8: Verify scheme and network match the x402_response
        assert header_json["scheme"] == x402_response["body"]["accepts"][0]["scheme"]
        assert header_json["network"] == x402_response["body"]["accepts"][0]["network"]

    @pytest.mark.skipif(
        not os.environ.get("TEST_PAYMENT_MANAGER_ARN"),
        reason="TEST_PAYMENT_MANAGER_ARN environment variable not set",
    )
    def test_generate_payment_header_v2_workflow(self):
        """Test complete v2 workflow for generatePaymentHeader.

        Verifies:
        - Payment instrument created with Ethereum network
        - Payment session created successfully
        - generatePaymentHeader called with v2 X.402 response (base64-encoded header)
        - Header starts with "PAYMENT-SIGNATURE:"
        - Header is base64 encoded
        - Decoded header contains required fields
        """
        # Step 1: Create payment instrument with Ethereum network
        instrument_response = self.manager.create_payment_instrument(
            user_id=self.user_id,
            payment_connector_id=self.payment_connector_id,
            payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
            payment_instrument_details={"embeddedCryptoWallet": {"network": "ETHEREUM"}},
            client_token=str(uuid.uuid4()),
        )
        instrument_id = instrument_response["paymentInstrumentId"]
        self.__class__.instrument_ids.append(instrument_id)

        # Step 2: Create payment session
        session_response = self.manager.create_payment_session(
            user_id=self.user_id,
            expiry_time_in_minutes=60,
            limits={"maxSpendAmount": {"value": "100.00", "currency": "USD"}},
            client_token=str(uuid.uuid4()),
        )
        session_id = session_response["paymentSessionId"]
        self.__class__.session_ids.append(session_id)

        # Step 3: Create v2 X.402 response with Base Mainnet network
        x402_response = self.create_v2_x402_response(
            scheme="exact",
            network="eip155:8453",
        )

        # Step 4: Call generatePaymentHeader
        header = self.manager.generate_payment_header(
            user_id=self.user_id,
            payment_instrument_id=instrument_id,
            payment_session_id=session_id,
            payment_required_request=x402_response,
        )

        # Step 5: Verify header format
        assert header is not None
        assert isinstance(header, dict)
        assert "PAYMENT-SIGNATURE" in header
        assert header["PAYMENT-SIGNATURE"] is not None
        assert header["PAYMENT-SIGNATURE"] != ""

        # Step 6: Verify header is base64 encoded
        import base64
        import json

        header_value = header["PAYMENT-SIGNATURE"]
        decoded_header = base64.b64decode(header_value).decode()
        header_json = json.loads(decoded_header)

        # Step 7: Verify decoded header contains required fields
        assert "x402Version" in header_json
        assert header_json["x402Version"] == 2
        assert "resource" in header_json
        assert header_json["resource"] is not None
        assert "accepted" in header_json
        assert header_json["accepted"] is not None
        assert "payload" in header_json
        assert header_json["payload"] is not None
        assert "extension" in header_json

        # Step 8: Verify scheme and network match the x402_response
        import json as json_module

        v2_payload_from_response = json_module.loads(
            base64.b64decode(x402_response["headers"]["payment-required"]).decode()
        )
        assert header_json["accepted"]["scheme"] == v2_payload_from_response["accepts"][0]["scheme"]
        assert header_json["accepted"]["network"] == v2_payload_from_response["accepts"][0]["network"]

    @pytest.mark.skipif(
        not os.environ.get("TEST_PAYMENT_MANAGER_ARN"),
        reason="TEST_PAYMENT_MANAGER_ARN environment variable not set",
    )
    def test_generate_payment_header_with_client_token(self):
        """Test generatePaymentHeader with provided client_token.

        Verifies:
        - Payment instrument created
        - Payment session created
        - generatePaymentHeader called with explicit client_token
        - Call succeeds with provided client_token
        """
        # Step 1: Create payment instrument
        instrument_response = self.manager.create_payment_instrument(
            user_id=self.user_id,
            payment_connector_id=self.payment_connector_id,
            payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
            payment_instrument_details={"embeddedCryptoWallet": {"network": "ETHEREUM"}},
            client_token=str(uuid.uuid4()),
        )
        instrument_id = instrument_response["paymentInstrumentId"]
        self.__class__.instrument_ids.append(instrument_id)

        # Step 2: Create payment session
        session_response = self.manager.create_payment_session(
            user_id=self.user_id,
            expiry_time_in_minutes=60,
            limits={"maxSpendAmount": {"value": "100.00", "currency": "USD"}},
            client_token=str(uuid.uuid4()),
        )
        session_id = session_response["paymentSessionId"]
        self.__class__.session_ids.append(session_id)

        # Step 3: Create v1 X.402 response
        x402_response = self.create_v1_x402_response()

        # Step 4: Call generatePaymentHeader with explicit client_token
        client_token = str(uuid.uuid4())
        header = self.manager.generate_payment_header(
            user_id=self.user_id,
            payment_instrument_id=instrument_id,
            payment_session_id=session_id,
            payment_required_request=x402_response,
            client_token=client_token,
        )

        # Step 5: Verify the call succeeds
        assert header is not None
        assert isinstance(header, dict)
        assert "X-PAYMENT" in header
        assert header["X-PAYMENT"] is not None
        assert header["X-PAYMENT"] != ""

    @pytest.mark.skipif(
        not os.environ.get("TEST_PAYMENT_MANAGER_ARN"),
        reason="TEST_PAYMENT_MANAGER_ARN environment variable not set",
    )
    def test_generate_payment_header_invalid_status_code(self):
        """Test generatePaymentHeader with non-402 status code.

        Verifies:
        - Payment instrument created
        - Payment session created
        - generatePaymentHeader called with x402_response statusCode != 402
        - Returns empty string or raises error without processing
        """
        from bedrock_agentcore.payments.manager import PaymentError

        # Step 1: Create payment instrument
        instrument_response = self.manager.create_payment_instrument(
            user_id=self.user_id,
            payment_connector_id=self.payment_connector_id,
            payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
            payment_instrument_details={"embeddedCryptoWallet": {"network": "ETHEREUM"}},
            client_token=str(uuid.uuid4()),
        )
        instrument_id = instrument_response["paymentInstrumentId"]
        self.__class__.instrument_ids.append(instrument_id)

        # Step 2: Create payment session
        session_response = self.manager.create_payment_session(
            user_id=self.user_id,
            expiry_time_in_minutes=60,
            limits={"maxSpendAmount": {"value": "100.00", "currency": "USD"}},
            client_token=str(uuid.uuid4()),
        )
        session_id = session_response["paymentSessionId"]
        self.__class__.session_ids.append(session_id)

        # Step 3: Create X.402 response with invalid status code
        x402_response = self.create_v1_x402_response()
        x402_response["statusCode"] = 400  # Invalid status code

        # Step 4: Call generatePaymentHeader and expect error
        with pytest.raises(PaymentError) as exc_info:
            self.manager.generate_payment_header(
                user_id=self.user_id,
                payment_instrument_id=instrument_id,
                payment_session_id=session_id,
                payment_required_request=x402_response,
            )

        # Step 5: Verify error message includes context
        assert "402" in str(exc_info.value)
        assert "400" in str(exc_info.value)


@pytest.mark.integration
class TestPaymentManagerAgentNameHeader:
    """Integration tests for agent_name header propagation in PaymentManager.

    These tests verify that the X-Amzn-Bedrock-AgentCore-Payments-Agent-Name
    header is correctly registered and injected into data-plane API calls.
    """

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        default_arn = "arn:aws:bedrock:us-west-2:123456789012:payment-manager/pm-test"
        cls.payment_manager_arn = os.environ.get("TEST_PAYMENT_MANAGER_ARN", default_arn)

    def test_manager_initialization_with_agent_name(self):
        """Test PaymentManager stores agent_name and registers event handler."""
        manager = PaymentManager(
            payment_manager_arn=self.payment_manager_arn,
            region_name=self.region,
            agent_name="integ-test-agent",
        )

        assert manager._agent_name == "integ-test-agent"
        assert manager._payment_client is not None

    def test_manager_initialization_without_agent_name(self):
        """Test PaymentManager works without agent_name (backward compatible)."""
        manager = PaymentManager(
            payment_manager_arn=self.payment_manager_arn,
            region_name=self.region,
        )

        assert manager._agent_name is None
        assert manager._payment_client is not None

    def test_agent_name_header_injected_into_request(self):
        """Test that _add_agent_name_header injects the correct header value."""
        manager = PaymentManager(
            payment_manager_arn=self.payment_manager_arn,
            region_name=self.region,
            agent_name="my-payment-agent",
        )

        # Simulate a request object with a real dict for headers
        class MockRequest:
            def __init__(self):
                self.headers = {}

        request = MockRequest()
        manager._add_agent_name_header(request)

        assert "X-Amzn-Bedrock-AgentCore-Payments-Agent-Name" in request.headers
        assert request.headers["X-Amzn-Bedrock-AgentCore-Payments-Agent-Name"] == "my-payment-agent"

    def test_agent_name_header_does_not_overwrite_existing_headers(self):
        """Test that injecting agent_name header preserves other headers."""
        manager = PaymentManager(
            payment_manager_arn=self.payment_manager_arn,
            region_name=self.region,
            agent_name="my-agent",
        )

        class MockRequest:
            def __init__(self):
                self.headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer test-token",
                }

        request = MockRequest()
        manager._add_agent_name_header(request)

        assert request.headers["X-Amzn-Bedrock-AgentCore-Payments-Agent-Name"] == "my-agent"
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["Authorization"] == "Bearer test-token"

    def test_agent_name_with_boto3_session(self):
        """Test agent_name works alongside boto3_session parameter."""
        session = boto3.Session(region_name=self.region)
        manager = PaymentManager(
            payment_manager_arn=self.payment_manager_arn,
            boto3_session=session,
            agent_name="session-agent",
        )

        assert manager._agent_name == "session-agent"
        assert manager.region_name == self.region

    @pytest.mark.skipif(
        not os.environ.get("TEST_PAYMENT_MANAGER_ARN"),
        reason="TEST_PAYMENT_MANAGER_ARN environment variable not set",
    )
    def test_agent_name_header_in_real_api_call(self):
        """Test that agent_name header is sent in a real API call.

        This test makes a real list_payment_instruments call with agent_name set
        and verifies the call succeeds (the service accepts the header).
        """
        manager = PaymentManager(
            payment_manager_arn=self.payment_manager_arn,
            region_name=self.region,
            agent_name="integ-test-agent",
        )

        user_id = os.environ.get("TEST_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")

        # This call should succeed with the agent_name header injected
        result = manager.list_payment_instruments(user_id=user_id)

        assert isinstance(result, dict)
        assert "paymentInstruments" in result


# ============================================================================
# PaymentManager Bearer Token Auth Integration Tests
# ============================================================================


@pytest.mark.integration
class TestPaymentManagerBearerTokenAuth:
    """Integration tests for bearer token authentication in PaymentManager.

    These tests verify that bearer_token and token_provider correctly
    configure the PaymentManager for CUSTOM_JWT auth flows.

    The test class creates its own Cognito User Pool, Resource Server, App Client,
    and CUSTOM_JWT Payment Manager during setup, and tears them all down after.
    No external configuration required beyond AWS credentials and BEDROCK_TEST_REGION.
    """

    SCOPE_NAME = "payments-api/invoke"
    RESOURCE_SERVER_ID = "payments-api"

    @classmethod
    def setup_class(cls):
        """Create Cognito + CUSTOM_JWT Payment Manager infrastructure for testing."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-east-1")
        cls._cognito_client = boto3.client("cognito-idp", region_name=cls.region)
        cls._created_resources = {}

        try:
            cls._create_test_infrastructure()
        except Exception as e:
            cls._cleanup_resources()
            raise RuntimeError(f"Failed to create bearer token test infrastructure: {e}") from e

    @classmethod
    def _create_test_infrastructure(cls):
        """Create Cognito User Pool, Resource Server, App Client, and Payment Manager."""
        import json
        import time

        from bedrock_agentcore.payments.client import PaymentClient

        run_id = uuid.uuid4().hex[:8]

        # 1. Create Cognito User Pool
        pool_resp = cls._cognito_client.create_user_pool(
            PoolName=f"payments-integ-test-{run_id}",
            Policies={"PasswordPolicy": {"MinimumLength": 8}},
        )
        cls._created_resources["user_pool_id"] = pool_resp["UserPool"]["Id"]

        # 2. Create Resource Server (defines the OAuth scope)
        cls._cognito_client.create_resource_server(
            UserPoolId=cls._created_resources["user_pool_id"],
            Identifier=cls.RESOURCE_SERVER_ID,
            Name="Payments API",
            Scopes=[{"ScopeName": "invoke", "ScopeDescription": "Invoke payment operations"}],
        )

        # 3. Create App Client with client_credentials grant
        client_resp = cls._cognito_client.create_user_pool_client(
            UserPoolId=cls._created_resources["user_pool_id"],
            ClientName=f"payments-test-client-{run_id}",
            GenerateSecret=True,
            AllowedOAuthFlows=["client_credentials"],
            AllowedOAuthScopes=[f"{cls.RESOURCE_SERVER_ID}/invoke"],
            AllowedOAuthFlowsUserPoolClient=True,
        )
        cls.client_id = client_resp["UserPoolClient"]["ClientId"]
        cls.client_secret = client_resp["UserPoolClient"]["ClientSecret"]
        cls._created_resources["client_id"] = cls.client_id

        # 4. Create/get domain for token endpoint
        domain_prefix = f"payments-test-{run_id}"
        cls._cognito_client.create_user_pool_domain(
            Domain=domain_prefix,
            UserPoolId=cls._created_resources["user_pool_id"],
        )
        cls._created_resources["domain_prefix"] = domain_prefix
        cls.token_url = f"https://{domain_prefix}.auth.{cls.region}.amazoncognito.com/oauth2/token"
        cls.scope = f"{cls.RESOURCE_SERVER_ID}/invoke"

        # 5. Build the discovery URL (OIDC issuer) for CUSTOM_JWT authorizer
        discovery_url = (
            f"https://cognito-idp.{cls.region}.amazonaws.com/{cls._created_resources['user_pool_id']}"
            "/.well-known/openid-configuration"
        )

        # 6. Create CUSTOM_JWT Payment Manager via PaymentClient
        payment_client = PaymentClient(region_name=cls.region)

        # Create IAM role for the payment manager with required trust + permissions
        iam = boto3.client("iam", region_name=cls.region)
        sts = boto3.client("sts", region_name=cls.region)
        account_id = sts.get_caller_identity()["Account"]
        role_name = f"bearertest{run_id}Role"

        trust_policy = json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": [
                                "bedrock-agentcore.amazonaws.com",
                                "preprod.genesis-service.aws.internal",
                                "developer.genesis-service.aws.internal",
                            ]
                        },
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
        )

        try:
            role_resp = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=trust_policy,
                Description="Temp role for bearer token integ test",
            )
            role_arn = role_resp["Role"]["Arn"]
        except iam.exceptions.EntityAlreadyExistsException:
            role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"

        cls._created_resources["iam_role_name"] = role_name

        # Attach required permissions for resource retrieval
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="AllowResourceRetrieval",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": "bedrock-agentcore:*",
                            "Resource": [
                                f"arn:aws:bedrock-agentcore:*:{account_id}:token-vault/default",
                                f"arn:aws:bedrock-agentcore:*:{account_id}:token-vault/default/*",
                                f"arn:aws:bedrock-agentcore:*:{account_id}:workload-identity-directory/default",
                                f"arn:aws:bedrock-agentcore:*:{account_id}:workload-identity-directory/default/workload-identity/*",
                            ],
                        },
                        {
                            "Effect": "Allow",
                            "Action": "secretsmanager:GetSecretValue",
                            "Resource": f"arn:aws:secretsmanager:*:{account_id}:secret:*",
                        },
                        {
                            "Effect": "Allow",
                            "Action": "sts:SetContext",
                            "Resource": f"arn:aws:sts::{account_id}:self",
                        },
                    ],
                }
            ),
        )

        # Wait for Cognito domain + IAM role to propagate
        time.sleep(15)

        manager_resp = payment_client.create_payment_manager(
            name=f"bearertest{run_id}",
            role_arn=role_arn,
            authorizer_type="CUSTOM_JWT",
            authorizer_configuration={
                "customJWTAuthorizer": {
                    "discoveryUrl": discovery_url,
                    "allowedClients": [cls.client_id],
                }
            },
            description="Created for bearer token integ tests",
            wait_for_ready=True,
            max_wait=120,
        )
        cls.payment_manager_arn = manager_resp["paymentManagerArn"]
        cls._created_resources["payment_manager_id"] = manager_resp["paymentManagerId"]

    @classmethod
    def teardown_class(cls):
        """Tear down all created test infrastructure."""
        cls._cleanup_resources()

    @classmethod
    def _cleanup_resources(cls):
        """Best-effort cleanup of all created resources in reverse order."""

        from bedrock_agentcore.payments.client import PaymentClient

        # Delete payment manager
        if "payment_manager_id" in cls._created_resources:
            try:
                payment_client = PaymentClient(region_name=cls.region)
                payment_client.delete_payment_manager(cls._created_resources["payment_manager_id"])
            except Exception as e:
                print(f"Warning: failed to delete payment manager: {e}")

        # Delete Cognito domain
        if "domain_prefix" in cls._created_resources and "user_pool_id" in cls._created_resources:
            try:
                cls._cognito_client.delete_user_pool_domain(
                    Domain=cls._created_resources["domain_prefix"],
                    UserPoolId=cls._created_resources["user_pool_id"],
                )
            except Exception as e:
                print(f"Warning: failed to delete Cognito domain: {e}")

        # Delete Cognito User Pool (cascades resource server + app client)
        if "user_pool_id" in cls._created_resources:
            try:
                cls._cognito_client.delete_user_pool(
                    UserPoolId=cls._created_resources["user_pool_id"],
                )
            except Exception as e:
                print(f"Warning: failed to delete Cognito user pool: {e}")

        # Delete IAM role (must remove inline policy first)
        if "iam_role_name" in cls._created_resources:
            try:
                iam = boto3.client("iam", region_name=cls.region)
                iam.delete_role_policy(
                    RoleName=cls._created_resources["iam_role_name"],
                    PolicyName="AllowResourceRetrieval",
                )
                iam.delete_role(RoleName=cls._created_resources["iam_role_name"])
            except Exception as e:
                print(f"Warning: failed to delete IAM role: {e}")

    @staticmethod
    def _fetch_cognito_token(token_url: str, client_id: str, client_secret: str, scope: str) -> str:
        """Fetch a JWT from Cognito using client_credentials grant."""
        import base64
        import urllib.request

        credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        data = f"grant_type=client_credentials&scope={scope}".encode()
        req = urllib.request.Request(
            token_url,
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {credentials}",
            },
        )
        import json

        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())["access_token"]

    def test_manager_with_static_bearer_token(self):
        """Test PaymentManager initialization with static bearer_token."""
        manager = PaymentManager(
            payment_manager_arn=self.payment_manager_arn,
            region_name=self.region,
            bearer_token="test-static-token",
        )
        assert manager._bearer_token == "test-static-token"
        assert manager._token_provider is None

    def test_manager_with_token_provider(self):
        """Test PaymentManager initialization with token_provider callable."""
        call_count = {"n": 0}

        def provider():
            call_count["n"] += 1
            return f"token-{call_count['n']}"

        manager = PaymentManager(
            payment_manager_arn=self.payment_manager_arn,
            region_name=self.region,
            token_provider=provider,
        )
        assert manager._token_provider is not None
        assert manager._bearer_token is None

    def test_mutual_exclusivity(self):
        """Test that bearer_token and token_provider cannot both be set."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            PaymentManager(
                payment_manager_arn=self.payment_manager_arn,
                region_name=self.region,
                bearer_token="token",
                token_provider=lambda: "token",
            )

    def test_bearer_with_agent_name(self):
        """Test bearer_token works alongside agent_name."""
        manager = PaymentManager(
            payment_manager_arn=self.payment_manager_arn,
            region_name=self.region,
            bearer_token="my-jwt",
            agent_name="test-agent",
        )
        assert manager._bearer_token == "my-jwt"
        assert manager._agent_name == "test-agent"

    def test_bearer_does_not_inject_user_id_header(self):
        """Test that bearer mode does NOT inject user_id header — service derives it from JWT sub."""
        manager = PaymentManager(
            payment_manager_arn=self.payment_manager_arn,
            region_name=self.region,
            bearer_token="my-jwt",
        )

        class MockRequest:
            def __init__(self):
                self.headers = {}

        request = MockRequest()
        manager._inject_bearer_token(request)
        assert request.headers["Authorization"] == "Bearer my-jwt"
        assert "X-Amzn-Bedrock-AgentCore-Payments-User-Id" not in request.headers

    @staticmethod
    def _retry_with_backoff(fn, max_attempts=5, initial_delay=5):
        """Retry a callable with exponential backoff for propagation delays."""
        import time

        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(initial_delay * (2**attempt))

    def test_bearer_token_real_api_call(self):
        """Test a real API call using bearer token from Cognito.

        This test fetches a JWT from Cognito and uses it to call the payments
        data-plane API. Requires a CUSTOM_JWT payment manager and Cognito credentials.
        """
        token = self._fetch_cognito_token(self.token_url, self.client_id, self.client_secret, self.scope)

        manager = PaymentManager(
            payment_manager_arn=self.payment_manager_arn,
            region_name=self.region,
            bearer_token=token,
        )

        # Retry with backoff — OIDC discovery propagation may lag behind READY status
        result = self._retry_with_backoff(lambda: manager.list_payment_instruments(user_id="ignored"))

        assert isinstance(result, dict)
        assert "paymentInstruments" in result

    def test_token_provider_real_api_call(self):
        """Test a real API call using token_provider with Cognito.

        Verifies that the token_provider callable is invoked and the resulting
        token is used for authentication.
        """
        call_count = {"n": 0}

        def cognito_provider():
            call_count["n"] += 1
            return self._fetch_cognito_token(self.token_url, self.client_id, self.client_secret, self.scope)

        manager = PaymentManager(
            payment_manager_arn=self.payment_manager_arn,
            region_name=self.region,
            token_provider=cognito_provider,
        )

        result = self._retry_with_backoff(lambda: manager.list_payment_instruments(user_id="ignored"))

        assert isinstance(result, dict)
        assert "paymentInstruments" in result
        assert call_count["n"] >= 1

    def test_bearer_create_session_real_api_call(self):
        """Test creating a payment session using bearer token auth.

        Verifies that createPaymentSession succeeds with bearer auth and
        the service derives userId from the JWT sub claim.
        """
        token = self._fetch_cognito_token(self.token_url, self.client_id, self.client_secret, self.scope)

        manager = PaymentManager(
            payment_manager_arn=self.payment_manager_arn,
            region_name=self.region,
            bearer_token=token,
        )

        result = self._retry_with_backoff(
            lambda: manager.create_payment_session(
                user_id="ignored",
                expiry_time_in_minutes=15,
                limits={"maxSpendAmount": {"value": "1000", "currency": "USD"}},
            )
        )

        assert isinstance(result, dict)
        assert "paymentSessionId" in result
        assert result.get("userId") == self.client_id
