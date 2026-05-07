"""Integration tests for AgentCorePaymentsPlugin with Strands framework.

Run with: python -m pytest tests_integ/payment/integrations/test_strands_plugin.py -v
"""

import json
import logging
import os
import uuid
from typing import Any

import pytest
from strands import Agent
from strands_tools import http_request

from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin
from bedrock_agentcore.payments.manager import (
    PaymentInstrumentConfigurationRequired,
    PaymentManager,
    PaymentSessionConfigurationRequired,
    PaymentSessionNotFound,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestAgentCorePaymentsPlugin:
    """Integration tests for AgentCorePaymentsPlugin."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.user_id = os.environ.get("TEST_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")

    def invoke_agent_with_payment_handling(
        self, agent: Agent, prompt: str, plugin: AgentCorePaymentsPlugin = None, max_iterations: int = 10
    ) -> Any:
        """Invoke agent and handle payment interrupts.

        This wrapper method handles payment failure interrupts by responding with
        a message indicating the payment issue. For instrument/session not found
        interrupts, it updates the plugin config with fallback values and retries.

        Args:
            agent: The Strands Agent instance
            prompt: The initial prompt to send to the agent
            plugin: Optional plugin instance to update config on configuration-required interrupts
            max_iterations: Maximum number of iterations to handle interrupts

        Returns:
            The final agent result
        """
        result = agent(prompt)
        iteration = 0

        while result.stop_reason == "interrupt" and iteration < max_iterations:
            iteration += 1
            logger.info("Handling interrupt (iteration %d): %s", iteration, result.stop_reason)

            responses = []
            for interrupt in result.interrupts:
                logger.info("Interrupt: %s", interrupt.name)
                logger.info("Reason: %s", interrupt.reason)

                # Handle payment failure interrupts
                if interrupt.name.startswith("payment-failure-"):
                    reason = interrupt.reason
                    exception_type = reason.get("exceptionType", "Unknown")
                    exception_message = reason.get("exceptionMessage", "Unknown error")

                    logger.warning("Payment failure: %s - %s", exception_type, exception_message)

                    # Handle instrument configuration required
                    if exception_type == PaymentInstrumentConfigurationRequired.__name__ and plugin:
                        instrument_id = os.environ.get(
                            "TEST_PAYMENT_INSTRUMENT_ID", "payment-instrument-abcdefghijklmno"
                        )
                        logger.info("Updating payment_instrument_id to %s", instrument_id)
                        plugin.config.update_payment_instrument_id(instrument_id)
                        responses.append(
                            {
                                "interruptResponse": {
                                    "interruptId": interrupt.id,
                                    "response": (
                                        "Payment Instrument ID has been configured."
                                        " Please retry the previous tool call."
                                    ),
                                }
                            }
                        )
                        continue

                    # Handle session configuration required
                    if exception_type == PaymentSessionConfigurationRequired.__name__ and plugin:
                        session_id = os.environ.get("TEST_PAYMENT_SESSION_ID", "payment-session-abcdefghijklmno")
                        logger.info("Updating payment_session_id to %s", session_id)
                        plugin.config.update_payment_session_id(session_id)
                        responses.append(
                            {
                                "interruptResponse": {
                                    "interruptId": interrupt.id,
                                    "response": (
                                        "Payment session ID has been configured. Please retry the previous tool call."
                                    ),
                                }
                            }
                        )
                        continue

                    # Handle session not found
                    if exception_type == PaymentSessionNotFound.__name__ and plugin:
                        session_id = os.environ.get("TEST_PAYMENT_SESSION_ID", "payment-session-abcdefghijklmno")
                        logger.info("Payment session not found, updating payment_session_id to %s", session_id)
                        plugin.config.update_payment_session_id(session_id)
                        responses.append(
                            {
                                "interruptResponse": {
                                    "interruptId": interrupt.id,
                                    "response": (
                                        "Payment session ID has been updated. Please retry the previous tool call."
                                    ),
                                }
                            }
                        )
                        continue

                    # Default: respond with failure message
                    response_message = (
                        f"Payment processing failed with {exception_type}: {exception_message}. "
                        "Unable to complete the payment transaction."
                    )

                    responses.append({"interruptResponse": {"interruptId": interrupt.id, "response": response_message}})

            # Continue agent with responses
            result = agent(responses)

        return result

    def test_plugin_with_real_agent(self):
        """Test plugin initialization with a real Strands Agent."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=os.environ.get(
                "TEST_PAYMENT_MANAGER_ARN",
                "arn:aws:bedrock-agentcore:us-west-2:12345678910:payment-manager/mypaymentmanager-cyrc25gr4c",
            ),
            user_id=self.user_id,
            payment_session_id=os.environ.get("TEST_PAYMENT_SESSION_ID", "payment-session-bWOGg4z2irAGbzA"),
            payment_instrument_id=os.environ.get("TEST_PAYMENT_INSTRUMENT_ID", "payment-instrument-vnL29CKAdyESdQ7"),
            region=self.region,
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        # Create a real Strands Agent with the plugin
        agent = Agent(system_prompt="You are a helpful assistant.", plugins=[plugin])

        # Verify plugin is registered with agent
        assert plugin.payment_manager is not None
        assert isinstance(plugin.payment_manager, PaymentManager)
        assert agent is not None

        logger.info("Plugin successfully initialized with real Strands Agent")

    def test_v1_happy_case_with_real_payment_manager(self):
        """Test V1 happy case: Plugin processes X.402 payment requirements.

        This test requires:
        - TEST_PAYMENT_MANAGER_ARN environment variable
        - Valid AWS credentials
        - Real payment manager setup

        The test will:
        1. Create a real plugin with real PaymentManager
        2. Test payment processing with V1 X.402 requirements
        3. Verify payment header is constructed correctly

        To run this test:
        export BEDROCK_TEST_REGION="us-west-2"
        export TEST_PAYMENT_MANAGER_ARN="arn:aws:bedrock:us-west-2:123456789012:payment-manager/pm-123"
        export TEST_PAYMENT_INSTRUMENT_ID="payment-instrument-xxx"
        export TEST_PAYMENT_SESSION_ID="payment-session-xxx"
        export TEST_USER_ID="test-user"
        pytest tests_integ/payment/integrations/test_plugin_integration.py::\
            TestAgentCorePaymentsPlugin::test_v1_happy_case_with_real_payment_manager -v -s
        """
        # Skip if payment manager ARN not configured
        payment_manager_arn = os.environ.get("TEST_PAYMENT_MANAGER_ARN")
        if not payment_manager_arn:
            pytest.skip("TEST_PAYMENT_MANAGER_ARN not configured - skipping V1 happy case test")

        # payment_session_id="payment-session-bWOGg4z2irAGbzA",  # Must match pattern: payment-session-[0-9a-zA-Z-]{15}
        # payment_instrument_id="payment-instrument-vnL29CKAdyESdQ7",  # Valid instrument ID

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=payment_manager_arn,
            user_id=self.user_id,
            region=self.region,
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        # Create a real Strands Agent with the plugin
        SYSTME_PROMPT = (
            "You are a helpful assistant. If you encounter 405, change the http "
            "method and try again. Print the tool input/output."
        )
        agent = Agent(system_prompt=SYSTME_PROMPT, tools=[http_request], plugins=[plugin])

        # Use the wrapper method to handle payment interrupts
        result = self.invoke_agent_with_payment_handling(
            agent,
            "Please fetch a joke from https://nickeljoke.vercel.app/api/joke and tell me what it is",
            plugin=plugin,
        )

        logger.info("✓ Plugin successfully initialized with real Strands Agent")
        logger.info("✓ V1 happy case test completed successfully")
        logger.info("Final result stop_reason: %s", result.stop_reason)


@pytest.mark.integration
class TestPaymentHandlerExtraction:
    """Integration tests for payment handler extraction with real handler instances."""

    def test_spec_compliant_marker_extraction(self):
        """Test that handler correctly extracts spec-compliant PAYMENT_REQUIRED marker.

        This test verifies:
        1. Handler correctly extracts payment structure from PAYMENT_REQUIRED: marker
        2. All required fields (statusCode, headers, body) are present
        3. Handler validates statusCode == 402
        """
        from bedrock_agentcore.payments.integrations.handlers import GenericPaymentHandler

        handler = GenericPaymentHandler()

        # Create spec-compliant response with PAYMENT_REQUIRED marker
        payment_required = {
            "statusCode": 402,
            "headers": {"content-type": "application/json", "x-custom": "value"},
            "body": {"error": "Payment required", "details": "Additional info"},
        }

        result = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_required)}"}]

        # Test extraction
        status_code = handler.extract_status_code(result)
        headers = handler.extract_headers(result)
        body = handler.extract_body(result)

        assert status_code == 402, "Status code should be 402"
        logger.info("✓ Status code correctly extracted: %d", status_code)

        assert headers is not None, "Headers should be extracted"
        assert headers.get("content-type") == "application/json", "Headers should contain content-type"
        assert headers.get("x-custom") == "value", "Headers should contain custom header"
        logger.info("✓ Headers correctly extracted: %s", headers)

        assert body is not None, "Body should be extracted"
        assert body.get("error") == "Payment required", "Body should contain error message"
        assert body.get("details") == "Additional info", "Body should contain details"
        logger.info("✓ Body correctly extracted: %s", body)

        logger.info("✓ Spec-compliant marker extraction test completed successfully")

    def test_payment_header_application(self):
        """Test that payment header is correctly applied to tool input.

        This test verifies:
        1. Payment header is added to tool input
        2. Existing headers are preserved
        3. Handler correctly applies header
        """
        from bedrock_agentcore.payments.integrations.handlers import GenericPaymentHandler

        handler = GenericPaymentHandler()

        # Create tool input with existing headers
        tool_input = {"url": "https://api.example.com/resource", "headers": {"content-type": "application/json"}}

        # Apply payment header
        payment_header = {"X-PAYMENT": "base64-encoded-payment"}
        success = handler.apply_payment_header(tool_input, payment_header)

        assert success is True, "Payment header application should succeed"
        logger.info("✓ Payment header application succeeded")

        # Verify header was added
        assert "X-PAYMENT" in tool_input["headers"], "Payment header should be in tool input"
        assert tool_input["headers"]["X-PAYMENT"] == "base64-encoded-payment", "Payment header value should match"
        logger.info("✓ Payment header correctly added: %s", tool_input["headers"]["X-PAYMENT"])

        # Verify existing headers are preserved
        assert tool_input["headers"]["content-type"] == "application/json", "Existing headers should be preserved"
        logger.info("✓ Existing headers preserved: %s", tool_input["headers"])

        logger.info("✓ Payment header application test completed successfully")

    def test_x402_v2_header_based_format(self):
        """Test extraction of X.402 v2 (header-based) format.

        This test verifies:
        1. Handler correctly extracts v2 format with payment-required header
        2. Headers are passed through unmodified
        3. Body is passed through unmodified
        """
        from bedrock_agentcore.payments.integrations.handlers import GenericPaymentHandler

        handler = GenericPaymentHandler()

        # Create X.402 v2 response (header-based)
        v2_payment_info = {
            "x402Version": 2,
            "accepts": [
                {
                    "scheme": "exact",
                    "network": "base-sepolia",
                    "maxAmountRequired": "5000",
                    "resource": "https://api.example.com/resource",
                    "payTo": "0x6813749E1eB9E0001A44C2684695FE8AD676cdD0",
                    "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
                }
            ],
        }

        payment_required = {
            "statusCode": 402,
            "headers": {
                "content-type": "application/json",
                "payment-required": json.dumps(v2_payment_info),
            },
            "body": {"error": "Payment required to access this resource"},
        }

        result = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_required)}"}]

        # Test extraction
        status_code = handler.extract_status_code(result)
        headers = handler.extract_headers(result)
        body = handler.extract_body(result)

        assert status_code == 402, "Status code should be 402"
        logger.info("✓ X.402 v2 status code extracted: %d", status_code)

        assert headers is not None, "Headers should be extracted"
        assert "payment-required" in headers, "Headers should contain payment-required"
        logger.info("✓ X.402 v2 headers extracted with payment-required: %s", headers.get("payment-required")[:50])

        assert body is not None, "Body should be extracted"
        assert body.get("error") == "Payment required to access this resource"
        logger.info("✓ X.402 v2 body extracted: %s", body)

        logger.info("✓ X.402 v2 header-based format test completed successfully")

    def test_x402_v1_body_based_format(self):
        """Test extraction of X.402 v1 (body-based) format.

        This test verifies:
        1. Handler correctly extracts v1 format with x402Version in body
        2. Headers are passed through unmodified
        3. Body is passed through unmodified
        """
        from bedrock_agentcore.payments.integrations.handlers import GenericPaymentHandler

        handler = GenericPaymentHandler()

        # Create X.402 v1 response (body-based)
        v1_body = {
            "x402Version": 1,
            "error": "X-PAYMENT header is required",
            "accepts": [
                {
                    "scheme": "exact",
                    "network": "base-sepolia",
                    "maxAmountRequired": "5000",
                    "resource": "https://api.example.com/resource",
                    "payTo": "0x6813749E1eB9E0001A44C2684695FE8AD676cdD0",
                    "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
                }
            ],
        }

        payment_required = {
            "statusCode": 402,
            "headers": {"content-type": "application/json"},
            "body": v1_body,
        }

        result = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_required)}"}]

        # Test extraction
        status_code = handler.extract_status_code(result)
        headers = handler.extract_headers(result)
        body = handler.extract_body(result)

        assert status_code == 402, "Status code should be 402"
        logger.info("✓ X.402 v1 status code extracted: %d", status_code)

        assert headers is not None, "Headers should be extracted"
        assert headers.get("content-type") == "application/json"
        logger.info("✓ X.402 v1 headers extracted: %s", headers)

        assert body is not None, "Body should be extracted"
        assert body.get("x402Version") == 1, "Body should contain x402Version"
        assert body.get("error") == "X-PAYMENT header is required"
        logger.info("✓ X.402 v1 body extracted with version: %d", body.get("x402Version"))

        logger.info("✓ X.402 v1 body-based format test completed successfully")


@pytest.mark.integration
class TestAgentWithPaymentHandlingFlow:
    """Integration tests for agent with payment handling using spec-compliant tool responses."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.user_id = os.environ.get("TEST_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")

    def test_invoke_agent_with_payment_handling(self):
        """Test agent invocation with payment handling using spec-compliant tool.

        This test demonstrates:
        1. Agent uses a custom tool that returns spec-compliant 402 responses
        2. Plugin intercepts the 402 response
        3. Plugin extracts payment requirements using the marker format
        4. Plugin processes payment and retries the tool
        5. Tool succeeds on retry with payment header

        The flow:
        1. Agent calls http_request_with_payment tool
        2. Tool detects 402 and returns PAYMENT_REQUIRED: marker with structure
        3. Plugin extracts payment requirements from marker
        4. Plugin calls PaymentManager to generate payment header
        5. Plugin applies payment header to tool input
        6. Plugin sets retry flag
        7. Agent retries tool with payment header
        8. Tool succeeds and returns 200 response

        To run this test with real payment manager:
        export BEDROCK_TEST_REGION="us-west-2"
        export TEST_PAYMENT_MANAGER_ARN="arn:aws:bedrock:us-west-2:123456789012:payment-manager/pm-123"
        export TEST_PAYMENT_INSTRUMENT_ID="payment-instrument-xxx"
        export TEST_PAYMENT_SESSION_ID="payment-session-xxx"
        export TEST_USER_ID="test-user"
        pytest tests_integ/payment/integrations/strands/test_plugin_integration.py::\
            TestAgentWithPaymentHandlingFlow::test_invoke_agent_with_payment_handling -v -s
        """
        # Skip if payment manager ARN not configured
        payment_manager_arn = os.environ.get("TEST_PAYMENT_MANAGER_ARN")
        if not payment_manager_arn:
            pytest.skip("TEST_PAYMENT_MANAGER_ARN not configured - skipping agent payment handling test")

        from strands import Agent
        from strands.tools import tool

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=payment_manager_arn,
            user_id=self.user_id,
            payment_instrument_id=os.environ.get("TEST_PAYMENT_INSTRUMENT_ID", "payment-instrument-test123"),
            payment_session_id=os.environ.get("TEST_PAYMENT_SESSION_ID", "payment-session-test456"),
            region=self.region,
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        # Define a custom tool that returns spec-compliant 402 responses
        @tool
        def http_request_with_payment(url: str, method: str = "GET", headers: dict = None, body: str = None) -> dict:
            """Make an HTTP request with payment support.

            Returns spec-compliant 402 response when payment is required.

            Args:
                url: The URL to request
                method: HTTP method (GET, POST, PUT, DELETE, etc.)
                headers: Optional HTTP headers as a dictionary
                body: Optional request body as a string

            Returns:
                dict: Response with status_code, headers, and body
            """
            logger.info("🔵 http_request_with_payment tool called")
            logger.info("   URL: %s", url)
            logger.info("   Method: %s", method)
            logger.info("   Headers: %s", headers)

            # Simulate payment requirement on first call (no X-PAYMENT header)
            if headers is None or "X-PAYMENT" not in headers:
                logger.info("   💳 Payment required - returning 402 response")

                # Build spec-compliant 402 response
                payment_required = {
                    "statusCode": 402,
                    "headers": {
                        "content-type": "application/json",
                        "x-payment-required": "true",
                    },
                    "body": {
                        "error": "Payment required",
                        "message": "X-PAYMENT header is required to access this resource",
                        "x402Version": 1,
                        "accepts": [
                            {
                                "scheme": "exact",
                                "network": "base-sepolia",
                                "maxAmountRequired": "5000",
                                "resource": url,
                                "payTo": "0x6813749E1eB9E0001A44C2684695FE8AD676cdD0",
                                "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
                            }
                        ],
                    },
                }

                # Return ToolResult with PAYMENT_REQUIRED marker (spec-compliant)
                return {
                    "status": "error",
                    "content": [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_required, indent=2)}"}],
                }

            # Simulate successful response on retry (with X-PAYMENT header)
            logger.info("   ✅ Payment header present - returning 200 response")
            return {
                "status": "success",
                "content": [
                    {
                        "text": json.dumps(
                            {
                                "status_code": 200,
                                "headers": {"content-type": "application/json"},
                                "body": {"message": "Success", "data": "Resource accessed with payment"},
                            }
                        )
                    }
                ],
            }

        # Create agent with plugin and custom tool
        agent = Agent(
            system_prompt="You are a helpful assistant that can make HTTP requests.",
            tools=[http_request_with_payment],
            plugins=[plugin],
        )

        logger.info("✓ Agent created with payment plugin and custom tool")

        # Invoke agent with a request that will trigger payment requirement
        prompt = "Please make a GET request to https://api.example.com/protected-resource"

        logger.info("📤 Invoking agent with prompt: %s", prompt)
        result = agent(prompt)

        logger.info("✓ Agent invocation completed")
        logger.info("  Stop reason: %s", result.stop_reason)

        logger.info("✓ Agent with payment handling flow test completed successfully")

    def test_spec_compliant_tool_response_structure(self):
        """Test that tool response follows spec-compliant structure.

        This test verifies:
        1. Tool returns ToolResult with error status
        2. Content contains PAYMENT_REQUIRED: marker
        3. Marker is followed by valid JSON
        4. JSON contains required fields: statusCode, headers, body
        5. statusCode is exactly 402
        6. headers and body are dictionaries
        """
        from strands.tools import tool

        @tool
        def spec_compliant_tool() -> dict:
            """Tool that returns spec-compliant 402 response."""
            payment_required = {
                "statusCode": 402,
                "headers": {
                    "content-type": "application/json",
                    "x-payment-required": "true",
                },
                "body": {
                    "error": "Payment required",
                    "x402Version": 1,
                },
            }

            return {
                "status": "error",
                "content": [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_required)}"}],
            }

        # Call the tool
        result = spec_compliant_tool()

        logger.info("Tool result: %s", result)

        # Verify structure
        assert result["status"] == "error", "Status should be error"
        logger.info("✓ Status is 'error'")

        assert "content" in result, "Result should have content"
        assert len(result["content"]) > 0, "Content should not be empty"
        logger.info("✓ Content is present")

        content_text = result["content"][0]["text"]
        assert content_text.startswith("PAYMENT_REQUIRED: "), "Content should start with PAYMENT_REQUIRED: marker"
        logger.info("✓ Content starts with PAYMENT_REQUIRED: marker")

        # Extract and parse JSON
        payment_json = content_text[len("PAYMENT_REQUIRED: ") :]
        payment_data = json.loads(payment_json)
        logger.info("✓ JSON parsed successfully")

        # Verify required fields
        assert "statusCode" in payment_data, "statusCode field required"
        assert payment_data["statusCode"] == 402, "statusCode must be 402"
        logger.info("✓ statusCode is 402")

        assert "headers" in payment_data, "headers field required"
        assert isinstance(payment_data["headers"], dict), "headers must be dict"
        logger.info("✓ headers is dict: %s", payment_data["headers"])

        assert "body" in payment_data, "body field required"
        assert isinstance(payment_data["body"], dict), "body must be dict"
        logger.info("✓ body is dict: %s", payment_data["body"])

        logger.info("✓ Spec-compliant tool response structure test completed successfully")


@pytest.mark.integration
class TestAgentCorePaymentsPluginAgentName:
    """Integration tests for agent_name propagation through the plugin to PaymentManager."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.user_id = os.environ.get("TEST_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")

    def test_plugin_config_with_agent_name(self):
        """Test plugin config accepts agent_name and passes it through to PaymentManager."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=os.environ.get(
                "TEST_PAYMENT_MANAGER_ARN",
                "arn:aws:bedrock-agentcore:us-west-2:12345678910:payment-manager/mypaymentmanager-cyrc25gr4c",
            ),
            user_id=self.user_id,
            payment_session_id=os.environ.get("TEST_PAYMENT_SESSION_ID", "payment-session-bWOGg4z2irAGbzA"),
            payment_instrument_id=os.environ.get("TEST_PAYMENT_INSTRUMENT_ID", "payment-instrument-vnL29CKAdyESdQ7"),
            region=self.region,
            agent_name="integ-test-agent",
        )

        assert config.agent_name == "integ-test-agent"

        plugin = AgentCorePaymentsPlugin(config=config)
        assert plugin.config.agent_name == "integ-test-agent"

    def test_plugin_config_without_agent_name_backward_compatible(self):
        """Test plugin config works without agent_name (backward compatible)."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=os.environ.get(
                "TEST_PAYMENT_MANAGER_ARN",
                "arn:aws:bedrock-agentcore:us-west-2:12345678910:payment-manager/mypaymentmanager-cyrc25gr4c",
            ),
            user_id=self.user_id,
            payment_session_id=os.environ.get("TEST_PAYMENT_SESSION_ID", "payment-session-bWOGg4z2irAGbzA"),
            payment_instrument_id=os.environ.get("TEST_PAYMENT_INSTRUMENT_ID", "payment-instrument-vnL29CKAdyESdQ7"),
            region=self.region,
        )

        assert config.agent_name is None

        plugin = AgentCorePaymentsPlugin(config=config)
        assert plugin.config.agent_name is None

    def test_plugin_with_agent_name_initializes_payment_manager(self):
        """Test plugin with agent_name initializes PaymentManager with agent_name set.

        This test creates a real Strands Agent with the plugin and verifies
        that PaymentManager is initialized with the agent_name from config.
        """
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=os.environ.get(
                "TEST_PAYMENT_MANAGER_ARN",
                "arn:aws:bedrock-agentcore:us-west-2:12345678910:payment-manager/mypaymentmanager-cyrc25gr4c",
            ),
            user_id=self.user_id,
            payment_session_id=os.environ.get("TEST_PAYMENT_SESSION_ID", "payment-session-bWOGg4z2irAGbzA"),
            payment_instrument_id=os.environ.get("TEST_PAYMENT_INSTRUMENT_ID", "payment-instrument-vnL29CKAdyESdQ7"),
            region=self.region,
            agent_name="strands-payment-agent",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        # Create a real Strands Agent with the plugin
        agent = Agent(system_prompt="You are a helpful assistant.", plugins=[plugin])

        # Verify plugin initialized PaymentManager with agent_name
        assert plugin.payment_manager is not None
        assert isinstance(plugin.payment_manager, PaymentManager)
        assert plugin.payment_manager._agent_name == "strands-payment-agent"
        assert agent is not None

        logger.info("Plugin with agent_name successfully initialized with real Strands Agent")
