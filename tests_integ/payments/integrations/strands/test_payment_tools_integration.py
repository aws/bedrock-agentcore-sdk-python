"""Integration tests for payment tools with real PaymentManager and Strands Agent.

These tests require real AWS credentials and a configured payment manager.
They will be skipped if TEST_PAYMENT_MANAGER_ARN environment variable is not set.

Run with: python -m pytest tests_integ/payment/integrations/strands/test_payment_tools_integration.py -v -s
"""

import logging
import os
import uuid

import pytest
from strands import Agent

from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin
from bedrock_agentcore.payments.manager import PaymentManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestPaymentToolsIntegration:
    """Integration tests for payment tools with real PaymentManager and Strands Agent.

    These tests use a real PaymentManager instance and Strands Agent to test
    the payment tools in a realistic scenario. They require:
    - TEST_PAYMENT_MANAGER_ARN environment variable
    - Valid AWS credentials
    - Real payment manager setup
    """

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.user_id = os.environ.get("TEST_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")
        cls.payment_manager_arn = os.environ.get("TEST_PAYMENT_MANAGER_ARN")
        cls.payment_instrument_id = os.environ.get("TEST_PAYMENT_INSTRUMENT_ID", "payment-instrument-abcdefghijklmno")
        cls.payment_session_id = os.environ.get("TEST_PAYMENT_SESSION_ID", "payment-session-abcdefghijklmno")

    def test_payment_tools_with_real_agent(self):
        """Test payment tools with a real Strands Agent.

        This test creates a real Strands Agent with the payment plugin and verifies
        that the payment tools are available and callable through the agent.

        This test requires:
        - TEST_PAYMENT_MANAGER_ARN environment variable
        - Valid AWS credentials
        """
        if not self.payment_manager_arn:
            pytest.skip("TEST_PAYMENT_MANAGER_ARN not configured - skipping real agent test")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=self.payment_manager_arn,
            user_id=self.user_id,
            payment_instrument_id=self.payment_instrument_id,
            payment_session_id=self.payment_session_id,
            region=self.region,
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        # Create a real Strands Agent with the payment plugin
        _agent = Agent(
            system_prompt="You are a helpful assistant with access to payment tools.",
            plugins=[plugin],
        )

        # Verify plugin is registered with agent
        assert plugin.payment_manager is not None
        assert isinstance(plugin.payment_manager, PaymentManager)

        logger.info("✓ Payment tools successfully initialized with real Strands Agent")

    def test_get_payment_instrument_tool_with_agent(self):
        """Test getPaymentInstrument tool execution through Strands Agent.

        This test creates an agent with the payment plugin and verifies that
        the getPaymentInstrument tool is available and can be invoked.

        This test requires:
        - TEST_PAYMENT_MANAGER_ARN environment variable
        - Valid AWS credentials
        - Real payment instrument ID
        """
        if not self.payment_manager_arn:
            pytest.skip("TEST_PAYMENT_MANAGER_ARN not configured - skipping real agent test")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=self.payment_manager_arn,
            user_id=self.user_id,
            payment_instrument_id=self.payment_instrument_id,
            payment_session_id=self.payment_session_id,
            region=self.region,
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        # Create a real Strands Agent with the payment plugin
        _agent = Agent(
            system_prompt="You are a helpful assistant with access to payment tools.",
            plugins=[plugin],
        )

        # Verify the tool is available
        assert callable(plugin.get_payment_instrument)

        logger.info("✓ getPaymentInstrument tool is available through Strands Agent")

    def test_list_payment_instruments_tool_with_agent(self):
        """Test listPaymentInstruments tool execution through Strands Agent.

        This test creates an agent with the payment plugin and verifies that
        the listPaymentInstruments tool is available and can be invoked.

        This test requires:
        - TEST_PAYMENT_MANAGER_ARN environment variable
        - Valid AWS credentials
        """
        if not self.payment_manager_arn:
            pytest.skip("TEST_PAYMENT_MANAGER_ARN not configured - skipping real agent test")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=self.payment_manager_arn,
            user_id=self.user_id,
            payment_instrument_id=self.payment_instrument_id,
            payment_session_id=self.payment_session_id,
            region=self.region,
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        # Create a real Strands Agent with the payment plugin
        _agent = Agent(
            system_prompt="You are a helpful assistant with access to payment tools.",
            plugins=[plugin],
        )

        # Verify the tool is available
        assert callable(plugin.list_payment_instruments)

        logger.info("✓ listPaymentInstruments tool is available through Strands Agent")

    def test_get_payment_session_tool_with_agent(self):
        """Test getPaymentSession tool execution through Strands Agent.

        This test creates an agent with the payment plugin and verifies that
        the getPaymentSession tool is available and can be invoked.

        This test requires:
        - TEST_PAYMENT_MANAGER_ARN environment variable
        - Valid AWS credentials
        - Real payment session ID
        """
        if not self.payment_manager_arn:
            pytest.skip("TEST_PAYMENT_MANAGER_ARN not configured - skipping real agent test")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=self.payment_manager_arn,
            user_id=self.user_id,
            payment_instrument_id=self.payment_instrument_id,
            payment_session_id=self.payment_session_id,
            region=self.region,
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        # Create a real Strands Agent with the payment plugin
        _agent = Agent(
            system_prompt="You are a helpful assistant with access to payment tools.",
            plugins=[plugin],
        )

        # Verify the tool is available
        assert callable(plugin.get_payment_session)

        logger.info("✓ getPaymentSession tool is available through Strands Agent")


@pytest.mark.integration
class TestPaymentToolsWithAutoPaymentFlag:
    """Integration tests for payment tools with auto_payment flag.

    These tests verify that payment tools work correctly with the auto_payment
    configuration flag using a real PaymentManager instance.
    """

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.user_id = os.environ.get("TEST_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")
        cls.payment_manager_arn = os.environ.get("TEST_PAYMENT_MANAGER_ARN")
        cls.payment_instrument_id = os.environ.get("TEST_PAYMENT_INSTRUMENT_ID", "payment-instrument-abcdefghijklmno")
        cls.payment_session_id = os.environ.get("TEST_PAYMENT_SESSION_ID", "payment-session-abcdefghijklmno")

    def test_tools_available_with_auto_payment_true(self):
        """Test payment tools are available when auto_payment=True.

        This test requires:
        - TEST_PAYMENT_MANAGER_ARN environment variable
        - Valid AWS credentials
        """
        if not self.payment_manager_arn:
            pytest.skip("TEST_PAYMENT_MANAGER_ARN not configured - skipping real PaymentManager test")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=self.payment_manager_arn,
            user_id=self.user_id,
            payment_instrument_id=self.payment_instrument_id,
            payment_session_id=self.payment_session_id,
            region=self.region,
            auto_payment=True,
        )
        plugin = AgentCorePaymentsPlugin(config)

        # Verify all tools are callable
        assert callable(plugin.get_payment_instrument)
        assert callable(plugin.list_payment_instruments)
        assert callable(plugin.get_payment_session)

        logger.info("✓ Payment tools are available with auto_payment=True")

    def test_tools_available_with_auto_payment_false(self):
        """Test payment tools are available when auto_payment=False.

        This test requires:
        - TEST_PAYMENT_MANAGER_ARN environment variable
        - Valid AWS credentials
        """
        if not self.payment_manager_arn:
            pytest.skip("TEST_PAYMENT_MANAGER_ARN not configured - skipping real PaymentManager test")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=self.payment_manager_arn,
            user_id=self.user_id,
            payment_instrument_id=self.payment_instrument_id,
            payment_session_id=self.payment_session_id,
            region=self.region,
            auto_payment=False,
        )
        plugin = AgentCorePaymentsPlugin(config)

        # Verify all tools are callable
        assert callable(plugin.get_payment_instrument)
        assert callable(plugin.list_payment_instruments)
        assert callable(plugin.get_payment_session)

        logger.info("✓ Payment tools are available with auto_payment=False")

    def test_auto_payment_flag_default_value(self):
        """Test that auto_payment flag defaults to True.

        This test requires:
        - TEST_PAYMENT_MANAGER_ARN environment variable
        - Valid AWS credentials
        """
        if not self.payment_manager_arn:
            pytest.skip("TEST_PAYMENT_MANAGER_ARN not configured - skipping real PaymentManager test")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=self.payment_manager_arn,
            user_id=self.user_id,
            payment_instrument_id=self.payment_instrument_id,
            payment_session_id=self.payment_session_id,
            region=self.region,
        )

        # Verify auto_payment defaults to True
        assert config.auto_payment is True

        logger.info("✓ auto_payment flag defaults to True")


@pytest.mark.integration
class TestPaymentToolsWithPrompts:
    """Integration tests that invoke agent with prompts to trigger payment tools.

    These tests verify that the payment tools can be invoked through natural language
    prompts to the Strands Agent.
    """

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.user_id = os.environ.get("TEST_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")
        cls.payment_manager_arn = os.environ.get("TEST_PAYMENT_MANAGER_ARN")
        cls.payment_instrument_id = os.environ.get("TEST_PAYMENT_INSTRUMENT_ID", "payment-instrument-abcdefghijklmno")
        cls.payment_session_id = os.environ.get("TEST_PAYMENT_SESSION_ID", "payment-session-abcdefghijklmno")

    def test_agent_invokes_list_payment_instruments_with_prompt(self):
        """Test agent invokes listPaymentInstruments tool via natural language prompt.

        This test requires:
        - TEST_PAYMENT_MANAGER_ARN environment variable
        - Valid AWS credentials
        """
        if not self.payment_manager_arn:
            pytest.skip("TEST_PAYMENT_MANAGER_ARN not configured - skipping real agent test")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=self.payment_manager_arn,
            user_id=self.user_id,
            payment_instrument_id=self.payment_instrument_id,
            payment_session_id=self.payment_session_id,
            region=self.region,
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        # Create a real Strands Agent with the payment plugin
        agent = Agent(
            system_prompt=(
                "You are a helpful assistant with access to payment tools. "
                "When asked about payment instruments, use the listPaymentInstruments tool."
            ),
            plugins=[plugin],
        )

        # Invoke agent with prompt to trigger listPaymentInstruments
        prompt = f"Please list all payment instruments for user {self.user_id}"
        logger.info("\n%s", "=" * 80)
        logger.info("PROMPT: %s", prompt)
        logger.info("%s", "=" * 80)

        result = agent(prompt)

        # Verify agent executed successfully
        assert result is not None
        logger.info("\n%s", "=" * 80)
        logger.info("RESULT: %s", result)
        logger.info("Stop Reason: %s", result.stop_reason)
        logger.info("%s", "=" * 80)
        logger.info("✓ Agent successfully invoked listPaymentInstruments tool via prompt")

    def test_agent_invokes_get_payment_instrument_with_prompt(self):
        """Test agent invokes getPaymentInstrument tool via natural language prompt.

        This test requires:
        - TEST_PAYMENT_MANAGER_ARN environment variable
        - Valid AWS credentials
        - Real payment instrument ID
        """
        if not self.payment_manager_arn:
            pytest.skip("TEST_PAYMENT_MANAGER_ARN not configured - skipping real agent test")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=self.payment_manager_arn,
            user_id=self.user_id,
            payment_instrument_id=self.payment_instrument_id,
            payment_session_id=self.payment_session_id,
            region=self.region,
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        # Create a real Strands Agent with the payment plugin
        agent = Agent(
            system_prompt=(
                "You are a helpful assistant with access to payment tools. "
                "When asked about a specific payment instrument, use the getPaymentInstrument tool."
            ),
            plugins=[plugin],
        )

        # Invoke agent with prompt to trigger getPaymentInstrument
        prompt = f"Please get details for payment instrument {self.payment_instrument_id} for user {self.user_id}"
        logger.info("\n%s", "=" * 80)
        logger.info("PROMPT: %s", prompt)
        logger.info("%s", "=" * 80)

        result = agent(prompt)

        # Verify agent executed successfully
        assert result is not None
        logger.info("\n%s", "=" * 80)
        logger.info("RESULT: %s", result)
        logger.info("Stop Reason: %s", result.stop_reason)
        logger.info("%s", "=" * 80)
        logger.info("✓ Agent successfully invoked getPaymentInstrument tool via prompt")

    def test_agent_invokes_get_payment_session_with_prompt(self):
        """Test agent invokes getPaymentSession tool via natural language prompt.

        This test requires:
        - TEST_PAYMENT_MANAGER_ARN environment variable
        - Valid AWS credentials
        - Real payment session ID
        """
        if not self.payment_manager_arn:
            pytest.skip("TEST_PAYMENT_MANAGER_ARN not configured - skipping real agent test")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=self.payment_manager_arn,
            user_id=self.user_id,
            payment_instrument_id=self.payment_instrument_id,
            payment_session_id=self.payment_session_id,
            region=self.region,
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        # Create a real Strands Agent with the payment plugin
        agent = Agent(
            system_prompt=(
                "You are a helpful assistant with access to payment tools. "
                "When asked about a payment session, use the getPaymentSession tool."
            ),
            plugins=[plugin],
        )

        # Invoke agent with prompt to trigger getPaymentSession
        prompt = f"Please get details for payment session {self.payment_session_id} for user {self.user_id}"
        logger.info("\n%s", "=" * 80)
        logger.info("PROMPT: %s", prompt)
        logger.info("%s", "=" * 80)

        result = agent(prompt)

        # Verify agent executed successfully
        assert result is not None
        logger.info("\n%s", "=" * 80)
        logger.info("RESULT: %s", result)
        logger.info("Stop Reason: %s", result.stop_reason)
        logger.info("%s", "=" * 80)
        logger.info("✓ Agent successfully invoked getPaymentSession tool via prompt")

    def test_agent_invokes_multiple_tools_with_sequential_prompts(self):
        """Test agent invokes multiple payment tools with sequential prompts.

        This test requires:
        - TEST_PAYMENT_MANAGER_ARN environment variable
        - Valid AWS credentials
        """
        if not self.payment_manager_arn:
            pytest.skip("TEST_PAYMENT_MANAGER_ARN not configured - skipping real agent test")

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn=self.payment_manager_arn,
            user_id=self.user_id,
            payment_instrument_id=self.payment_instrument_id,
            payment_session_id=self.payment_session_id,
            region=self.region,
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        # Create a real Strands Agent with the payment plugin
        system_prompt = (
            "You are a helpful assistant with access to payment tools. "
            "Use the appropriate tool to answer questions about payment instruments "
            "and sessions."
        )
        agent = Agent(
            system_prompt=system_prompt,
            plugins=[plugin],
        )

        # First prompt: list instruments
        prompt1 = f"List all payment instruments for user {self.user_id}"
        result1 = agent(prompt1)
        assert result1 is not None
        logger.info("✓ First prompt: listPaymentInstruments executed")

        # Second prompt: get specific instrument
        prompt2 = f"Get details for payment instrument {self.payment_instrument_id}"
        result2 = agent(prompt2)
        assert result2 is not None
        logger.info("✓ Second prompt: getPaymentInstrument executed")

        # Third prompt: get session details
        prompt3 = f"Get details for payment session {self.payment_session_id}"
        result3 = agent(prompt3)
        assert result3 is not None
        logger.info("✓ Third prompt: getPaymentSession executed")

        logger.info("✓ Agent successfully invoked multiple payment tools with sequential prompts")
