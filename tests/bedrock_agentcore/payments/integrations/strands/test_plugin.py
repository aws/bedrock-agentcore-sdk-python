"""Unit tests for AgentCorePaymentsPlugin."""

import json
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin
from bedrock_agentcore.payments.manager import (
    PaymentError,
    PaymentInstrumentConfigurationRequired,
    PaymentInstrumentNotFound,
    PaymentSessionConfigurationRequired,
    PaymentSessionExpired,
)


def _create_mock_agent():
    """Create a mock agent with a dict-backed state object."""
    agent = MagicMock()
    state_store = {}

    def state_get(key=None):
        if key is None:
            return dict(state_store)
        return state_store.get(key)

    def state_set(key, value):
        state_store[key] = value

    def state_delete(key):
        state_store.pop(key, None)

    agent.state.get = MagicMock(side_effect=state_get)
    agent.state.set = MagicMock(side_effect=state_set)
    agent.state.delete = MagicMock(side_effect=state_delete)
    agent._state_store = state_store  # expose for test assertions
    return agent


def _setup_plugin_with_agent(config, mock_pm_instance=None):
    """Create a plugin and initialize it with a mock agent."""
    plugin = AgentCorePaymentsPlugin(config=config)
    agent = _create_mock_agent()
    if mock_pm_instance:
        with patch(
            "bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager", return_value=mock_pm_instance
        ):
            plugin.init_agent(agent)
    else:
        with patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager"):
            plugin.init_agent(agent)
    if mock_pm_instance:
        plugin.payment_manager = mock_pm_instance
    return plugin, agent


def _create_event_with_agent(event_attrs=None, agent=None):
    """Create a mock event with an agent that has dict-backed state."""
    if agent is None:
        agent = _create_mock_agent()
    event = MagicMock()
    event.agent = agent
    if event_attrs:
        for k, v in event_attrs.items():
            setattr(event, k, v)
    return event, agent


class TestAgentCorePaymentsPluginInitialization:
    """Tests for plugin initialization."""

    def test_plugin_init_with_valid_config(self):
        """Test plugin initialization with valid configuration."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
            region="us-west-2",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        assert plugin.config == config
        assert plugin.payment_manager is None
        assert plugin.name == "agent-core-payments-plugin"

    def test_plugin_init_with_network_preferences(self):
        """Test plugin initialization with network preferences."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
            region="us-west-2",
            network_preferences_config=["eip155:1", "solana:mainnet"],
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        assert plugin.config.network_preferences_config == ["eip155:1", "solana:mainnet"]

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_init_agent_success(self, mock_payment_manager_class):
        """Test successful agent initialization."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
            region="us-west-2",
        )

        mock_pm_instance = MagicMock()
        mock_payment_manager_class.return_value = mock_pm_instance

        plugin = AgentCorePaymentsPlugin(config=config)
        mock_agent = MagicMock()

        plugin.init_agent(mock_agent)

        mock_payment_manager_class.assert_called_once_with(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            region_name="us-west-2",
            agent_name=None,
            bearer_token=None,
            token_provider=None,
        )
        assert plugin.payment_manager == mock_pm_instance

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_init_agent_failure(self, mock_payment_manager_class):
        """Test agent initialization failure."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
            region="us-west-2",
        )

        mock_payment_manager_class.side_effect = Exception("Connection failed")

        plugin = AgentCorePaymentsPlugin(config=config)
        mock_agent = MagicMock()

        with pytest.raises(RuntimeError, match="Failed to initialize PaymentManager"):
            plugin.init_agent(mock_agent)


class TestBeforeToolCall:
    """Tests for before_tool_call hook."""

    def test_before_tool_call_no_payment_failure(self):
        """Test before_tool_call when no payment failure is stored."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        event = MagicMock()
        event.invocation_state = {}
        event.interrupt = MagicMock()

        plugin.before_tool_call(event)

        event.interrupt.assert_not_called()

    def test_before_tool_call_with_payment_failure(self):
        """Test before_tool_call when payment failure is stored."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        tool_use_id = "tool-use-123"
        failure_info = {
            "toolUseId": tool_use_id,
            "exceptionType": "PaymentError",
            "exceptionMessage": "Insufficient budget",
        }

        event, agent = _create_event_with_agent(
            {
                "invocation_state": {f"payment_failure_{tool_use_id}": failure_info},
                "tool_use": {"name": "test_tool"},
            }
        )
        event.interrupt = MagicMock()

        plugin.before_tool_call(event)

        event.interrupt.assert_called_once()
        call_args = event.interrupt.call_args
        assert call_args[0][0].startswith(f"payment-failure-{tool_use_id}")
        assert call_args[1]["reason"] == failure_info
        assert f"payment_failure_{tool_use_id}" not in event.invocation_state

    def test_before_tool_call_multiple_failures(self):
        """Test before_tool_call with multiple payment failures."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        tool_use_id_1 = "tool-use-123"
        tool_use_id_2 = "tool-use-456"
        failure_info_1 = {"toolUseId": tool_use_id_1, "exceptionType": "PaymentError"}
        failure_info_2 = {"toolUseId": tool_use_id_2, "exceptionType": "PaymentError"}

        event, agent = _create_event_with_agent(
            {
                "invocation_state": {
                    f"payment_failure_{tool_use_id_1}": failure_info_1,
                    f"payment_failure_{tool_use_id_2}": failure_info_2,
                },
                "tool_use": {"name": "test_tool"},
            }
        )
        event.interrupt = MagicMock()

        plugin.before_tool_call(event)

        # Should only interrupt for the first failure found
        event.interrupt.assert_called_once()


class TestAfterToolCall:
    """Tests for after_tool_call hook."""

    def test_after_tool_call_no_result(self):
        """Test after_tool_call when result is None."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        event, _ = _create_event_with_agent(
            {
                "result": None,
                "invocation_state": {},
                "retry": False,
                "tool_use": {"name": "test_tool", "toolUseId": "tool-123"},
            }
        )

        plugin.after_tool_call(event)

        # Should return early without processing
        assert event.retry is False

    def test_after_tool_call_non_402_status(self):
        """Test after_tool_call with non-402 status code."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        event, _ = _create_event_with_agent(
            {
                "result": [{"text": "Status Code: 200"}],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": {}},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        # Should not set retry for non-402 status
        assert event.retry is False

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_after_tool_call_no_handler_for_tool(self, mock_payment_manager_class):
        """Test after_tool_call with generic handler for unknown tools."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.return_value = {"X-PAYMENT": "base64-encoded"}

        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, agent = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "unknown_tool", "toolUseId": "tool-123", "input": {}},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        assert event.retry is True
        mock_pm_instance.generate_payment_header.assert_called_once()
        assert "X-PAYMENT" in event.tool_use["input"]["headers"]
        assert event.tool_use["input"]["headers"]["X-PAYMENT"] == "base64-encoded"

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_after_tool_call_402_payment_success(self, mock_payment_manager_class):
        """Test after_tool_call successfully processes 402 payment."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.return_value = {"X-PAYMENT": "base64-encoded-payment"}

        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, agent = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        assert event.retry is True
        mock_pm_instance.generate_payment_header.assert_called_once()
        assert "X-PAYMENT" in event.tool_use["input"]["headers"]

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_after_tool_call_payment_error(self, mock_payment_manager_class):
        """Test after_tool_call handles payment error."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.side_effect = PaymentInstrumentNotFound("Instrument not found")

        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, agent = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        assert "payment_failure_tool-123" in event.invocation_state

    def test_after_tool_call_402_validation_failure_skips_payment(self):
        """Test after_tool_call skips payment processing when tool input validation fails."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        mock_pm_instance = MagicMock()
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, _ = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": "not a dict"},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        # Should not call generate_payment_header
        mock_pm_instance.generate_payment_header.assert_not_called()
        assert event.retry is False
        assert "payment_failure_tool-123" in event.invocation_state

    def test_after_tool_call_retry_limit_reached(self):
        """Test after_tool_call when retry limit is reached."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        event = MagicMock()
        event.result = [{"text": "Status Code: 402"}]
        event.tool_use = {"name": "http_request", "toolUseId": "tool-123", "input": {}}
        event.invocation_state = {"payment_retry_count_tool-123": 3}
        event.retry = False

        plugin.after_tool_call(event)

        # Should return early without processing
        assert event.retry is False


class TestCheckPaymentRetryLimit:
    """Tests for _check_payment_retry_limit method."""

    def test_retry_limit_not_reached(self):
        """Test when retry limit has not been reached."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        event, agent = _create_event_with_agent(
            {
                "tool_use": {"toolUseId": "tool-123"},
                "invocation_state": {"payment_retry_count_tool-123": 1},
            }
        )

        result = plugin._check_payment_retry_limit(event)
        assert result is False

    def test_retry_limit_reached(self):
        """Test when retry limit has been reached."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        event, agent = _create_event_with_agent(
            {
                "tool_use": {"toolUseId": "tool-123"},
                "invocation_state": {"payment_retry_count_tool-123": 3},
            }
        )

        result = plugin._check_payment_retry_limit(event)
        assert result is True

    def test_retry_limit_no_prior_attempts(self):
        """Test when no prior retry attempts exist."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        event, agent = _create_event_with_agent(
            {
                "tool_use": {"toolUseId": "tool-123"},
                "invocation_state": {},
            }
        )

        result = plugin._check_payment_retry_limit(event)
        assert result is False


class TestIncrementPaymentRetryCount:
    """Tests for _increment_payment_retry_count method."""

    def test_increment_from_zero(self):
        """Test incrementing retry count from zero."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        event = MagicMock()
        event.tool_use = {"toolUseId": "tool-123"}
        event.invocation_state = {}

        plugin._increment_payment_retry_count(event)

        assert event.invocation_state["payment_retry_count_tool-123"] == 1

    def test_increment_from_existing_count(self):
        """Test incrementing retry count from existing value."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        event = MagicMock()
        event.tool_use = {"toolUseId": "tool-123"}
        event.invocation_state = {"payment_retry_count_tool-123": 2}

        plugin._increment_payment_retry_count(event)

        assert event.invocation_state["payment_retry_count_tool-123"] == 3


class TestStorePaymentFailureState:
    """Tests for _store_payment_failure_state method."""

    def test_store_payment_failure_state(self):
        """Test storing payment failure state."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        event = MagicMock()
        event.tool_use = {
            "toolUseId": "tool-123",
            "name": "http_request",
            "input": {"url": "https://example.com"},
        }
        event.invocation_state = {}

        exception = PaymentInstrumentNotFound("Instrument not found")

        plugin._store_payment_failure_state(event, exception)

        failure_key = "payment_failure_tool-123"
        assert failure_key in event.invocation_state

        failure_info = event.invocation_state[failure_key]
        assert failure_info["tool"] == "http_request"
        assert failure_info["toolUseId"] == "tool-123"
        assert failure_info["exceptionType"] == "PaymentInstrumentNotFound"
        assert failure_info["exceptionMessage"] == "Instrument not found"
        assert failure_info["maxRetries"] == 3

    def test_store_payment_failure_state_with_retry_count(self):
        """Test storing payment failure state with existing retry count."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        event = MagicMock()
        event.tool_use = {
            "toolUseId": "tool-123",
            "name": "http_request",
            "input": {},
        }
        event.invocation_state = {"payment_retry_count_tool-123": 2}

        exception = PaymentSessionExpired("Session expired")

        plugin._store_payment_failure_state(event, exception)

        failure_info = event.invocation_state["payment_failure_tool-123"]
        assert failure_info["retryAttempt"] == 2


class TestProcessX402Payment:
    """Tests for _process_x402_payment method."""

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_process_x402_payment_success(self, mock_payment_manager_class):
        """Test successful X.402 payment processing."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.return_value = {"X-PAYMENT": "base64-encoded"}
        mock_payment_manager_class.return_value = mock_pm_instance

        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        x402_response = {
            "statusCode": 402,
            "headers": {"X-Payment-Required": "true"},
            "body": {"scheme": "exact", "network": "ethereum"},
        }

        result = plugin._process_payment_required_request(x402_response)

        assert result == {"X-PAYMENT": "base64-encoded"}
        mock_pm_instance.generate_payment_header.assert_called_once()

    def test_process_x402_payment_no_payment_manager(self):
        """Test X.402 payment processing when PaymentManager is not initialized."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = None

        x402_response = {
            "statusCode": 402,
            "headers": {},
            "body": {},
        }

        with pytest.raises(PaymentError, match="PaymentManager not initialized"):
            plugin._process_payment_required_request(x402_response)

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_process_x402_payment_with_network_preferences(self, mock_payment_manager_class):
        """Test X.402 payment processing with network preferences."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
            network_preferences_config=["eip155:1", "solana:mainnet"],
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.return_value = {"X-PAYMENT": "base64-encoded"}
        mock_payment_manager_class.return_value = mock_pm_instance

        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        x402_response = {
            "statusCode": 402,
            "headers": {},
            "body": {},
        }

        plugin._process_payment_required_request(x402_response)

        # Verify network_preferences were passed
        call_kwargs = mock_pm_instance.generate_payment_header.call_args[1]
        assert call_kwargs["network_preferences"] == ["eip155:1", "solana:mainnet"]


class TestAfterToolCallEdgeCases:
    """Tests for edge cases in after_tool_call hook."""

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_after_tool_call_handler_apply_header_fails(self, mock_payment_manager_class):
        """Test after_tool_call when handler fails to apply payment header."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.return_value = {"X-PAYMENT": "base64-encoded"}

        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, agent = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": "not a dict"},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        # Should store failure state when handler fails
        assert "payment_failure_tool-123" in event.invocation_state

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_after_tool_call_unexpected_exception(self, mock_payment_manager_class):
        """Test after_tool_call handles unexpected exceptions."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.side_effect = Exception("Unexpected error")
        mock_payment_manager_class.return_value = mock_pm_instance

        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, _ = _create_event_with_agent(
            {
                "result": [
                    {"text": "Status Code: 402"},
                    {"text": "Headers: {}"},
                    {"text": "Body: {}"},
                ],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        # Should not set retry for unexpected exceptions
        assert event.retry is False


class TestProcessX402PaymentEdgeCases:
    """Tests for edge cases in _process_x402_payment method."""

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_process_x402_payment_with_uuid_generation(self, mock_payment_manager_class):
        """Test that _process_x402_payment generates a client token."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.return_value = {"X-PAYMENT": "base64-encoded"}
        mock_payment_manager_class.return_value = mock_pm_instance

        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        x402_response = {
            "statusCode": 402,
            "headers": {},
            "body": {},
        }

        plugin._process_payment_required_request(x402_response)

        # Verify client_token was passed and is a valid UUID string
        call_kwargs = mock_pm_instance.generate_payment_header.call_args[1]
        client_token = call_kwargs["client_token"]
        assert isinstance(client_token, str)
        # Verify it's a valid UUID format
        assert len(client_token) == 36  # UUID string length


class TestBeforeToolCallEdgeCases:
    """Tests for edge cases in before_tool_call hook."""

    def test_before_tool_call_with_non_payment_failure_keys(self):
        """Test before_tool_call ignores non-payment-failure keys."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        event = MagicMock()
        event.invocation_state = {
            "other_key": "value",
            "payment_retry_count_tool-123": 1,
        }
        event.interrupt = MagicMock()

        plugin.before_tool_call(event)

        # Should not interrupt for non-payment-failure keys
        event.interrupt.assert_not_called()


class TestPluginMaxRetries:
    """Tests for MAX_PAYMENT_RETRIES constant."""

    def test_max_payment_retries_constant(self):
        """Test that MAX_PAYMENT_RETRIES is set correctly."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        assert plugin.MAX_PAYMENT_RETRIES == 3

    def test_retry_count_increments_to_max(self):
        """Test that retry count can increment to MAX_PAYMENT_RETRIES."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        plugin = AgentCorePaymentsPlugin(config=config)

        event = MagicMock()
        event.tool_use = {"toolUseId": "tool-123"}
        event.invocation_state = {}

        # Increment to max
        for _ in range(plugin.MAX_PAYMENT_RETRIES):
            plugin._increment_payment_retry_count(event)

        assert event.invocation_state["payment_retry_count_tool-123"] == plugin.MAX_PAYMENT_RETRIES

        # Next check should return True (limit reached)
        result = plugin._check_payment_retry_limit(event)
        assert result is True


class TestProcessPaymentRequiredConfigChecks:
    """Tests for _process_payment_required_request config validation."""

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_raises_instrument_config_required_when_none(self, mock_payment_manager_class):
        """Test that PaymentInstrumentConfigurationRequired is raised when instrument_id is None."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_session_id="payment-session-456",
        )

        mock_pm_instance = MagicMock()
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        with pytest.raises(PaymentInstrumentConfigurationRequired, match="payment_instrument_id is required"):
            plugin._process_payment_required_request({"statusCode": 402, "headers": {}, "body": {}})

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_raises_session_config_required_when_none(self, mock_payment_manager_class):
        """Test that PaymentSessionConfigurationRequired is raised when session_id is None."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
        )

        mock_pm_instance = MagicMock()
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        with pytest.raises(PaymentSessionConfigurationRequired, match="payment_session_id is required"):
            plugin._process_payment_required_request({"statusCode": 402, "headers": {}, "body": {}})

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_raises_instrument_before_session_when_both_none(self, mock_payment_manager_class):
        """Test that instrument check comes before session check when both are None."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
        )

        mock_pm_instance = MagicMock()
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        with pytest.raises(PaymentInstrumentConfigurationRequired):
            plugin._process_payment_required_request({"statusCode": 402, "headers": {}, "body": {}})

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_succeeds_after_config_update(self, mock_payment_manager_class):
        """Test that payment processing succeeds after updating config."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.return_value = {"X-PAYMENT": "base64"}
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        # First call fails
        with pytest.raises(PaymentInstrumentConfigurationRequired):
            plugin._process_payment_required_request({"statusCode": 402, "headers": {}, "body": {}})

        # Update config
        config.update_payment_instrument_id("instrument-123")
        config.update_payment_session_id("session-456")

        # Second call succeeds
        result = plugin._process_payment_required_request({"statusCode": 402, "headers": {}, "body": {}})
        assert result == {"X-PAYMENT": "base64"}


class TestInterruptRetryLimit:
    """Tests for interrupt retry limit functionality."""

    def test_check_interrupt_retry_limit_no_agent(self):
        """Test that limit is reached when agent is None."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        assert plugin._check_interrupt_retry_limit(None, "tool-123") is True

    def test_check_interrupt_retry_limit_zero_max(self):
        """Test that limit is reached when max_interrupt_retries is 0."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            max_interrupt_retries=0,
        )
        plugin = AgentCorePaymentsPlugin(config=config)
        agent = _create_mock_agent()

        assert plugin._check_interrupt_retry_limit(agent, "tool-123") is True

    def test_check_interrupt_retry_limit_not_reached(self):
        """Test that limit is not reached with count below max."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            max_interrupt_retries=5,
        )
        plugin = AgentCorePaymentsPlugin(config=config)
        agent = _create_mock_agent()

        assert plugin._check_interrupt_retry_limit(agent, "tool-123") is False

    def test_check_interrupt_retry_limit_reached(self):
        """Test that limit is reached when count equals max."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            max_interrupt_retries=3,
        )
        plugin = AgentCorePaymentsPlugin(config=config)
        agent = _create_mock_agent()

        for _ in range(3):
            plugin._increment_interrupt_retry_count(agent, "tool-123")

        assert plugin._check_interrupt_retry_limit(agent, "tool-123") is True

    def test_increment_interrupt_retry_count(self):
        """Test incrementing interrupt retry count in agent state."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
        )
        plugin = AgentCorePaymentsPlugin(config=config)
        agent = _create_mock_agent()

        plugin._increment_interrupt_retry_count(agent, "tool-123")
        assert agent._state_store["payment_interrupt_retry_tool-123"] == 1

        plugin._increment_interrupt_retry_count(agent, "tool-123")
        assert agent._state_store["payment_interrupt_retry_tool-123"] == 2

    def test_increment_interrupt_retry_count_no_agent(self):
        """Test that increment is a no-op when agent is None."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        # Should not raise
        plugin._increment_interrupt_retry_count(None, "tool-123")

    def test_interrupt_retry_independent_per_tool(self):
        """Test that interrupt retry counts are independent per tool_use_id."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            max_interrupt_retries=2,
        )
        plugin = AgentCorePaymentsPlugin(config=config)
        agent = _create_mock_agent()

        plugin._increment_interrupt_retry_count(agent, "tool-1")
        plugin._increment_interrupt_retry_count(agent, "tool-1")
        plugin._increment_interrupt_retry_count(agent, "tool-2")

        assert plugin._check_interrupt_retry_limit(agent, "tool-1") is True
        assert plugin._check_interrupt_retry_limit(agent, "tool-2") is False


class TestBeforeToolCallInterruptRetry:
    """Tests for interrupt retry limit in before_tool_call."""

    def test_before_tool_call_skips_interrupt_when_limit_reached(self):
        """Test that before_tool_call skips interrupt when interrupt retry limit is reached."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            max_interrupt_retries=1,
        )
        plugin = AgentCorePaymentsPlugin(config=config)
        agent = _create_mock_agent()

        # Simulate one interrupt already happened
        plugin._increment_interrupt_retry_count(agent, "tool-123")

        failure_info = {
            "toolUseId": "tool-123",
            "exceptionType": "PaymentError",
            "exceptionMessage": "Some error",
        }
        event, _ = _create_event_with_agent(
            {
                "invocation_state": {"payment_failure_tool-123": failure_info},
                "tool_use": {"name": "test_tool"},
            },
            agent=agent,
        )
        event.interrupt = MagicMock()

        plugin.before_tool_call(event)

        event.interrupt.assert_not_called()
        assert "payment_failure_tool-123" not in event.invocation_state

    def test_before_tool_call_raises_interrupt_when_below_limit(self):
        """Test that before_tool_call raises interrupt when below limit."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            max_interrupt_retries=5,
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        failure_info = {
            "toolUseId": "tool-123",
            "exceptionType": "PaymentInstrumentConfigurationRequired",
            "exceptionMessage": "payment_instrument_id is not set.",
        }
        event, agent = _create_event_with_agent(
            {
                "invocation_state": {"payment_failure_tool-123": failure_info},
                "tool_use": {"name": "test_tool"},
            }
        )
        event.interrupt = MagicMock()

        plugin.before_tool_call(event)

        event.interrupt.assert_called_once()
        assert agent._state_store["payment_interrupt_retry_tool-123"] == 1


class TestCheckPaymentRetryLimitWithInterrupt:
    """Tests for _check_payment_retry_limit (payment-only, decoupled from interrupt)."""

    def test_returns_false_when_interrupt_limit_reached_but_payment_below(self):
        """Test that _check_payment_retry_limit returns False even when interrupt limit is reached.

        Interrupt limits only gate interrupts in before_tool_call, not 402 payment processing.
        """
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            max_interrupt_retries=2,
        )
        plugin = AgentCorePaymentsPlugin(config=config)
        agent = _create_mock_agent()

        plugin._increment_interrupt_retry_count(agent, "tool-123")
        plugin._increment_interrupt_retry_count(agent, "tool-123")

        event, _ = _create_event_with_agent(
            {
                "tool_use": {"toolUseId": "tool-123"},
                "invocation_state": {},
            },
            agent=agent,
        )

        result = plugin._check_payment_retry_limit(event)
        assert result is False

    def test_returns_false_when_payment_retries_below_limit(self):
        """Test returns False when payment retries are below limit."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            max_interrupt_retries=5,
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        event, agent = _create_event_with_agent(
            {
                "tool_use": {"toolUseId": "tool-123"},
                "invocation_state": {"payment_retry_count_tool-123": 1},
            }
        )

        result = plugin._check_payment_retry_limit(event)
        assert result is False

    def test_max_interrupt_retries_zero_does_not_block_payment_processing(self):
        """Test that max_interrupt_retries=0 does not block 402 payment processing."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
            max_interrupt_retries=0,
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.return_value = {"X-PAYMENT": "base64"}
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, agent = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        assert event.retry is True
        mock_pm_instance.generate_payment_header.assert_called_once()


class TestResetInterruptRetryCount:
    """Tests for _reset_interrupt_retry_count method."""

    def test_reset_clears_state(self):
        """Test that reset deletes the interrupt retry key from agent state."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
        )
        plugin = AgentCorePaymentsPlugin(config=config)
        agent = _create_mock_agent()

        plugin._increment_interrupt_retry_count(agent, "tool-123")
        assert agent._state_store.get("payment_interrupt_retry_tool-123") == 1

        event, _ = _create_event_with_agent(
            {
                "tool_use": {"toolUseId": "tool-123"},
            },
            agent=agent,
        )
        plugin._reset_interrupt_retry_count(event)
        assert "payment_interrupt_retry_tool-123" not in agent._state_store

    def test_reset_no_agent_is_noop(self):
        """Test that reset is a no-op when agent is None."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
        )
        plugin = AgentCorePaymentsPlugin(config=config)
        event = MagicMock()
        event.agent = None
        event.tool_use = {"toolUseId": "tool-123"}
        plugin._reset_interrupt_retry_count(event)

    def test_reset_after_successful_payment(self):
        """Test that successful payment processing resets the interrupt retry count."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
            max_interrupt_retries=3,
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.return_value = {"X-PAYMENT": "base64"}
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        agent = _create_mock_agent()
        # Simulate prior interrupt retries
        plugin._increment_interrupt_retry_count(agent, "tool-123")
        plugin._increment_interrupt_retry_count(agent, "tool-123")
        assert agent._state_store["payment_interrupt_retry_tool-123"] == 2

        # Trigger successful payment processing via after_tool_call
        event, _ = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {},
                "retry": False,
            },
            agent=agent,
        )

        plugin.after_tool_call(event)

        assert event.retry is True
        assert "payment_interrupt_retry_tool-123" not in agent._state_store


class TestAgentCorePaymentsPluginAgentName:
    """Tests for agent_name propagation from config to PaymentManager."""

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_init_agent_passes_agent_name_to_payment_manager(self, mock_payment_manager_class):
        """Test that agent_name from config is passed to PaymentManager."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
            region="us-west-2",
            agent_name="my-agent",
        )

        mock_pm_instance = MagicMock()
        mock_payment_manager_class.return_value = mock_pm_instance

        plugin = AgentCorePaymentsPlugin(config=config)
        mock_agent = MagicMock()

        plugin.init_agent(mock_agent)

        mock_payment_manager_class.assert_called_once_with(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            region_name="us-west-2",
            agent_name="my-agent",
            bearer_token=None,
            token_provider=None,
        )

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_init_agent_passes_none_agent_name_when_not_set(self, mock_payment_manager_class):
        """Test that agent_name defaults to None when not set in config."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            region="us-west-2",
        )

        mock_pm_instance = MagicMock()
        mock_payment_manager_class.return_value = mock_pm_instance

        plugin = AgentCorePaymentsPlugin(config=config)
        mock_agent = MagicMock()

        plugin.init_agent(mock_agent)

        mock_payment_manager_class.assert_called_once_with(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            region_name="us-west-2",
            agent_name=None,
            bearer_token=None,
            token_provider=None,
        )


class TestIsPostPaymentFailure:
    """Tests for _is_post_payment_failure method."""

    def test_returns_false_on_first_attempt(self):
        """Test that first 402 is not treated as a post-payment failure."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        event, _ = _create_event_with_agent(
            {
                "tool_use": {"toolUseId": "tool-123"},
                "invocation_state": {"payment_retry_count_tool-123": 1},
            }
        )

        body = {"error": "invalid_exact_evm_insufficient_balance"}
        assert plugin._is_post_payment_failure(event, body) is False

    def test_returns_true_on_second_attempt_with_non_payment_error(self):
        """Test that a non-'payment required' error on retry is detected as post-payment failure."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        event, _ = _create_event_with_agent(
            {
                "tool_use": {"toolUseId": "tool-123"},
                "invocation_state": {"payment_retry_count_tool-123": 2},
            }
        )

        body = {"error": "invalid_exact_evm_insufficient_balance"}
        assert plugin._is_post_payment_failure(event, body) is True

    def test_returns_false_on_second_attempt_with_payment_required_error(self):
        """Test that 'payment required' error on retry is NOT treated as post-payment failure."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        event, _ = _create_event_with_agent(
            {
                "tool_use": {"toolUseId": "tool-123"},
                "invocation_state": {"payment_retry_count_tool-123": 2},
            }
        )

        body = {"error": "Payment required"}
        assert plugin._is_post_payment_failure(event, body) is False

    def test_returns_false_when_body_is_none(self):
        """Test that None body is not treated as post-payment failure."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        event, _ = _create_event_with_agent(
            {
                "tool_use": {"toolUseId": "tool-123"},
                "invocation_state": {"payment_retry_count_tool-123": 2},
            }
        )

        assert plugin._is_post_payment_failure(event, None) is False

    def test_returns_false_when_body_has_empty_error(self):
        """Test that empty error string is not treated as post-payment failure."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        event, _ = _create_event_with_agent(
            {
                "tool_use": {"toolUseId": "tool-123"},
                "invocation_state": {"payment_retry_count_tool-123": 2},
            }
        )

        body = {"error": ""}
        assert plugin._is_post_payment_failure(event, body) is False

    def test_returns_false_when_body_has_no_error_key(self):
        """Test that body without error key is not treated as post-payment failure."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
        )
        plugin = AgentCorePaymentsPlugin(config=config)

        event, _ = _create_event_with_agent(
            {
                "tool_use": {"toolUseId": "tool-123"},
                "invocation_state": {"payment_retry_count_tool-123": 2},
            }
        )

        body = {"statusCode": 402}
        assert plugin._is_post_payment_failure(event, body) is False


class TestExtractPaymentErrorMessage:
    """Tests for _extract_payment_error_message static method."""

    def test_extracts_error_from_body(self):
        """Test extracting error message from body dict."""
        body = {"error": "invalid_exact_evm_insufficient_balance"}
        assert AgentCorePaymentsPlugin._extract_payment_error_message(body) == "invalid_exact_evm_insufficient_balance"

    def test_returns_unknown_for_none_body(self):
        """Test returns 'unknown error' when body is None."""
        assert AgentCorePaymentsPlugin._extract_payment_error_message(None) == "unknown error"

    def test_returns_unknown_for_empty_error(self):
        """Test returns 'unknown error' when error is empty string."""
        body = {"error": ""}
        assert AgentCorePaymentsPlugin._extract_payment_error_message(body) == "unknown error"

    def test_returns_unknown_for_missing_error_key(self):
        """Test returns 'unknown error' when error key is missing."""
        body = {"statusCode": 402}
        assert AgentCorePaymentsPlugin._extract_payment_error_message(body) == "unknown error"

    def test_returns_unknown_for_non_string_error(self):
        """Test returns 'unknown error' when error is not a string."""
        body = {"error": 42}
        assert AgentCorePaymentsPlugin._extract_payment_error_message(body) == "unknown error"


class TestAfterToolCallPostPaymentFailure:
    """Tests for the post-payment failure path in after_tool_call."""

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_post_payment_failure_stores_failure_and_does_not_retry(self, mock_payment_manager_class):
        """Test that a 402 with non-payment-required error after retry stores failure without retrying."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        mock_pm_instance = MagicMock()
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        # Simulate second 402 after a payment retry (retry count already at 1)
        error_body = {
            "x402Version": 2,
            "error": "invalid_exact_evm_insufficient_balance",
            "resource": {"url": "https://example.com"},
            "accepts": [{"scheme": "exact", "network": "eip155:84532", "amount": "1000"}],
        }
        event, _ = _create_event_with_agent(
            {
                "result": [
                    {"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': error_body})}"}
                ],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {"payment_retry_count_tool-123": 1},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        # Should NOT retry
        assert event.retry is False
        # Should store failure state
        assert "payment_failure_tool-123" in event.invocation_state
        failure = event.invocation_state["payment_failure_tool-123"]
        assert "invalid_exact_evm_insufficient_balance" in failure["exceptionMessage"]
        # Should NOT have called generate_payment_header
        mock_pm_instance.generate_payment_header.assert_not_called()

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_first_402_with_payment_required_error_proceeds_normally(self, mock_payment_manager_class):
        """Test that the first 402 with 'Payment required' error processes payment normally."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.return_value = {"X-PAYMENT": "base64-encoded"}
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        payment_body = {
            "x402Version": 2,
            "error": "Payment required",
            "accepts": [{"scheme": "exact"}],
        }
        event, _ = _create_event_with_agent(
            {
                "result": [
                    {
                        "text": (
                            f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': payment_body})}"
                        )
                    }
                ],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        # Should retry with payment
        assert event.retry is True
        mock_pm_instance.generate_payment_header.assert_called_once()


class TestAfterToolCallAutoPaymentDisabled:
    """Tests for auto_payment=False in after_tool_call."""

    def test_auto_payment_disabled_skips_processing(self):
        """Test that after_tool_call skips payment processing when auto_payment is False."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
            auto_payment=False,
        )

        mock_pm_instance = MagicMock()
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, _ = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        assert event.retry is False
        mock_pm_instance.generate_payment_header.assert_not_called()


class TestAfterToolCallPaymentErrorNoRetry:
    """Tests for PaymentError exception path — should NOT set retry."""

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_payment_error_does_not_set_retry(self, mock_payment_manager_class):
        """Test that PaymentError stores failure but does NOT set event.retry=True."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="payment-instrument-123",
            payment_session_id="payment-session-456",
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.side_effect = PaymentError("Payment processing failed")
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, _ = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        # Should NOT retry on PaymentError
        assert event.retry is False
        # Should store failure
        assert "payment_failure_tool-123" in event.invocation_state


class TestAfterToolCallConfigurationErrors:
    """Tests for PaymentInstrumentConfigurationRequired and PaymentSessionConfigurationRequired in after_tool_call."""

    def test_instrument_config_required_stores_failure_no_retry(self):
        """Test that PaymentInstrumentConfigurationRequired stores failure without retry."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            # No payment_instrument_id — will trigger PaymentInstrumentConfigurationRequired
        )

        mock_pm_instance = MagicMock()
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, _ = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        assert event.retry is False
        assert "payment_failure_tool-123" in event.invocation_state
        failure = event.invocation_state["payment_failure_tool-123"]
        assert failure["exceptionType"] == "PaymentInstrumentConfigurationRequired"

    def test_session_config_required_stores_failure_no_retry(self):
        """Test that PaymentSessionConfigurationRequired stores failure without retry."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="instrument-123",
            # No payment_session_id — will trigger PaymentSessionConfigurationRequired
        )

        mock_pm_instance = MagicMock()
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, _ = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        assert event.retry is False
        assert "payment_failure_tool-123" in event.invocation_state
        failure = event.invocation_state["payment_failure_tool-123"]
        assert failure["exceptionType"] == "PaymentSessionConfigurationRequired"

    @patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager")
    def test_generic_exception_stores_failure_no_retry(self, mock_payment_manager_class):
        """Test that unexpected Exception stores failure without retry."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="instrument-123",
            payment_session_id="session-456",
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.side_effect = RuntimeError("Something unexpected")
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, _ = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "http_request", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        assert event.retry is False
        assert "payment_failure_tool-123" in event.invocation_state
        failure = event.invocation_state["payment_failure_tool-123"]
        assert failure["exceptionType"] == "RuntimeError"


class TestPaymentToolAllowlist:
    """Tests for payment_tool_allowlist behavior in after_tool_call."""

    def test_allowlist_blocks_non_listed_tool(self):
        """Test that a tool not in the allowlist is skipped for payment processing."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="instrument-123",
            payment_session_id="session-456",
            payment_tool_allowlist=["http_request"],
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.return_value = {"X-PAYMENT": "base64token"}
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, _ = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "not_in_allowlist", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {},
                "retry": False,
            }
        )

        plugin.after_tool_call(event)

        assert event.retry is False
        mock_pm_instance.generate_payment_header.assert_not_called()

    def test_allowlist_none_allows_all_tools(self):
        """Test that allowlist=None allows all tools (default behavior)."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="instrument-123",
            payment_session_id="session-456",
            payment_tool_allowlist=None,
        )

        mock_pm_instance = MagicMock()
        mock_pm_instance.generate_payment_header.return_value = {"X-PAYMENT": "base64token"}
        plugin = AgentCorePaymentsPlugin(config=config)
        plugin.payment_manager = mock_pm_instance

        event, _ = _create_event_with_agent(
            {
                "result": [{"text": f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {}})}"}],
                "tool_use": {"name": "any_tool_name", "toolUseId": "tool-123", "input": {"headers": {}}},
                "invocation_state": {},
                "retry": False,
            }
        )

        with patch("bedrock_agentcore.payments.integrations.strands.plugin.get_payment_handler") as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.extract_status_code.return_value = 402
            mock_handler.extract_headers.return_value = {}
            mock_handler.extract_body.return_value = {}
            mock_handler.validate_tool_input.return_value = True
            mock_handler.apply_payment_header.return_value = True
            mock_get_handler.return_value = mock_handler

            plugin.after_tool_call(event)

            mock_handler.extract_status_code.assert_called_once()
            mock_pm_instance.generate_payment_header.assert_called_once()
