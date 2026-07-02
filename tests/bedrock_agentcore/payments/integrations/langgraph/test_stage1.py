"""Tests for LangGraph AgentCorePaymentsConfig and AgentCorePaymentsMiddleware."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bedrock_agentcore.payments.integrations.handlers import GenericPaymentHandler
from bedrock_agentcore.payments.integrations.langgraph import AgentCorePaymentsConfig
from bedrock_agentcore.payments.integrations.langgraph.middleware import AgentCorePaymentsMiddleware

# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestAgentCorePaymentsConfigValidation:
    """Test AgentCorePaymentsConfig field validation."""

    def test_valid_minimal_config(self):
        config = AgentCorePaymentsConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            user_id="user-123",
        )
        assert config.payment_manager_arn.startswith("arn:")
        assert config.user_id == "user-123"
        assert config.auto_payment is True
        assert config.provide_http_request is True
        assert config.post_payment_retry_delay_seconds == 3.0
        assert config.custom_handlers is None

    def test_empty_arn_raises(self):
        with pytest.raises(ValueError, match="payment_manager_arn is required"):
            AgentCorePaymentsConfig(payment_manager_arn="", user_id="u")

    def test_invalid_arn_format_raises(self):
        with pytest.raises(ValueError, match="Invalid ARN format"):
            AgentCorePaymentsConfig(payment_manager_arn="not-an-arn", user_id="u")

    def test_user_id_required_for_sigv4(self):
        with pytest.raises(ValueError, match="user_id is required for SigV4"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1"
            )

    def test_user_id_optional_with_bearer_token(self):
        config = AgentCorePaymentsConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            bearer_token="jwt-token",
        )
        assert config.user_id is None

    def test_user_id_optional_with_token_provider(self):
        config = AgentCorePaymentsConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            token_provider=lambda: "fresh",
        )
        assert config.user_id is None

    def test_whitespace_user_id_raises(self):
        with pytest.raises(ValueError, match="user_id cannot be whitespace-only"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="   ",
            )

    def test_bearer_token_and_token_provider_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="u",
                bearer_token="tok",
                token_provider=lambda: "tok",
            )

    def test_bearer_token_must_be_string(self):
        with pytest.raises(ValueError, match="bearer_token must be a string"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="u",
                bearer_token=123,  # type: ignore
            )

    def test_token_provider_must_be_callable(self):
        with pytest.raises(ValueError, match="token_provider must be callable"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="u",
                token_provider="not-callable",  # type: ignore
            )

    def test_auto_payment_must_be_bool(self):
        with pytest.raises(ValueError, match="auto_payment must be a boolean"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="u",
                auto_payment="yes",  # type: ignore
            )

    def test_provide_http_request_must_be_bool(self):
        with pytest.raises(ValueError, match="provide_http_request must be a boolean"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="u",
                provide_http_request=1,  # type: ignore
            )

    def test_allowlist_must_be_list(self):
        with pytest.raises(ValueError, match="payment_tool_allowlist must be a list"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="u",
                payment_tool_allowlist="http_request",  # type: ignore
            )

    def test_allowlist_entries_must_be_strings(self):
        with pytest.raises(ValueError, match="All entries in payment_tool_allowlist must be strings"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="u",
                payment_tool_allowlist=["ok", 123],  # type: ignore
            )

    def test_delay_must_be_numeric(self):
        with pytest.raises(ValueError, match="post_payment_retry_delay_seconds must be a number"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="u",
                post_payment_retry_delay_seconds="3",  # type: ignore
            )

    def test_delay_must_be_non_negative(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="u",
                post_payment_retry_delay_seconds=-1,
            )

    def test_delay_bool_rejected(self):
        """Booleans are technically int subclass but should be rejected."""
        with pytest.raises(ValueError, match="post_payment_retry_delay_seconds must be a number"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="u",
                post_payment_retry_delay_seconds=True,  # type: ignore
            )

    def test_custom_handlers_must_be_dict(self):
        with pytest.raises(ValueError, match="custom_handlers must be a dict"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="u",
                custom_handlers=["not", "a", "dict"],  # type: ignore
            )

    def test_custom_handlers_keys_must_be_strings(self):
        with pytest.raises(ValueError, match="All keys in custom_handlers must be strings"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="u",
                custom_handlers={123: GenericPaymentHandler()},  # type: ignore
            )

    def test_custom_handlers_values_must_be_handler_instances(self):
        with pytest.raises(ValueError, match="All values in custom_handlers must be PaymentResponseHandler"):
            AgentCorePaymentsConfig(
                payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
                user_id="u",
                custom_handlers={"my_tool": "not-a-handler"},  # type: ignore
            )

    def test_custom_handlers_valid(self):
        handler = GenericPaymentHandler()
        config = AgentCorePaymentsConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            user_id="u",
            custom_handlers={"my_tool": handler},
        )
        assert config.custom_handlers == {"my_tool": handler}

    def test_valid_full_config(self):
        config = AgentCorePaymentsConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            user_id="user-1",
            payment_instrument_id="instr-1",
            payment_session_id="sess-1",
            payment_connector_id="conn-1",
            region="us-west-2",
            network_preferences_config=["eip155:84532"],
            auto_payment=False,
            agent_name="my-agent",
            payment_tool_allowlist=["http_request"],
            provide_http_request=False,
            post_payment_retry_delay_seconds=5.0,
        )
        assert config.auto_payment is False
        assert config.provide_http_request is False
        assert config.post_payment_retry_delay_seconds == 5.0
        assert config.payment_tool_allowlist == ["http_request"]


# ---------------------------------------------------------------------------
# Middleware instantiation tests
# ---------------------------------------------------------------------------


class TestAgentCorePaymentsMiddlewareInstantiation:
    """Test middleware creation and PaymentManager initialization."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_middleware_creates_payment_manager(self, mock_pm_cls):
        """PaymentManager is created with config values during __init__."""
        config = AgentCorePaymentsConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            user_id="user-1",
            region="us-west-2",
            agent_name="test-agent",
        )
        mw = AgentCorePaymentsMiddleware(config)

        mock_pm_cls.assert_called_once_with(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            region_name="us-west-2",
            agent_name="test-agent",
            bearer_token=None,
            token_provider=None,
        )
        assert mw.config is config
        assert mw.payment_manager is mock_pm_cls.return_value

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_middleware_passes_bearer_token(self, mock_pm_cls):
        config = AgentCorePaymentsConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            bearer_token="my-jwt",
        )
        AgentCorePaymentsMiddleware(config)
        mock_pm_cls.assert_called_once_with(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            region_name=None,
            agent_name=None,
            bearer_token="my-jwt",
            token_provider=None,
        )

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_middleware_raises_runtime_error_on_pm_failure(self, mock_pm_cls):
        """RuntimeError raised if PaymentManager constructor throws."""
        mock_pm_cls.side_effect = Exception("boto3 broke")
        config = AgentCorePaymentsConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            user_id="u",
        )
        with pytest.raises(RuntimeError, match="Failed to initialize PaymentManager"):
            AgentCorePaymentsMiddleware(config)


# ---------------------------------------------------------------------------
# Pass-through behavior tests
# ---------------------------------------------------------------------------


class TestAgentCorePaymentsMiddlewarePassThrough:
    """Test that wrap_tool_call and awrap_tool_call pass through correctly."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_wrap_tool_call_passes_through(self, mock_pm_cls):
        """Sync wrap_tool_call calls handler and returns its result."""
        config = AgentCorePaymentsConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            user_id="u",
        )
        mw = AgentCorePaymentsMiddleware(config)

        mock_request = MagicMock()
        mock_result = MagicMock()
        mock_handler = MagicMock(return_value=mock_result)

        result = mw.wrap_tool_call(mock_request, mock_handler)

        mock_handler.assert_called_once_with(mock_request)
        assert result is mock_result

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_awrap_tool_call_passes_through(self, mock_pm_cls):
        """Async awrap_tool_call awaits handler and returns its result."""
        config = AgentCorePaymentsConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            user_id="u",
        )
        mw = AgentCorePaymentsMiddleware(config)

        mock_request = MagicMock()
        mock_result = MagicMock()
        mock_handler = AsyncMock(return_value=mock_result)

        result = asyncio.run(mw.awrap_tool_call(mock_request, mock_handler))

        mock_handler.assert_called_once_with(mock_request)
        assert result is mock_result

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_wrap_tool_call_propagates_exceptions(self, mock_pm_cls):
        """Exceptions from handler propagate through wrap_tool_call."""
        config = AgentCorePaymentsConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            user_id="u",
        )
        mw = AgentCorePaymentsMiddleware(config)

        mock_request = MagicMock()
        mock_handler = MagicMock(side_effect=RuntimeError("tool exploded"))

        with pytest.raises(RuntimeError, match="tool exploded"):
            mw.wrap_tool_call(mock_request, mock_handler)

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_awrap_tool_call_propagates_exceptions(self, mock_pm_cls):
        """Exceptions from async handler propagate through awrap_tool_call."""
        config = AgentCorePaymentsConfig(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
            user_id="u",
        )
        mw = AgentCorePaymentsMiddleware(config)

        mock_request = MagicMock()
        mock_handler = AsyncMock(side_effect=ValueError("async boom"))

        with pytest.raises(ValueError, match="async boom"):
            asyncio.run(mw.awrap_tool_call(mock_request, mock_handler))
