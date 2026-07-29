"""Tests for auto_session lazy session creation in AgentCorePaymentsMiddleware."""

import json
from unittest.mock import MagicMock, patch

from langchain.messages import ToolMessage

from bedrock_agentcore.payments.integrations.langgraph import AgentCorePaymentsConfig
from bedrock_agentcore.payments.integrations.langgraph.middleware import AgentCorePaymentsMiddleware


def _make_config(**overrides):
    defaults = {
        "payment_manager_arn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
        "user_id": "user-1",
        "payment_instrument_id": "instr-1",
        "payment_session_id": None,  # no session pre-configured
        "auto_session": True,
        "auto_session_budget": "5.00",
        "auto_session_expiry_minutes": 30,
        "post_payment_retry_delay_seconds": 0,
    }
    defaults.update(overrides)
    return AgentCorePaymentsConfig(**defaults)


def _make_request(tool_name="http_request", tool_args=None, tool_id="tc-1"):
    req = MagicMock()
    req.tool_call = {
        "name": tool_name,
        "args": tool_args if tool_args is not None else {"url": "http://paid-api.com", "headers": {}},
        "id": tool_id,
    }
    return req


def _402_content():
    payload = json.dumps({"statusCode": 402, "headers": {"x-pay": "v"}, "body": {"x402Version": 1}})
    return f"PAYMENT_REQUIRED: {payload}"


def _200_content():
    return json.dumps({"statusCode": 200, "body": {"data": "paid content"}})


class TestAutoSession:
    """Test auto_session lazy session creation on first 402."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_auto_session_creates_session_on_first_402(self, mock_pm_cls):
        """When auto_session=True and no session_id, a session is created on 402."""
        mock_pm = mock_pm_cls.return_value
        mock_pm.create_payment_session.return_value = {"paymentSessionId": "auto-sess-123"}
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config()
        mw = AgentCorePaymentsMiddleware(config)

        assert config.payment_session_id is None

        request = _make_request(tool_args={"url": "http://paid-api.com", "headers": {}})
        call_count = [0]

        def mock_handler(req):
            call_count[0] += 1
            if call_count[0] == 1:
                return ToolMessage(content=_402_content(), tool_call_id="tc-1")
            return ToolMessage(content=_200_content(), tool_call_id="tc-1")

        result = mw.wrap_tool_call(request, mock_handler)

        # Session was created
        mock_pm.create_payment_session.assert_called_once_with(
            user_id="user-1",
            limits={"maxSpendAmount": {"value": "5.00", "currency": "USD"}},
            expiry_time_in_minutes=30,
        )
        # Config was mutated with the new session ID
        assert config.payment_session_id == "auto-sess-123"
        # Payment header was generated and tool was retried
        mock_pm.generate_payment_header.assert_called_once()
        assert call_count[0] == 2
        # Final result is the 200 response
        assert "paid content" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_auto_session_uses_session_for_subsequent_calls(self, mock_pm_cls):
        """After auto_session creates a session, subsequent 402s reuse it."""
        mock_pm = mock_pm_cls.return_value
        mock_pm.create_payment_session.return_value = {"paymentSessionId": "auto-sess-456"}
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config()
        mw = AgentCorePaymentsMiddleware(config)

        # First tool call — triggers session creation
        request1 = _make_request(tool_args={"url": "http://api1.com", "headers": {}})
        mw.wrap_tool_call(
            request1,
            MagicMock(
                side_effect=[
                    ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                    ToolMessage(content=_200_content(), tool_call_id="tc-1"),
                ]
            ),
        )

        # Second tool call — session already exists, no new creation
        request2 = _make_request(tool_args={"url": "http://api2.com", "headers": {}}, tool_id="tc-2")
        mw.wrap_tool_call(
            request2,
            MagicMock(
                side_effect=[
                    ToolMessage(content=_402_content(), tool_call_id="tc-2"),
                    ToolMessage(content=_200_content(), tool_call_id="tc-2"),
                ]
            ),
        )

        # create_payment_session only called once (first 402)
        assert mock_pm.create_payment_session.call_count == 1
        # generate_payment_header called twice (once per 402)
        assert mock_pm.generate_payment_header.call_count == 2

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_auto_session_disabled_raises_without_session(self, mock_pm_cls):
        """When auto_session=False and no session_id, 402 produces a PAYMENT ERROR."""
        mock_pm = mock_pm_cls.return_value

        config = _make_config(auto_session=False)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://paid-api.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, mock_handler)

        # Should produce a payment error, not create a session
        assert isinstance(result, ToolMessage)
        assert "PAYMENT ERROR" in result.content
        mock_pm.create_payment_session.assert_not_called()

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_auto_session_passes_budget_and_expiry(self, mock_pm_cls):
        """auto_session uses configured budget and expiry values."""
        mock_pm = mock_pm_cls.return_value
        mock_pm.create_payment_session.return_value = {"paymentSessionId": "sess-custom"}
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config(auto_session_budget="10.00", auto_session_expiry_minutes=120)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mw.wrap_tool_call(
            request,
            MagicMock(
                side_effect=[
                    ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                    ToolMessage(content=_200_content(), tool_call_id="tc-1"),
                ]
            ),
        )

        mock_pm.create_payment_session.assert_called_once_with(
            user_id="user-1",
            limits={"maxSpendAmount": {"value": "10.00", "currency": "USD"}},
            expiry_time_in_minutes=120,
        )

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_auto_session_skipped_when_session_already_set(self, mock_pm_cls):
        """If payment_session_id is already set, auto_session does nothing."""
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config(payment_session_id="pre-existing-sess")
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mw.wrap_tool_call(
            request,
            MagicMock(
                side_effect=[
                    ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                    ToolMessage(content=_200_content(), tool_call_id="tc-1"),
                ]
            ),
        )

        # No session creation since one was already configured
        mock_pm.create_payment_session.assert_not_called()
        # But payment still processed
        mock_pm.generate_payment_header.assert_called_once()
