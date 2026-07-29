"""Tests for Stage 4: Error Handling — deterministic messages + broad exception guard."""

import json
from unittest.mock import MagicMock, patch

from langchain.messages import ToolMessage

from bedrock_agentcore.payments.integrations.langgraph import AgentCorePaymentsConfig
from bedrock_agentcore.payments.integrations.langgraph.middleware import AgentCorePaymentsMiddleware
from bedrock_agentcore.payments.manager import (
    InsufficientBudget,
    PaymentError,
    PaymentInstrumentNotFound,
    PaymentSessionExpired,
    PaymentSessionNotFound,
)


def _make_config(**overrides):
    defaults = {
        "payment_manager_arn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
        "user_id": "user-1",
        "payment_instrument_id": "instr-1",
        "payment_session_id": "sess-1",
        "post_payment_retry_delay_seconds": 0,
    }
    defaults.update(overrides)
    return AgentCorePaymentsConfig(**defaults)


def _make_request(tool_name="http_request", tool_args=None, tool_id="tc-1"):
    req = MagicMock()
    req.tool_call = {
        "name": tool_name,
        "args": tool_args if tool_args is not None else {"url": "http://x.com", "headers": {}},
        "id": tool_id,
    }
    return req


def _402_content():
    payload = json.dumps({"statusCode": 402, "headers": {}, "body": {"x402Version": 1}})
    return f"PAYMENT_REQUIRED: {payload}"


# ---------------------------------------------------------------------------
# Deterministic error message tests
# ---------------------------------------------------------------------------


class TestDeterministicErrorMessages:
    """Each exception type produces the correct deterministic message."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_instrument_config_required(self, mock_pm_cls):
        config = _make_config(payment_instrument_id=None)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, handler)
        assert "No payment instrument configured" in result.content
        assert "Do not retry this call" in result.content
        assert result.status == "error"

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_session_config_required(self, mock_pm_cls):
        config = _make_config(payment_session_id=None)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, handler)
        assert "No payment session configured" in result.content
        assert "Do not retry this call" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_instrument_not_found(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.side_effect = PaymentInstrumentNotFound("not found")
        mw = AgentCorePaymentsMiddleware(_make_config())

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, handler)
        assert "Payment instrument not found" in result.content
        assert "Do not retry this call" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_session_not_found(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.side_effect = PaymentSessionNotFound("gone")
        mw = AgentCorePaymentsMiddleware(_make_config())

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, handler)
        assert "Payment session not found" in result.content
        assert "Do not retry this call" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_session_expired(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.side_effect = PaymentSessionExpired("expired")
        mw = AgentCorePaymentsMiddleware(_make_config())

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, handler)
        assert "Payment session has expired" in result.content
        assert "Do not retry this call" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_insufficient_budget(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.side_effect = InsufficientBudget("over limit")
        mw = AgentCorePaymentsMiddleware(_make_config())

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, handler)
        assert "Insufficient budget" in result.content
        assert "Do not retry this call" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_generic_payment_error(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.side_effect = PaymentError("something broke")
        mw = AgentCorePaymentsMiddleware(_make_config())

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, handler)
        assert "Payment processing failed" in result.content
        assert "something broke" in result.content
        assert "Do not retry this call" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_post_payment_rejection(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.return_value = {"X-PAYMENT": "sig"}
        mw = AgentCorePaymentsMiddleware(_make_config())

        payload_with_error = json.dumps({"statusCode": 402, "headers": {}, "body": {"error": "bad_sig"}})
        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        handler = MagicMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=f"PAYMENT_REQUIRED: {payload_with_error}", tool_call_id="tc-1"),
            ]
        )

        result = mw.wrap_tool_call(request, handler)
        assert "signed but rejected" in result.content
        assert "bad_sig" in result.content
        assert "Do not retry this call" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_tool_input_validation_failed(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.return_value = {"X-PAYMENT": "sig"}
        mw = AgentCorePaymentsMiddleware(_make_config())

        request = _make_request(tool_args="not-a-dict")
        handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, handler)
        assert "Could not apply payment credentials" in result.content
        assert "Do not retry this call" in result.content


# ---------------------------------------------------------------------------
# Broad exception guard tests
# ---------------------------------------------------------------------------


class TestUnexpectedExceptionHandling:
    """Unexpected exceptions are caught and returned as error ToolMessages."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_unexpected_runtime_error(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.side_effect = RuntimeError("boom")
        mw = AgentCorePaymentsMiddleware(_make_config())

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        # Should NOT raise — returns error ToolMessage
        result = mw.wrap_tool_call(request, handler)
        assert isinstance(result, ToolMessage)
        assert "unexpected error" in result.content
        assert "boom" in result.content
        assert "Do not retry this call" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_unexpected_type_error(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.side_effect = TypeError("bad type")
        mw = AgentCorePaymentsMiddleware(_make_config())

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, handler)
        assert isinstance(result, ToolMessage)
        assert "unexpected error" in result.content
        assert "bad type" in result.content


# ---------------------------------------------------------------------------
# Guard regression tests
# ---------------------------------------------------------------------------


class TestGuardRegression:
    """Guards bypass payment processing entirely — no error messages leak."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_auto_payment_false_no_error(self, mock_pm_cls):
        config = _make_config(auto_payment=False)
        mw = AgentCorePaymentsMiddleware(config)

        tool_msg = ToolMessage(content=_402_content(), tool_call_id="tc-1")
        request = _make_request()
        handler = MagicMock(return_value=tool_msg)

        result = mw.wrap_tool_call(request, handler)
        assert result is tool_msg
        assert "PAYMENT ERROR" not in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_not_in_allowlist_no_error(self, mock_pm_cls):
        config = _make_config(payment_tool_allowlist=["other_tool"])
        mw = AgentCorePaymentsMiddleware(config)

        tool_msg = ToolMessage(content=_402_content(), tool_call_id="tc-1")
        request = _make_request(tool_name="http_request")
        handler = MagicMock(return_value=tool_msg)

        result = mw.wrap_tool_call(request, handler)
        assert result is tool_msg
        assert "PAYMENT ERROR" not in result.content
