"""Tests for Stage 7: Error Handler Callback."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.messages import ToolMessage
from langgraph.types import Command

from bedrock_agentcore.payments.integrations.langgraph.config import AgentCorePaymentsConfig
from bedrock_agentcore.payments.integrations.langgraph.errors import ErrorResolution, PaymentErrorContext
from bedrock_agentcore.payments.integrations.langgraph.middleware import AgentCorePaymentsMiddleware
from bedrock_agentcore.payments.manager import (
    InsufficientBudget,
    PaymentError,
    PaymentInstrumentConfigurationRequired,
    PaymentSessionExpired,
)


def _402_content():
    payload = json.dumps({"statusCode": 402, "headers": {}, "body": {"x402Version": 1}})
    return f"PAYMENT_REQUIRED: {payload}"


def _200_content():
    return json.dumps({"statusCode": 200, "body": {"data": "paid"}})


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
    req.tool_call = {"name": tool_name, "args": tool_args if tool_args is not None else {"url": "http://x.com", "headers": {}}, "id": tool_id}
    return req


# ---------------------------------------------------------------------------
# Basic callback flow
# ---------------------------------------------------------------------------


class TestErrorHandlerBasicFlow:
    """Callback invoked on payment errors and can resolve them."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_callback_fixes_missing_instrument(self, mock_pm_cls):
        """Callback sets instrument_id, returns RETRY → payment succeeds."""
        mock_pm = mock_pm_cls.return_value
        # First call fails (no instrument), after callback fixes it, second call succeeds
        mock_pm.generate_payment_header.side_effect = [
            PaymentInstrumentConfigurationRequired("missing"),
            {"X-PAYMENT": "sig"},
        ]

        def handler_cb(ctx):
            ctx.config.payment_instrument_id = "fixed-instr"
            return ErrorResolution.RETRY

        config = _make_config(payment_instrument_id=None, on_payment_error=handler_cb)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(side_effect=[
            ToolMessage(content=_402_content(), tool_call_id="tc-1"),
            ToolMessage(content=_200_content(), tool_call_id="tc-1"),
        ])

        result = mw.wrap_tool_call(request, mock_handler)
        assert "PAYMENT ERROR" not in result.content
        assert json.loads(result.content)["statusCode"] == 200

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_callback_not_invoked_when_none(self, mock_pm_cls):
        """No callback → standard error ToolMessage."""
        mock_pm_cls.return_value.generate_payment_header.side_effect = PaymentError("fail")
        config = _make_config(on_payment_error=None)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, mock_handler)
        assert "PAYMENT ERROR" in result.content


# ---------------------------------------------------------------------------
# PROPAGATE resolution
# ---------------------------------------------------------------------------


class TestPropagateResolution:
    """Callback returns PROPAGATE or custom string → error message to LLM."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_propagate_falls_through(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.side_effect = InsufficientBudget("over")
        config = _make_config(on_payment_error=lambda ctx: ErrorResolution.PROPAGATE)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, mock_handler)
        assert "Insufficient budget" in result.content
        assert "Do not retry" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_string_return_custom_message(self, mock_pm_cls):
        """Returning a string sends that custom message to the LLM."""
        mock_pm_cls.return_value.generate_payment_header.side_effect = PaymentError("fail")

        def cb(ctx):
            return "Please visit https://myapp.com/setup to configure your wallet."

        config = _make_config(on_payment_error=cb)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, mock_handler)
        assert "PAYMENT ERROR" in result.content
        assert "https://myapp.com/setup" in result.content
        assert result.status == "error"
        assert result.tool_call_id == "tc-1"


# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------


class TestRetryLoop:
    """Callback can retry multiple times up to max_error_retries."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_max_retries_exhausted(self, mock_pm_cls):
        """Callback always retries but PM always fails → exhausts max retries."""
        mock_pm_cls.return_value.generate_payment_header.side_effect = PaymentError("always fails")
        config = _make_config(on_payment_error=lambda ctx: ErrorResolution.RETRY, max_error_retries=3)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, mock_handler)
        assert "PAYMENT ERROR" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_retry_count_increments(self, mock_pm_cls):
        """retry_count passed to callback increments each time."""
        mock_pm_cls.return_value.generate_payment_header.side_effect = PaymentError("fail")
        counts = []

        def cb(ctx):
            counts.append(ctx.retry_count)
            return ErrorResolution.RETRY

        config = _make_config(on_payment_error=cb, max_error_retries=3)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        mw.wrap_tool_call(request, mock_handler)
        assert counts == [0, 1, 2]


# ---------------------------------------------------------------------------
# Exception safety
# ---------------------------------------------------------------------------


class TestCallbackExceptionSafety:
    """Buggy callbacks don't crash the agent."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_callback_raises(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.side_effect = PaymentError("orig")

        def bad_cb(ctx):
            raise RuntimeError("callback bug")

        config = _make_config(on_payment_error=bad_cb)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, mock_handler)
        # Falls through to original error message, no crash
        assert "PAYMENT ERROR" in result.content
        assert "orig" in result.content or "Payment processing failed" in result.content


# ---------------------------------------------------------------------------
# Context populated correctly
# ---------------------------------------------------------------------------


class TestContextPopulated:
    """PaymentErrorContext has correct data."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_context_fields(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.side_effect = PaymentSessionExpired("expired!")
        captured = []

        def cb(ctx):
            captured.append(ctx)
            return ErrorResolution.PROPAGATE

        config = _make_config(on_payment_error=cb)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_name="my_tool", tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        mw.wrap_tool_call(request, mock_handler)

        ctx = captured[0]
        assert ctx.exception_type == "PaymentSessionExpired"
        assert ctx.exception_message == "expired!"
        assert ctx.tool_name == "my_tool"
        assert ctx.tool_args == {"url": "http://x.com", "headers": {}}
        assert ctx.config is config
        assert ctx.retry_count == 0
        assert ctx.payment_required_request is not None
        assert ctx.payment_required_request["statusCode"] == 402


# ---------------------------------------------------------------------------
# Async callback support
# ---------------------------------------------------------------------------


class TestAsyncErrorHandler:
    """Async callbacks are awaited correctly."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_async_callback_awaited(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.side_effect = [
            PaymentError("first fail"),
            {"X-PAYMENT": "sig"},
        ]

        async def async_cb(ctx):
            ctx.config.payment_session_id = "new-sess"
            return ErrorResolution.RETRY

        config = _make_config(on_payment_error=async_cb)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        handler = AsyncMock(side_effect=[
            ToolMessage(content=_402_content(), tool_call_id="tc-1"),
            ToolMessage(content=_200_content(), tool_call_id="tc-1"),
        ])

        result = asyncio.run(mw.awrap_tool_call(request, handler))
        assert "PAYMENT ERROR" not in result.content
        assert json.loads(result.content)["statusCode"] == 200

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_sync_callback_in_async_path(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.side_effect = [
            PaymentError("fail"),
            {"X-PAYMENT": "sig"},
        ]

        def sync_cb(ctx):
            return ErrorResolution.RETRY

        config = _make_config(on_payment_error=sync_cb)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        handler = AsyncMock(side_effect=[
            ToolMessage(content=_402_content(), tool_call_id="tc-1"),
            ToolMessage(content=_200_content(), tool_call_id="tc-1"),
        ])

        result = asyncio.run(mw.awrap_tool_call(request, handler))
        assert "PAYMENT ERROR" not in result.content


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """New config fields are validated."""

    def test_on_payment_error_must_be_callable(self):
        with pytest.raises(ValueError, match="on_payment_error must be callable"):
            _make_config(on_payment_error="not callable")

    def test_on_payment_error_none_is_valid(self):
        config = _make_config(on_payment_error=None)
        assert config.on_payment_error is None

    def test_max_error_retries_must_be_int(self):
        with pytest.raises(ValueError, match="max_error_retries must be an int"):
            _make_config(max_error_retries="three")

    def test_max_error_retries_must_be_non_negative(self):
        with pytest.raises(ValueError, match="max_error_retries must be >= 0"):
            _make_config(max_error_retries=-1)

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_max_error_retries_zero_disables_callback(self, mock_pm_cls):
        """max_error_retries=0 means callback is never invoked."""
        mock_pm_cls.return_value.generate_payment_header.side_effect = PaymentError("fail")
        called = []

        def cb(ctx):
            called.append(True)
            return ErrorResolution.RETRY

        config = _make_config(on_payment_error=cb, max_error_retries=0)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, mock_handler)
        assert called == []
        assert "PAYMENT ERROR" in result.content


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """No regression when on_payment_error is None."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_no_callback_instrument_missing(self, mock_pm_cls):
        config = _make_config(payment_instrument_id=None, on_payment_error=None)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, mock_handler)
        assert "No payment instrument configured" in result.content
        assert "Do not retry" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_no_callback_session_expired(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.side_effect = PaymentSessionExpired("exp")
        config = _make_config(on_payment_error=None)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, mock_handler)
        assert "Payment session has expired" in result.content
