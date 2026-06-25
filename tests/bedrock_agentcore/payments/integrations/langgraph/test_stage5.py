"""Tests for Stage 5: Async awrap_tool_call."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from langchain.messages import ToolMessage
from langgraph.types import Command

from bedrock_agentcore.payments.integrations.langgraph import AgentCorePaymentsConfig
from bedrock_agentcore.payments.integrations.langgraph.middleware import AgentCorePaymentsMiddleware


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


def _200_content():
    return json.dumps({"statusCode": 200, "body": {"data": "paid"}})


# ---------------------------------------------------------------------------
# Basic async pass-through
# ---------------------------------------------------------------------------


class TestAsyncPassThrough:
    """Test basic async behavior for non-payment cases."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_non_402_passes_through(self, mock_pm_cls):
        mw = AgentCorePaymentsMiddleware(_make_config())
        tool_msg = ToolMessage(content="normal output", tool_call_id="tc-1")
        handler = AsyncMock(return_value=tool_msg)

        result = asyncio.run(mw.awrap_tool_call(_make_request(), handler))
        assert result is tool_msg

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_command_passes_through(self, mock_pm_cls):
        mw = AgentCorePaymentsMiddleware(_make_config())
        cmd = Command(update={"k": "v"})
        handler = AsyncMock(return_value=cmd)

        result = asyncio.run(mw.awrap_tool_call(_make_request(), handler))
        assert result is cmd


# ---------------------------------------------------------------------------
# Full retry flow
# ---------------------------------------------------------------------------


class TestAsyncRetryFlow:
    """Test 402 → sign → retry → 200 async flow."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_402_then_200_on_retry(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.return_value = {"X-PAYMENT": "sig"}
        mw = AgentCorePaymentsMiddleware(_make_config())

        success_msg = ToolMessage(content=_200_content(), tool_call_id="tc-1")
        handler = AsyncMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                success_msg,
            ]
        )

        result = asyncio.run(
            mw.awrap_tool_call(
                _make_request(tool_args={"url": "http://x.com", "headers": {}}),
                handler,
            )
        )
        assert result is success_msg

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_handler_awaited_twice(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.return_value = {"X-PAYMENT": "sig"}
        mw = AgentCorePaymentsMiddleware(_make_config())

        handler = AsyncMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=_200_content(), tool_call_id="tc-1"),
            ]
        )

        asyncio.run(mw.awrap_tool_call(_make_request(tool_args={"url": "http://x.com", "headers": {}}), handler))
        assert handler.await_count == 2


# ---------------------------------------------------------------------------
# asyncio.sleep verification
# ---------------------------------------------------------------------------


class TestAsyncSleepUsed:
    """Verify asyncio.sleep is used (not time.sleep)."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_asyncio_sleep_called(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.return_value = {"X-PAYMENT": "sig"}
        config = _make_config(post_payment_retry_delay_seconds=3.0)
        mw = AgentCorePaymentsMiddleware(config)

        handler = AsyncMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=_200_content(), tool_call_id="tc-1"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_async_sleep:
            asyncio.run(mw.awrap_tool_call(_make_request(tool_args={"url": "http://x.com", "headers": {}}), handler))
            mock_async_sleep.assert_called_once_with(3.0)

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.time.sleep")
    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_time_sleep_not_called(self, mock_pm_cls, mock_time_sleep):
        mock_pm_cls.return_value.generate_payment_header.return_value = {"X-PAYMENT": "sig"}
        config = _make_config(post_payment_retry_delay_seconds=3.0)
        mw = AgentCorePaymentsMiddleware(config)

        handler = AsyncMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=_200_content(), tool_call_id="tc-1"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(mw.awrap_tool_call(_make_request(tool_args={"url": "http://x.com", "headers": {}}), handler))

        mock_time_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# asyncio.to_thread verification
# ---------------------------------------------------------------------------


class TestAsyncToThread:
    """Verify _generate_payment_header runs via asyncio.to_thread."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_generate_header_runs_in_thread(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.return_value = {"X-PAYMENT": "sig"}
        mw = AgentCorePaymentsMiddleware(_make_config())

        handler = AsyncMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=_200_content(), tool_call_id="tc-1"),
            ]
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value={"X-PAYMENT": "sig"}) as mock_to_thread:
            asyncio.run(mw.awrap_tool_call(_make_request(tool_args={"url": "http://x.com", "headers": {}}), handler))
            mock_to_thread.assert_called_once()
            # First arg is the method
            assert mock_to_thread.call_args[0][0] == mw._generate_payment_header


# ---------------------------------------------------------------------------
# Async error handling
# ---------------------------------------------------------------------------


class TestAsyncErrorHandling:
    """Async path produces same error messages as sync path."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_missing_instrument_error(self, mock_pm_cls):
        config = _make_config(payment_instrument_id=None)
        mw = AgentCorePaymentsMiddleware(config)

        handler = AsyncMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = asyncio.run(
            mw.awrap_tool_call(
                _make_request(tool_args={"url": "http://x.com", "headers": {}}),
                handler,
            )
        )
        assert "PAYMENT ERROR" in result.content
        assert "No payment instrument configured" in result.content
        assert result.status == "error"

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_post_payment_rejection(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.return_value = {"X-PAYMENT": "sig"}
        mw = AgentCorePaymentsMiddleware(_make_config())

        handler = AsyncMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
            ]
        )

        result = asyncio.run(
            mw.awrap_tool_call(
                _make_request(tool_args={"url": "http://x.com", "headers": {}}),
                handler,
            )
        )
        assert "signed but rejected" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_unexpected_exception(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.side_effect = RuntimeError("async boom")
        mw = AgentCorePaymentsMiddleware(_make_config())

        handler = AsyncMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = asyncio.run(
            mw.awrap_tool_call(
                _make_request(tool_args={"url": "http://x.com", "headers": {}}),
                handler,
            )
        )
        assert isinstance(result, ToolMessage)
        assert "unexpected error" in result.content
        assert "async boom" in result.content


# ---------------------------------------------------------------------------
# Async guards
# ---------------------------------------------------------------------------


class TestAsyncGuards:
    """Guards work identically in async path."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_auto_payment_false_skips(self, mock_pm_cls):
        config = _make_config(auto_payment=False)
        mw = AgentCorePaymentsMiddleware(config)

        tool_msg = ToolMessage(content=_402_content(), tool_call_id="tc-1")
        handler = AsyncMock(return_value=tool_msg)

        result = asyncio.run(mw.awrap_tool_call(_make_request(), handler))
        assert result is tool_msg
        assert "PAYMENT ERROR" not in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_allowlist_skips(self, mock_pm_cls):
        config = _make_config(payment_tool_allowlist=["other_tool"])
        mw = AgentCorePaymentsMiddleware(config)

        tool_msg = ToolMessage(content=_402_content(), tool_call_id="tc-1")
        handler = AsyncMock(return_value=tool_msg)

        result = asyncio.run(mw.awrap_tool_call(_make_request(tool_name="http_request"), handler))
        assert result is tool_msg
