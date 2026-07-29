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


# ---------------------------------------------------------------------------
# Async auto_session
# ---------------------------------------------------------------------------


class TestAsyncAutoSession:
    """Test auto_session in the async path."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_auto_session_creates_session_on_first_402_async(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.create_payment_session.return_value = {"paymentSessionId": "async-sess-1"}
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config(payment_session_id=None, auto_session=True, auto_session_budget="5.00")
        mw = AgentCorePaymentsMiddleware(config)

        handler = AsyncMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=_200_content(), tool_call_id="tc-1"),
            ]
        )

        result = asyncio.run(
            mw.awrap_tool_call(_make_request(tool_args={"url": "http://x.com", "headers": {}}), handler)
        )

        mock_pm.create_payment_session.assert_called_once()
        assert config.payment_session_id == "async-sess-1"
        assert "paid" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_auto_session_reuses_session_async(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.create_payment_session.return_value = {"paymentSessionId": "async-sess-2"}
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config(payment_session_id=None, auto_session=True)
        mw = AgentCorePaymentsMiddleware(config)

        # First call
        handler1 = AsyncMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=_200_content(), tool_call_id="tc-1"),
            ]
        )
        asyncio.run(mw.awrap_tool_call(_make_request(tool_args={"url": "http://a.com", "headers": {}}), handler1))

        # Second call — session already exists
        handler2 = AsyncMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-2"),
                ToolMessage(content=_200_content(), tool_call_id="tc-2"),
            ]
        )
        asyncio.run(
            mw.awrap_tool_call(
                _make_request(tool_args={"url": "http://b.com", "headers": {}}, tool_id="tc-2"),
                handler2,
            )
        )

        assert mock_pm.create_payment_session.call_count == 1


# ---------------------------------------------------------------------------
# Async post-payment rejection with raw JSON fallback
# ---------------------------------------------------------------------------


class TestAsyncPostPaymentRejection:
    """Post-payment rejection detection works in async path."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_raw_json_rejection_async(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.return_value = {"X-PAYMENT": "sig"}
        mw = AgentCorePaymentsMiddleware(_make_config())

        # Retry returns raw JSON 402 (no marker)
        raw_402 = json.dumps({"statusCode": 402, "body": {"error": "insufficient funds"}})
        handler = AsyncMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=raw_402, tool_call_id="tc-1"),
            ]
        )

        result = asyncio.run(
            mw.awrap_tool_call(_make_request(tool_args={"url": "http://x.com", "headers": {}}), handler)
        )

        assert "PAYMENT ERROR" in result.content
        assert "rejected" in result.content


# ---------------------------------------------------------------------------
# Async name-based handler fallback detection
# ---------------------------------------------------------------------------


class TestAsyncNameBasedFallback:
    """Legacy text-block format detected via name-based handler in async path."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_legacy_format_detected_async(self, mock_pm_cls):
        mock_pm_cls.return_value.generate_payment_header.return_value = {"X-PAYMENT": "sig"}
        mw = AgentCorePaymentsMiddleware(_make_config())

        legacy_content = 'Status Code: 402\nHeaders: {}\nBody: {"x402Version": 1, "accepts": []}'
        handler = AsyncMock(
            side_effect=[
                ToolMessage(content=legacy_content, tool_call_id="tc-1"),
                ToolMessage(content=_200_content(), tool_call_id="tc-1"),
            ]
        )

        asyncio.run(
            mw.awrap_tool_call(
                _make_request(tool_name="http_request", tool_args={"url": "http://x.com", "headers": {}}),
                handler,
            )
        )

        # Should detect via HttpRequestPaymentHandler and retry successfully
        assert handler.await_count == 2


# ---------------------------------------------------------------------------
# Async custom handler
# ---------------------------------------------------------------------------


class TestAsyncCustomHandler:
    """Custom handlers work in async path with raw content."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_custom_handler_receives_raw_content_async(self, mock_pm_cls):
        from bedrock_agentcore.payments.integrations.handlers import PaymentResponseHandler

        mock_pm_cls.return_value.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        class RawJsonHandler(PaymentResponseHandler):
            """Handler that expects raw JSON string content."""

            def extract_status_code(self, result):
                # result should be the raw string content
                if isinstance(result, str):
                    parsed = json.loads(result)
                    return parsed.get("code")
                return None

            def extract_headers(self, result):
                if isinstance(result, str):
                    parsed = json.loads(result)
                    return parsed.get("hdrs", {})
                return {}

            def extract_body(self, result):
                if isinstance(result, str):
                    parsed = json.loads(result)
                    return parsed.get("payment", {})
                return {}

            def validate_tool_input(self, tool_input):
                return isinstance(tool_input, dict) and "headers" in tool_input

            def apply_payment_header(self, tool_input, payment_header):
                tool_input["headers"].update(payment_header)
                return True

        custom = RawJsonHandler()
        config = _make_config(custom_handlers={"my_tool": custom})
        mw = AgentCorePaymentsMiddleware(config)

        # Non-standard 402 format that only our custom handler understands
        raw_content = json.dumps({"code": 402, "hdrs": {"x-pay": "v"}, "payment": {"x402Version": 1}})

        handler = AsyncMock(
            side_effect=[
                ToolMessage(content=raw_content, tool_call_id="tc-1"),
                ToolMessage(content=_200_content(), tool_call_id="tc-1"),
            ]
        )

        result = asyncio.run(
            mw.awrap_tool_call(
                _make_request(tool_name="my_tool", tool_args={"url": "http://x.com", "headers": {}}),
                handler,
            )
        )

        # Custom handler detected 402, payment signed, retry succeeded
        assert handler.await_count == 2
        assert "paid" in result.content


# ---------------------------------------------------------------------------
# Async error handler callback
# ---------------------------------------------------------------------------


class TestAsyncErrorHandlerCallback:
    """Error handler callback works in async path."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_async_error_handler_retries_successfully(self, mock_pm_cls):
        from bedrock_agentcore.payments.integrations.langgraph.errors import ErrorResolution
        from bedrock_agentcore.payments.manager import PaymentError

        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.side_effect = [
            PaymentError("session expired"),
            {"X-PAYMENT": "sig"},
        ]

        async def fix_session(ctx):
            ctx.config.payment_session_id = "fresh-sess"
            return ErrorResolution.RETRY

        config = _make_config(on_payment_error=fix_session)
        mw = AgentCorePaymentsMiddleware(config)

        handler = AsyncMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=_200_content(), tool_call_id="tc-1"),
            ]
        )

        result = asyncio.run(
            mw.awrap_tool_call(_make_request(tool_args={"url": "http://x.com", "headers": {}}), handler)
        )

        assert "PAYMENT ERROR" not in result.content
        assert config.payment_session_id == "fresh-sess"

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_async_error_handler_propagates(self, mock_pm_cls):
        from bedrock_agentcore.payments.integrations.langgraph.errors import ErrorResolution
        from bedrock_agentcore.payments.manager import PaymentError

        mock_pm_cls.return_value.generate_payment_header.side_effect = PaymentError("fatal")

        async def propagate(ctx):
            return ErrorResolution.PROPAGATE

        config = _make_config(on_payment_error=propagate)
        mw = AgentCorePaymentsMiddleware(config)

        handler = AsyncMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = asyncio.run(
            mw.awrap_tool_call(_make_request(tool_args={"url": "http://x.com", "headers": {}}), handler)
        )

        assert "PAYMENT ERROR" in result.content
