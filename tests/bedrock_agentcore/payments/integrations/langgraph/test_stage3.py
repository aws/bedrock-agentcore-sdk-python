"""Tests for Stage 3: Payment Signing + Retry."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain.messages import ToolMessage

from bedrock_agentcore.payments.integrations.handlers import PaymentResponseHandler
from bedrock_agentcore.payments.integrations.langgraph import AgentCorePaymentsConfig
from bedrock_agentcore.payments.integrations.langgraph.middleware import AgentCorePaymentsMiddleware
from bedrock_agentcore.payments.manager import (
    PaymentError,
    PaymentInstrumentConfigurationRequired,
    PaymentSessionConfigurationRequired,
)


class _BespokeHandler(PaymentResponseHandler):
    """Custom handler for a tool with a non-standard 402 shape: {"custom_status": 402, ...}.

    Receives the raw ToolMessage.content (a str) per the custom-handler contract.
    """

    def extract_status_code(self, result):
        if isinstance(result, str):
            try:
                return json.loads(result).get("custom_status")
            except (ValueError, TypeError):
                return None
        return None

    def extract_headers(self, result):
        return {}

    def extract_body(self, result):
        if isinstance(result, str):
            try:
                return json.loads(result)
            except (ValueError, TypeError):
                return {}
        return {}

    def validate_tool_input(self, tool_input):
        return isinstance(tool_input, dict)

    def apply_payment_header(self, tool_input, payment_header):
        tool_input.setdefault("headers", {}).update(payment_header)
        return True


def _make_config(**overrides):
    defaults = {
        "payment_manager_arn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
        "user_id": "user-1",
        "payment_instrument_id": "instr-1",
        "payment_session_id": "sess-1",
        "post_payment_retry_delay_seconds": 0,  # no delay in tests
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
    payload = json.dumps({"statusCode": 402, "headers": {"x-pay": "v"}, "body": {"x402Version": 1}})
    return f"PAYMENT_REQUIRED: {payload}"


def _200_content():
    return json.dumps({"statusCode": 200, "body": {"data": "paid content"}})


# ---------------------------------------------------------------------------
# _generate_payment_header tests
# ---------------------------------------------------------------------------


class TestGeneratePaymentHeader:
    """Test _generate_payment_header method."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_calls_pm_with_correct_params(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "signed"}

        config = _make_config()
        mw = AgentCorePaymentsMiddleware(config)

        payment_req = {"statusCode": 402, "headers": {"h": "1"}, "body": {"b": "2"}}
        result = mw._generate_payment_header(payment_req)

        assert result == {"X-PAYMENT": "signed"}
        mock_pm.generate_payment_header.assert_called_once()
        call_kwargs = mock_pm.generate_payment_header.call_args[1]
        assert call_kwargs["user_id"] == "user-1"
        assert call_kwargs["payment_instrument_id"] == "instr-1"
        assert call_kwargs["payment_session_id"] == "sess-1"
        assert call_kwargs["payment_required_request"] is payment_req
        assert call_kwargs["client_token"]  # uuid string, non-empty

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_raises_if_no_instrument_id(self, mock_pm_cls):
        config = _make_config(payment_instrument_id=None)
        mw = AgentCorePaymentsMiddleware(config)

        with pytest.raises(PaymentInstrumentConfigurationRequired):
            mw._generate_payment_header({"statusCode": 402, "headers": {}, "body": {}})

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_raises_if_no_session_id(self, mock_pm_cls):
        config = _make_config(payment_session_id=None)
        mw = AgentCorePaymentsMiddleware(config)

        with pytest.raises(PaymentSessionConfigurationRequired):
            mw._generate_payment_header({"statusCode": 402, "headers": {}, "body": {}})


# ---------------------------------------------------------------------------
# Header injection tests
# ---------------------------------------------------------------------------


class TestHeaderInjection:
    """Test that payment headers are correctly injected into tool args."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_header_injected_into_tool_args(self, mock_pm_cls):
        """After signing, the payment header appears in tool_args['headers']."""
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig123"}

        config = _make_config()
        mw = AgentCorePaymentsMiddleware(config)

        tool_args = {"url": "http://x.com", "headers": {}}
        request = _make_request(tool_args=tool_args)

        call_count = [0]

        def mock_handler(req):
            call_count[0] += 1
            if call_count[0] == 1:
                return ToolMessage(content=_402_content(), tool_call_id="tc-1")
            return ToolMessage(content=_200_content(), tool_call_id="tc-1")

        mw.wrap_tool_call(request, mock_handler)

        # Verify header was injected into tool_args
        assert tool_args["headers"]["X-PAYMENT"] == "sig123"


# ---------------------------------------------------------------------------
# Successful retry tests
# ---------------------------------------------------------------------------


class TestSuccessfulRetry:
    """Test the full 402 → sign → retry → 200 flow."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_402_then_200_on_retry(self, mock_pm_cls):
        """Tool returns 402, middleware signs, retries, gets 200."""
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config()
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        success_msg = ToolMessage(content=_200_content(), tool_call_id="tc-1")

        call_count = [0]

        def mock_handler(req):
            call_count[0] += 1
            if call_count[0] == 1:
                return ToolMessage(content=_402_content(), tool_call_id="tc-1")
            return success_msg

        result = mw.wrap_tool_call(request, mock_handler)

        assert result is success_msg
        assert call_count[0] == 2
        mock_pm.generate_payment_header.assert_called_once()

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_handler_called_twice(self, mock_pm_cls):
        """The execute handler is called exactly twice: initial + retry."""
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config()
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=_200_content(), tool_call_id="tc-1"),
            ]
        )

        mw.wrap_tool_call(request, mock_handler)
        assert mock_handler.call_count == 2


# ---------------------------------------------------------------------------
# Post-payment rejection tests
# ---------------------------------------------------------------------------


class TestPostPaymentRejection:
    """Test detection of 402 after successful signing."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_402_after_signing_returns_error(self, mock_pm_cls):
        """If retry still returns 402, return error ToolMessage."""
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config()
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})

        # Both calls return 402
        mock_handler = MagicMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
            ]
        )

        result = mw.wrap_tool_call(request, mock_handler)

        assert isinstance(result, ToolMessage)
        assert "PAYMENT ERROR" in result.content
        assert "rejected" in result.content
        assert result.tool_call_id == "tc-1"

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_rejection_error_includes_body_error(self, mock_pm_cls):
        """Error message from 402 body is included in the rejection message."""
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config()
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})

        payload_with_error = json.dumps(
            {
                "statusCode": 402,
                "headers": {},
                "body": {"error": "insufficient_balance"},
            }
        )
        content_402_with_error = f"PAYMENT_REQUIRED: {payload_with_error}"

        mock_handler = MagicMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=content_402_with_error, tool_call_id="tc-1"),
            ]
        )

        result = mw.wrap_tool_call(request, mock_handler)
        assert "insufficient_balance" in result.content


# ---------------------------------------------------------------------------
# Delay tests
# ---------------------------------------------------------------------------


class TestRetryDelay:
    """Test configurable delay before retry."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.time.sleep")
    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_delay_applied_before_retry(self, mock_pm_cls, mock_sleep):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config(post_payment_retry_delay_seconds=3.0)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=_200_content(), tool_call_id="tc-1"),
            ]
        )

        mw.wrap_tool_call(request, mock_handler)
        mock_sleep.assert_called_once_with(3.0)

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.time.sleep")
    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_zero_delay_skips_sleep(self, mock_pm_cls, mock_sleep):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config(post_payment_retry_delay_seconds=0)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(
            side_effect=[
                ToolMessage(content=_402_content(), tool_call_id="tc-1"),
                ToolMessage(content=_200_content(), tool_call_id="tc-1"),
            ]
        )

        mw.wrap_tool_call(request, mock_handler)
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Error ToolMessage tests
# ---------------------------------------------------------------------------


class TestErrorToolMessage:
    """Test error messages returned for various failure cases."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_missing_instrument_returns_error_msg(self, mock_pm_cls):
        config = _make_config(payment_instrument_id=None)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, mock_handler)
        assert "PAYMENT ERROR" in result.content
        assert "payment instrument" in result.content
        assert result.tool_call_id == "tc-1"

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_missing_session_returns_error_msg(self, mock_pm_cls):
        config = _make_config(payment_session_id=None)
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, mock_handler)
        assert "PAYMENT ERROR" in result.content
        assert "payment session" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_pm_error_returns_error_msg(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.side_effect = PaymentError("budget exceeded")

        config = _make_config()
        mw = AgentCorePaymentsMiddleware(config)

        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, mock_handler)
        assert "PAYMENT ERROR" in result.content
        assert "budget exceeded" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_validate_tool_input_fails_returns_error(self, mock_pm_cls):
        """If handler can't validate tool input shape, return error."""
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config()
        mw = AgentCorePaymentsMiddleware(config)

        # Tool args not a dict — force validate_tool_input to fail
        # We pass args as a non-dict via direct manipulation
        request = _make_request(tool_args="not-a-dict")

        mock_handler = MagicMock(return_value=ToolMessage(content=_402_content(), tool_call_id="tc-1"))

        result = mw.wrap_tool_call(request, mock_handler)
        assert "PAYMENT ERROR" in result.content
        assert "request format" in result.content


# ---------------------------------------------------------------------------
# Custom-handler retry rejection (custom handler owns post-payment 402 detection)
# ---------------------------------------------------------------------------


class TestCustomHandlerRetryRejection:
    """A registered custom handler decides whether the post-payment retry is still a 402."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_custom_format_rejection_detected_after_payment(self, mock_pm_cls):
        """Retry still-402 in the custom format is detected and its error detail surfaced."""
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config(custom_handlers={"bespoke_tool": _BespokeHandler()})
        mw = AgentCorePaymentsMiddleware(config)

        bespoke_402 = json.dumps({"custom_status": 402, "error": "bespoke_reject", "accepts": []})
        request = _make_request(tool_name="bespoke_tool", tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(
            side_effect=[
                ToolMessage(content=bespoke_402, tool_call_id="tc-1"),  # initial 402
                ToolMessage(content=bespoke_402, tool_call_id="tc-1"),  # retry still 402
            ]
        )

        result = mw.wrap_tool_call(request, mock_handler)
        assert isinstance(result, ToolMessage)
        assert "signed but rejected" in result.content
        assert "bespoke_reject" in result.content  # real detail, not "unknown"

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_custom_format_success_after_payment_passes_through(self, mock_pm_cls):
        """When the retry succeeds in the custom format, the success result is returned (no false rejection)."""
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}

        config = _make_config(custom_handlers={"bespoke_tool": _BespokeHandler()})
        mw = AgentCorePaymentsMiddleware(config)

        bespoke_402 = json.dumps({"custom_status": 402, "error": "pay", "accepts": []})
        success = ToolMessage(content=json.dumps({"custom_status": 200, "data": "ok"}), tool_call_id="tc-1")
        request = _make_request(tool_name="bespoke_tool", tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(side_effect=[ToolMessage(content=bespoke_402, tool_call_id="tc-1"), success])

        result = mw.wrap_tool_call(request, mock_handler)
        assert result is success
        assert "PAYMENT ERROR" not in result.content


# ---------------------------------------------------------------------------
# Header injection when the tool call arrives without an "args" key
# ---------------------------------------------------------------------------


class TestHeaderInjectionMissingArgs:
    """A tool call with no 'args' key still gets the payment header written where the retry sees it."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_header_injected_when_args_key_absent(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig123"}

        config = _make_config()
        mw = AgentCorePaymentsMiddleware(config)

        # tool_call dict deliberately has NO "args" key
        request = MagicMock()
        request.tool_call = {"name": "http_request", "id": "tc-1"}

        call_count = [0]

        def mock_handler(req):
            call_count[0] += 1
            if call_count[0] == 1:
                return ToolMessage(content=_402_content(), tool_call_id="tc-1")
            return ToolMessage(content=_200_content(), tool_call_id="tc-1")

        result = mw.wrap_tool_call(request, mock_handler)

        # args was created on request.tool_call and the header injected into it,
        # so the retried handler would have sent it.
        assert "args" in request.tool_call
        assert request.tool_call["args"]["headers"]["X-PAYMENT"] == "sig123"
        assert call_count[0] == 2
        assert json.loads(result.content)["statusCode"] == 200
