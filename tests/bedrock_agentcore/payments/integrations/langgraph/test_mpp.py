"""Tests for MPP (Machine Payments Protocol) support in the LangGraph middleware."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain.messages import ToolMessage

from bedrock_agentcore.payments.integrations.langgraph import AgentCorePaymentsConfig
from bedrock_agentcore.payments.integrations.langgraph.middleware import AgentCorePaymentsMiddleware
from bedrock_agentcore.payments.manager import PaymentError

CHALLENGE = (
    'Payment id="evm-1", realm="api.example.com", method="evm", intent="charge", request="eyJhbW91bnQiOiIxMDAwIn0"'
)
MPP_CREDENTIAL = "Payment eyJjaGFsbGVuZ2UiOnt9fQ"


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


def _mpp_402_content(challenge=CHALLENGE):
    """A spec-compliant PAYMENT_REQUIRED marker carrying an MPP challenge."""
    payload = json.dumps(
        {
            "statusCode": 402,
            "headers": {"WWW-Authenticate": challenge, "content-type": "application/json"},
            "body": {},
        }
    )
    return f"PAYMENT_REQUIRED: {payload}"


def _mpp_402_raw_content(challenge=CHALLENGE):
    """A raw JSON 402 with no marker — exercises fallback detection."""
    return json.dumps({"responseHeaders": {"WWW-Authenticate": challenge}, "structuredContent": {}})


def _200_content():
    return json.dumps({"statusCode": 200, "body": {"data": "paid content"}})


class TestMppFallbackDetection:
    """A header-only MPP 402 must be detected without an explicit status code."""

    def test_detects_mpp_challenge_in_response_headers(self):
        detected = AgentCorePaymentsMiddleware._fallback_detect_402(_mpp_402_raw_content())

        assert detected is not None
        assert detected["statusCode"] == 402
        assert detected["headers"]["WWW-Authenticate"] == CHALLENGE

    def test_detects_mpp_challenge_under_headers_key(self):
        content = json.dumps({"headers": {"WWW-Authenticate": CHALLENGE}})

        detected = AgentCorePaymentsMiddleware._fallback_detect_402(content)

        assert detected is not None
        assert detected["headers"]["WWW-Authenticate"] == CHALLENGE

    def test_ignores_bearer_challenge(self):
        content = json.dumps({"responseHeaders": {"WWW-Authenticate": 'Bearer realm="x"'}})

        assert AgentCorePaymentsMiddleware._fallback_detect_402(content) is None

    def test_x402_detection_still_takes_precedence(self):
        content = json.dumps({"x402Version": 1, "accepts": [{"network": "base-sepolia"}]})

        detected = AgentCorePaymentsMiddleware._fallback_detect_402(content)

        assert detected["body"]["x402Version"] == 1

    def test_non_payment_response_is_not_detected(self):
        assert AgentCorePaymentsMiddleware._fallback_detect_402(json.dumps({"statusCode": 200})) is None


class TestMppSyncFlow:
    """The sync path must settle MPP 402s and retry with the Authorization header."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_402_then_200_on_retry(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        mw = AgentCorePaymentsMiddleware(_make_config())
        request = _make_request()
        success = ToolMessage(content=_200_content(), tool_call_id="tc-1")

        calls = [0]

        def handler(req):
            calls[0] += 1
            if calls[0] == 1:
                return ToolMessage(content=_mpp_402_content(), tool_call_id="tc-1")
            return success

        result = mw.wrap_tool_call(request, handler)

        assert result is success
        assert calls[0] == 2
        mock_pm.generate_payment_header.assert_called_once()

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_authorization_header_injected_into_tool_args(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        mw = AgentCorePaymentsMiddleware(_make_config())
        request = _make_request(tool_args={"url": "http://x.com", "headers": {}})

        mw.wrap_tool_call(
            request,
            MagicMock(
                side_effect=[
                    ToolMessage(content=_mpp_402_content(), tool_call_id="tc-1"),
                    ToolMessage(content=_200_content(), tool_call_id="tc-1"),
                ]
            ),
        )

        assert request.tool_call["args"]["headers"]["Authorization"] == MPP_CREDENTIAL

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_challenge_reaches_payment_manager(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        mw = AgentCorePaymentsMiddleware(_make_config())

        mw.wrap_tool_call(
            _make_request(),
            MagicMock(
                side_effect=[
                    ToolMessage(content=_mpp_402_content(), tool_call_id="tc-1"),
                    ToolMessage(content=_200_content(), tool_call_id="tc-1"),
                ]
            ),
        )

        request_arg = mock_pm.generate_payment_header.call_args.kwargs["payment_required_request"]
        assert request_arg["headers"]["WWW-Authenticate"] == CHALLENGE

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_selection_failure_returns_error_tool_message(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.side_effect = PaymentError("MPP Challenge Selection: No matching challenge")
        mw = AgentCorePaymentsMiddleware(_make_config())

        result = mw.wrap_tool_call(
            _make_request(),
            MagicMock(return_value=ToolMessage(content=_mpp_402_content(), tool_call_id="tc-1")),
        )

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "tc-1"

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_auto_payment_disabled_skips_mpp(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mw = AgentCorePaymentsMiddleware(_make_config(auto_payment=False))
        msg = ToolMessage(content=_mpp_402_content(), tool_call_id="tc-1")

        result = mw.wrap_tool_call(_make_request(), MagicMock(return_value=msg))

        assert result is msg
        mock_pm.generate_payment_header.assert_not_called()

    @pytest.mark.parametrize("configured", [True, False, None])
    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_buyer_pays_gas_fees_is_forwarded_from_config(self, mock_pm_cls, configured):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        mw = AgentCorePaymentsMiddleware(_make_config(buyer_pays_gas_fees=configured))

        mw.wrap_tool_call(
            _make_request(),
            MagicMock(
                side_effect=[
                    ToolMessage(content=_mpp_402_content(), tool_call_id="tc-1"),
                    ToolMessage(content=_200_content(), tool_call_id="tc-1"),
                ]
            ),
        )

        assert mock_pm.generate_payment_header.call_args.kwargs["buyer_pays_gas_fees"] is configured

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_x402_flow_is_unaffected(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "sig"}
        mw = AgentCorePaymentsMiddleware(_make_config())
        request = _make_request()
        x402 = f"PAYMENT_REQUIRED: {json.dumps({'statusCode': 402, 'headers': {}, 'body': {'x402Version': 1}})}"

        mw.wrap_tool_call(
            request,
            MagicMock(
                side_effect=[
                    ToolMessage(content=x402, tool_call_id="tc-1"),
                    ToolMessage(content=_200_content(), tool_call_id="tc-1"),
                ]
            ),
        )

        assert request.tool_call["args"]["headers"]["X-PAYMENT"] == "sig"


class TestMppAsyncFlow:
    """The async path must behave identically to the sync path."""

    @pytest.mark.asyncio
    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    async def test_async_402_then_200_on_retry(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        mw = AgentCorePaymentsMiddleware(_make_config())
        request = _make_request()
        success = ToolMessage(content=_200_content(), tool_call_id="tc-1")

        calls = [0]

        async def handler(req):
            calls[0] += 1
            if calls[0] == 1:
                return ToolMessage(content=_mpp_402_content(), tool_call_id="tc-1")
            return success

        result = await mw.awrap_tool_call(request, handler)

        assert result is success
        assert calls[0] == 2
        assert request.tool_call["args"]["headers"]["Authorization"] == MPP_CREDENTIAL

    @pytest.mark.asyncio
    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    async def test_async_selection_failure_returns_error_message(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.generate_payment_header.side_effect = PaymentError("MPP Challenge Selection: No matching challenge")
        mw = AgentCorePaymentsMiddleware(_make_config())

        async def handler(req):
            return ToolMessage(content=_mpp_402_content(), tool_call_id="tc-1")

        result = await mw.awrap_tool_call(_make_request(), handler)

        assert isinstance(result, ToolMessage)
