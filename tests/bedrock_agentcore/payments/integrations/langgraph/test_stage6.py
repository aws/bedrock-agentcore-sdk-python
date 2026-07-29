"""Tests for Stage 6: Built-in Tools."""

import json
from unittest.mock import MagicMock, patch

from bedrock_agentcore.payments.integrations.langgraph import AgentCorePaymentsConfig
from bedrock_agentcore.payments.integrations.langgraph.middleware import AgentCorePaymentsMiddleware


def _make_config(**overrides):
    defaults = {
        "payment_manager_arn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
        "user_id": "user-1",
        "payment_instrument_id": "instr-1",
        "payment_session_id": "sess-1",
        "payment_connector_id": "conn-1",
        "post_payment_retry_delay_seconds": 0,
    }
    defaults.update(overrides)
    return AgentCorePaymentsConfig(**defaults)


def _get_tool_by_name(mw, name):
    for t in mw.tools:
        if t.name == name:
            return t
    return None


# ---------------------------------------------------------------------------
# Conditional registration tests
# ---------------------------------------------------------------------------


class TestConditionalRegistration:
    """Test provide_http_request flag controls tool registration."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_provide_http_request_true_includes_tool(self, mock_pm_cls):
        mw = AgentCorePaymentsMiddleware(_make_config(provide_http_request=True))
        names = [t.name for t in mw.tools]
        assert "http_request" in names

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_provide_http_request_false_excludes_tool(self, mock_pm_cls):
        mw = AgentCorePaymentsMiddleware(_make_config(provide_http_request=False))
        names = [t.name for t in mw.tools]
        assert "http_request" not in names

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_query_tools_always_registered(self, mock_pm_cls):
        mw = AgentCorePaymentsMiddleware(_make_config(provide_http_request=False))
        names = [t.name for t in mw.tools]
        assert "get_payment_instrument" in names
        assert "list_payment_instruments" in names
        assert "get_payment_instrument_balance" in names
        assert "get_payment_session" in names


# ---------------------------------------------------------------------------
# http_request tool tests
# ---------------------------------------------------------------------------


class TestHttpRequestTool:
    """Test http_request tool behavior."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_402_returns_payment_required_marker(self, mock_pm_cls):
        mw = AgentCorePaymentsMiddleware(_make_config())
        tool = _get_tool_by_name(mw, "http_request")

        mock_resp = MagicMock()
        mock_resp.status_code = 402
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {"x402Version": 1, "accepts": []}

        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client_cls.return_value)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.request.return_value = mock_resp

            result = tool.invoke({"url": "http://example.com"})

        assert result.startswith("PAYMENT_REQUIRED: ")
        parsed = json.loads(result[len("PAYMENT_REQUIRED: ") :])
        assert parsed["statusCode"] == 402

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_200_returns_json_payload(self, mock_pm_cls):
        mw = AgentCorePaymentsMiddleware(_make_config())
        tool = _get_tool_by_name(mw, "http_request")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {"data": "content"}

        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client_cls.return_value)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.request.return_value = mock_resp

            result = tool.invoke({"url": "http://example.com"})

        parsed = json.loads(result)
        assert parsed["statusCode"] == 200
        assert parsed["body"] == {"data": "content"}

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_network_error_returns_error(self, mock_pm_cls):
        import httpx as _httpx

        mw = AgentCorePaymentsMiddleware(_make_config())
        tool = _get_tool_by_name(mw, "http_request")

        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client_cls.return_value)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.request.side_effect = _httpx.ConnectError("connection refused")

            result = tool.invoke({"url": "http://example.com"})

        parsed = json.loads(result)
        assert parsed["statusCode"] == 0
        assert "connection refused" in parsed["error"]

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_post_with_json_body(self, mock_pm_cls):
        mw = AgentCorePaymentsMiddleware(_make_config())
        tool = _get_tool_by_name(mw, "http_request")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.json.return_value = {"ok": True}

        with patch("httpx.Client") as mock_client_cls:
            client = mock_client_cls.return_value
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.request.return_value = mock_resp

            tool.invoke({"url": "http://x.com", "method": "POST", "body": {"k": "v"}})
            client.request.assert_called_once_with("POST", "http://x.com", headers={}, json={"k": "v"})


# ---------------------------------------------------------------------------
# Payment query tool tests
# ---------------------------------------------------------------------------


class TestPaymentQueryTools:
    """Test payment query tools call PaymentManager correctly."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_get_payment_instrument(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.get_payment_instrument.return_value = {"paymentInstrumentId": "instr-1"}

        mw = AgentCorePaymentsMiddleware(_make_config())
        tool = _get_tool_by_name(mw, "get_payment_instrument")

        result = tool.invoke({"payment_instrument_id": "instr-99", "user_id": "u2"})
        mock_pm.get_payment_instrument.assert_called_once_with(
            user_id="u2",
            payment_instrument_id="instr-99",
            payment_connector_id=None,
        )
        assert result == {"paymentInstrumentId": "instr-1"}

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_get_payment_instrument_falls_back_to_config(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.get_payment_instrument.return_value = {"paymentInstrumentId": "instr-1"}

        mw = AgentCorePaymentsMiddleware(_make_config())
        tool = _get_tool_by_name(mw, "get_payment_instrument")

        tool.invoke({})
        mock_pm.get_payment_instrument.assert_called_once_with(
            user_id="user-1",
            payment_instrument_id="instr-1",
            payment_connector_id=None,
        )

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_list_payment_instruments(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.list_payment_instruments.return_value = {"paymentInstruments": []}

        mw = AgentCorePaymentsMiddleware(_make_config())
        tool = _get_tool_by_name(mw, "list_payment_instruments")

        result = tool.invoke({"user_id": "u2"})
        mock_pm.list_payment_instruments.assert_called_once_with(
            user_id="u2",
            payment_connector_id=None,
            max_results=100,
            next_token=None,
        )
        assert result == {"paymentInstruments": []}

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_get_payment_instrument_balance(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.get_payment_instrument_balance.return_value = {"tokenBalance": {"amount": "10.0"}}

        mw = AgentCorePaymentsMiddleware(_make_config())
        tool = _get_tool_by_name(mw, "get_payment_instrument_balance")

        result = tool.invoke({"payment_instrument_id": "instr-1", "chain": "BASE_SEPOLIA"})
        mock_pm.get_payment_instrument_balance.assert_called_once_with(
            payment_connector_id="conn-1",
            payment_instrument_id="instr-1",
            chain="BASE_SEPOLIA",
            token="USDC",
            user_id="user-1",
        )
        assert result["tokenBalance"]["amount"] == "10.0"

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_get_payment_session(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.get_payment_session.return_value = {"paymentSessionId": "sess-1"}

        mw = AgentCorePaymentsMiddleware(_make_config())
        tool = _get_tool_by_name(mw, "get_payment_session")

        result = tool.invoke({"payment_session_id": "sess-99", "user_id": "u3"})
        mock_pm.get_payment_session.assert_called_once_with(
            user_id="u3",
            payment_session_id="sess-99",
        )
        assert result == {"paymentSessionId": "sess-1"}

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_get_payment_session_falls_back_to_config(self, mock_pm_cls):
        mock_pm = mock_pm_cls.return_value
        mock_pm.get_payment_session.return_value = {"paymentSessionId": "sess-1"}

        mw = AgentCorePaymentsMiddleware(_make_config())
        tool = _get_tool_by_name(mw, "get_payment_session")

        tool.invoke({})
        mock_pm.get_payment_session.assert_called_once_with(
            user_id="user-1",
            payment_session_id="sess-1",
        )
