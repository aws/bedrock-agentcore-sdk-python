"""Tests for Stage 2: 402 Detection + Adapter."""

import json
from unittest.mock import MagicMock, patch

from langchain.messages import ToolMessage
from langgraph.types import Command

from bedrock_agentcore.payments.integrations.handlers import (
    GenericPaymentHandler,
    HttpRequestPaymentHandler,
    MCPRequestPaymentHandler,
    PaymentResponseHandler,
)
from bedrock_agentcore.payments.integrations.langgraph import AgentCorePaymentsConfig
from bedrock_agentcore.payments.integrations.langgraph.middleware import AgentCorePaymentsMiddleware


def _make_config(**overrides):
    defaults = {
        "payment_manager_arn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-1",
        "user_id": "user-1",
    }
    defaults.update(overrides)
    return AgentCorePaymentsConfig(**defaults)


def _make_request(tool_name="http_request", tool_args=None, tool_id="tc-1"):
    req = MagicMock()
    req.tool_call = {"name": tool_name, "args": tool_args or {}, "id": tool_id}
    return req


PAYMENT_402_PAYLOAD = json.dumps({"statusCode": 402, "headers": {"x-pay": "v"}, "body": {"x402Version": 1}})


# ---------------------------------------------------------------------------
# _prepare_for_handler tests
# ---------------------------------------------------------------------------


class TestPrepareForHandler:
    """Test the adapter that normalizes ToolMessage.content for handlers."""

    def test_string_content_wrapped(self):
        content = f"PAYMENT_REQUIRED: {PAYMENT_402_PAYLOAD}"
        result = AgentCorePaymentsMiddleware._prepare_for_handler(content)
        assert result == {"content": [{"text": content}]}

    def test_list_of_dicts_passed_through(self):
        content = [{"text": "PAYMENT_REQUIRED: " + PAYMENT_402_PAYLOAD}]
        result = AgentCorePaymentsMiddleware._prepare_for_handler(content)
        assert result == {"content": content}

    def test_list_of_strings_wrapped(self):
        content = ["PAYMENT_REQUIRED: data", "other"]
        result = AgentCorePaymentsMiddleware._prepare_for_handler(content)
        assert result == {"content": [{"text": "PAYMENT_REQUIRED: data"}, {"text": "other"}]}

    def test_mixed_list(self):
        content = [{"text": "foo"}, "bar"]
        result = AgentCorePaymentsMiddleware._prepare_for_handler(content)
        assert result == {"content": [{"text": "foo"}, {"text": "bar"}]}

    def test_empty_string(self):
        result = AgentCorePaymentsMiddleware._prepare_for_handler("")
        assert result == {"content": [{"text": ""}]}

    def test_none_returns_none(self):
        assert AgentCorePaymentsMiddleware._prepare_for_handler(None) is None

    def test_non_str_non_list_returns_none(self):
        assert AgentCorePaymentsMiddleware._prepare_for_handler(12345) is None


# ---------------------------------------------------------------------------
# _get_handler tests
# ---------------------------------------------------------------------------


class TestGetHandler:
    """Test handler resolution priority."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_custom_handler_takes_priority(self, mock_pm):
        custom = MagicMock(spec=PaymentResponseHandler)
        config = _make_config(custom_handlers={"my_tool": custom})
        mw = AgentCorePaymentsMiddleware(config)
        assert mw._get_handler("my_tool", {}) is custom

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_custom_handler_only_for_registered_name(self, mock_pm):
        custom = MagicMock(spec=PaymentResponseHandler)
        config = _make_config(custom_handlers={"my_tool": custom})
        mw = AgentCorePaymentsMiddleware(config)
        handler = mw._get_handler("other_tool", {})
        assert handler is not custom
        assert isinstance(handler, GenericPaymentHandler)

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_builtin_name_based_handler(self, mock_pm):
        mw = AgentCorePaymentsMiddleware(_make_config())
        handler = mw._get_handler("http_request", {})
        assert isinstance(handler, HttpRequestPaymentHandler)

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_mcp_shape_detected(self, mock_pm):
        mw = AgentCorePaymentsMiddleware(_make_config())
        handler = mw._get_handler("proxy_tool", {"toolName": "x", "parameters": {}})
        assert isinstance(handler, MCPRequestPaymentHandler)

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_generic_fallback(self, mock_pm):
        mw = AgentCorePaymentsMiddleware(_make_config())
        handler = mw._get_handler("unknown_tool", {"url": "http://example.com"})
        assert isinstance(handler, GenericPaymentHandler)


# ---------------------------------------------------------------------------
# 402 detection tests
# ---------------------------------------------------------------------------


class TestDetection402:
    """Test that 402 is correctly detected from various formats."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_402_detected_from_marker_string(self, mock_pm):
        mw = AgentCorePaymentsMiddleware(_make_config())
        content = f"PAYMENT_REQUIRED: {PAYMENT_402_PAYLOAD}"
        tool_msg = ToolMessage(content=content, tool_call_id="tc-1")

        request = _make_request()
        mock_handler = MagicMock(return_value=tool_msg)

        result = mw.wrap_tool_call(request, mock_handler)

        # 402 detected → signing attempted → error (no instrument configured)
        assert isinstance(result, ToolMessage)
        assert "PAYMENT ERROR" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_402_detected_from_content_list(self, mock_pm):
        mw = AgentCorePaymentsMiddleware(_make_config())
        content = [{"text": f"PAYMENT_REQUIRED: {PAYMENT_402_PAYLOAD}"}]
        tool_msg = ToolMessage(content=content, tool_call_id="tc-1")

        request = _make_request()
        mock_handler = MagicMock(return_value=tool_msg)

        result = mw.wrap_tool_call(request, mock_handler)

        # 402 detected → signing attempted → error (no instrument configured)
        assert isinstance(result, ToolMessage)
        assert "PAYMENT ERROR" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_non_402_status_passes_through(self, mock_pm):
        mw = AgentCorePaymentsMiddleware(_make_config())
        payload = json.dumps({"statusCode": 200, "headers": {}, "body": {}})
        content = f"PAYMENT_REQUIRED: {payload}"
        tool_msg = ToolMessage(content=content, tool_call_id="tc-1")

        request = _make_request()
        mock_handler = MagicMock(return_value=tool_msg)
        result = mw.wrap_tool_call(request, mock_handler)
        assert result is tool_msg

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_no_marker_passes_through(self, mock_pm):
        mw = AgentCorePaymentsMiddleware(_make_config())
        tool_msg = ToolMessage(content="Just normal output", tool_call_id="tc-1")

        request = _make_request()
        mock_handler = MagicMock(return_value=tool_msg)
        result = mw.wrap_tool_call(request, mock_handler)
        assert result is tool_msg


# ---------------------------------------------------------------------------
# Guard condition tests
# ---------------------------------------------------------------------------


class TestGuardConditions:
    """Test that guards correctly bypass 402 detection."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_command_result_passes_through(self, mock_pm):
        mw = AgentCorePaymentsMiddleware(_make_config())
        cmd = Command(update={"key": "val"})

        request = _make_request()
        mock_handler = MagicMock(return_value=cmd)

        with patch.object(mw, "_get_handler") as spy:
            result = mw.wrap_tool_call(request, mock_handler)
            spy.assert_not_called()

        assert result is cmd

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_auto_payment_false_skips(self, mock_pm):
        config = _make_config(auto_payment=False)
        mw = AgentCorePaymentsMiddleware(config)
        content = f"PAYMENT_REQUIRED: {PAYMENT_402_PAYLOAD}"
        tool_msg = ToolMessage(content=content, tool_call_id="tc-1")

        request = _make_request()
        mock_handler = MagicMock(return_value=tool_msg)

        with patch.object(mw, "_get_handler") as spy:
            result = mw.wrap_tool_call(request, mock_handler)
            spy.assert_not_called()

        assert result is tool_msg

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_tool_not_in_allowlist_skips(self, mock_pm):
        config = _make_config(payment_tool_allowlist=["other_tool"])
        mw = AgentCorePaymentsMiddleware(config)
        content = f"PAYMENT_REQUIRED: {PAYMENT_402_PAYLOAD}"
        tool_msg = ToolMessage(content=content, tool_call_id="tc-1")

        request = _make_request(tool_name="http_request")
        mock_handler = MagicMock(return_value=tool_msg)

        with patch.object(mw, "_get_handler") as spy:
            result = mw.wrap_tool_call(request, mock_handler)
            spy.assert_not_called()

        assert result is tool_msg

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_tool_in_allowlist_proceeds(self, mock_pm):
        config = _make_config(payment_tool_allowlist=["http_request"])
        mw = AgentCorePaymentsMiddleware(config)
        content = f"PAYMENT_REQUIRED: {PAYMENT_402_PAYLOAD}"
        tool_msg = ToolMessage(content=content, tool_call_id="tc-1")

        request = _make_request(tool_name="http_request")
        mock_handler = MagicMock(return_value=tool_msg)

        result = mw.wrap_tool_call(request, mock_handler)

        # 402 detected and processing attempted (error because no instrument)
        assert isinstance(result, ToolMessage)
        assert "PAYMENT ERROR" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_none_allowlist_processes_all(self, mock_pm):
        config = _make_config(payment_tool_allowlist=None)
        mw = AgentCorePaymentsMiddleware(config)
        content = f"PAYMENT_REQUIRED: {PAYMENT_402_PAYLOAD}"
        tool_msg = ToolMessage(content=content, tool_call_id="tc-1")

        request = _make_request(tool_name="any_tool")
        mock_handler = MagicMock(return_value=tool_msg)

        result = mw.wrap_tool_call(request, mock_handler)

        # 402 detected and processing attempted (error because no instrument)
        assert isinstance(result, ToolMessage)
        assert "PAYMENT ERROR" in result.content


# ---------------------------------------------------------------------------
# Fallback detection tests (raw JSON without PAYMENT_REQUIRED: marker)
# ---------------------------------------------------------------------------


class TestFallbackDetection:
    """Test lenient fallback that detects 402 from raw JSON."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_raw_status_code_402_detected(self, mock_pm):
        """Raw JSON with statusCode:402 triggers payment processing."""
        mw = AgentCorePaymentsMiddleware(_make_config())
        content = json.dumps({"statusCode": 402, "headers": {"h": "v"}, "body": {"x402Version": 1}})
        tool_msg = ToolMessage(content=content, tool_call_id="tc-1")

        request = _make_request()
        mock_handler = MagicMock(return_value=tool_msg)
        result = mw.wrap_tool_call(request, mock_handler)

        # 402 detected via fallback → payment processing attempted
        assert "PAYMENT ERROR" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_x402_payload_detected(self, mock_pm):
        """Raw JSON with x402Version + accepts triggers payment processing."""
        mw = AgentCorePaymentsMiddleware(_make_config())
        content = json.dumps({"x402Version": 1, "accepts": [{"network": "base-sepolia"}]})
        tool_msg = ToolMessage(content=content, tool_call_id="tc-1")

        request = _make_request()
        mock_handler = MagicMock(return_value=tool_msg)
        result = mw.wrap_tool_call(request, mock_handler)

        assert "PAYMENT ERROR" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_non_402_raw_json_passes_through(self, mock_pm):
        """Raw JSON with statusCode:200 is not detected as 402."""
        mw = AgentCorePaymentsMiddleware(_make_config())
        content = json.dumps({"statusCode": 200, "body": {"data": "ok"}})
        tool_msg = ToolMessage(content=content, tool_call_id="tc-1")

        request = _make_request()
        mock_handler = MagicMock(return_value=tool_msg)
        result = mw.wrap_tool_call(request, mock_handler)

        assert result is tool_msg

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_plain_text_not_detected(self, mock_pm):
        """Non-JSON text is not detected as 402."""
        mw = AgentCorePaymentsMiddleware(_make_config())
        tool_msg = ToolMessage(content="Hello world", tool_call_id="tc-1")

        request = _make_request()
        mock_handler = MagicMock(return_value=tool_msg)
        result = mw.wrap_tool_call(request, mock_handler)

        assert result is tool_msg


# ---------------------------------------------------------------------------
# Legacy text-block format detection (name-based handler fallback)
# ---------------------------------------------------------------------------


class TestLegacyTextBlockDetection:
    """Test that legacy Status Code: format is detected via name-based handler."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_legacy_format_detected_for_http_request_tool(self, mock_pm):
        """Legacy 'Status Code: 402' format detected via HttpRequestPaymentHandler."""
        mw = AgentCorePaymentsMiddleware(_make_config())
        # Legacy text-block format (no PAYMENT_REQUIRED: marker, not valid JSON)
        content = 'Status Code: 402\nHeaders: {}\nBody: {"x402Version": 1, "accepts": []}'
        tool_msg = ToolMessage(content=content, tool_call_id="tc-1")

        request = _make_request(tool_name="http_request")
        mock_handler = MagicMock(return_value=tool_msg)

        result = mw.wrap_tool_call(request, mock_handler)

        # 402 detected → payment processing attempted → error (no instrument configured)
        assert isinstance(result, ToolMessage)
        assert "PAYMENT ERROR" in result.content

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_legacy_format_not_detected_for_unknown_tool(self, mock_pm):
        """Legacy format is NOT detected for a tool that doesn't resolve to HttpRequestPaymentHandler."""
        mw = AgentCorePaymentsMiddleware(_make_config())
        content = 'Status Code: 402\nHeaders: {}\nBody: {"x402Version": 1}'
        tool_msg = ToolMessage(content=content, tool_call_id="tc-1")

        # Tool named "my_custom_tool" resolves to GenericPaymentHandler, which can't parse this
        request = _make_request(tool_name="my_custom_tool")
        mock_handler = MagicMock(return_value=tool_msg)

        result = mw.wrap_tool_call(request, mock_handler)

        # GenericPaymentHandler also can't parse it — passes through
        assert result is tool_msg


# ---------------------------------------------------------------------------
# Custom handler raw content contract
# ---------------------------------------------------------------------------


class TestCustomHandlerRawContentContract:
    """Verify custom handlers receive raw content, not the prepared shape."""

    @patch("bedrock_agentcore.payments.integrations.langgraph.middleware.PaymentManager")
    def test_custom_handler_receives_raw_string(self, mock_pm):
        """Custom handler extract_status_code gets the raw string content."""
        from bedrock_agentcore.payments.integrations.handlers import PaymentResponseHandler

        received_inputs = []

        class SpyHandler(PaymentResponseHandler):
            def extract_status_code(self, result):
                received_inputs.append(result)
                # Parse raw JSON directly
                if isinstance(result, str):
                    parsed = json.loads(result)
                    return parsed.get("code")
                return None

            def extract_headers(self, result):
                return {}

            def extract_body(self, result):
                return {"x402Version": 1}

            def validate_tool_input(self, tool_input):
                return True

            def apply_payment_header(self, tool_input, payment_header):
                return True

        config = _make_config(custom_handlers={"my_tool": SpyHandler()})
        mock_pm.return_value.generate_payment_header.return_value = {"X-PAYMENT": "sig"}
        mw = AgentCorePaymentsMiddleware(config)

        raw_json = json.dumps({"code": 402, "data": "custom format"})
        tool_msg = ToolMessage(content=raw_json, tool_call_id="tc-1")

        request = _make_request(tool_name="my_tool", tool_args={"url": "http://x.com", "headers": {}})
        mock_handler = MagicMock(
            side_effect=[
                tool_msg,
                ToolMessage(content="success", tool_call_id="tc-1"),
            ]
        )

        mw.wrap_tool_call(request, mock_handler)

        # Verify the handler received the raw string, not {"content": [{"text": ...}]}
        assert len(received_inputs) == 1
        assert received_inputs[0] == raw_json
        assert isinstance(received_inputs[0], str)
