"""Unit tests for MPP support in the Strands payments plugin."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig
from bedrock_agentcore.payments.integrations.strands.plugin import AgentCorePaymentsPlugin
from bedrock_agentcore.payments.manager import PaymentError

PAYMENT_MANAGER_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/test"

CHALLENGE = (
    'Payment id="evm-1", realm="api.example.com", method="evm", intent="charge", request="eyJhbW91bnQiOiIxMDAwIn0"'
)
MPP_CREDENTIAL = "Payment eyJjaGFsbGVuZ2UiOnt9fQ"


def _create_mock_agent():
    """Create a mock agent with a dict-backed state object."""
    agent = MagicMock()
    state_store = {}

    def state_get(key=None):
        if key is None:
            return dict(state_store)
        return state_store.get(key)

    agent.state.get = MagicMock(side_effect=state_get)
    agent.state.set = MagicMock(side_effect=lambda k, v: state_store.__setitem__(k, v))
    agent.state.delete = MagicMock(side_effect=lambda k: state_store.pop(k, None))
    agent._state_store = state_store
    return agent


def _create_event(result, tool_input=None, agent=None, invocation_state=None):
    """Create a mock AfterToolCallEvent carrying *result*.

    Pass the same *invocation_state* dict across events to simulate consecutive
    attempts on one tool use — the plugin tracks signing and failure state there,
    not on agent.state.
    """
    event = MagicMock()
    event.agent = agent or _create_mock_agent()
    event.result = result
    event.tool_use = {
        "name": "http_request",
        "toolUseId": "tool-123",
        "input": tool_input if tool_input is not None else {"headers": {}},
    }
    event.invocation_state = invocation_state if invocation_state is not None else {}
    event.retry = False
    return event


def _config(**overrides):
    kwargs = {
        "payment_manager_arn": PAYMENT_MANAGER_ARN,
        "user_id": "test-user",
        "payment_instrument_id": "payment-instrument-123",
        "payment_session_id": "payment-session-456",
        "post_payment_retry_delay_seconds": 0,
    }
    kwargs.update(overrides)
    return AgentCorePaymentsPluginConfig(**kwargs)


def _mpp_402_result(challenge=CHALLENGE):
    """Build an http_request tool result carrying an MPP 402."""
    payload = {
        "statusCode": 402,
        "headers": {"WWW-Authenticate": challenge, "content-type": "application/json"},
        "body": {},
    }
    return [{"text": f"PAYMENT_REQUIRED: {json.dumps(payload)}"}]


def _plugin_with_manager(mock_pm, config=None):
    plugin = AgentCorePaymentsPlugin(config=config or _config())
    plugin.payment_manager = mock_pm
    return plugin


class TestStrandsPluginMppFlow:
    """The plugin must settle MPP 402s and retry with the Authorization header."""

    def test_mpp_402_triggers_payment_and_retry(self):
        mock_pm = MagicMock()
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        plugin = _plugin_with_manager(mock_pm)
        event = _create_event(_mpp_402_result())

        plugin.after_tool_call(event)

        assert event.retry is True
        mock_pm.generate_payment_header.assert_called_once()

    def test_authorization_header_is_applied_to_tool_input(self):
        mock_pm = MagicMock()
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        plugin = _plugin_with_manager(mock_pm)
        event = _create_event(_mpp_402_result())

        plugin.after_tool_call(event)

        assert event.tool_use["input"]["headers"]["Authorization"] == MPP_CREDENTIAL

    def test_www_authenticate_challenge_reaches_payment_manager(self):
        """The plugin must forward the challenge header so selection can run."""
        mock_pm = MagicMock()
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        plugin = _plugin_with_manager(mock_pm)
        event = _create_event(_mpp_402_result())

        plugin.after_tool_call(event)

        request = mock_pm.generate_payment_header.call_args.kwargs["payment_required_request"]
        assert request["statusCode"] == 402
        assert request["headers"]["WWW-Authenticate"] == CHALLENGE

    def test_existing_request_headers_are_preserved(self):
        mock_pm = MagicMock()
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        plugin = _plugin_with_manager(mock_pm)
        event = _create_event(_mpp_402_result(), tool_input={"headers": {"Accept": "application/json"}})

        plugin.after_tool_call(event)

        headers = event.tool_use["input"]["headers"]
        assert headers["Accept"] == "application/json"
        assert headers["Authorization"] == MPP_CREDENTIAL

    def test_headers_dict_is_created_when_absent(self):
        mock_pm = MagicMock()
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        plugin = _plugin_with_manager(mock_pm)
        event = _create_event(_mpp_402_result(), tool_input={"url": "https://api.example.com/resource"})

        plugin.after_tool_call(event)

        assert event.tool_use["input"]["headers"]["Authorization"] == MPP_CREDENTIAL

    def test_network_preferences_are_passed_through(self):
        mock_pm = MagicMock()
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        plugin = _plugin_with_manager(mock_pm, _config(network_preferences_config=["eip155:1"]))
        event = _create_event(_mpp_402_result())

        plugin.after_tool_call(event)

        assert mock_pm.generate_payment_header.call_args.kwargs["network_preferences"] == ["eip155:1"]

    @pytest.mark.parametrize("configured", [True, False])
    def test_buyer_pays_gas_fees_is_forwarded_from_config(self, configured):
        mock_pm = MagicMock()
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        plugin = _plugin_with_manager(mock_pm, _config(buyer_pays_gas_fees=configured))
        event = _create_event(_mpp_402_result())

        plugin.after_tool_call(event)

        assert mock_pm.generate_payment_header.call_args.kwargs["buyer_pays_gas_fees"] is configured

    def test_buyer_pays_gas_fees_defaults_to_none(self):
        """Unset config must not assert a gas-fee choice on the caller's behalf."""
        mock_pm = MagicMock()
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        plugin = _plugin_with_manager(mock_pm)
        event = _create_event(_mpp_402_result())

        plugin.after_tool_call(event)

        assert mock_pm.generate_payment_header.call_args.kwargs["buyer_pays_gas_fees"] is None

    def test_non_boolean_buyer_pays_gas_fees_is_rejected_by_config(self):
        with pytest.raises(ValueError, match="buyer_pays_gas_fees must be a boolean"):
            _config(buyer_pays_gas_fees="yes")

    def test_x402_flow_is_unaffected(self):
        """Adding MPP must not change the existing x402 behavior."""
        mock_pm = MagicMock()
        mock_pm.generate_payment_header.return_value = {"X-PAYMENT": "base64-encoded"}
        plugin = _plugin_with_manager(mock_pm)
        payload = {
            "statusCode": 402,
            "headers": {},
            "body": {"x402Version": 1, "accepts": [{"network": "base-sepolia"}]},
        }
        event = _create_event([{"text": f"PAYMENT_REQUIRED: {json.dumps(payload)}"}])

        plugin.after_tool_call(event)

        assert event.retry is True
        assert event.tool_use["input"]["headers"]["X-PAYMENT"] == "base64-encoded"


class TestStrandsPluginMppErrors:
    """Failures on the MPP path must surface like any other payment failure."""

    def test_unsatisfiable_challenge_does_not_retry(self):
        mock_pm = MagicMock()
        mock_pm.generate_payment_header.side_effect = PaymentError(
            "MPP Challenge Selection: No matching challenge - no advertised payment method"
        )
        plugin = _plugin_with_manager(mock_pm)
        event = _create_event(_mpp_402_result())

        plugin.after_tool_call(event)

        assert event.retry is False
        assert "Authorization" not in event.tool_use["input"].get("headers", {})

    def test_payment_failure_state_is_recorded(self):
        mock_pm = MagicMock()
        mock_pm.generate_payment_header.side_effect = PaymentError("MPP selection failed")
        plugin = _plugin_with_manager(mock_pm)
        invocation_state = {}
        event = _create_event(_mpp_402_result(), invocation_state=invocation_state)

        plugin.after_tool_call(event)

        failure = invocation_state.get("payment_failure_tool-123")
        assert failure is not None, f"Expected payment failure state, got {invocation_state}"
        assert "MPP selection failed" in str(failure)

    def test_second_402_after_signing_is_not_retried(self):
        """A 402 after a successful signing means server-side rejection."""
        mock_pm = MagicMock()
        mock_pm.generate_payment_header.return_value = {"Authorization": MPP_CREDENTIAL}
        plugin = _plugin_with_manager(mock_pm)
        # invocation_state is shared across attempts on the same tool use.
        invocation_state = {}

        first = _create_event(_mpp_402_result(), invocation_state=invocation_state)
        plugin.after_tool_call(first)
        assert first.retry is True

        second = _create_event(_mpp_402_result(), invocation_state=invocation_state)
        plugin.after_tool_call(second)

        assert second.retry is False
        assert mock_pm.generate_payment_header.call_count == 1
        assert "payment_failure_tool-123" in invocation_state

    def test_auto_payment_disabled_skips_mpp(self):
        mock_pm = MagicMock()
        plugin = _plugin_with_manager(mock_pm, _config(auto_payment=False))
        event = _create_event(_mpp_402_result())

        plugin.after_tool_call(event)

        mock_pm.generate_payment_header.assert_not_called()

    def test_tool_not_in_allowlist_skips_mpp(self):
        mock_pm = MagicMock()
        plugin = _plugin_with_manager(mock_pm, _config(payment_tool_allowlist=["other_tool"]))
        event = _create_event(_mpp_402_result())

        plugin.after_tool_call(event)

        mock_pm.generate_payment_header.assert_not_called()

    def test_missing_instrument_id_raises_configuration_error(self):
        """The setup hint must not be x402-specific now that MPP shares this path."""
        with patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager"):
            plugin = AgentCorePaymentsPlugin(config=_config(payment_instrument_id=None))
        plugin.payment_manager = MagicMock()

        with pytest.raises(Exception) as exc_info:
            plugin._process_payment_required_request(
                {"statusCode": 402, "headers": {"WWW-Authenticate": CHALLENGE}, "body": {}}
            )

        assert "payment_instrument_id is required" in str(exc_info.value)


class TestStrandsHttpRequestToolPreservesChallenges:
    """The built-in http_request tool must not strip WWW-Authenticate."""

    def test_402_response_headers_are_forwarded_verbatim(self):
        with patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager"):
            plugin = AgentCorePaymentsPlugin(config=_config())

        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_response.headers = {"WWW-Authenticate": CHALLENGE, "content-type": "application/json"}
        mock_response.json.return_value = {}

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.request.return_value = mock_response
            result = plugin.http_request(url="https://api.example.com/resource")

        text = result["content"][0]["text"]
        assert text.startswith("PAYMENT_REQUIRED: ")
        payload = json.loads(text[len("PAYMENT_REQUIRED: ") :])
        assert payload["statusCode"] == 402
        assert payload["headers"]["WWW-Authenticate"] == CHALLENGE

    def test_round_trip_challenge_survives_json_encoding(self):
        """Base64url request bytes must be byte-identical after the tool's JSON hop."""
        request_bytes = base64.urlsafe_b64encode(b'{"amount":"1000"}').decode().rstrip("=")
        challenge = f'Payment id="x", method="evm", intent="charge", request="{request_bytes}"'

        with patch("bedrock_agentcore.payments.integrations.strands.plugin.PaymentManager"):
            plugin = AgentCorePaymentsPlugin(config=_config())

        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_response.headers = {"WWW-Authenticate": challenge}
        mock_response.json.return_value = {}

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.request.return_value = mock_response
            result = plugin.http_request(url="https://api.example.com/resource")

        text = result["content"][0]["text"]
        payload = json.loads(text[len("PAYMENT_REQUIRED: ") :])
        assert payload["headers"]["WWW-Authenticate"] == challenge
        assert request_bytes in payload["headers"]["WWW-Authenticate"]
