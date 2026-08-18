"""Unit tests for MPP detection in the integration payment handlers."""

import json

import pytest

from bedrock_agentcore.payments.integrations.handlers import (
    GenericPaymentHandler,
    HttpRequestPaymentHandler,
    MCPRequestPaymentHandler,
    has_mpp_challenge,
)
from bedrock_agentcore.payments.mpp import is_mpp_payment_required

CHALLENGE = 'Payment id="evm-1", realm="api.example.com", method="evm", intent="charge", request="eyJhIjoxfQ"'


class TestHasMppChallenge:
    """Tests for the header-level MPP challenge probe."""

    def test_detects_canonical_header(self):
        assert has_mpp_challenge({"WWW-Authenticate": CHALLENGE}) is True

    @pytest.mark.parametrize("name", ["www-authenticate", "WWW-Authenticate", "Www-Authenticate"])
    def test_header_name_is_case_insensitive(self, name):
        assert has_mpp_challenge({name: CHALLENGE}) is True

    def test_detects_in_list_valued_header(self):
        assert has_mpp_challenge({"www-authenticate": ['Bearer realm="x"', CHALLENGE]}) is True

    def test_scheme_name_is_case_insensitive(self):
        assert has_mpp_challenge({"www-authenticate": CHALLENGE.replace("Payment ", "payment ", 1)}) is True

    def test_rejects_bearer_scheme(self):
        assert has_mpp_challenge({"WWW-Authenticate": 'Bearer realm="x"'}) is False

    def test_detects_payment_challenge_following_another_scheme(self):
        """A server may offer Bearer and Payment; the Payment option must still be found."""
        header = f'Bearer realm="api", error="invalid_token", {CHALLENGE}'

        assert has_mpp_challenge({"WWW-Authenticate": header}) is True

    def test_rejects_scheme_merely_prefixed_with_payment(self):
        """`PaymentXYZ` is a different scheme, not MPP."""
        assert has_mpp_challenge({"WWW-Authenticate": 'PaymentXYZ realm="x"'}) is False

    def test_rejects_bare_payment_token_with_no_auth_params(self):
        """Nothing to fulfill, so this must not be reported as a payment requirement."""
        assert has_mpp_challenge({"WWW-Authenticate": "Payment"}) is False

    @pytest.mark.parametrize(
        "header_value",
        [
            CHALLENGE,
            f'Bearer realm="api", {CHALLENGE}',
            'PaymentXYZ realm="x"',
            "Payment",
            'Bearer realm="x"',
            "",
        ],
    )
    def test_agrees_with_the_mpp_parser(self, header_value):
        """Detection must never disagree with the parser that does the real work."""
        headers = {"WWW-Authenticate": header_value}

        assert has_mpp_challenge(headers) == is_mpp_payment_required({"headers": headers})

    def test_rejects_missing_header(self):
        assert has_mpp_challenge({"content-type": "application/json"}) is False

    @pytest.mark.parametrize("headers", [None, "string", 42, [], {"www-authenticate": 42}])
    def test_malformed_input_is_false(self, headers):
        assert has_mpp_challenge(headers) is False


class TestGenericHandlerForwardsMppHeaders:
    """The generic handler must surface WWW-Authenticate so MPP reaches PaymentManager."""

    def _result(self, payload):
        return {"content": [{"text": f"PAYMENT_REQUIRED: {json.dumps(payload)}"}]}

    def test_extracts_402_status(self):
        handler = GenericPaymentHandler()
        result = self._result({"statusCode": 402, "headers": {"WWW-Authenticate": CHALLENGE}, "body": {}})

        assert handler.extract_status_code(result) == 402

    def test_forwards_www_authenticate_header_intact(self):
        handler = GenericPaymentHandler()
        result = self._result({"statusCode": 402, "headers": {"WWW-Authenticate": CHALLENGE}, "body": {}})

        headers = handler.extract_headers(result)

        assert headers["WWW-Authenticate"] == CHALLENGE

    def test_applies_authorization_header_to_tool_input(self):
        handler = GenericPaymentHandler()
        tool_input = {"url": "https://api.example.com/resource"}

        applied = handler.apply_payment_header(tool_input, {"Authorization": "Payment eyJhIjoxfQ"})

        assert applied is True
        assert tool_input["headers"]["Authorization"] == "Payment eyJhIjoxfQ"

    def test_authorization_header_merges_with_existing_headers(self):
        handler = GenericPaymentHandler()
        tool_input = {"url": "https://x", "headers": {"Accept": "application/json"}}

        handler.apply_payment_header(tool_input, {"Authorization": "Payment tok"})

        assert tool_input["headers"] == {"Accept": "application/json", "Authorization": "Payment tok"}


class TestHttpRequestHandlerForwardsMppHeaders:
    """The http_request handler's legacy text format must also carry MPP challenges."""

    def test_legacy_headers_block_python_repr(self):
        handler = HttpRequestPaymentHandler()
        result = {
            "content": [
                {"text": "Status Code: 402"},
                {"text": "Headers: " + repr({"WWW-Authenticate": CHALLENGE})},
            ]
        }

        assert handler.extract_status_code(result) == 402
        assert handler.extract_headers(result)["WWW-Authenticate"] == CHALLENGE

    def test_spec_compliant_marker_takes_precedence(self):
        handler = HttpRequestPaymentHandler()
        payload = {"statusCode": 402, "headers": {"WWW-Authenticate": CHALLENGE}, "body": {}}
        result = {"content": [{"text": f"PAYMENT_REQUIRED: {json.dumps(payload)}"}]}

        assert handler.extract_headers(result)["WWW-Authenticate"] == CHALLENGE


class TestMcpHandlerMppDetection:
    """MCP Gateway returns 200 with the requirement embedded; MPP lives in headers."""

    def test_infers_402_from_mpp_challenge_in_response_headers(self):
        handler = MCPRequestPaymentHandler()
        result = {"responseHeaders": {"WWW-Authenticate": CHALLENGE}, "structuredContent": {}}

        assert handler.extract_status_code(result) == 402

    def test_forwards_response_headers_for_mpp(self):
        handler = MCPRequestPaymentHandler()
        result = {"responseHeaders": {"WWW-Authenticate": CHALLENGE}, "structuredContent": {}}

        assert handler.extract_headers(result)["WWW-Authenticate"] == CHALLENGE

    def test_surfaces_structured_content_as_body_for_mpp(self):
        handler = MCPRequestPaymentHandler()
        result = {
            "responseHeaders": {"WWW-Authenticate": CHALLENGE},
            "structuredContent": {"error": "payment required"},
        }

        assert handler.extract_body(result) == {"error": "payment required"}

    def test_x402_detection_is_unchanged(self):
        handler = MCPRequestPaymentHandler()
        x402 = {"x402Version": 1, "accepts": [{"network": "base-sepolia"}]}
        result = {"structuredContent": x402}

        assert handler.extract_status_code(result) == 402
        assert handler.extract_body(result) == x402
        # No upstream headers present, so the synthetic content-type is used.
        assert handler.extract_headers(result) == {"content-type": "application/json"}

    def test_x402_prefers_real_response_headers_when_present(self):
        handler = MCPRequestPaymentHandler()
        result = {
            "responseHeaders": {"content-type": "application/json", "x-request-id": "abc"},
            "structuredContent": {"x402Version": 1, "accepts": []},
        }

        assert handler.extract_headers(result)["x-request-id"] == "abc"

    def test_no_payment_requirement_returns_none(self):
        handler = MCPRequestPaymentHandler()
        result = {"responseHeaders": {"content-type": "application/json"}, "structuredContent": {"ok": True}}

        assert handler.extract_status_code(result) is None
        assert handler.extract_headers(result) is None
        assert handler.extract_body(result) is None

    def test_bearer_challenge_is_not_a_payment_requirement(self):
        handler = MCPRequestPaymentHandler()
        result = {"responseHeaders": {"WWW-Authenticate": 'Bearer realm="x"'}, "structuredContent": {}}

        assert handler.extract_status_code(result) is None

    def test_applies_authorization_header_inside_parameters(self):
        handler = MCPRequestPaymentHandler()
        tool_input = {"toolName": "fetch", "parameters": {"url": "https://x"}}

        applied = handler.apply_payment_header(tool_input, {"Authorization": "Payment tok"})

        assert applied is True
        assert tool_input["parameters"]["headers"]["Authorization"] == "Payment tok"
