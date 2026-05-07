"""Tests for payment response handlers.

Tests for both the generic handler and tool-specific handlers.
"""

import json

import pytest

from bedrock_agentcore.payments.integrations.handlers import (
    GenericPaymentHandler,
    HttpRequestPaymentHandler,
    MCPRequestPaymentHandler,
    get_payment_handler,
)


class TestGenericPaymentHandler:
    """Tests for GenericPaymentHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GenericPaymentHandler()

    # PAYMENT_REQUIRED Marker Extraction Tests
    def test_extract_status_code_from_payment_required_marker(self):
        """Test extracting status code from PAYMENT_REQUIRED marker in content blocks."""
        payment_structure = {"statusCode": 402, "headers": {}, "body": {}}
        result = [
            {"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"},
        ]
        assert self.handler.extract_status_code(result) == 402

    def test_extract_headers_from_payment_required_marker(self):
        """Test extracting headers from PAYMENT_REQUIRED marker."""
        headers = {"content-type": "application/json"}
        payment_structure = {"statusCode": 402, "headers": headers, "body": {}}
        result = [
            {"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"},
        ]
        assert self.handler.extract_headers(result) == headers

    def test_extract_body_from_payment_required_marker(self):
        """Test extracting body from PAYMENT_REQUIRED marker."""
        body = {"error": "Payment required"}
        payment_structure = {"statusCode": 402, "headers": {}, "body": body}
        result = [
            {"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"},
        ]
        assert self.handler.extract_body(result) == body

    def test_extract_all_fields_from_payment_required_marker(self):
        """Test extracting all fields from PAYMENT_REQUIRED marker."""
        headers = {"content-type": "application/json"}
        body = {"error": "Payment required"}
        payment_structure = {"statusCode": 402, "headers": headers, "body": body}
        result = [
            {"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"},
        ]
        assert self.handler.extract_status_code(result) == 402
        assert self.handler.extract_headers(result) == headers
        assert self.handler.extract_body(result) == body

    # Fallback: Direct Dict Response Tests
    def test_extract_status_code_from_dict_direct(self):
        """Test extracting status code from marker format."""
        payment_structure = {"statusCode": 402, "headers": {}, "body": {}}
        result = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"}]
        assert self.handler.extract_status_code(result) == 402

    def test_extract_headers_from_dict_direct(self):
        """Test extracting headers from marker format."""
        headers = {"content-type": "application/json"}
        payment_structure = {"statusCode": 402, "headers": headers, "body": {}}
        result = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"}]
        assert self.handler.extract_headers(result) == headers

    def test_extract_body_from_dict_direct(self):
        """Test extracting body from marker format."""
        body = {"error": "Payment required"}
        payment_structure = {"statusCode": 402, "headers": {}, "body": body}
        result = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"}]
        assert self.handler.extract_body(result) == body

    def test_extract_all_fields_from_dict_response(self):
        """Test extracting all fields from marker format response."""
        headers = {"content-type": "application/json"}
        body = {"error": "Payment required"}
        payment_structure = {
            "statusCode": 402,
            "headers": headers,
            "body": body,
        }
        result = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"}]
        assert self.handler.extract_status_code(result) == 402
        assert self.handler.extract_headers(result) == headers
        assert self.handler.extract_body(result) == body

    # Error Handling Tests
    def test_extract_status_code_returns_none_when_not_found(self):
        """Test that None is returned when status code not found."""
        result = [{"text": 'PAYMENT_REQUIRED: {"headers": {}, "body": {}}'}]
        assert self.handler.extract_status_code(result) is None

    def test_extract_status_code_returns_none_for_non_402_status(self):
        """Test that status code is returned even for non-402 status codes."""
        payment_structure = {"statusCode": 200, "headers": {}, "body": {}}
        result = [
            {"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"},
        ]
        assert self.handler.extract_status_code(result) == 200

    def test_extract_headers_returns_none_when_not_found(self):
        """Test that None is returned when headers not found."""
        result = [{"text": 'PAYMENT_REQUIRED: {"statusCode": 402, "body": {}}'}]
        assert self.handler.extract_headers(result) is None

    def test_extract_body_returns_none_when_not_found(self):
        """Test that None is returned when body not found."""
        result = {"statusCode": 402, "headers": {}}
        assert self.handler.extract_body(result) is None

    def test_extract_status_code_handles_invalid_marker_format(self):
        """Test that invalid marker format is handled gracefully."""
        result = [{"text": "PAYMENT_REQUIRED: invalid json"}]
        assert self.handler.extract_status_code(result) is None

    def test_extract_status_code_handles_empty_content_blocks(self):
        """Test that empty content blocks are handled gracefully."""
        result = []
        assert self.handler.extract_status_code(result) is None

        status_code = self.handler.extract_status_code(result)

        assert status_code is None

    def test_extract_status_code_invalid_format(self):
        """Test with invalid status code format."""
        handler = HttpRequestPaymentHandler()

        result = [{"text": "Status Code: invalid"}]

        status_code = handler.extract_status_code(result)

        assert status_code is None

    def test_extract_status_code_with_reason_text(self):
        """Test extracting status code when reason text follows the number."""
        handler = HttpRequestPaymentHandler()

        result = [{"text": "Status Code: 402 Payment Required"}]

        status_code = handler.extract_status_code(result)

        assert status_code == 402

    def test_extract_status_code_with_multiple_reason_words(self):
        """Test extracting status code with multiple words in reason."""
        handler = HttpRequestPaymentHandler()

        result = [{"text": "Status Code: 500 Internal Server Error"}]

        status_code = handler.extract_status_code(result)

        assert status_code == 500

    # Verify structuredContent is NOT handled by GenericPaymentHandler
    def test_generic_handler_does_not_handle_structured_content(self):
        """Test that GenericPaymentHandler ignores structuredContent x402 data."""
        result = {
            "structuredContent": {
                "x402Version": 1,
                "accepts": [{"scheme": "exact"}],
            }
        }
        assert self.handler.extract_status_code(result) is None
        assert self.handler.extract_headers(result) is None
        assert self.handler.extract_body(result) is None

    # Verify MCP-shaped input is NOT handled by GenericPaymentHandler.apply_payment_header
    def test_generic_handler_apply_header_uses_top_level_for_mcp_shaped_input(self):
        """Test that GenericPaymentHandler puts headers at top level even for MCP-shaped input."""
        tool_input = {"toolName": "some_tool", "parameters": {"url": "https://example.com"}}
        payment_header = {"X-PAYMENT": "base64"}
        assert self.handler.apply_payment_header(tool_input, payment_header) is True
        # Headers go at top level, NOT inside parameters
        assert tool_input["headers"] == {"X-PAYMENT": "base64"}
        assert "headers" not in tool_input["parameters"]


class TestMCPRequestPaymentHandler:
    """Tests for MCPRequestPaymentHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MCPRequestPaymentHandler()
        self.x402_result = {
            "structuredContent": {
                "x402Version": 1,
                "accepts": [{"scheme": "exact", "network": "base-sepolia"}],
            }
        }

    # extract_status_code tests
    def test_extract_status_code_returns_402_for_x402_data(self):
        """Test returns 402 when structuredContent has x402 payment data."""
        assert self.handler.extract_status_code(self.x402_result) == 402

    def test_extract_status_code_returns_none_for_non_x402(self):
        """Test returns None when structuredContent lacks x402 fields."""
        result = {"structuredContent": {"someOther": "data"}}
        assert self.handler.extract_status_code(result) is None

    def test_extract_status_code_returns_none_for_missing_structured_content(self):
        """Test returns None when no structuredContent key."""
        assert self.handler.extract_status_code({"other": "data"}) is None

    def test_extract_status_code_returns_none_for_non_dict_result(self):
        """Test returns None for non-dict result."""
        assert self.handler.extract_status_code("not a dict") is None
        assert self.handler.extract_status_code([]) is None
        assert self.handler.extract_status_code(None) is None

    def test_extract_status_code_returns_none_for_partial_x402(self):
        """Test returns None when only x402Version present without accepts."""
        result = {"structuredContent": {"x402Version": 1}}
        assert self.handler.extract_status_code(result) is None

    # extract_headers tests
    def test_extract_headers_returns_content_type_for_x402(self):
        """Test returns content-type header for x402 data."""
        assert self.handler.extract_headers(self.x402_result) == {"content-type": "application/json"}

    def test_extract_headers_returns_none_for_non_x402(self):
        """Test returns None when no x402 data."""
        result = {"structuredContent": {"other": "data"}}
        assert self.handler.extract_headers(result) is None

    def test_extract_headers_returns_none_for_non_dict(self):
        """Test returns None for non-dict result."""
        assert self.handler.extract_headers("string") is None

    # extract_body tests
    def test_extract_body_returns_structured_content_for_x402(self):
        """Test returns structuredContent dict directly for x402 data."""
        body = self.handler.extract_body(self.x402_result)
        assert body == self.x402_result["structuredContent"]
        assert body["x402Version"] == 1
        assert body["accepts"] == [{"scheme": "exact", "network": "base-sepolia"}]

    def test_extract_body_returns_none_for_non_x402(self):
        """Test returns None when no x402 data."""
        result = {"structuredContent": {"other": "data"}}
        assert self.handler.extract_body(result) is None

    def test_extract_body_returns_none_for_non_dict(self):
        """Test returns None for non-dict result."""
        assert self.handler.extract_body([]) is None

    # apply_payment_header tests
    def test_apply_header_places_in_parameters_headers(self):
        """Test places headers inside parameters.headers."""
        tool_input = {"toolName": "proxy_tool_call", "parameters": {"url": "https://example.com"}}
        payment_header = {"X-PAYMENT": "base64-encoded"}
        assert self.handler.apply_payment_header(tool_input, payment_header) is True
        assert tool_input["parameters"]["headers"] == {"X-PAYMENT": "base64-encoded"}

    def test_apply_header_adds_to_existing_parameters_headers(self):
        """Test adds to existing parameters.headers."""
        tool_input = {
            "toolName": "proxy_tool_call",
            "parameters": {"headers": {"existing": "value"}},
        }
        payment_header = {"X-PAYMENT": "base64"}
        assert self.handler.apply_payment_header(tool_input, payment_header) is True
        assert tool_input["parameters"]["headers"]["existing"] == "value"
        assert tool_input["parameters"]["headers"]["X-PAYMENT"] == "base64"

    def test_apply_header_creates_parameters_headers_if_missing(self):
        """Test creates headers dict inside parameters if missing."""
        tool_input = {"toolName": "proxy_tool_call", "parameters": {}}
        payment_header = {"X-PAYMENT": "base64"}
        assert self.handler.apply_payment_header(tool_input, payment_header) is True
        assert tool_input["parameters"]["headers"] == {"X-PAYMENT": "base64"}

    def test_apply_header_returns_false_for_non_dict_parameters(self):
        """Test returns False when parameters is not a dict."""
        tool_input = {"toolName": "proxy_tool_call", "parameters": "not a dict"}
        payment_header = {"X-PAYMENT": "base64"}
        assert self.handler.apply_payment_header(tool_input, payment_header) is False

    def test_validate_tool_input_returns_false_for_non_dict_input(self):
        """Test validate_tool_input returns False for non-dict tool input."""
        assert self.handler.validate_tool_input("not a dict") is False

    def test_validate_tool_input_returns_false_without_tool_name(self):
        """Test validate_tool_input returns False when toolName is missing."""
        tool_input = {"parameters": {"url": "https://example.com"}}
        assert self.handler.validate_tool_input(tool_input) is False

    def test_validate_tool_input_returns_false_without_parameters(self):
        """Test validate_tool_input returns False when parameters is missing."""
        tool_input = {"toolName": "proxy_tool_call"}
        assert self.handler.validate_tool_input(tool_input) is False

    def test_validate_tool_input_returns_false_for_non_dict_parameters(self):
        """Test validate_tool_input returns False when parameters is not a dict."""
        tool_input = {"toolName": "proxy_tool_call", "parameters": "not a dict"}
        assert self.handler.validate_tool_input(tool_input) is False

    def test_validate_tool_input_returns_true_for_valid_input(self):
        """Test validate_tool_input returns True for valid MCP-shaped input."""
        tool_input = {"toolName": "proxy_tool_call", "parameters": {"url": "https://example.com"}}
        assert self.handler.validate_tool_input(tool_input) is True

    # Full flow test
    def test_full_mcp_x402_extraction(self):
        """Test complete extraction flow for MCP x402 response."""
        result = {
            "structuredContent": {
                "x402Version": 2,
                "accepts": [
                    {"scheme": "exact", "network": "eip155:8453", "maxAmountRequired": "1000"},
                ],
                "error": "Payment Required",
            }
        }
        assert self.handler.extract_status_code(result) == 402
        assert self.handler.extract_headers(result) == {"content-type": "application/json"}
        body = self.handler.extract_body(result)
        assert body["x402Version"] == 2
        assert body["accepts"][0]["network"] == "eip155:8453"


class TestHttpRequestPaymentHandlerExtractHeaders:
    """Tests for HttpRequestPaymentHandler.extract_headers."""

    def test_extract_headers_json_format(self):
        """Test extracting headers in JSON format."""
        handler = HttpRequestPaymentHandler()

        headers_dict = {"Content-Type": "application/json", "X-Custom": "value"}
        result = [{"text": f"Headers: {json.dumps(headers_dict)}"}]

        headers = handler.extract_headers(result)

        assert headers == headers_dict

    def test_extract_headers_dict_string_format(self):
        """Test extracting headers in Python dict string format (single-quoted keys)."""
        handler = HttpRequestPaymentHandler()

        result = [{"text": "Headers: {'Content-Type': 'application/json'}"}]

        headers = handler.extract_headers(result)

        assert headers == {"Content-Type": "application/json"}

    def test_extract_headers_not_found(self):
        """Test when headers are not found."""
        handler = HttpRequestPaymentHandler()

        result = [{"text": "Some other text"}]

        headers = handler.extract_headers(result)

        assert headers is None

    def test_extract_headers_empty_content(self):
        """Test with empty content."""
        handler = HttpRequestPaymentHandler()

        result = []

        headers = handler.extract_headers(result)

        assert headers is None

    def test_extract_headers_invalid_json(self):
        """Test with invalid JSON in headers."""
        handler = HttpRequestPaymentHandler()

        result = [{"text": "Headers: {invalid json}"}]

        headers = handler.extract_headers(result)

        assert headers is None


class TestHttpRequestPaymentHandlerExtractBody:
    """Tests for HttpRequestPaymentHandler.extract_body."""

    def test_extract_body_json_format(self):
        """Test extracting body in JSON format."""
        handler = HttpRequestPaymentHandler()

        body_dict = {"scheme": "exact", "network": "ethereum"}
        result = [{"text": f"Body: {json.dumps(body_dict)}"}]

        body = handler.extract_body(result)

        assert body == body_dict

    def test_extract_body_not_found(self):
        """Test when body is not found."""
        handler = HttpRequestPaymentHandler()

        result = [{"text": "Some other text"}]

        body = handler.extract_body(result)

        assert body is None

    def test_extract_body_empty_content(self):
        """Test with empty content."""
        handler = HttpRequestPaymentHandler()

        result = []

        body = handler.extract_body(result)

        assert body is None

    def test_extract_body_invalid_json(self):
        """Test with invalid JSON in body."""
        handler = HttpRequestPaymentHandler()

        result = [{"text": "Body: {invalid json}"}]

        body = handler.extract_body(result)

        assert body is None


class TestHttpRequestPaymentHandlerApplyPaymentHeader:
    """Tests for HttpRequestPaymentHandler.apply_payment_header."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = HttpRequestPaymentHandler()

    def test_apply_payment_header_success(self):
        """Test successfully applying payment header."""
        handler = HttpRequestPaymentHandler()

        tool_input = {"url": "https://example.com", "headers": {}}
        payment_header = {"X-PAYMENT": "base64-encoded"}

        result = handler.apply_payment_header(tool_input, payment_header)

        assert result is True
        assert tool_input["headers"]["X-PAYMENT"] == "base64-encoded"

    def test_apply_payment_header_creates_headers_dict(self):
        """Test that headers dict is created if it doesn't exist."""
        tool_input = {}
        payment_header = {"X-PAYMENT": "base64-encoded-value"}
        assert self.handler.apply_payment_header(tool_input, payment_header) is True
        assert tool_input["headers"] == payment_header

    def test_apply_payment_header_adds_to_existing_headers(self):
        """Test that payment header is added to existing headers."""
        tool_input = {"headers": {"content-type": "application/json"}}
        payment_header = {"X-PAYMENT": "base64-encoded-value"}
        assert self.handler.apply_payment_header(tool_input, payment_header) is True
        assert tool_input["headers"]["X-PAYMENT"] == "base64-encoded-value"
        assert tool_input["headers"]["content-type"] == "application/json"

    def test_apply_payment_header_returns_false_for_non_dict_input(self):
        """Test that False is returned for non-dict input."""
        tool_input = "not a dict"
        assert self.handler.validate_tool_input(tool_input) is False

    def test_validate_tool_input_returns_true_for_valid_input(self):
        """Test that True is returned for valid dict input."""
        tool_input = {"url": "https://example.com"}
        assert self.handler.validate_tool_input(tool_input) is True

    def test_apply_payment_header_returns_false_for_non_dict_headers(self):
        """Test that False is returned when headers is not a dict."""
        tool_input = {"headers": "not a dict"}
        payment_header = {"X-PAYMENT": "base64-encoded-value"}
        assert self.handler.apply_payment_header(tool_input, payment_header) is False

    # Content Block Extraction Tests
    def test_extract_from_content_dict_with_marker(self):
        """Test extracting from dict with content key containing marker."""
        payment_structure = {"statusCode": 402, "headers": {"x-test": "value"}, "body": {"msg": "test"}}
        result = {
            "content": [
                {"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"},
            ]
        }
        assert self.handler.extract_status_code(result) == 402
        assert self.handler.extract_headers(result) == {"x-test": "value"}
        assert self.handler.extract_body(result) == {"msg": "test"}

    def test_handler_with_object_attributes(self):
        """Test handler with objects that have attributes."""

        class ContentBlock:
            def __init__(self, text):
                self.text = text

        class Response:
            def __init__(self, content):
                self.content = content

        payment_structure = {"statusCode": 402, "headers": {"x-test": "value"}, "body": {"msg": "test"}}
        result = Response(
            [
                ContentBlock(f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"),
            ]
        )
        assert self.handler.extract_status_code(result) == 402
        assert self.handler.extract_headers(result) == {"x-test": "value"}
        assert self.handler.extract_body(result) == {"msg": "test"}


class TestHttpRequestPaymentHandler:
    """Tests for HttpRequestPaymentHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = HttpRequestPaymentHandler()

    def test_http_request_handler_inherits_from_generic(self):
        """Test that HttpRequestPaymentHandler inherits from GenericPaymentHandler."""
        assert isinstance(self.handler, GenericPaymentHandler)

    def test_http_request_handler_extracts_status_code_from_marker(self):
        """Test that http_request handler can extract status code from marker."""
        payment_structure = {"statusCode": 402, "headers": {}, "body": {}}
        result = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"}]
        assert self.handler.extract_status_code(result) == 402

    def test_http_request_handler_extracts_headers_from_marker(self):
        """Test that http_request handler can extract headers from marker."""
        headers = {"content-type": "application/json"}
        payment_structure = {"statusCode": 402, "headers": headers, "body": {}}
        result = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"}]
        assert self.handler.extract_headers(result) == headers

    def test_http_request_handler_extracts_body_from_marker(self):
        """Test that http_request handler can extract body from marker."""
        body = {"error": "Payment required"}
        payment_structure = {"statusCode": 402, "headers": {}, "body": body}
        result = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"}]
        assert self.handler.extract_body(result) == body


class TestHandlerRegistry:
    """Tests for handler registry and resolution."""

    def test_get_payment_handler_returns_http_request_handler(self):
        """Test that http_request tool gets the specific handler."""
        handler = get_payment_handler("http_request", {})
        assert isinstance(handler, HttpRequestPaymentHandler)

    def test_get_payment_handler_returns_generic_for_unknown_tool(self):
        """Test that unknown tools get the generic handler."""
        handler = get_payment_handler("unknown_tool", {})
        assert isinstance(handler, GenericPaymentHandler)
        assert not isinstance(handler, HttpRequestPaymentHandler)

    def test_get_payment_handler_returns_generic_for_custom_tool(self):
        """Test that custom tools get the generic handler."""
        handler = get_payment_handler("my_custom_http_tool", {})
        assert isinstance(handler, GenericPaymentHandler)
        assert not isinstance(handler, HttpRequestPaymentHandler)

    def test_get_payment_handler_returns_mcp_for_mcp_shaped_input(self):
        """Test that MCP Gateway shaped input returns MCPRequestPaymentHandler."""
        tool_input = {"toolName": "proxy_tool_call", "parameters": {"url": "https://example.com"}}
        handler = get_payment_handler("some_mcp_tool", tool_input)
        assert isinstance(handler, MCPRequestPaymentHandler)

    def test_get_payment_handler_mcp_detection_requires_both_keys(self):
        """Test that MCP detection requires both toolName and parameters."""
        # Only toolName
        handler = get_payment_handler("some_tool", {"toolName": "proxy_tool_call"})
        assert isinstance(handler, GenericPaymentHandler)

        # Only parameters
        handler = get_payment_handler("some_tool", {"parameters": {}})
        assert isinstance(handler, GenericPaymentHandler)

    def test_get_payment_handler_name_registry_takes_precedence_over_shape(self):
        """Test that name-based registry match takes precedence over MCP shape detection."""
        tool_input = {"toolName": "proxy_tool_call", "parameters": {}}
        handler = get_payment_handler("http_request", tool_input)
        assert isinstance(handler, HttpRequestPaymentHandler)

    def test_get_payment_handler_handles_empty_args(self):
        """Test that empty args fall back to generic handler."""
        handler = get_payment_handler("", {})
        assert isinstance(handler, GenericPaymentHandler)

    def test_get_payment_handler_handles_non_dict_input(self):
        """Test that non-dict input field falls back to generic handler."""
        handler = get_payment_handler("some_tool", "not a dict")
        assert isinstance(handler, GenericPaymentHandler)

    def test_generic_handler_works_with_marker_format(self):
        """Test that generic handler works with PAYMENT_REQUIRED marker format."""
        handler = get_payment_handler("some_unknown_tool", {})

        # Test with marker format
        tool_input = {
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer token",
            }
        }
        payment_header = {"X-PAYMENT": "base64-encoded"}

        result = handler.apply_payment_header(tool_input, payment_header)

        assert result is True
        assert tool_input["headers"]["Content-Type"] == "application/json"
        assert tool_input["headers"]["Authorization"] == "Bearer token"
        assert tool_input["headers"]["X-PAYMENT"] == "base64-encoded"


class TestPaymentResponseHandlerAbstractClass:
    """Tests for PaymentResponseHandler abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that PaymentResponseHandler cannot be instantiated directly."""
        from bedrock_agentcore.payments.integrations.handlers import PaymentResponseHandler

        with pytest.raises(TypeError):
            PaymentResponseHandler()


class TestHttpRequestPaymentHandlerEdgeCasesWithNonTextBlocks:
    """Tests for edge cases with non-text content blocks."""

    def test_extract_status_code_with_non_text_blocks(self):
        """Test extracting status code when blocks don't have text."""
        handler = HttpRequestPaymentHandler()

        payment_structure = {
            "statusCode": 402,
            "headers": {"x-payment-required": "true"},
            "body": {"error": "Payment required"},
        }
        marker_response = [
            {"other_key": "value"},
            {"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"},
        ]
        assert handler.extract_status_code(marker_response) == 402
        assert handler.extract_headers(marker_response) is not None
        assert handler.extract_body(marker_response) is not None

    def test_generic_handler_works_with_dict_format(self):
        """Test that generic handler works with marker format in content array."""
        handler = get_payment_handler("some_unknown_tool", {})

        # Test with marker format (spec-compliant)
        payment_structure = {
            "statusCode": 402,
            "headers": {"x-payment-required": "true"},
            "body": {"error": "Payment required"},
        }
        marker_response = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"}]

        assert handler.extract_status_code(marker_response) == 402
        assert handler.extract_headers(marker_response) is not None
        assert handler.extract_body(marker_response) is not None


class TestGenericPaymentHandlerEdgeCases:
    """Tests for GenericPaymentHandler edge cases and exception paths."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GenericPaymentHandler()

    def test_extract_content_array_from_object_with_content_attr(self):
        """Test _extract_content_array with an object that has a content attribute."""

        class MockResult:
            content = [{"text": "hello"}]

        result = self.handler._extract_content_array(MockResult())
        assert result == [{"text": "hello"}]

    def test_extract_content_array_returns_none_for_non_matching(self):
        """Test _extract_content_array returns None for unsupported types."""
        assert self.handler._extract_content_array(42) is None
        assert self.handler._extract_content_array("string") is None

    def test_extract_text_from_block_with_text_attribute(self):
        """Test _extract_text_from_block with an object that has a text attribute."""

        class MockBlock:
            text = "some text"

        assert self.handler._extract_text_from_block(MockBlock()) == "some text"

    def test_extract_text_from_block_returns_none_for_non_matching(self):
        """Test _extract_text_from_block returns None for unsupported types."""
        assert self.handler._extract_text_from_block(42) is None
        assert self.handler._extract_text_from_block({"other": "key"}) is None

    def test_parse_json_or_dict_returns_none_for_non_dict_json(self):
        """Test _parse_json_or_dict returns None when JSON parses to non-dict."""
        assert self.handler._parse_json_or_dict("[1, 2, 3]") is None
        assert self.handler._parse_json_or_dict('"just a string"') is None

    def test_parse_json_or_dict_returns_none_for_invalid_json(self):
        """Test _parse_json_or_dict returns None for invalid JSON."""
        assert self.handler._parse_json_or_dict("not json at all") is None

    def test_extract_payment_required_structure_exception_path(self):
        """Test _extract_payment_required_structure handles exceptions gracefully."""

        # A result that causes an exception during iteration
        class BadResult:
            @property
            def content(self):
                raise RuntimeError("boom")

        assert self.handler._extract_payment_required_structure(BadResult()) is None

    def test_extract_status_code_exception_path(self):
        """Test extract_status_code handles exceptions in _extract_payment_required_structure."""
        # Patch to raise an exception
        from unittest.mock import patch

        with patch.object(self.handler, "_extract_payment_required_structure", side_effect=Exception("boom")):
            assert self.handler.extract_status_code([]) is None

    def test_extract_headers_exception_path(self):
        """Test extract_headers handles exceptions gracefully."""
        from unittest.mock import patch

        with patch.object(self.handler, "_extract_payment_required_structure", side_effect=Exception("boom")):
            assert self.handler.extract_headers([]) is None

    def test_extract_body_exception_path(self):
        """Test extract_body handles exceptions gracefully."""
        from unittest.mock import patch

        with patch.object(self.handler, "_extract_payment_required_structure", side_effect=Exception("boom")):
            assert self.handler.extract_body([]) is None

    def test_apply_payment_header_non_dict_headers_returns_false(self):
        """Test apply_payment_header returns False when headers is not a dict."""
        tool_input = {"headers": "not-a-dict"}
        assert self.handler.apply_payment_header(tool_input, {"X-PAYMENT": "val"}) is False

    def test_apply_payment_header_exception_path(self):
        """Test apply_payment_header handles exceptions gracefully."""

        # Frozen dict that raises on update
        class FrozenDict(dict):
            def __setitem__(self, key, value):
                raise TypeError("frozen")

        tool_input = FrozenDict()
        assert self.handler.apply_payment_header(tool_input, {"X-PAYMENT": "val"}) is False

    def test_extract_status_code_non_int_status_code(self):
        """Test extract_status_code returns None when statusCode is not an int."""
        payment_structure = {"statusCode": "402", "headers": {}, "body": {}}
        result = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"}]
        assert self.handler.extract_status_code(result) is None

    def test_extract_headers_non_dict_headers(self):
        """Test extract_headers returns None when headers is not a dict."""
        payment_structure = {"statusCode": 402, "headers": "not-a-dict", "body": {}}
        result = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"}]
        assert self.handler.extract_headers(result) is None

    def test_extract_body_non_dict_body(self):
        """Test extract_body returns None when body is not a dict."""
        payment_structure = {"statusCode": 402, "headers": {}, "body": "not-a-dict"}
        result = [{"text": f"PAYMENT_REQUIRED: {json.dumps(payment_structure)}"}]
        assert self.handler.extract_body(result) is None


class TestMCPRequestPaymentHandlerExceptionPaths:
    """Tests for MCPRequestPaymentHandler exception handling paths."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MCPRequestPaymentHandler()

    def test_extract_status_code_exception_path(self):
        """Test extract_status_code handles exceptions gracefully."""

        # Object whose .get raises
        class BadDict(dict):
            def get(self, key, default=None):
                raise RuntimeError("boom")

        assert self.handler.extract_status_code(BadDict()) is None

    def test_extract_headers_exception_path(self):
        """Test extract_headers handles exceptions gracefully."""

        class BadDict(dict):
            def get(self, key, default=None):
                raise RuntimeError("boom")

        assert self.handler.extract_headers(BadDict()) is None

    def test_extract_body_exception_path(self):
        """Test extract_body handles exceptions gracefully."""

        class BadDict(dict):
            def get(self, key, default=None):
                raise RuntimeError("boom")

        assert self.handler.extract_body(BadDict()) is None

    def test_apply_payment_header_non_dict_headers_returns_false(self):
        """Test apply_payment_header returns False when parameters.headers is not a dict."""
        tool_input = {
            "toolName": "proxy_tool_call",
            "parameters": {"headers": "not-a-dict"},
        }
        assert self.handler.apply_payment_header(tool_input, {"X-PAYMENT": "val"}) is False

    def test_apply_payment_header_exception_path(self):
        """Test apply_payment_header handles exceptions gracefully."""
        # Missing 'parameters' key entirely
        tool_input = {}
        assert self.handler.apply_payment_header(tool_input, {"X-PAYMENT": "val"}) is False


class TestHttpRequestPaymentHandlerExceptionPaths:
    """Tests for HttpRequestPaymentHandler exception handling paths."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = HttpRequestPaymentHandler()

    def test_extract_status_code_outer_exception_path(self):
        """Test extract_status_code handles outer exceptions gracefully."""
        from unittest.mock import patch

        with patch.object(HttpRequestPaymentHandler, "_extract_content_array", side_effect=Exception("boom")):
            assert self.handler.extract_status_code([]) is None

    def test_extract_headers_outer_exception_path(self):
        """Test extract_headers handles outer exceptions gracefully."""
        from unittest.mock import patch

        with patch.object(HttpRequestPaymentHandler, "_extract_content_array", side_effect=Exception("boom")):
            assert self.handler.extract_headers([]) is None

    def test_extract_body_outer_exception_path(self):
        """Test extract_body handles outer exceptions gracefully."""
        from unittest.mock import patch

        with patch.object(HttpRequestPaymentHandler, "_extract_content_array", side_effect=Exception("boom")):
            assert self.handler.extract_body([]) is None

    def test_extract_body_invalid_json_continues(self):
        """Test extract_body continues past invalid JSON body blocks."""
        result = [
            {"text": "Body: not-valid-json"},
            {"text": 'Body: {"valid": "json"}'},
        ]
        body = self.handler.extract_body(result)
        assert body == {"valid": "json"}

    def test_extract_headers_with_non_text_blocks_skipped(self):
        """Test extract_headers skips non-text content blocks."""
        result = [
            {"image": "data"},
            {"text": 'Headers: {"x-test": "value"}'},
        ]
        headers = self.handler.extract_headers(result)
        assert headers == {"x-test": "value"}

    def test_extract_body_with_non_text_blocks_skipped(self):
        """Test extract_body skips non-text content blocks."""
        result = [
            {"image": "data"},
            {"text": 'Body: {"key": "value"}'},
        ]
        body = self.handler.extract_body(result)
        assert body == {"key": "value"}

    def test_extract_status_code_empty_status_code_text(self):
        """Test extract_status_code with 'Status Code:' but no number."""
        result = [{"text": "Status Code: "}]
        assert self.handler.extract_status_code(result) is None


class TestHttpRequestPaymentHandlerX402V2:
    """Tests for HttpRequestPaymentHandler X402 v2 support.

    X402 v2 conveys the payment requirement via a Payment-Required HTTP response header
    whose value is a base64-encoded JSON payload. The http_request tool includes this
    header in its Headers: {...} text block (using Python dict repr with single quotes).
    """

    def setup_method(self):
        """Set up test fixtures."""
        import base64

        self.handler = HttpRequestPaymentHandler()

        # Build a realistic x402 v2 payload
        self.x402_v2_payload = {
            "x402Version": 2,
            "error": "Payment required",
            "resource": {
                "url": "https://example.com/weather",
                "description": "Weather report",
                "mimeType": "application/json",
            },
            "accepts": [
                {
                    "scheme": "exact",
                    "network": "eip155:84532",
                    "amount": "1000",
                    "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF71",
                    "payTo": "0x2eDbF699657ae1A09D9C3833FD162A6b59344364",
                    "maxTimeoutSeconds": 300,
                    "extra": {"name": "USDC", "version": "2"},
                }
            ],
        }
        self.x402_v2_base64 = base64.b64encode(json.dumps(self.x402_v2_payload).encode()).decode()

    def test_extract_headers_with_payment_required_single_quotes(self):
        """Test extracting headers containing Payment-Required in Python dict repr format."""
        # This is what the http_request tool actually produces: str(dict) with single quotes
        result = [
            {"text": "Status Code: 402"},
            {"text": f"Headers: {{'payment-required': '{self.x402_v2_base64}', 'Content-Type': 'application/json'}}"},
            {"text": "Body: {}"},
        ]

        headers = self.handler.extract_headers(result)

        assert headers is not None
        assert "payment-required" in headers
        assert headers["payment-required"] == self.x402_v2_base64
        assert headers["Content-Type"] == "application/json"

    def test_extract_headers_with_payment_required_json_format(self):
        """Test extracting headers containing Payment-Required in JSON format."""
        import json as json_mod

        headers_dict = {"Payment-Required": self.x402_v2_base64, "Content-Type": "application/json"}
        result = [
            {"text": "Status Code: 402"},
            {"text": f"Headers: {json_mod.dumps(headers_dict)}"},
            {"text": "Body: {}"},
        ]

        headers = self.handler.extract_headers(result)

        assert headers is not None
        assert "Payment-Required" in headers
        assert headers["Payment-Required"] == self.x402_v2_base64

    def test_extract_status_code_from_402_v2_response(self):
        """Test extracting 402 status code from a v2 response."""
        result = [
            {"text": "Status Code: 402"},
            {"text": f"Headers: {{'payment-required': '{self.x402_v2_base64}', 'Content-Type': 'application/json'}}"},
            {"text": "Body: {}"},
        ]

        status_code = self.handler.extract_status_code(result)
        assert status_code == 402

    def test_full_v2_extraction_flow(self):
        """Test the complete v2 extraction flow: status code, headers, and body.

        This simulates the full flow where:
        1. Handler extracts status_code=402
        2. Handler extracts headers containing Payment-Required
        3. Plugin builds payment_required_request
        4. PaymentManager._extract_x402_payload finds the header and decodes it
        """
        import base64

        result = [
            {"text": "Status Code: 402"},
            {"text": f"Headers: {{'payment-required': '{self.x402_v2_base64}', 'Content-Type': 'application/json'}}"},
            {"text": 'Body: {"error": "Payment required"}'},
        ]

        # Step 1: Extract status code
        status_code = self.handler.extract_status_code(result)
        assert status_code == 402

        # Step 2: Extract headers (with Payment-Required)
        headers = self.handler.extract_headers(result)
        assert headers is not None
        assert "payment-required" in headers

        # Step 3: Extract body
        body = self.handler.extract_body(result)
        assert body == {"error": "Payment required"}

        # Step 4: Simulate what PaymentManager._extract_x402_payload does
        payment_required_header = None
        for key, value in headers.items():
            if key.lower() == "payment-required":
                payment_required_header = value
                break

        assert payment_required_header is not None

        # Step 5: Decode the base64 payload
        decoded = base64.b64decode(payment_required_header)
        x402_payload = json.loads(decoded)

        assert x402_payload["x402Version"] == 2
        assert "accepts" in x402_payload
        assert x402_payload["accepts"][0]["scheme"] == "exact"
        assert x402_payload["accepts"][0]["network"] == "eip155:84532"

    def test_extract_headers_case_insensitive_payment_required(self):
        """Test that various casings of Payment-Required are preserved in extracted headers."""
        # Server might return different casings
        for header_name in ["Payment-Required", "payment-required", "PAYMENT-REQUIRED"]:
            result = [
                {"text": f"Headers: {{'{header_name}': '{self.x402_v2_base64}'}}"},
            ]
            headers = self.handler.extract_headers(result)
            assert headers is not None
            assert header_name in headers
            assert headers[header_name] == self.x402_v2_base64

    def test_apply_payment_signature_header(self):
        """Test applying PAYMENT-SIGNATURE header (v2 format) to tool input."""
        tool_input = {"method": "GET", "url": "https://example.com/weather", "headers": {}}
        payment_header = {"PAYMENT-SIGNATURE": "base64-encoded-v2-signature"}

        result = self.handler.apply_payment_header(tool_input, payment_header)

        assert result is True
        assert tool_input["headers"]["PAYMENT-SIGNATURE"] == "base64-encoded-v2-signature"

    def test_v2_body_without_x402version_still_extracts(self):
        """Test that body without x402Version is still extracted (v2 uses headers, not body)."""
        result = [
            {"text": "Status Code: 402"},
            {"text": f"Headers: {{'payment-required': '{self.x402_v2_base64}', 'Content-Type': 'application/json'}}"},
            {"text": 'Body: {"error": "Payment required", "message": "Use Payment-Required header"}'},
        ]

        body = self.handler.extract_body(result)
        assert body is not None
        assert body["error"] == "Payment required"
        # Body does NOT need x402Version for v2 — that's in the header

    def test_parse_headers_string_with_base64_value(self):
        """Test _parse_headers_string handles base64 values with special chars correctly."""
        # Base64 can contain +, /, = which are valid in Python string literals
        headers_str = f"{{'payment-required': '{self.x402_v2_base64}'}}"

        parsed = HttpRequestPaymentHandler._parse_headers_string(headers_str)

        assert parsed is not None
        assert parsed["payment-required"] == self.x402_v2_base64

    def test_parse_headers_string_returns_none_for_invalid_input(self):
        """Test _parse_headers_string returns None for unparseable strings."""
        assert HttpRequestPaymentHandler._parse_headers_string("not a dict at all") is None
        assert HttpRequestPaymentHandler._parse_headers_string("{broken: syntax[") is None
        assert HttpRequestPaymentHandler._parse_headers_string("") is None
