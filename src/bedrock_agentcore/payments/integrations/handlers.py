"""Tool-specific handlers for X.402 payment processing.

This module provides handlers for extracting payment information from different tool responses.
Each handler is responsible for parsing tool-specific response formats and extracting
HTTP status codes and X.402 payment requirements.
"""

import ast
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PaymentResponseHandler(ABC):
    """Abstract base class for tool-specific payment response handlers."""

    @abstractmethod
    def extract_status_code(self, result: Any) -> Optional[int]:
        """Extract HTTP status code from tool result.

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            Status code if found, None otherwise
        """
        pass

    @abstractmethod
    def extract_headers(self, result: Any) -> Optional[Dict[str, Any]]:
        """Extract HTTP headers from tool result.

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            Headers dictionary if found, None otherwise
        """
        pass

    @abstractmethod
    def extract_body(self, result: Any) -> Optional[Dict[str, Any]]:
        """Extract HTTP response body from tool result.

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            Body dictionary if found, None otherwise
        """
        pass

    def validate_tool_input(self, tool_input: Any) -> bool:
        """Validate that tool input is suitable for applying payment headers.

        Args:
            tool_input: The tool input to validate

        Returns:
            True if tool input is valid, False otherwise
        """
        if not isinstance(tool_input, dict):
            logger.warning("Tool input is not a dict, cannot add payment header")
            return False
        return True

    @abstractmethod
    def apply_payment_header(self, tool_input: Dict[str, Any], payment_header: Dict[str, str]) -> bool:
        """Apply payment header to tool input.

        Args:
            tool_input: The tool input dictionary to modify
            payment_header: The payment header to add (e.g., {"X-PAYMENT": "base64..."})

        Returns:
            True if header was successfully applied, False otherwise
        """
        pass


class GenericPaymentHandler(PaymentResponseHandler):
    """Generic handler for extracting payment information from tool responses.

    This handler extracts payment information from tool responses following the
    402 PaymentRequired Standard Response Structure Specification v1.0.

    Tools MUST return responses with the PAYMENT_REQUIRED marker containing:
    {
        "statusCode": 402,
        "headers": dict,
        "body": dict
    }

    This handler supports:
    - Standard PAYMENT_REQUIRED: marker format (spec-compliant)
    - Direct dictionary responses with statusCode, headers, body
    - Content arrays with text blocks (Anthropic format)
    - Fallback extraction for backward compatibility
    """

    PAYMENT_REQUIRED_MARKER = "PAYMENT_REQUIRED: "

    @staticmethod
    def _extract_content_array(result: Any) -> Optional[list]:
        """Extract content array from result in various formats.

        Handles:
        1. A list of content blocks
        2. A dict with 'content' key
        3. An object with 'content' attribute

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            Content array if found, None otherwise
        """
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return result.get("content")
        elif hasattr(result, "content"):
            return getattr(result, "content", None)
        return None

    @staticmethod
    def _extract_text_from_block(content_block: Any) -> Optional[str]:
        """Extract text from a content block.

        Handles:
        1. Objects with 'text' attribute
        2. Dicts with 'text' key

        Args:
            content_block: A single content block

        Returns:
            Text string if found, None otherwise
        """
        if hasattr(content_block, "text"):
            return getattr(content_block, "text", None)
        elif isinstance(content_block, dict) and "text" in content_block:
            return content_block.get("text")
        return None

    @staticmethod
    def _parse_json_or_dict(value_str: str) -> Optional[Dict[str, Any]]:
        """Parse a string as JSON.

        Args:
            value_str: String to parse

        Returns:
            Parsed dictionary if successful, None otherwise
        """
        try:
            result = json.loads(value_str)
            if isinstance(result, dict):
                return result
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    def _extract_payment_required_structure(self, result: Any) -> Optional[Dict[str, Any]]:
        """Extract payment_required structure from result.

        Follows the 402 PaymentRequired Standard Response Structure Specification v1.0.
        Looks for the PAYMENT_REQUIRED: marker in content blocks.

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            Parsed payment_required dict if found, None otherwise
        """
        try:
            # Try to find PAYMENT_REQUIRED marker in content blocks
            content = self._extract_content_array(result)
            if content:
                for content_block in content:
                    text_data = self._extract_text_from_block(content_block)
                    if isinstance(text_data, str) and text_data.startswith(self.PAYMENT_REQUIRED_MARKER):
                        # Extract JSON after marker
                        payment_json = text_data[len(self.PAYMENT_REQUIRED_MARKER) :]
                        parsed = self._parse_json_or_dict(payment_json)
                        if parsed and isinstance(parsed, dict):
                            logger.debug("Extracted payment_required structure from PAYMENT_REQUIRED marker")
                            return parsed

            return None
        except Exception as e:
            logger.debug("Error extracting payment_required structure: %s", str(e))
            return None

    def extract_status_code(self, result: Any) -> Optional[int]:
        """Extract HTTP status code from tool result.

        Follows the 402 PaymentRequired Standard Response Structure Specification v1.0.

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            Status code if found, None otherwise
        """
        try:
            # Extract from payment_required structure (spec-compliant)
            payment_required = self._extract_payment_required_structure(result)
            if payment_required:
                status_code = payment_required.get("statusCode")
                if isinstance(status_code, int):
                    return status_code

            return None
        except Exception as e:
            logger.error("Error extracting status code from result: %s", str(e))
            return None

    def extract_headers(self, result: Any) -> Optional[Dict[str, Any]]:
        """Extract HTTP headers from tool result.

        Follows the 402 PaymentRequired Standard Response Structure Specification v1.0.

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            Headers dictionary if found, None otherwise
        """
        try:
            # Extract from payment_required structure (spec-compliant)
            payment_required = self._extract_payment_required_structure(result)
            if payment_required:
                headers = payment_required.get("headers")
                if isinstance(headers, dict):
                    return headers

            return None
        except Exception as e:
            logger.error("Error extracting headers from result: %s", str(e))
            return None

    def extract_body(self, result: Any) -> Optional[Dict[str, Any]]:
        """Extract HTTP response body from tool result.

        Follows the 402 PaymentRequired Standard Response Structure Specification v1.0.

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            Body dictionary if found, None otherwise
        """
        try:
            # Extract from payment_required structure
            payment_required = self._extract_payment_required_structure(result)
            if payment_required:
                body = payment_required.get("body")
                if isinstance(body, dict):
                    return body

            return None
        except Exception as e:
            logger.debug("Error extracting body from result: %s", str(e))
            return None

    def apply_payment_header(self, tool_input: Dict[str, Any], payment_header: Dict[str, str]) -> bool:
        """Apply payment header to tool input.

        Adds the payment header to the headers dictionary in the tool input.

        Args:
            tool_input: The tool input dictionary to modify
            payment_header: The payment header to add (e.g., {"X-PAYMENT": "base64..."})

        Returns:
            True if header was successfully applied, False otherwise
        """
        try:
            # Ensure headers dict exists
            if "headers" not in tool_input:
                tool_input["headers"] = {}

            # Add payment header to the headers dict
            if isinstance(tool_input["headers"], dict):
                tool_input["headers"].update(payment_header)
                logger.info("Added payment header to tool input headers: %s", list(payment_header.keys()))
                return True
            else:
                logger.warning("Tool input headers is not a dict, cannot add payment header")
                return False
        except Exception as e:
            logger.error("Error applying payment header to tool input: %s", str(e))
            return False


class MCPRequestPaymentHandler(PaymentResponseHandler):
    """Handler for MCP Gateway proxy_tool_call responses.

    This handler extracts payment information from MCP Gateway responses where
    x402 payment data is returned in the structuredContent field, and applies
    payment headers inside parameters.headers for MCP-shaped tool inputs.
    """

    def validate_tool_input(self, tool_input: Any) -> bool:
        """Validate that tool input has MCP Gateway shape suitable for payment headers.

        Args:
            tool_input: The tool input to validate

        Returns:
            True if tool input is valid MCP Gateway shape, False otherwise
        """
        if not super().validate_tool_input(tool_input):
            return False
        if "toolName" not in tool_input or "parameters" not in tool_input:
            logger.warning("Tool input does not have MCP Gateway shape (toolName + parameters)")
            return False
        if not isinstance(tool_input["parameters"], dict):
            logger.warning("Tool input parameters is not a dict, cannot add payment header")
            return False
        return True

    @staticmethod
    def _is_x402_payment_data(data: Any) -> bool:
        """Check if a dictionary contains x402 payment required data.

        Args:
            data: Dictionary to check

        Returns:
            True if it contains x402Version and accepts fields
        """
        return isinstance(data, dict) and "x402Version" in data and "accepts" in data

    def extract_status_code(self, result: Any) -> Optional[int]:
        """Extract status code from MCP Gateway tool result.

        MCP Gateway returns HTTP 200 with x402 payment data embedded in
        structuredContent, so there is no explicit 402 status code. We infer
        402 from the presence of x402Version + accepts fields.

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            402 if x402 payment data found, None otherwise
        """
        try:
            if isinstance(result, dict) and self._is_x402_payment_data(result.get("structuredContent")):
                return 402
            return None
        except Exception as e:
            logger.error("Error extracting status code from MCP result: %s", str(e))
            return None

    def extract_headers(self, result: Any) -> Optional[Dict[str, Any]]:
        """Extract headers from MCP Gateway tool result.

        Returns content-type header when structuredContent contains x402 data.

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            Headers dict if x402 data found, None otherwise
        """
        try:
            if isinstance(result, dict) and self._is_x402_payment_data(result.get("structuredContent")):
                return {"content-type": "application/json"}
            return None
        except Exception as e:
            logger.error("Error extracting headers from MCP result: %s", str(e))
            return None

    def extract_body(self, result: Any) -> Optional[Dict[str, Any]]:
        """Extract body from MCP Gateway tool result.

        Returns the structuredContent dict directly when it contains x402 data.

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            structuredContent dict if x402 data found, None otherwise
        """
        try:
            if isinstance(result, dict):
                sc = result.get("structuredContent")
                if self._is_x402_payment_data(sc):
                    return sc
            return None
        except Exception as e:
            logger.debug("Error extracting body from MCP result: %s", str(e))
            return None

    def apply_payment_header(self, tool_input: Dict[str, Any], payment_header: Dict[str, str]) -> bool:
        """Apply payment header to MCP Gateway tool input.

        Places headers inside parameters.headers and sets method to POST.

        Args:
            tool_input: The tool input dictionary to modify
            payment_header: The payment header to add

        Returns:
            True if header was successfully applied, False otherwise
        """
        try:
            params = tool_input["parameters"]
            if "headers" not in params:
                params["headers"] = {}
            if isinstance(params["headers"], dict):
                params["headers"].update(payment_header)
                logger.info("Added payment header to parameters.headers: %s", list(payment_header.keys()))
                return True

            logger.warning("parameters.headers is not a dict, cannot add payment header")
            return False
        except Exception as e:
            logger.error("Error applying payment header to MCP tool input: %s", str(e))
            return False


class HttpRequestPaymentHandler(GenericPaymentHandler):
    """Handler for http_request tool responses.

    See: https://github.com/strands-agents/tools/blob/main/src/strands_tools/http_request.py
    This handler supports both x402Version 1 and x402Version 2.

    This handler extends GenericPaymentHandler with http_request-specific optimizations,
    adding support for legacy "Status Code:", "Headers:", "Body:" text block format.

    For x402 v2, the payment requirement is conveyed via a ``Payment-Required`` HTTP
    response header whose value is a base64-encoded JSON payload.  The http_request tool
    includes ``Payment-Required`` in its important-headers filter, so the header appears
    inside the ``Headers: {...}`` text block.  This handler parses that text block
    (which uses Python dict repr with single quotes) via ``ast.literal_eval`` as a
    fallback when ``json.loads`` fails, ensuring the ``Payment-Required`` header value
    is available for downstream extraction by ``PaymentManager._extract_x402_payload``.
    """

    @staticmethod
    def _parse_headers_string(headers_str: str) -> Optional[Dict[str, Any]]:
        """Parse a headers string that may be JSON or Python dict repr.

        The http_request tool formats headers with ``f"Headers: {headers_text}"``
        where *headers_text* is a Python dict.  ``str(dict)`` produces single-quoted
        keys/values which are not valid JSON but can be parsed by
        ``ast.literal_eval``.

        Args:
            headers_str: The string after ``Headers:`` prefix.

        Returns:
            Parsed dictionary if successful, None otherwise.
        """
        # Try JSON first (double-quoted keys)
        try:
            result = json.loads(headers_str)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: Python dict repr (single-quoted keys from str(dict))
        try:
            result = ast.literal_eval(headers_str)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass

        return None

    def extract_status_code(self, result: Any) -> Optional[int]:
        """Extract HTTP status code from http_request tool result.

        First tries spec-compliant format via parent class, then falls back to legacy format.

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            Status code if found, None otherwise
        """
        try:
            # Try spec-compliant format first (via parent class)
            status_code = super().extract_status_code(result)
            if status_code is not None:
                return status_code

            # Fallback to legacy format
            content = self._extract_content_array(result)
            if not content:
                return None

            for content_block in content:
                text_data = self._extract_text_from_block(content_block)
                if isinstance(text_data, str) and text_data.startswith("Status Code:"):
                    try:
                        status_code_str = text_data.replace("Status Code:", "").strip().split()[0]
                        return int(status_code_str)
                    except (ValueError, IndexError):
                        logger.error("Failed to parse status code: %s", status_code_str)
                        continue

            return None
        except Exception as e:
            logger.error("Error extracting status code from result: %s", str(e))
            return None

    def extract_headers(self, result: Any) -> Optional[Dict[str, Any]]:
        """Extract HTTP headers from http_request tool result.

        First tries spec-compliant format via parent class, then falls back to legacy format.
        The legacy format supports both JSON and Python dict repr (single-quoted keys)
        to handle the http_request tool's ``f"Headers: {headers_text}"`` output.

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            Headers dictionary if found, None otherwise
        """
        try:
            # Try spec-compliant format first (via parent class)
            headers = super().extract_headers(result)
            if headers is not None:
                return headers

            # Fallback to legacy format
            content = self._extract_content_array(result)
            if not content:
                return None

            for content_block in content:
                text_data = self._extract_text_from_block(content_block)
                if not isinstance(text_data, str):
                    continue

                # Check for "Headers: {...}" format
                if text_data.startswith("Headers:"):
                    headers_str = text_data.replace("Headers:", "", 1).strip()
                    parsed = self._parse_headers_string(headers_str)
                    if parsed:
                        return parsed
                    logger.error("Failed to parse headers string: %s", headers_str)
                    continue

            return None
        except Exception as e:
            logger.error("Error extracting headers from result: %s", str(e))
            return None

    def extract_body(self, result: Any) -> Optional[Dict[str, Any]]:
        """Extract HTTP response body from http_request tool result.

        First tries spec-compliant format via parent class, then falls back to legacy format.

        Args:
            result: The tool result from AfterToolCallEvent

        Returns:
            Body dictionary if found, None otherwise
        """
        try:
            # Try spec-compliant format first (via parent class)
            body = super().extract_body(result)
            if body is not None:
                return body

            # Fallback to legacy format
            content = self._extract_content_array(result)
            if not content:
                return None

            for content_block in content:
                text_data = self._extract_text_from_block(content_block)
                if not isinstance(text_data, str):
                    continue

                # Check for "Body: {...}" format
                if text_data.startswith("Body:"):
                    body_str = text_data.replace("Body:", "", 1).strip()
                    try:
                        return json.loads(body_str)
                    except json.JSONDecodeError as e:
                        logger.debug("Failed to parse body as JSON: %s", str(e))
                        continue
            return None
        except Exception as e:
            logger.debug("Error extracting body from result: %s", str(e))
            return None


# Registry of tool handlers (name-based)
PAYMENT_HANDLERS: Dict[str, PaymentResponseHandler] = {
    "http_request": HttpRequestPaymentHandler(),
}

# Singleton handler instances
_GENERIC_HANDLER = GenericPaymentHandler()
_MCP_HANDLER = MCPRequestPaymentHandler()


def get_payment_handler(tool_name: str, tool_input: Dict[str, Any]) -> PaymentResponseHandler:
    """Get the payment handler for a specific tool.

    This function implements a handler resolution strategy:
    1. First, try to get a tool-specific handler from the name-based registry
    2. Then, detect MCP Gateway shape from tool input (toolName + parameters keys)
    3. If not found, return the generic handler as a fallback

    Args:
        tool_name: Name of the tool
        tool_input: The tool input dictionary

    Returns:
        PaymentResponseHandler (tool-specific, MCP, or generic fallback)
    """
    # First, try to get tool-specific handler by name
    handler = PAYMENT_HANDLERS.get(tool_name)
    if handler:
        logger.debug("Using tool-specific handler for tool: %s", tool_name)
        return handler

    # Detect MCP Gateway shape from tool input
    if isinstance(tool_input, dict) and "toolName" in tool_input and "parameters" in tool_input:
        logger.debug("Using MCP handler for tool: %s (detected toolName+parameters shape)", tool_name)
        return _MCP_HANDLER

    # Fall back to generic handler
    logger.debug("Using generic handler for tool: %s", tool_name)
    return _GENERIC_HANDLER
