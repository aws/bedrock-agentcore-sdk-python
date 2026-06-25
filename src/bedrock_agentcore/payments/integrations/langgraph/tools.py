"""Built-in tools for AgentCorePaymentsMiddleware."""

import json
import logging
from typing import Any, Dict, Optional, Union

import httpx
from langchain.tools import tool

logger = logging.getLogger(__name__)


def make_http_request_tool(middleware: Any):
    """Create an http_request tool that closes over the middleware instance.

    Returns PAYMENT_REQUIRED: marker on 402 for automatic payment processing.
    """

    @tool
    def http_request(
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Union[Dict[str, Any], str]] = None,
    ) -> str:
        """Call an HTTP endpoint. 402 Payment Required responses are settled automatically.

        Args:
            url: The full URL to request.
            method: HTTP method. Defaults to GET.
            headers: Optional request headers.
            body: Optional request body. Dict is sent as JSON; str is sent as-is.

        Returns:
            JSON string with statusCode, headers, and body. Prefixed with
            PAYMENT_REQUIRED: on 402 for automatic payment processing.
        """
        request_headers = dict(headers) if headers else {}
        method_upper = method.upper()

        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                if body is None or method_upper in ("GET", "HEAD"):
                    resp = client.request(method_upper, url, headers=request_headers)
                elif isinstance(body, str):
                    resp = client.request(method_upper, url, headers=request_headers, content=body)
                else:
                    resp = client.request(method_upper, url, headers=request_headers, json=body)
        except httpx.RequestError as exc:
            return json.dumps({"statusCode": 0, "error": f"Request failed: {exc}", "url": url})

        response_headers = dict(resp.headers)
        try:
            response_body: Any = resp.json()
        except Exception:
            response_body = {"text": resp.text}

        payload = {
            "statusCode": resp.status_code,
            "headers": response_headers,
            "body": response_body,
        }

        if resp.status_code == 402:
            return f"PAYMENT_REQUIRED: {json.dumps(payload)}"

        return json.dumps(payload)

    return http_request


def make_get_payment_instrument_tool(middleware: Any):
    """Create a get_payment_instrument tool."""

    @tool
    def get_payment_instrument(
        payment_instrument_id: Optional[str] = None,
        user_id: Optional[str] = None,
        payment_connector_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve details about a specific payment instrument.

        Args:
            payment_instrument_id: Instrument ID (falls back to middleware config).
            user_id: User ID (falls back to middleware config).
            payment_connector_id: Connector ID (optional).

        Returns:
            Payment instrument details dictionary.
        """
        resolved_id = (
            payment_instrument_id.strip() if payment_instrument_id else None
        ) or middleware.config.payment_instrument_id
        resolved_user = (user_id.strip() if user_id else None) or middleware.config.user_id
        return middleware.payment_manager.get_payment_instrument(
            user_id=resolved_user,
            payment_instrument_id=resolved_id,
            payment_connector_id=payment_connector_id,
        )

    return get_payment_instrument


def make_list_payment_instruments_tool(middleware: Any):
    """Create a list_payment_instruments tool."""

    @tool
    def list_payment_instruments(
        user_id: Optional[str] = None,
        payment_connector_id: Optional[str] = None,
        max_results: int = 100,
        next_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List all payment instruments for a user.

        Args:
            user_id: User ID (falls back to middleware config).
            payment_connector_id: Filter by connector (optional).
            max_results: Maximum results to return (default 100).
            next_token: Pagination token (optional).

        Returns:
            Dictionary with paymentInstruments list and optional nextToken.
        """
        resolved_user = (user_id.strip() if user_id else None) or middleware.config.user_id
        return middleware.payment_manager.list_payment_instruments(
            user_id=resolved_user,
            payment_connector_id=payment_connector_id,
            max_results=max_results,
            next_token=next_token,
        )

    return list_payment_instruments


def make_get_payment_instrument_balance_tool(middleware: Any):
    """Create a get_payment_instrument_balance tool."""

    @tool
    def get_payment_instrument_balance(
        payment_instrument_id: str,
        chain: str = "BASE_SEPOLIA",
        token: str = "USDC",
        payment_connector_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get the token balance for a payment instrument on a blockchain.

        Args:
            payment_instrument_id: Instrument ID (required).
            chain: Blockchain chain (e.g., BASE_SEPOLIA, SOLANA_DEVNET).
            token: Token to query (e.g., USDC).
            payment_connector_id: Connector ID (falls back to config).
            user_id: User ID (falls back to config).

        Returns:
            Dictionary with balance information.
        """
        resolved_user = (user_id.strip() if user_id else None) or middleware.config.user_id
        resolved_connector = payment_connector_id or middleware.config.payment_connector_id
        return middleware.payment_manager.get_payment_instrument_balance(
            payment_connector_id=resolved_connector,
            payment_instrument_id=payment_instrument_id,
            chain=chain,
            token=token,
            user_id=resolved_user,
        )

    return get_payment_instrument_balance


def make_get_payment_session_tool(middleware: Any):
    """Create a get_payment_session tool."""

    @tool
    def get_payment_session(
        payment_session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve details about a specific payment session.

        Args:
            payment_session_id: Session ID (falls back to middleware config).
            user_id: User ID (falls back to middleware config).

        Returns:
            Payment session details dictionary.
        """
        resolved_id = (
            payment_session_id.strip() if payment_session_id else None
        ) or middleware.config.payment_session_id
        resolved_user = (user_id.strip() if user_id else None) or middleware.config.user_id
        return middleware.payment_manager.get_payment_session(
            user_id=resolved_user,
            payment_session_id=resolved_id,
        )

    return get_payment_session
