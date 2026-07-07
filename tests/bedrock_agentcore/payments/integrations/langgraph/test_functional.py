"""Functional test: full 402 → sign → retry → 200 flow against a real testnet.

PRE-MERGE REQUIREMENT:
    These tests are skipped in CI (env vars not set). Any PR that modifies the
    langgraph middleware MUST be validated against live testnet before merge.
    Run this suite locally with a funded wallet on Base Sepolia to confirm the
    real 402 → sign → settle → retry flow works end-to-end.

Run:
    python -m pytest tests/bedrock_agentcore/payments/integrations/langgraph/test_functional.py -v -s

Required environment variables:
    ACP_PAYMENT_MANAGER_ARN   - PaymentManager ARN
    ACP_USER_ID               - User ID for payment processing
    ACP_PAYMENT_INSTRUMENT_ID - Payment instrument ID (funded wallet)
    ACP_REGION                - AWS region (e.g., us-east-1)
    ACP_TESTNET_URL           - x402-enabled testnet endpoint that returns 402

Optional:
    ACP_PAYMENT_SESSION_ID    - Active payment session ID (auto-created if not set)
    ACP_PAYMENT_CONNECTOR_ID  - Payment connector ID
    ACP_RETRY_DELAY           - Override retry delay (default 3.0s)
"""

import json
import os
from unittest.mock import MagicMock

import pytest
from langchain.messages import ToolMessage

from bedrock_agentcore.payments.integrations.langgraph import (
    AgentCorePaymentsConfig,
    AgentCorePaymentsMiddleware,
)
from bedrock_agentcore.payments.manager import PaymentManager

# Skip entire module if required env vars not configured
pytestmark = pytest.mark.skipif(
    not all(
        os.environ.get(k)
        for k in [
            "ACP_PAYMENT_MANAGER_ARN",
            "ACP_USER_ID",
            "ACP_PAYMENT_INSTRUMENT_ID",
            "ACP_REGION",
            "ACP_TESTNET_URL",
        ]
    ),
    reason=(
        "Testnet env vars not set (ACP_PAYMENT_MANAGER_ARN, ACP_USER_ID,"
        " ACP_PAYMENT_INSTRUMENT_ID, ACP_REGION, ACP_TESTNET_URL)"
    ),
)


@pytest.fixture(scope="module")
def payment_session_id():
    """Use existing session ID from env, or auto-create one ($1.00, 60 min)."""
    existing = os.environ.get("ACP_PAYMENT_SESSION_ID")
    if existing:
        print(f"\n[Using existing session: {existing}]")
        return existing

    print("\n[ACP_PAYMENT_SESSION_ID not set — auto-creating session ($1.00, 60 min)]")
    pm = PaymentManager(
        payment_manager_arn=os.environ["ACP_PAYMENT_MANAGER_ARN"],
        region_name=os.environ["ACP_REGION"],
    )
    session = pm.create_payment_session(
        user_id=os.environ["ACP_USER_ID"],
        limits={"maxSpendAmount": {"value": "1.00", "currency": "USD"}},
        expiry_time_in_minutes=60,
    )
    session_id = session["paymentSessionId"]
    print(f"[Created session: {session_id}]")
    return session_id


@pytest.fixture(scope="module")
def config(payment_session_id):
    return AgentCorePaymentsConfig(
        payment_manager_arn=os.environ["ACP_PAYMENT_MANAGER_ARN"],
        user_id=os.environ["ACP_USER_ID"],
        payment_instrument_id=os.environ["ACP_PAYMENT_INSTRUMENT_ID"],
        payment_session_id=payment_session_id,
        payment_connector_id=os.environ.get("ACP_PAYMENT_CONNECTOR_ID"),
        region=os.environ["ACP_REGION"],
        post_payment_retry_delay_seconds=float(os.environ.get("ACP_RETRY_DELAY", "3.0")),
    )


@pytest.fixture(scope="module")
def middleware(config):
    return AgentCorePaymentsMiddleware(config)


@pytest.fixture(scope="module")
def testnet_url():
    return os.environ["ACP_TESTNET_URL"]


class TestFullPaymentFlow:
    """End-to-end: http_request tool → 402 → middleware signs → retry → 200."""

    def test_http_request_tool_gets_402(self, middleware, testnet_url):
        """The built-in http_request tool returns PAYMENT_REQUIRED on 402."""
        tool = next(t for t in middleware.tools if t.name == "http_request")
        result = tool.invoke({"url": testnet_url})

        print(f"\n[http_request raw result]: {result[:200]}...")
        assert "PAYMENT_REQUIRED:" in result, f"Expected 402 from testnet, got: {result[:100]}"

        parsed = json.loads(result[len("PAYMENT_REQUIRED: ") :])
        assert parsed["statusCode"] == 402
        print(f"[402 body keys]: {list(parsed.get('body', {}).keys())}")

    def test_wrap_tool_call_full_flow(self, middleware, testnet_url):
        """wrap_tool_call intercepts 402, signs payment, retries, gets 200."""
        # Simulate what LangGraph does: create a ToolCallRequest-like object
        # with tool_call dict, then a handler that calls http_request
        tool = next(t for t in middleware.tools if t.name == "http_request")

        tool_args = {"url": testnet_url, "method": "GET", "headers": {}}

        request = MagicMock()
        request.tool_call = {
            "name": "http_request",
            "args": tool_args,
            "id": "functional-test-1",
        }

        call_count = [0]

        def handler(req):
            """Simulate LangGraph's tool execution — calls the actual http_request tool."""
            call_count[0] += 1
            content = tool.invoke(req.tool_call["args"])
            return ToolMessage(content=content, tool_call_id=req.tool_call["id"])

        print(f"\n[Calling wrap_tool_call against {testnet_url}]")
        result = middleware.wrap_tool_call(request, handler)

        print(f"[Handler called {call_count[0]} time(s)]")
        print(f"[Result content]: {result.content[:200]}...")

        # Should have been called twice: initial 402 + retry with payment header
        assert call_count[0] == 2, f"Expected 2 calls (402 + retry), got {call_count[0]}"

        # Result should NOT be a payment error
        assert "PAYMENT ERROR" not in result.content, f"Payment failed: {result.content}"

        # Result should be successful (200)
        parsed = json.loads(result.content)
        assert parsed["statusCode"] == 200, f"Expected 200 on retry, got: {parsed.get('statusCode')}"
        print(f"[Success] Paid content received: {json.dumps(parsed['body'], indent=2)[:200]}")

    def test_wrap_tool_call_mcp_gateway_shape(self, middleware, testnet_url):
        """wrap_tool_call works with MCP Gateway shaped tool input (parameters.headers)."""
        import httpx

        # MCP Gateway tools have args like: {"toolName": "...", "parameters": {"url": ..., "headers": {}}}
        # The MCPRequestPaymentHandler injects headers into parameters.headers
        tool_args = {
            "toolName": "fetch_paid_content",
            "parameters": {"url": testnet_url, "method": "GET", "headers": {}},
        }

        request = MagicMock()
        request.tool_call = {
            "name": "mcp_proxy_tool",
            "args": tool_args,
            "id": "functional-mcp-test",
        }

        call_count = [0]

        def handler(req):
            """Simulate MCP proxy: uses parameters.url and parameters.headers to make the real call."""
            call_count[0] += 1
            params = req.tool_call["args"]["parameters"]
            url = params["url"]
            headers = params.get("headers", {})

            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.request("GET", url, headers=headers)

            resp_headers = dict(resp.headers)
            try:
                resp_body = resp.json()
            except Exception:
                resp_body = {"text": resp.text}

            payload = {"statusCode": resp.status_code, "headers": resp_headers, "body": resp_body}

            if resp.status_code == 402:
                content = f"PAYMENT_REQUIRED: {json.dumps(payload)}"
            else:
                content = json.dumps(payload)

            return ToolMessage(content=content, tool_call_id=req.tool_call["id"])

        print(f"\n[MCP Gateway shape test against {testnet_url}]")
        result = middleware.wrap_tool_call(request, handler)

        print(f"[Handler called {call_count[0]} time(s)]")
        print(f"[Result content]: {result.content[:200]}...")

        assert call_count[0] == 2, f"Expected 2 calls (402 + retry), got {call_count[0]}"
        assert "PAYMENT ERROR" not in result.content, f"Payment failed: {result.content}"

        parsed = json.loads(result.content)
        assert parsed["statusCode"] == 200, f"Expected 200, got: {parsed.get('statusCode')}"

        # Verify header was injected into parameters.headers (not top-level headers)
        injected = tool_args["parameters"]["headers"]
        print(f"[MCP parameters.headers]: {list(injected.keys())}")
        assert len(injected) > 0, "No payment header injected into parameters.headers"
        print("[MCP Gateway flow succeeded with 200]")

    def test_payment_header_was_injected(self, middleware, testnet_url):
        """After wrap_tool_call, the tool_args dict has a payment header."""
        tool = next(t for t in middleware.tools if t.name == "http_request")

        tool_args = {"url": testnet_url, "method": "GET", "headers": {}}

        request = MagicMock()
        request.tool_call = {
            "name": "http_request",
            "args": tool_args,
            "id": "functional-test-2",
        }

        def handler(req):
            content = tool.invoke(req.tool_call["args"])
            return ToolMessage(content=content, tool_call_id=req.tool_call["id"])

        middleware.wrap_tool_call(request, handler)

        # After the flow, headers should contain a payment header
        injected_headers = tool_args.get("headers", {})
        print(f"\n[Injected headers]: {list(injected_headers.keys())}")
        assert len(injected_headers) > 0, "No payment header was injected"
        # Common header names: X-PAYMENT (v1) or PAYMENT-SIGNATURE (v2)
        has_payment_header = any(k.upper() in ("X-PAYMENT", "PAYMENT-SIGNATURE", "PAYMENT") for k in injected_headers)
        assert has_payment_header, f"Expected payment header, got: {list(injected_headers.keys())}"


class TestPaymentQueryTools:
    """Functional tests for payment query tools against real PaymentManager."""

    def test_get_payment_instrument(self, middleware):
        """get_payment_instrument returns real instrument data."""
        tool = next(t for t in middleware.tools if t.name == "get_payment_instrument")
        result = tool.invoke({})
        print(f"\n[get_payment_instrument]: {str(result)[:300]}")
        assert "paymentInstrumentId" in result or "payment_instrument_id" in str(result).lower()

    def test_get_payment_session(self, middleware):
        """get_payment_session returns real session data."""
        tool = next(t for t in middleware.tools if t.name == "get_payment_session")
        result = tool.invoke({})
        print(f"\n[get_payment_session]: {str(result)[:300]}")
        assert "paymentSessionId" in result or "payment_session_id" in str(result).lower()


class TestFallbackDetectionFunctional:
    """Functional test: tool returns raw JSON (no PAYMENT_REQUIRED: marker) and fallback detects 402."""

    def test_raw_json_tool_full_flow(self, middleware, testnet_url):
        """A tool returning raw JSON without the marker still gets payment processing via fallback."""
        import httpx

        tool_args = {"url": testnet_url, "headers": {}}

        request = MagicMock()
        request.tool_call = {"name": "raw_api_tool", "args": tool_args, "id": "fallback-test"}

        call_count = [0]

        def handler(req):
            """Tool that returns raw JSON — NO PAYMENT_REQUIRED: prefix."""
            call_count[0] += 1
            url = req.tool_call["args"]["url"]
            headers = req.tool_call["args"].get("headers", {})

            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.request("GET", url, headers=headers)

            resp_headers = dict(resp.headers)
            try:
                resp_body = resp.json()
            except Exception:
                resp_body = {"text": resp.text}

            # Raw JSON — no marker
            payload = json.dumps({"statusCode": resp.status_code, "headers": resp_headers, "body": resp_body})
            return ToolMessage(content=payload, tool_call_id=req.tool_call["id"])

        print(f"\n[Fallback detection test (no marker, no custom handler) against {testnet_url}]")
        result = middleware.wrap_tool_call(request, handler)

        print(f"[Handler called {call_count[0]} time(s)]")
        print(f"[Result content]: {result.content[:200]}...")

        # Fallback detected 402 from raw JSON and processed payment
        assert call_count[0] == 2, f"Expected 2 calls (402 + retry), got {call_count[0]}"
        assert "PAYMENT ERROR" not in result.content, f"Payment failed: {result.content}"

        parsed = json.loads(result.content)
        assert parsed["statusCode"] == 200, f"Expected 200, got {parsed.get('statusCode')}"
        print("[Fallback detection flow succeeded — no marker, no custom handler, got 200]")


class TestCustomHandlerRegistry:
    """Functional test: custom handler resolves and processes payment correctly."""

    def test_custom_handler_full_flow(self, config, testnet_url):
        """A custom handler registered for a tool name is used for detection, extraction, and injection."""
        from bedrock_agentcore.payments.integrations.handlers import GenericPaymentHandler

        # Custom handler that tracks all three phases
        class TrackingHandler(GenericPaymentHandler):
            def __init__(self):
                self.detect_called = False
                self.extract_called = False
                self.inject_called = False

            @staticmethod
            def _to_prepared(result):
                """Wrap raw content into prepared shape for GenericPaymentHandler."""
                if isinstance(result, str):
                    return {"content": [{"text": result}]}
                elif isinstance(result, list):
                    return {"content": result}
                return result

            def extract_status_code(self, result):
                self.detect_called = True
                return super().extract_status_code(self._to_prepared(result))

            def extract_headers(self, result):
                self.extract_called = True
                return super().extract_headers(self._to_prepared(result))

            def apply_payment_header(self, tool_input, payment_header):
                self.inject_called = True
                return super().apply_payment_header(tool_input, payment_header)

        custom_handler = TrackingHandler()

        from dataclasses import replace

        custom_config = replace(config, custom_handlers={"my_http_tool": custom_handler})
        mw = AgentCorePaymentsMiddleware(custom_config)

        http_tool = next(t for t in mw.tools if t.name == "http_request")
        tool_args = {"url": testnet_url, "method": "GET", "headers": {}}

        request = MagicMock()
        request.tool_call = {"name": "my_http_tool", "args": tool_args, "id": "custom-handler-test"}

        call_count = [0]

        def handler(req):
            call_count[0] += 1
            content = http_tool.invoke(req.tool_call["args"])
            return ToolMessage(content=content, tool_call_id=req.tool_call["id"])

        print(f"\n[Testing custom handler registry against {testnet_url}]")
        result = mw.wrap_tool_call(request, handler)

        assert custom_handler.detect_called, "Custom handler's extract_status_code was not invoked"
        assert custom_handler.extract_called, "Custom handler's extract_headers was not invoked"
        assert custom_handler.inject_called, "Custom handler's apply_payment_header was not invoked"
        print("[Custom handler used for detection: ✓, extraction: ✓, injection: ✓]")

        assert call_count[0] == 2, f"Expected 2 calls, got {call_count[0]}"
        assert "PAYMENT ERROR" not in result.content, f"Payment failed: {result.content}"

        parsed = json.loads(result.content)
        assert parsed["statusCode"] == 200
        print("[Custom handler flow succeeded with 200]")

    def test_custom_handler_non_marker_tool(self, config, testnet_url):
        """Custom handler detects 402 from a tool that does NOT use the PAYMENT_REQUIRED: marker."""
        import httpx

        from bedrock_agentcore.payments.integrations.handlers import PaymentResponseHandler

        # Custom handler that detects 402 from raw JSON (no marker prefix)
        class RawJsonHandler(PaymentResponseHandler):
            """Handles tools that return raw JSON like {"statusCode": 402, "headers": {...}, "body": {...}}"""

            def __init__(self):
                self.detect_called = False

            def extract_status_code(self, result):
                self.detect_called = True
                import json as _json

                # Custom handlers now receive raw content (str or list)
                texts = []
                if isinstance(result, str):
                    texts.append(result)
                elif isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                        elif isinstance(item, str):
                            texts.append(item)

                for text in texts:
                    try:
                        parsed = _json.loads(text)
                        if isinstance(parsed, dict):
                            return parsed.get("statusCode")
                    except (ValueError, TypeError):
                        pass
                return None

            def extract_headers(self, result):
                import json as _json

                texts = []
                if isinstance(result, str):
                    texts.append(result)
                elif isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                        elif isinstance(item, str):
                            texts.append(item)

                for text in texts:
                    try:
                        parsed = _json.loads(text)
                        if isinstance(parsed, dict):
                            return parsed.get("headers", {})
                    except (ValueError, TypeError):
                        pass
                return None

            def extract_body(self, result):
                import json as _json

                texts = []
                if isinstance(result, str):
                    texts.append(result)
                elif isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                        elif isinstance(item, str):
                            texts.append(item)

                for text in texts:
                    try:
                        parsed = _json.loads(text)
                        if isinstance(parsed, dict):
                            return parsed.get("body", {})
                    except (ValueError, TypeError):
                        pass
                return None

            def validate_tool_input(self, tool_input):
                return isinstance(tool_input, dict)

            def apply_payment_header(self, tool_input, payment_header):
                if "headers" not in tool_input:
                    tool_input["headers"] = {}
                tool_input["headers"].update(payment_header)
                return True

        custom_handler = RawJsonHandler()

        from dataclasses import replace

        custom_config = replace(config, custom_handlers={"raw_http_tool": custom_handler})
        mw = AgentCorePaymentsMiddleware(custom_config)

        tool_args = {"url": testnet_url, "headers": {}}

        request = MagicMock()
        request.tool_call = {"name": "raw_http_tool", "args": tool_args, "id": "non-marker-test"}

        call_count = [0]

        def handler(req):
            """Tool that returns raw JSON WITHOUT the PAYMENT_REQUIRED: marker."""
            call_count[0] += 1
            url = req.tool_call["args"]["url"]
            headers = req.tool_call["args"].get("headers", {})

            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.request("GET", url, headers=headers)

            resp_headers = dict(resp.headers)
            try:
                resp_body = resp.json()
            except Exception:
                resp_body = {"text": resp.text}

            # NO PAYMENT_REQUIRED: prefix — just raw JSON
            payload = json.dumps({"statusCode": resp.status_code, "headers": resp_headers, "body": resp_body})
            return ToolMessage(content=payload, tool_call_id=req.tool_call["id"])

        print(f"\n[Non-marker tool with custom handler against {testnet_url}]")
        result = mw.wrap_tool_call(request, handler)

        print(f"[Handler called {call_count[0]} time(s)]")
        print(f"[Custom handler detect_called: {custom_handler.detect_called}]")
        print(f"[Result content]: {result.content[:200]}...")

        # Custom handler detected 402 from raw JSON
        assert custom_handler.detect_called, "Custom handler was not invoked for detection"

        # Full flow worked
        assert call_count[0] == 2, f"Expected 2 calls (402 + retry), got {call_count[0]}"
        assert "PAYMENT ERROR" not in result.content, f"Payment failed: {result.content}"

        parsed = json.loads(result.content)
        assert parsed["statusCode"] == 200, f"Expected 200, got {parsed.get('statusCode')}"
        print("[Non-marker custom handler flow succeeded with 200]")
        from bedrock_agentcore.payments.integrations.handlers import GenericPaymentHandler

        # Custom handler that tracks all three phases
        class TrackingHandler(GenericPaymentHandler):
            def __init__(self):
                self.detect_called = False
                self.extract_called = False
                self.inject_called = False

            @staticmethod
            def _to_prepared(result):
                """Wrap raw content into prepared shape for GenericPaymentHandler."""
                if isinstance(result, str):
                    return {"content": [{"text": result}]}
                elif isinstance(result, list):
                    return {"content": result}
                return result

            def extract_status_code(self, result):
                self.detect_called = True
                return super().extract_status_code(self._to_prepared(result))

            def extract_headers(self, result):
                self.extract_called = True
                return super().extract_headers(self._to_prepared(result))

            def apply_payment_header(self, tool_input, payment_header):
                self.inject_called = True
                return super().apply_payment_header(tool_input, payment_header)

        custom_handler = TrackingHandler()

        # Create middleware with custom handler for "my_http_tool"
        from dataclasses import replace

        custom_config = replace(config, custom_handlers={"my_http_tool": custom_handler})
        mw = AgentCorePaymentsMiddleware(custom_config)

        # Use the real http_request tool under the hood, but the tool_call name is "my_http_tool"
        http_tool = next(t for t in mw.tools if t.name == "http_request")
        tool_args = {"url": testnet_url, "method": "GET", "headers": {}}

        request = MagicMock()
        request.tool_call = {"name": "my_http_tool", "args": tool_args, "id": "custom-handler-test"}

        call_count = [0]

        def handler(req):
            call_count[0] += 1
            content = http_tool.invoke(req.tool_call["args"])
            return ToolMessage(content=content, tool_call_id=req.tool_call["id"])

        print(f"\n[Testing custom handler registry against {testnet_url}]")
        result = mw.wrap_tool_call(request, handler)

        # Custom handler was used for all three phases
        assert custom_handler.detect_called, "Custom handler's extract_status_code was not invoked"
        assert custom_handler.extract_called, "Custom handler's extract_headers was not invoked"
        assert custom_handler.inject_called, "Custom handler's apply_payment_header was not invoked"
        print("[Custom handler used for detection: ✓, extraction: ✓, injection: ✓]")

        # Full flow still works (402 → sign → retry → 200)
        assert call_count[0] == 2, f"Expected 2 calls, got {call_count[0]}"
        assert "PAYMENT ERROR" not in result.content, f"Payment failed: {result.content}"

        parsed = json.loads(result.content)
        assert parsed["statusCode"] == 200
        print("[Custom handler flow succeeded with 200]")
