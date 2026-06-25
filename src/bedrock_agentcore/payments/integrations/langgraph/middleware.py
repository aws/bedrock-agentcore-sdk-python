"""AgentCorePaymentsMiddleware for LangGraph agents."""

import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from bedrock_agentcore.payments.integrations.error_messages import get_payment_error_message
from bedrock_agentcore.payments.integrations.handlers import (
    PaymentResponseHandler,
    get_payment_handler,
)
from bedrock_agentcore.payments.manager import (
    PaymentError,
    PaymentInstrumentConfigurationRequired,
    PaymentManager,
    PaymentSessionConfigurationRequired,
)

from ..config import AgentCorePaymentsConfig
from .tools import (
    make_get_payment_instrument_balance_tool,
    make_get_payment_instrument_tool,
    make_get_payment_session_tool,
    make_http_request_tool,
    make_list_payment_instruments_tool,
)

logger = logging.getLogger(__name__)

# Deterministic error messages per exception type.


class _FallbackHandler:
    """Minimal handler wrapping pre-parsed 402 data from fallback detection."""

    def __init__(self, parsed: Dict[str, Any]):
        self._parsed = parsed

    def extract_status_code(self, result: Any) -> Optional[int]:
        return self._parsed.get("statusCode")

    def extract_headers(self, result: Any) -> Optional[Dict[str, Any]]:
        return self._parsed.get("headers", {})

    def extract_body(self, result: Any) -> Optional[Dict[str, Any]]:
        return self._parsed.get("body", {})


class AgentCorePaymentsMiddleware(AgentMiddleware):
    """Middleware that intercepts tool calls to handle x402 Payment Required responses.

    This middleware wraps tool execution to automatically detect HTTP 402 responses,
    process x402 payment requirements via PaymentManager, and retry the tool call
    with payment credentials.

    Usage:
        config = AgentCorePaymentsConfig(
            payment_manager_arn="arn:aws:...",
            user_id="user-123",
            payment_instrument_id="instrument-456",
            payment_session_id="session-789",
        )
        middleware = AgentCorePaymentsMiddleware(config)
        agent = create_agent(model=..., tools=[...], middleware=[middleware])
    """

    def __init__(self, config: AgentCorePaymentsConfig) -> None:
        """Initialize middleware with config and create PaymentManager.

        Args:
            config: Payment configuration.

        Raises:
            RuntimeError: If PaymentManager initialization fails.
        """
        super().__init__()
        self.config = config
        try:
            self.payment_manager = PaymentManager(
                payment_manager_arn=config.payment_manager_arn,
                region_name=config.region,
                agent_name=config.agent_name,
                bearer_token=config.bearer_token,
                token_provider=config.token_provider,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PaymentManager: {e}") from e
        self.tools = self._build_tools()
        logger.info("AgentCorePaymentsMiddleware initialized")

    def _build_tools(self) -> list:
        """Build the list of tools to register with the agent.

        Returns:
            List of tool callables. Includes http_request if provide_http_request=True.
        """
        tools = []
        if self.config.provide_http_request:
            tools.append(make_http_request_tool(self))
        tools.append(make_get_payment_instrument_tool(self))
        tools.append(make_list_payment_instruments_tool(self))
        tools.append(make_get_payment_instrument_balance_tool(self))
        tools.append(make_get_payment_session_tool(self))
        return tools

    @staticmethod
    def _prepare_for_handler(content: Any) -> Optional[Dict[str, List[Dict[str, str]]]]:
        """Normalize ToolMessage.content into handler-compatible shape.

        Args:
            content: ToolMessage.content — either a str, list, or None.

        Returns:
            Dict with "content" key containing list of {"text": ...} blocks, or None.
        """
        if content is None:
            return None
        if isinstance(content, str):
            return {"content": [{"text": content}]}
        if isinstance(content, list):
            blocks = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    blocks.append(item)
                elif isinstance(item, str):
                    blocks.append({"text": item})
                else:
                    blocks.append(item)
            return {"content": blocks}
        return None

    def _get_handler(self, tool_name: str, tool_args: Dict[str, Any]) -> PaymentResponseHandler:
        """Resolve the payment response handler for a tool.

        Resolution priority:
        1. Custom handlers from config (highest priority)
        2. Built-in registry (name-based → MCP shape → generic fallback)

        Args:
            tool_name: Name of the tool.
            tool_args: Tool call arguments dict.

        Returns:
            The resolved PaymentResponseHandler instance.
        """
        if self.config.custom_handlers and tool_name in self.config.custom_handlers:
            logger.debug("Using custom handler for tool: %s", tool_name)
            return self.config.custom_handlers[tool_name]
        return get_payment_handler(tool_name, tool_args)

    @staticmethod
    def _fallback_detect_402(content: Any) -> Optional[Dict[str, Any]]:
        """Lenient fallback: detect 402 from raw JSON without the PAYMENT_REQUIRED: marker.

        Handles MCP tools and other tools that return raw JSON responses like:
        - {"statusCode": 402, "headers": {...}, "body": {...}}
        - {"x402Version": 1, "accepts": [...]}

        Args:
            content: ToolMessage.content (str or list).

        Returns:
            Parsed payment-required dict if 402 detected, None otherwise.
        """
        import json as _json

        texts = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)

        for text in texts:
            try:
                parsed = _json.loads(text)
            except (ValueError, TypeError):
                continue
            if not isinstance(parsed, dict):
                continue

            # Check for statusCode: 402
            if parsed.get("statusCode") == 402:
                logger.debug("Fallback detection: found statusCode 402 in raw JSON")
                return parsed

            # Check for httpStatus: 402 (MCP structuredContent format)
            if parsed.get("httpStatus") == 402:
                logger.debug("Fallback detection: found httpStatus 402 (MCP format)")
                return {
                    "statusCode": 402,
                    "headers": parsed.get("responseHeaders", {}),
                    "body": parsed.get("structuredContent", {}),
                }

            # Check for x402 payload (x402Version + accepts) at top level
            if "x402Version" in parsed and "accepts" in parsed:
                logger.debug("Fallback detection: found x402 payload in raw JSON")
                return {"statusCode": 402, "headers": {}, "body": parsed}

            # Check for x402 payload nested in structuredContent
            sc = parsed.get("structuredContent")
            if isinstance(sc, dict) and "x402Version" in sc and "accepts" in sc:
                logger.debug("Fallback detection: found x402 payload in structuredContent")
                return {
                    "statusCode": 402,
                    "headers": parsed.get("responseHeaders", {}),
                    "body": sc,
                }

        return None

    def _generate_payment_header(self, payment_required_request: Dict[str, Any]) -> Dict[str, str]:
        """Generate payment header via PaymentManager.

        Args:
            payment_required_request: Dict with statusCode, headers, body from the 402 response.

        Returns:
            Dict with payment header name → value.

        Raises:
            PaymentInstrumentConfigurationRequired: If payment_instrument_id not set.
            PaymentSessionConfigurationRequired: If payment_session_id not set.
            PaymentError: If payment processing fails.
        """
        if self.config.payment_instrument_id is None:
            raise PaymentInstrumentConfigurationRequired("payment_instrument_id is required for x402 payments.")
        if self.config.payment_session_id is None:
            if self.config.auto_session:
                self._create_auto_session()
            else:
                raise PaymentSessionConfigurationRequired("payment_session_id is required for x402 payments.")

        return self.payment_manager.generate_payment_header(
            user_id=self.config.user_id,
            payment_instrument_id=self.config.payment_instrument_id,
            payment_session_id=self.config.payment_session_id,
            payment_required_request=payment_required_request,
            network_preferences=self.config.network_preferences_config,
            client_token=str(uuid.uuid4()),
            payment_connector_id=self.config.payment_connector_id,
        )

    def _create_auto_session(self) -> None:
        """Lazily create a payment session on first 402 when auto_session=True."""
        logger.info(
            "auto_session: creating payment session (budget=$%s, expiry=%dmin)",
            self.config.auto_session_budget,
            self.config.auto_session_expiry_minutes,
        )
        session = self.payment_manager.create_payment_session(
            user_id=self.config.user_id,
            limits={"maxSpendAmount": {"value": self.config.auto_session_budget, "currency": "USD"}},
            expiry_time_in_minutes=self.config.auto_session_expiry_minutes,
        )
        self.config.payment_session_id = session["paymentSessionId"]
        logger.info("auto_session: created session %s", self.config.payment_session_id)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Union[ToolMessage, Command]],
    ) -> Union[ToolMessage, Command]:
        """Wrap tool execution with 402 payment detection, signing, and retry.

        Args:
            request: The tool call request.
            handler: Callable that executes the tool.

        Returns:
            The tool execution result or an error ToolMessage.
        """
        result = handler(request)

        # Guard: Command results have no tool output to inspect
        if isinstance(result, Command):
            return result

        # Guard: auto_payment disabled
        if not self.config.auto_payment:
            return result

        # Guard: allowlist filtering
        tool_name = request.tool_call["name"]
        if self.config.payment_tool_allowlist is not None:
            if tool_name not in self.config.payment_tool_allowlist:
                return result

        # 402 detection
        # Priority: custom handler → GenericPaymentHandler (marker) → lenient fallback (raw JSON)
        tool_args = request.tool_call.get("args", {})
        prepared = self._prepare_for_handler(result.content)
        if prepared is None:
            return result

        has_custom_handler = self.config.custom_handlers is not None and tool_name in self.config.custom_handlers

        if has_custom_handler:
            detection_handler = self.config.custom_handlers[tool_name]
        else:
            from bedrock_agentcore.payments.integrations.handlers import GenericPaymentHandler

            detection_handler = GenericPaymentHandler()

        status_code = detection_handler.extract_status_code(prepared)

        # Lenient fallback: if no custom handler and marker detection didn't find 402,
        # try parsing raw JSON for statusCode:402 or x402Version fields.
        # This handles MCP tools and other tools that return raw JSON without the marker.
        if status_code != 402 and not has_custom_handler:
            fallback = self._fallback_detect_402(result.content)
            if fallback is not None:
                status_code = 402
                # Switch detection_handler to a wrapper that returns the parsed data
                detection_handler = _FallbackHandler(fallback)

        if status_code != 402:
            return result

        logger.info("Detected 402 Payment Required from tool: %s", tool_name)

        # Payment processing with comprehensive error handling
        payment_required_request = None
        try:
            # Extract payment requirement details
            headers_402 = detection_handler.extract_headers(prepared) or {}
            body_402 = detection_handler.extract_body(prepared) or {}
            payment_required_request = {
                "statusCode": 402,
                "headers": headers_402,
                "body": body_402,
            }

            # Generate payment header
            payment_header = self._generate_payment_header(payment_required_request)

            # Resolve handler for header injection
            # Custom handler handles all phases; otherwise resolve by tool shape
            if has_custom_handler:
                injection_handler = detection_handler
            else:
                injection_handler = self._get_handler(tool_name, tool_args)

            # Inject header into tool args
            if not injection_handler.validate_tool_input(tool_args):
                return self._error_tool_message(
                    request,
                    PaymentError("Could not apply payment credentials to this tool's request format."),
                )
            if not injection_handler.apply_payment_header(tool_args, payment_header):
                return self._error_tool_message(
                    request,
                    PaymentError("Could not apply payment credentials to this tool's request format."),
                )

            # Blockchain timing delay
            delay = self.config.post_payment_retry_delay_seconds
            if delay > 0:
                logger.info("Waiting %.1fs before retry for blockchain timing", delay)
                time.sleep(delay)

            # Re-execute the tool with payment credentials
            retry_result = handler(request)

            if isinstance(retry_result, Command):
                return retry_result

            # Post-payment rejection detection
            retry_prepared = self._prepare_for_handler(retry_result.content)
            if retry_prepared is not None:
                # Use fresh detection on the retry result (not the frozen fallback handler)
                from bedrock_agentcore.payments.integrations.handlers import GenericPaymentHandler as _GH

                _retry_handler = _GH()
                retry_status = _retry_handler.extract_status_code(retry_prepared)
                # Also check via fallback if marker not found
                if retry_status != 402:
                    retry_fallback = self._fallback_detect_402(retry_result.content)
                    if retry_fallback is not None:
                        retry_status = 402
                        _retry_handler = _FallbackHandler(retry_fallback)
                if retry_status == 402:
                    retry_body = _retry_handler.extract_body(retry_prepared) or {}
                    error_detail = (
                        retry_body.get("error", "unknown error") if isinstance(retry_body, dict) else "unknown error"
                    )
                    return self._error_tool_message(
                        request,
                        PaymentError(f"Payment was signed but rejected by the server ({error_detail})."),
                    )

            return retry_result

        except Exception as e:
            logger.error("Payment processing error for tool %s: %s: %s", tool_name, type(e).__name__, e)
            if self.config.on_payment_error is not None and self.config.max_error_retries > 0:
                resolution = self._invoke_error_handler(
                    exception=e,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    payment_required_request=payment_required_request,
                    request=request,
                    handler=handler,
                )
                if resolution is not None:
                    return resolution
            return self._error_tool_message(request, e)

    def _invoke_error_handler(
        self,
        exception: Exception,
        tool_name: str,
        tool_args: Dict[str, Any],
        payment_required_request: Optional[Dict[str, Any]],
        request: "ToolCallRequest",
        handler: Callable,
    ) -> Optional[Union[ToolMessage, Command]]:
        """Invoke on_payment_error callback and retry if requested.

        Returns:
            ToolMessage/Command if retry succeeded or max retries exhausted.
            None if callback returned PROPAGATE (caller uses default error path).
        """
        from .errors import ErrorResolution, PaymentErrorContext

        retry_count = 0
        current_exception = exception

        while retry_count < self.config.max_error_retries:
            ctx = PaymentErrorContext(
                exception=current_exception,
                exception_type=type(current_exception).__name__,
                exception_message=str(current_exception),
                tool_name=tool_name,
                tool_args=tool_args,
                payment_required_request=payment_required_request,
                config=self.config,
                retry_count=retry_count,
            )

            try:
                resolution = self.config.on_payment_error(ctx)
            except Exception as cb_err:
                logger.error("on_payment_error callback raised: %s", cb_err)
                return None

            # str return = custom message to the LLM
            if isinstance(resolution, str):
                return ToolMessage(
                    content=f"PAYMENT ERROR: {resolution}",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )

            if resolution != ErrorResolution.RETRY:
                return None

            retry_count += 1
            logger.info("on_payment_error returned RETRY (attempt %d/%d)", retry_count, self.config.max_error_retries)

            try:
                payment_header = self._generate_payment_header(payment_required_request or {})

                has_custom = self.config.custom_handlers and tool_name in self.config.custom_handlers
                injection_handler = (
                    self.config.custom_handlers[tool_name] if has_custom else self._get_handler(tool_name, tool_args)
                )

                if not injection_handler.validate_tool_input(tool_args):
                    return self._error_tool_message(
                        request,
                        PaymentError("Could not apply payment credentials after error recovery."),
                    )
                if not injection_handler.apply_payment_header(tool_args, payment_header):
                    return self._error_tool_message(
                        request,
                        PaymentError("Could not apply payment credentials after error recovery."),
                    )

                delay = self.config.post_payment_retry_delay_seconds
                if delay > 0:
                    time.sleep(delay)

                retry_result = handler(request)
                if isinstance(retry_result, Command):
                    return retry_result

                # Post-payment rejection check
                retry_prepared = self._prepare_for_handler(retry_result.content)
                if retry_prepared is not None:
                    from bedrock_agentcore.payments.integrations.handlers import GenericPaymentHandler as _GH

                    _rh = _GH()
                    retry_status = _rh.extract_status_code(retry_prepared)
                    if retry_status != 402:
                        fallback = self._fallback_detect_402(retry_result.content)
                        if fallback:
                            retry_status = 402
                    if retry_status == 402:
                        retry_body = _rh.extract_body(retry_prepared) or {}
                        detail = retry_body.get("error", "unknown") if isinstance(retry_body, dict) else "unknown"
                        return self._error_tool_message(
                            request,
                            PaymentError(f"Payment signed but rejected after recovery ({detail})."),
                        )

                return retry_result

            except Exception as retry_err:
                logger.error("Payment retry after error handler failed: %s", retry_err)
                current_exception = retry_err
                continue

        logger.warning("max_error_retries (%d) exhausted", self.config.max_error_retries)
        return self._error_tool_message(request, current_exception)

    @staticmethod
    def _error_tool_message(request: ToolCallRequest, exception: Exception) -> ToolMessage:
        """Create a ToolMessage with a deterministic error message for the LLM.

        Looks up the exception type in the error message map. Falls back to a
        generic message that includes the exception string for unrecognized types.

        Args:
            request: The original tool call request (for tool_call_id).
            exception: The exception to report.

        Returns:
            ToolMessage with status="error" and deterministic content.
        """
        msg = get_payment_error_message(exception)

        return ToolMessage(
            content=f"PAYMENT ERROR: {msg}",
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Union[ToolMessage, Command]]],
    ) -> Union[ToolMessage, Command]:
        """Async version of wrap_tool_call.

        Uses asyncio.sleep for non-blocking delay and asyncio.to_thread for
        the synchronous PaymentManager.generate_payment_header call.

        Args:
            request: The tool call request.
            handler: Async callable that executes the tool.

        Returns:
            The tool execution result or an error ToolMessage.
        """
        import asyncio

        result = await handler(request)

        if isinstance(result, Command):
            return result
        if not self.config.auto_payment:
            return result

        tool_name = request.tool_call["name"]
        if self.config.payment_tool_allowlist is not None:
            if tool_name not in self.config.payment_tool_allowlist:
                return result

        tool_args = request.tool_call.get("args", {})
        prepared = self._prepare_for_handler(result.content)
        if prepared is None:
            return result

        has_custom_handler = self.config.custom_handlers is not None and tool_name in self.config.custom_handlers

        if has_custom_handler:
            detection_handler = self.config.custom_handlers[tool_name]
        else:
            from bedrock_agentcore.payments.integrations.handlers import GenericPaymentHandler

            detection_handler = GenericPaymentHandler()

        status_code = detection_handler.extract_status_code(prepared)

        # Lenient fallback for async path (same as sync)
        if status_code != 402 and not has_custom_handler:
            fallback = self._fallback_detect_402(result.content)
            if fallback is not None:
                status_code = 402
                detection_handler = _FallbackHandler(fallback)

        if status_code != 402:
            return result

        logger.info("Detected 402 Payment Required from tool (async): %s", tool_name)

        payment_required_request = None
        try:
            headers_402 = detection_handler.extract_headers(prepared) or {}
            body_402 = detection_handler.extract_body(prepared) or {}
            payment_required_request = {
                "statusCode": 402,
                "headers": headers_402,
                "body": body_402,
            }

            payment_header = await asyncio.to_thread(self._generate_payment_header, payment_required_request)

            if has_custom_handler:
                injection_handler = detection_handler
            else:
                injection_handler = self._get_handler(tool_name, tool_args)

            if not injection_handler.validate_tool_input(tool_args):
                return self._error_tool_message(
                    request,
                    PaymentError("Could not apply payment credentials to this tool's request format."),
                )
            if not injection_handler.apply_payment_header(tool_args, payment_header):
                return self._error_tool_message(
                    request,
                    PaymentError("Could not apply payment credentials to this tool's request format."),
                )

            delay = self.config.post_payment_retry_delay_seconds
            if delay > 0:
                logger.info("Waiting %.1fs before retry for blockchain timing (async)", delay)
                await asyncio.sleep(delay)

            retry_result = await handler(request)

            if isinstance(retry_result, Command):
                return retry_result

            retry_prepared = self._prepare_for_handler(retry_result.content)
            if retry_prepared is not None:
                from bedrock_agentcore.payments.integrations.handlers import GenericPaymentHandler as _GH

                _retry_handler = _GH()
                retry_status = _retry_handler.extract_status_code(retry_prepared)
                if retry_status != 402:
                    retry_fallback = self._fallback_detect_402(retry_result.content)
                    if retry_fallback is not None:
                        retry_status = 402
                        _retry_handler = _FallbackHandler(retry_fallback)
                if retry_status == 402:
                    retry_body = _retry_handler.extract_body(retry_prepared) or {}
                    error_detail = (
                        retry_body.get("error", "unknown error") if isinstance(retry_body, dict) else "unknown error"
                    )
                    return self._error_tool_message(
                        request,
                        PaymentError(f"Payment was signed but rejected by the server ({error_detail})."),
                    )

            return retry_result

        except Exception as e:
            logger.error("Payment processing error (async) for tool %s: %s: %s", tool_name, type(e).__name__, e)
            if self.config.on_payment_error is not None and self.config.max_error_retries > 0:
                resolution = await self._ainvoke_error_handler(
                    exception=e,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    payment_required_request=payment_required_request,
                    request=request,
                    handler=handler,
                )
                if resolution is not None:
                    return resolution
            return self._error_tool_message(request, e)

    async def _ainvoke_error_handler(
        self,
        exception: Exception,
        tool_name: str,
        tool_args: Dict[str, Any],
        payment_required_request: Optional[Dict[str, Any]],
        request: "ToolCallRequest",
        handler: Callable,
    ) -> Optional[Union[ToolMessage, Command]]:
        """Async version of _invoke_error_handler. Supports async callbacks."""
        import asyncio
        import inspect

        from .errors import ErrorResolution, PaymentErrorContext

        retry_count = 0
        current_exception = exception

        while retry_count < self.config.max_error_retries:
            ctx = PaymentErrorContext(
                exception=current_exception,
                exception_type=type(current_exception).__name__,
                exception_message=str(current_exception),
                tool_name=tool_name,
                tool_args=tool_args,
                payment_required_request=payment_required_request,
                config=self.config,
                retry_count=retry_count,
            )

            try:
                if inspect.iscoroutinefunction(self.config.on_payment_error):
                    resolution = await self.config.on_payment_error(ctx)
                else:
                    resolution = self.config.on_payment_error(ctx)
            except Exception as cb_err:
                logger.error("on_payment_error callback raised (async): %s", cb_err)
                return None

            # str return = custom message to the LLM
            if isinstance(resolution, str):
                return ToolMessage(
                    content=f"PAYMENT ERROR: {resolution}",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )

            if resolution != ErrorResolution.RETRY:
                return None

            retry_count += 1
            logger.info(
                "on_payment_error returned RETRY (async, attempt %d/%d)",
                retry_count,
                self.config.max_error_retries,
            )

            try:
                payment_header = await asyncio.to_thread(self._generate_payment_header, payment_required_request or {})

                has_custom = self.config.custom_handlers and tool_name in self.config.custom_handlers
                injection_handler = (
                    self.config.custom_handlers[tool_name] if has_custom else self._get_handler(tool_name, tool_args)
                )

                if not injection_handler.validate_tool_input(tool_args):
                    return self._error_tool_message(
                        request,
                        PaymentError("Could not apply payment credentials after error recovery."),
                    )
                if not injection_handler.apply_payment_header(tool_args, payment_header):
                    return self._error_tool_message(
                        request,
                        PaymentError("Could not apply payment credentials after error recovery."),
                    )

                delay = self.config.post_payment_retry_delay_seconds
                if delay > 0:
                    await asyncio.sleep(delay)

                retry_result = await handler(request)
                if isinstance(retry_result, Command):
                    return retry_result

                retry_prepared = self._prepare_for_handler(retry_result.content)
                if retry_prepared is not None:
                    from bedrock_agentcore.payments.integrations.handlers import GenericPaymentHandler as _GH

                    _rh = _GH()
                    retry_status = _rh.extract_status_code(retry_prepared)
                    if retry_status != 402:
                        fallback = self._fallback_detect_402(retry_result.content)
                        if fallback:
                            retry_status = 402
                    if retry_status == 402:
                        retry_body = _rh.extract_body(retry_prepared) or {}
                        detail = retry_body.get("error", "unknown") if isinstance(retry_body, dict) else "unknown"
                        return self._error_tool_message(
                            request,
                            PaymentError(f"Payment signed but rejected after recovery ({detail})."),
                        )

                return retry_result

            except Exception as retry_err:
                logger.error("Payment retry after error handler failed (async): %s", retry_err)
                current_exception = retry_err
                continue

        logger.warning("max_error_retries (%d) exhausted (async)", self.config.max_error_retries)
        return self._error_tool_message(request, current_exception)
