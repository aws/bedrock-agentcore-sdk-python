"""AgentCorePaymentsMiddleware for LangGraph agents."""

import asyncio
import functools
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from bedrock_agentcore.payments.integrations.error_messages import get_payment_error_message
from bedrock_agentcore.payments.integrations.handlers import (
    GenericPaymentHandler,
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
from .errors import ErrorResolution, PaymentErrorContext
from .tools import (
    make_get_payment_instrument_balance_tool,
    make_get_payment_instrument_tool,
    make_get_payment_session_tool,
    make_http_request_tool,
    make_list_payment_instruments_tool,
)

logger = logging.getLogger(__name__)


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


@dataclass
class _DetectionResult:
    """Result of 402 detection phase."""

    detection_handler: Any
    has_custom_handler: bool
    prepared: Dict[str, List[Dict[str, str]]]
    raw_content: Any  # Original ToolMessage.content for custom handlers
    tool_name: str
    tool_args: Dict[str, Any]


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
        """Initialize middleware with config and create PaymentManager."""
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
        """Build the list of tools to register with the agent."""
        tools = []
        if self.config.provide_http_request:
            tools.append(make_http_request_tool(self))
        tools.append(make_get_payment_instrument_tool(self))
        tools.append(make_list_payment_instruments_tool(self))
        tools.append(make_get_payment_instrument_balance_tool(self))
        tools.append(make_get_payment_session_tool(self))
        return tools

    # -------------------------------------------------------------------------
    # Shared helpers (pure logic, no async/sync split)
    # -------------------------------------------------------------------------

    @staticmethod
    def _prepare_for_handler(content: Any) -> Optional[Dict[str, List[Dict[str, str]]]]:
        """Normalize ToolMessage.content into handler-compatible shape."""
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
        """Resolve the payment response handler for a tool."""
        if self.config.custom_handlers and tool_name in self.config.custom_handlers:
            logger.debug("Using custom handler for tool: %s", tool_name)
            return self.config.custom_handlers[tool_name]
        return get_payment_handler(tool_name, tool_args)

    @staticmethod
    def _fallback_detect_402(content: Any) -> Optional[Dict[str, Any]]:
        """Lenient fallback: detect 402 from raw JSON without the PAYMENT_REQUIRED: marker."""
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
                parsed = json.loads(text)
            except (ValueError, TypeError):
                continue
            if not isinstance(parsed, dict):
                continue

            if parsed.get("statusCode") == 402:
                logger.debug("Fallback detection: found statusCode 402 in raw JSON")
                return parsed

            if parsed.get("httpStatus") == 402:
                logger.debug("Fallback detection: found httpStatus 402 (MCP format)")
                return {
                    "statusCode": 402,
                    "headers": parsed.get("responseHeaders", {}),
                    "body": parsed.get("structuredContent", {}),
                }

            if "x402Version" in parsed and "accepts" in parsed:
                logger.debug("Fallback detection: found x402 payload in raw JSON")
                return {"statusCode": 402, "headers": {}, "body": parsed}

            sc = parsed.get("structuredContent")
            if isinstance(sc, dict) and "x402Version" in sc and "accepts" in sc:
                logger.debug("Fallback detection: found x402 payload in structuredContent")
                return {
                    "statusCode": 402,
                    "headers": parsed.get("responseHeaders", {}),
                    "body": sc,
                }

        return None

    def _check_guards(
        self, request: ToolCallRequest, result: Union[ToolMessage, Command]
    ) -> Optional[Union[ToolMessage, Command]]:
        """Check early-exit guards. Returns the result to pass through, or None to continue."""
        if isinstance(result, Command):
            return result
        if not self.config.auto_payment:
            return result
        tool_name = request.tool_call["name"]
        if self.config.payment_tool_allowlist is not None:
            if tool_name not in self.config.payment_tool_allowlist:
                return result
        return None

    def _detect_402(self, request: ToolCallRequest, result: ToolMessage) -> Optional[_DetectionResult]:
        """Run 402 detection. Returns detection context if 402 found, None otherwise."""
        tool_name = request.tool_call["name"]
        # Store args back on the tool call so an injected payment header reaches the retried handler.
        tool_args = request.tool_call.setdefault("args", {})
        prepared = self._prepare_for_handler(result.content)
        if prepared is None:
            return None

        has_custom_handler = self.config.custom_handlers is not None and tool_name in self.config.custom_handlers

        if has_custom_handler:
            detection_handler = self.config.custom_handlers[tool_name]
            # Custom handlers receive raw content (not the internal prepared shape)
            status_code = detection_handler.extract_status_code(result.content)
        else:
            detection_handler = GenericPaymentHandler()
            status_code = detection_handler.extract_status_code(prepared)

        # Lenient fallback if marker detection didn't find 402
        if status_code != 402 and not has_custom_handler:
            fallback = self._fallback_detect_402(result.content)
            if fallback is not None:
                status_code = 402
                detection_handler = _FallbackHandler(fallback)

        # Name-based handler fallback (e.g. HttpRequestPaymentHandler for legacy text-block format)
        if status_code != 402 and not has_custom_handler:
            name_handler = self._get_handler(tool_name, tool_args)
            name_status = name_handler.extract_status_code(prepared)
            if name_status == 402:
                status_code = 402
                detection_handler = name_handler

        if status_code != 402:
            return None

        logger.info("Detected 402 Payment Required from tool: %s", tool_name)
        return _DetectionResult(
            detection_handler=detection_handler,
            has_custom_handler=has_custom_handler,
            prepared=prepared,
            raw_content=result.content,
            tool_name=tool_name,
            tool_args=tool_args,
        )

    def _extract_payment_request(self, detection: _DetectionResult) -> Dict[str, Any]:
        """Extract the payment-required request dict from a detection result."""
        # Custom handlers receive raw content; built-in handlers receive prepared shape
        handler_input = detection.raw_content if detection.has_custom_handler else detection.prepared
        headers_402 = detection.detection_handler.extract_headers(handler_input) or {}
        body_402 = detection.detection_handler.extract_body(handler_input) or {}
        return {
            "statusCode": 402,
            "headers": headers_402,
            "body": body_402,
        }

    def _inject_payment_header(
        self,
        request: ToolCallRequest,
        detection: _DetectionResult,
        payment_header: Dict[str, str],
    ) -> Optional[ToolMessage]:
        """Inject payment header into tool args. Returns error ToolMessage on failure, None on success."""
        if detection.has_custom_handler:
            injection_handler = detection.detection_handler
        else:
            injection_handler = self._get_handler(detection.tool_name, detection.tool_args)

        if not injection_handler.validate_tool_input(detection.tool_args):
            return self._error_tool_message(
                request,
                PaymentError("Could not apply payment credentials to this tool's request format."),
            )
        if not injection_handler.apply_payment_header(detection.tool_args, payment_header):
            return self._error_tool_message(
                request,
                PaymentError("Could not apply payment credentials to this tool's request format."),
            )
        return None

    def _check_retry_rejection(
        self,
        request: ToolCallRequest,
        retry_result: Union[ToolMessage, Command],
        error_context: str = "by the server",
    ) -> Optional[Union[ToolMessage, Command]]:
        """Check if a retry result is still a 402 (payment rejected).

        Args:
            request: The original tool call request.
            retry_result: The result from the retry attempt.
            error_context: Context string for the error message (e.g. "by the server", "after recovery").

        Returns:
            Error ToolMessage if 402 detected, the Command if result is a Command, or None.
        """
        if isinstance(retry_result, Command):
            return retry_result

        tool_name = request.tool_call["name"]
        if self.config.custom_handlers and tool_name in self.config.custom_handlers:
            # Mirror _detect_402: a custom handler owns detection for its tool, so it must
            # also decide whether the post-payment retry is still a 402. It receives raw content.
            handler = self.config.custom_handlers[tool_name]
            retry_status = handler.extract_status_code(retry_result.content)
            retry_body = handler.extract_body(retry_result.content) if retry_status == 402 else None
        else:
            retry_prepared = self._prepare_for_handler(retry_result.content)
            if retry_prepared is None:
                return None

            _rh = GenericPaymentHandler()
            retry_status = _rh.extract_status_code(retry_prepared)
            if retry_status != 402:
                fallback = self._fallback_detect_402(retry_result.content)
                if fallback is not None:
                    retry_status = 402
                    _rh = _FallbackHandler(fallback)
            retry_body = _rh.extract_body(retry_prepared) if retry_status == 402 else None

        if retry_status == 402:
            body = retry_body or {}
            detail = body.get("error", "unknown error") if isinstance(body, dict) else "unknown error"
            return self._error_tool_message(
                request,
                PaymentError(f"Payment was signed but rejected {error_context} ({detail})."),
            )
        return None

    def _inject_for_error_retry(
        self,
        request: ToolCallRequest,
        tool_name: str,
        tool_args: Dict[str, Any],
        payment_header: Dict[str, str],
    ) -> Optional[ToolMessage]:
        """Inject payment header during error handler retry. Returns error ToolMessage on failure."""
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
        return None

    def _generate_payment_header(self, payment_required_request: Dict[str, Any]) -> Dict[str, str]:
        """Generate payment header via PaymentManager."""
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

    @staticmethod
    def _is_async_callback(callback: Any) -> bool:
        """True if the callback ultimately resolves to an async coroutine function.

        Covers the wrappings a plain ``inspect.iscoroutinefunction`` can miss: callable
        objects with an ``async def __call__``, ``functools.partial`` (whose async-ness is
        not reliably visible through the partial across Python versions), and partials of
        either. (``callable()`` — ruff B004's suggestion — cannot distinguish sync from async.)
        """
        target = callback
        while isinstance(target, functools.partial):
            target = target.func
        if inspect.iscoroutinefunction(target):
            return True
        call = getattr(target, "__call__", None)  # noqa: B004
        return inspect.iscoroutinefunction(call)

    @staticmethod
    def _error_tool_message(request: ToolCallRequest, exception: Exception) -> ToolMessage:
        """Create a ToolMessage with a deterministic error message for the LLM."""
        msg = get_payment_error_message(exception)
        return ToolMessage(
            content=f"PAYMENT ERROR: {msg}",
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def _build_error_context(
        self,
        current_exception: Exception,
        tool_name: str,
        tool_args: Dict[str, Any],
        payment_required_request: Optional[Dict[str, Any]],
        retry_count: int,
    ) -> Any:
        """Build a PaymentErrorContext for the error handler callback."""
        return PaymentErrorContext(
            exception=current_exception,
            exception_type=type(current_exception).__name__,
            exception_message=str(current_exception),
            tool_name=tool_name,
            tool_args=tool_args,
            payment_required_request=payment_required_request,
            config=self.config,
            retry_count=retry_count,
        )

    def _handle_callback_resolution(
        self, resolution: Any, request: ToolCallRequest
    ) -> Optional[Union[ToolMessage, None]]:
        """Handle the callback return value. Returns ToolMessage for str, None for PROPAGATE, raises for RETRY."""
        if isinstance(resolution, str):
            return ToolMessage(
                content=f"PAYMENT ERROR: {resolution}",
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        if resolution != ErrorResolution.RETRY:
            # Signal: caller should return None (PROPAGATE)
            return "PROPAGATE"  # type: ignore[return-value]
        # Signal: caller should continue with retry
        return None

    # -------------------------------------------------------------------------
    # Sync path
    # -------------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Union[ToolMessage, Command]],
    ) -> Union[ToolMessage, Command]:
        """Wrap tool execution with 402 payment detection, signing, and retry."""
        result = handler(request)

        guard_result = self._check_guards(request, result)
        if guard_result is not None:
            return guard_result

        detection = self._detect_402(request, result)
        if detection is None:
            return result

        payment_required_request = None
        try:
            payment_required_request = self._extract_payment_request(detection)
            payment_header = self._generate_payment_header(payment_required_request)

            inject_error = self._inject_payment_header(request, detection, payment_header)
            if inject_error is not None:
                return inject_error

            delay = self.config.post_payment_retry_delay_seconds
            if delay > 0:
                logger.info("Waiting %.1fs before retry for blockchain timing", delay)
                time.sleep(delay)

            retry_result = handler(request)

            rejection = self._check_retry_rejection(request, retry_result, "by the server")
            if rejection is not None:
                return rejection

            return retry_result

        except Exception as e:
            logger.error(
                "Payment processing error for tool %s: %s: %s",
                detection.tool_name,
                type(e).__name__,
                e,
            )
            if self.config.on_payment_error is not None and self.config.max_error_retries > 0:
                resolution = self._invoke_error_handler(
                    exception=e,
                    tool_name=detection.tool_name,
                    tool_args=detection.tool_args,
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
        """Invoke on_payment_error callback and retry if requested."""
        # Fail loudly: an async callback on the sync path can never be awaited here, so a
        # silent PROPAGATE would hide a programming error. Raise before the loop (outside the
        # callback try/except) so the TypeError propagates instead of being swallowed.
        if self._is_async_callback(self.config.on_payment_error):
            raise TypeError(
                "async on_payment_error callback cannot be used with sync .invoke(). "
                "Use agent.ainvoke() or provide a synchronous callback."
            )

        retry_count = 0
        current_exception = exception

        while retry_count < self.config.max_error_retries:
            ctx = self._build_error_context(
                current_exception, tool_name, tool_args, payment_required_request, retry_count
            )

            try:
                resolution = self.config.on_payment_error(ctx)
            except Exception as cb_err:
                logger.error("on_payment_error callback raised: %s", cb_err)
                return None

            handled = self._handle_callback_resolution(resolution, request)
            if handled == "PROPAGATE":
                return None
            if handled is not None:
                return handled

            retry_count += 1
            logger.info("on_payment_error returned RETRY (attempt %d/%d)", retry_count, self.config.max_error_retries)

            try:
                payment_header = self._generate_payment_header(payment_required_request or {})

                inject_error = self._inject_for_error_retry(request, tool_name, tool_args, payment_header)
                if inject_error is not None:
                    return inject_error

                delay = self.config.post_payment_retry_delay_seconds
                if delay > 0:
                    time.sleep(delay)

                retry_result = handler(request)

                rejection = self._check_retry_rejection(request, retry_result, "after recovery")
                if rejection is not None:
                    return rejection

                return retry_result

            except Exception as retry_err:
                logger.error("Payment retry after error handler failed: %s", retry_err)
                current_exception = retry_err
                continue

        logger.warning("max_error_retries (%d) exhausted", self.config.max_error_retries)
        return self._error_tool_message(request, current_exception)

    # -------------------------------------------------------------------------
    # Async path
    # -------------------------------------------------------------------------

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Union[ToolMessage, Command]]],
    ) -> Union[ToolMessage, Command]:
        """Async version of wrap_tool_call.

        Uses asyncio.sleep for non-blocking delay and asyncio.to_thread for
        the synchronous PaymentManager.generate_payment_header call.
        """
        result = await handler(request)

        guard_result = self._check_guards(request, result)
        if guard_result is not None:
            return guard_result

        detection = self._detect_402(request, result)
        if detection is None:
            return result

        payment_required_request = None
        try:
            payment_required_request = self._extract_payment_request(detection)
            payment_header = await asyncio.to_thread(self._generate_payment_header, payment_required_request)

            inject_error = self._inject_payment_header(request, detection, payment_header)
            if inject_error is not None:
                return inject_error

            delay = self.config.post_payment_retry_delay_seconds
            if delay > 0:
                logger.info("Waiting %.1fs before retry for blockchain timing (async)", delay)
                await asyncio.sleep(delay)

            retry_result = await handler(request)

            rejection = self._check_retry_rejection(request, retry_result, "by the server")
            if rejection is not None:
                return rejection

            return retry_result

        except Exception as e:
            logger.error(
                "Payment processing error (async) for tool %s: %s: %s",
                detection.tool_name,
                type(e).__name__,
                e,
            )
            if self.config.on_payment_error is not None and self.config.max_error_retries > 0:
                resolution = await self._ainvoke_error_handler(
                    exception=e,
                    tool_name=detection.tool_name,
                    tool_args=detection.tool_args,
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
        retry_count = 0
        current_exception = exception

        while retry_count < self.config.max_error_retries:
            ctx = self._build_error_context(
                current_exception, tool_name, tool_args, payment_required_request, retry_count
            )

            try:
                if self._is_async_callback(self.config.on_payment_error):
                    resolution = await self.config.on_payment_error(ctx)
                else:
                    resolution = self.config.on_payment_error(ctx)
            except Exception as cb_err:
                logger.error("on_payment_error callback raised (async): %s", cb_err)
                return None

            handled = self._handle_callback_resolution(resolution, request)
            if handled == "PROPAGATE":
                return None
            if handled is not None:
                return handled

            retry_count += 1
            logger.info(
                "on_payment_error returned RETRY (async, attempt %d/%d)",
                retry_count,
                self.config.max_error_retries,
            )

            try:
                payment_header = await asyncio.to_thread(self._generate_payment_header, payment_required_request or {})

                inject_error = self._inject_for_error_retry(request, tool_name, tool_args, payment_header)
                if inject_error is not None:
                    return inject_error

                delay = self.config.post_payment_retry_delay_seconds
                if delay > 0:
                    await asyncio.sleep(delay)

                retry_result = await handler(request)

                rejection = self._check_retry_rejection(request, retry_result, "after recovery")
                if rejection is not None:
                    return rejection

                return retry_result

            except Exception as retry_err:
                logger.error("Payment retry after error handler failed (async): %s", retry_err)
                current_exception = retry_err
                continue

        logger.warning("max_error_retries (%d) exhausted (async)", self.config.max_error_retries)
        return self._error_tool_message(request, current_exception)
