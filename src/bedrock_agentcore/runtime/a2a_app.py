"""Bedrock AgentCore A2A application implementation.

Provides a Starlette-based web server for A2A (Agent-to-Agent) protocol communication.
"""

import asyncio
import contextvars
import inspect
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Sequence
from typing import Any, Callable, Dict, Optional

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.types import Lifespan

from .a2a_models import (
    A2A_DEFAULT_PORT,
    AgentCard,
    JsonRpcErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
)
from .context import BedrockAgentCoreContext, RequestContext
from .models import (
    ACCESS_TOKEN_HEADER,
    AUTHORIZATION_HEADER,
    CUSTOM_HEADER_PREFIX,
    OAUTH2_CALLBACK_URL_HEADER,
    REQUEST_ID_HEADER,
    SESSION_HEADER,
    PingStatus,
)


class A2ARequestContextFormatter(logging.Formatter):
    """Formatter including request and session IDs for A2A applications."""

    def format(self, record):
        """Format log record as AWS Lambda JSON."""
        from datetime import datetime, timezone

        log_entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "protocol": "A2A",
        }

        request_id = BedrockAgentCoreContext.get_request_id()
        if request_id:
            log_entry["requestId"] = request_id

        session_id = BedrockAgentCoreContext.get_session_id()
        if session_id:
            log_entry["sessionId"] = session_id

        if record.exc_info:
            import traceback

            log_entry["errorType"] = record.exc_info[0].__name__
            log_entry["errorMessage"] = str(record.exc_info[1])
            log_entry["stackTrace"] = traceback.format_exception(*record.exc_info)
            log_entry["location"] = f"{record.pathname}:{record.funcName}:{record.lineno}"

        return json.dumps(log_entry, ensure_ascii=False)


class BedrockAgentCoreA2AApp(Starlette):
    """Bedrock AgentCore A2A application class for agent-to-agent communication.

    This class implements the A2A protocol contract for AgentCore Runtime,
    supporting JSON-RPC 2.0 messaging and agent discovery via Agent Cards.

    Example:
        ```python
        from bedrock_agentcore.runtime import BedrockAgentCoreA2AApp, AgentCard, AgentSkill

        agent_card = AgentCard(
            name="Calculator Agent",
            description="A calculator agent",
            skills=[AgentSkill(id="calc", name="Calculator", description="Math ops")]
        )

        app = BedrockAgentCoreA2AApp(agent_card=agent_card)

        @app.entrypoint
        def handle_message(request, context):
            # Process JSON-RPC request
            message = request.params["message"]
            user_text = message["parts"][0]["text"]

            # Return result (will be wrapped in JSON-RPC response)
            return {
                "artifacts": [{
                    "artifactId": str(uuid.uuid4()),
                    "name": "response",
                    "parts": [{"kind": "text", "text": f"Result: {user_text}"}]
                }]
            }

        app.run()  # Runs on port 9000
        ```
    """

    def __init__(
        self,
        agent_card: AgentCard,
        debug: bool = False,
        lifespan: Optional[Lifespan] = None,
        middleware: Sequence[Middleware] | None = None,
    ):
        """Initialize Bedrock AgentCore A2A application.

        Args:
            agent_card: AgentCard containing agent metadata for discovery
            debug: Enable debug mode for verbose logging (default: False)
            lifespan: Optional lifespan context manager for startup/shutdown
            middleware: Optional sequence of Starlette Middleware objects
        """
        self.agent_card = agent_card
        self.handlers: Dict[str, Callable] = {}
        self._ping_handler: Optional[Callable] = None
        self._active_tasks: Dict[int, Dict[str, Any]] = {}
        self._task_counter_lock: threading.Lock = threading.Lock()
        self._forced_ping_status: Optional[PingStatus] = None
        self._last_status_update_time: float = time.time()

        routes = [
            Route("/", self._handle_jsonrpc, methods=["POST"]),
            Route("/.well-known/agent-card.json", self._handle_agent_card, methods=["GET"]),
            Route("/ping", self._handle_ping, methods=["GET"]),
        ]
        super().__init__(routes=routes, lifespan=lifespan, middleware=middleware)
        self.debug = debug

        self.logger = logging.getLogger("bedrock_agentcore.a2a_app")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = A2ARequestContextFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

    def entrypoint(self, func: Callable) -> Callable:
        """Decorator to register a function as the main message handler.

        The handler receives the JSON-RPC request and context, and should return
        a result that will be wrapped in a JSON-RPC response.

        Args:
            func: The function to register as entrypoint.
                  Signature: (request: JsonRpcRequest, context: RequestContext) -> Any
                  Or for streaming: async generator yielding response chunks

        Returns:
            The decorated function with added run method
        """
        self.handlers["main"] = func
        func.run = lambda port=A2A_DEFAULT_PORT, host=None: self.run(port, host)
        return func

    def ping(self, func: Callable) -> Callable:
        """Decorator to register a custom ping status handler.

        Args:
            func: The function to register as ping status handler

        Returns:
            The decorated function
        """
        self._ping_handler = func
        return func

    def get_current_ping_status(self) -> PingStatus:
        """Get current ping status (forced > custom > automatic)."""
        current_status = None

        if self._forced_ping_status is not None:
            current_status = self._forced_ping_status
        elif self._ping_handler:
            try:
                result = self._ping_handler()
                if isinstance(result, str):
                    current_status = PingStatus(result)
                else:
                    current_status = result
            except Exception as e:
                self.logger.warning(
                    "Custom ping handler failed, falling back to automatic: %s: %s", type(e).__name__, e
                )

        if current_status is None:
            current_status = PingStatus.HEALTHY_BUSY if self._active_tasks else PingStatus.HEALTHY

        if not hasattr(self, "_last_known_status") or self._last_known_status != current_status:
            self._last_known_status = current_status
            self._last_status_update_time = time.time()

        return current_status

    def _get_runtime_url(self) -> Optional[str]:
        """Get the runtime URL from environment variable.

        Returns:
            The runtime URL if set, None otherwise.
        """
        return os.environ.get("AGENTCORE_RUNTIME_URL")

    def _build_request_context(self, request) -> RequestContext:
        """Build request context and setup all context variables."""
        try:
            headers = request.headers
            request_id = headers.get(REQUEST_ID_HEADER)
            if not request_id:
                request_id = str(uuid.uuid4())

            session_id = headers.get(SESSION_HEADER)
            BedrockAgentCoreContext.set_request_context(request_id, session_id)

            agent_identity_token = headers.get(ACCESS_TOKEN_HEADER)
            if agent_identity_token:
                BedrockAgentCoreContext.set_workload_access_token(agent_identity_token)

            oauth2_callback_url = headers.get(OAUTH2_CALLBACK_URL_HEADER)
            if oauth2_callback_url:
                BedrockAgentCoreContext.set_oauth2_callback_url(oauth2_callback_url)

            # Collect relevant request headers
            request_headers = {}

            authorization_header = headers.get(AUTHORIZATION_HEADER)
            if authorization_header is not None:
                request_headers[AUTHORIZATION_HEADER] = authorization_header

            for header_name, header_value in headers.items():
                if header_name.lower().startswith(CUSTOM_HEADER_PREFIX.lower()):
                    request_headers[header_name] = header_value

            if request_headers:
                BedrockAgentCoreContext.set_request_headers(request_headers)

            req_headers = BedrockAgentCoreContext.get_request_headers()

            return RequestContext(
                session_id=session_id,
                request_headers=req_headers,
                request=request,
            )
        except Exception as e:
            self.logger.warning("Failed to build request context: %s: %s", type(e).__name__, e)
            request_id = str(uuid.uuid4())
            BedrockAgentCoreContext.set_request_context(request_id, None)
            return RequestContext(session_id=None, request=None)

    def _takes_context(self, handler: Callable) -> bool:
        """Check if handler accepts context parameter."""
        try:
            params = list(inspect.signature(handler).parameters.keys())
            return len(params) >= 2 and params[1] == "context"
        except Exception:
            return False

    async def _handle_jsonrpc(self, request):
        """Handle JSON-RPC 2.0 requests at root endpoint."""
        request_context = self._build_request_context(request)
        start_time = time.time()

        try:
            body = await request.json()
            self.logger.debug("Processing JSON-RPC request: %s", body.get("method", "unknown"))

            # Validate JSON-RPC format
            if body.get("jsonrpc") != "2.0":
                return self._jsonrpc_error_response(
                    body.get("id"),
                    JsonRpcErrorCode.INVALID_REQUEST,
                    "Invalid JSON-RPC version",
                )

            method = body.get("method")
            if not method:
                return self._jsonrpc_error_response(
                    body.get("id"),
                    JsonRpcErrorCode.INVALID_REQUEST,
                    "Missing method",
                )

            jsonrpc_request = JsonRpcRequest.from_dict(body)

            handler = self.handlers.get("main")
            if not handler:
                self.logger.error("No entrypoint defined")
                return self._jsonrpc_error_response(
                    jsonrpc_request.id,
                    JsonRpcErrorCode.INTERNAL_ERROR,
                    "No entrypoint defined",
                )

            takes_context = self._takes_context(handler)

            self.logger.debug("Invoking handler for method: %s", method)
            result = await self._invoke_handler(handler, request_context, takes_context, jsonrpc_request)

            duration = time.time() - start_time

            # Handle streaming responses
            if inspect.isasyncgen(result):
                self.logger.info("Returning streaming response (%.3fs)", duration)
                return StreamingResponse(
                    self._stream_jsonrpc_response(result, jsonrpc_request.id),
                    media_type="text/event-stream",
                )
            elif inspect.isgenerator(result):
                self.logger.info("Returning streaming response (sync generator) (%.3fs)", duration)
                return StreamingResponse(
                    self._sync_stream_jsonrpc_response(result, jsonrpc_request.id),
                    media_type="text/event-stream",
                )

            # Non-streaming response
            self.logger.info("Request completed successfully (%.3fs)", duration)
            response = JsonRpcResponse.success(jsonrpc_request.id, result)
            return JSONResponse(response.to_dict())

        except json.JSONDecodeError as e:
            duration = time.time() - start_time
            self.logger.warning("Invalid JSON in request (%.3fs): %s", duration, e)
            return self._jsonrpc_error_response(
                None,
                JsonRpcErrorCode.PARSE_ERROR,
                f"Parse error: {str(e)}",
            )
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception("Request failed (%.3fs)", duration)
            return self._jsonrpc_error_response(
                body.get("id") if "body" in dir() else None,
                JsonRpcErrorCode.INTERNAL_ERROR,
                str(e),
            )

    def _jsonrpc_error_response(
        self,
        request_id: Optional[str],
        code: int,
        message: str,
        data: Optional[Any] = None,
    ) -> JSONResponse:
        """Create a JSON-RPC error response."""
        response = JsonRpcResponse.error_response(request_id, code, message, data)
        return JSONResponse(response.to_dict())

    async def _stream_jsonrpc_response(self, generator, request_id):
        """Wrap async generator for SSE streaming with JSON-RPC format."""
        try:
            async for value in generator:
                # Wrap each chunk in JSON-RPC format
                chunk_response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": value,
                }
                yield self._to_sse(chunk_response)
        except Exception as e:
            self.logger.exception("Error in async streaming")
            error_response = JsonRpcResponse.error_response(
                request_id,
                JsonRpcErrorCode.INTERNAL_ERROR,
                str(e),
            )
            yield self._to_sse(error_response.to_dict())

    def _sync_stream_jsonrpc_response(self, generator, request_id):
        """Wrap sync generator for SSE streaming with JSON-RPC format."""
        try:
            for value in generator:
                chunk_response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": value,
                }
                yield self._to_sse(chunk_response)
        except Exception as e:
            self.logger.exception("Error in sync streaming")
            error_response = JsonRpcResponse.error_response(
                request_id,
                JsonRpcErrorCode.INTERNAL_ERROR,
                str(e),
            )
            yield self._to_sse(error_response.to_dict())

    def _to_sse(self, data: Any) -> bytes:
        """Convert data to SSE format."""
        json_string = json.dumps(data, ensure_ascii=False)
        return f"data: {json_string}\n\n".encode("utf-8")

    def _handle_agent_card(self, request):
        """Handle GET /.well-known/agent-card.json endpoint."""
        try:
            runtime_url = self._get_runtime_url()
            card_dict = self.agent_card.to_dict(url=runtime_url)

            self.logger.debug("Serving Agent Card: %s", self.agent_card.name)
            return JSONResponse(card_dict)
        except Exception as e:
            self.logger.exception("Failed to serve Agent Card")
            return JSONResponse({"error": str(e)}, status_code=500)

    def _handle_ping(self, request):
        """Handle GET /ping health check endpoint."""
        try:
            status = self.get_current_ping_status()
            self.logger.debug("Ping request - status: %s", status.value)
            return JSONResponse({"status": status.value, "time_of_last_update": int(self._last_status_update_time)})
        except Exception:
            self.logger.exception("Ping endpoint failed")
            return JSONResponse({"status": PingStatus.HEALTHY.value, "time_of_last_update": int(time.time())})

    async def _invoke_handler(self, handler, request_context, takes_context, jsonrpc_request):
        """Invoke the handler with appropriate arguments."""
        try:
            args = (jsonrpc_request, request_context) if takes_context else (jsonrpc_request,)

            if asyncio.iscoroutinefunction(handler):
                return await handler(*args)
            else:
                loop = asyncio.get_event_loop()
                ctx = contextvars.copy_context()
                return await loop.run_in_executor(None, ctx.run, handler, *args)
        except Exception:
            handler_name = getattr(handler, "__name__", "unknown")
            self.logger.debug("Handler '%s' execution failed", handler_name)
            raise

    def run(self, port: int = A2A_DEFAULT_PORT, host: Optional[str] = None, **kwargs):
        """Start the Bedrock AgentCore A2A server.

        Args:
            port: Port to serve on, defaults to 9000 (A2A standard)
            host: Host to bind to, auto-detected if None
            **kwargs: Additional arguments passed to uvicorn.run()
        """
        import uvicorn

        if host is None:
            if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
                host = "0.0.0.0"  # nosec B104 - Docker needs this to expose the port
            else:
                host = "127.0.0.1"

        uvicorn_params = {
            "host": host,
            "port": port,
            "access_log": self.debug,
            "log_level": "info" if self.debug else "warning",
        }
        uvicorn_params.update(kwargs)

        self.logger.info("Starting A2A server on %s:%d", host, port)
        uvicorn.run(self, **uvicorn_params)
