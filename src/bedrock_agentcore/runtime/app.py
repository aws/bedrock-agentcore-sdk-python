"""Bedrock AgentCore HTTP implementation.

Provides a Starlette-based web server that wraps user functions as HTTP endpoints.
"""

import inspect
import json
import time
from collections.abc import Sequence
from typing import Any, Callable, Dict, Optional

from starlette.middleware import Middleware
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.types import Lifespan
from starlette.websockets import WebSocket, WebSocketDisconnect

from .base_app import _BaseAgentCoreApp, _BaseRequestContextFormatter
from .models import (
    TASK_ACTION_CLEAR_FORCED_STATUS,
    TASK_ACTION_FORCE_BUSY,
    TASK_ACTION_FORCE_HEALTHY,
    TASK_ACTION_JOB_STATUS,
    TASK_ACTION_PING_STATUS,
    PingStatus,
)


class RequestContextFormatter(_BaseRequestContextFormatter):
    """Formatter including request and session IDs for HTTP applications."""

    pass


class BedrockAgentCoreApp(_BaseAgentCoreApp):
    """Bedrock AgentCore application class for HTTP protocol deployment."""

    _default_port = 8080

    def __init__(
        self,
        debug: bool = False,
        lifespan: Optional[Lifespan] = None,
        middleware: Sequence[Middleware] | None = None,
    ):
        """Initialize Bedrock AgentCore application.

        Args:
            debug: Enable debug actions for task management (default: False)
            lifespan: Optional lifespan context manager for startup/shutdown
            middleware: Optional sequence of Starlette Middleware objects (or Middleware(...) entries)
        """
        self._websocket_handler: Optional[Callable] = None

        routes = [
            Route("/invocations", self._handle_invocation, methods=["POST"]),
            Route("/ping", self._handle_ping, methods=["GET"]),
            WebSocketRoute("/ws", self._handle_websocket),
        ]
        super().__init__(
            routes=routes,
            debug=debug,
            lifespan=lifespan,
            middleware=middleware,
            logger_name="bedrock_agentcore.app",
            log_formatter=RequestContextFormatter(),
        )

    def websocket(self, func: Callable) -> Callable:
        """Decorator to register a WebSocket handler at /ws endpoint.

        Args:
            func: The function to register as WebSocket handler

        Returns:
            The decorated function

        Example:
            @app.websocket
            async def handler(websocket, context):
                await websocket.accept()
                # ... handle messages ...
        """
        self._websocket_handler = func
        return func

    async def _handle_invocation(self, request):
        request_context = self._build_request_context(request)

        start_time = time.time()

        try:
            payload = await request.json()
            self.logger.debug("Processing invocation request")

            if self.debug:
                task_response = self._handle_task_action(payload)
                if task_response:
                    duration = time.time() - start_time
                    self.logger.info("Debug action completed (%.3fs)", duration)
                    return task_response

            handler = self.handlers.get("main")
            if not handler:
                self.logger.error("No entrypoint defined")
                return JSONResponse({"error": "No entrypoint defined"}, status_code=500)

            takes_context = self._takes_context(handler)

            handler_name = handler.__name__ if hasattr(handler, "__name__") else "unknown"
            self.logger.debug("Invoking handler: %s", handler_name)
            result = await self._invoke_handler(handler, request_context, takes_context, payload)

            duration = time.time() - start_time
            if inspect.isgenerator(result):
                self.logger.info("Returning streaming response (generator) (%.3fs)", duration)
                return StreamingResponse(self._sync_stream_with_error_handling(result), media_type="text/event-stream")
            elif inspect.isasyncgen(result):
                self.logger.info("Returning streaming response (async generator) (%.3fs)", duration)
                return StreamingResponse(self._stream_with_error_handling(result), media_type="text/event-stream")

            self.logger.info("Invocation completed successfully (%.3fs)", duration)
            # Use safe serialization for consistency with streaming paths
            safe_json_string = self._safe_serialize_to_json_string(result)
            return Response(safe_json_string, media_type="application/json")

        except json.JSONDecodeError as e:
            duration = time.time() - start_time
            self.logger.warning("Invalid JSON in request (%.3fs): %s", duration, e)
            return JSONResponse({"error": "Invalid JSON", "details": str(e)}, status_code=400)
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception("Invocation failed (%.3fs)", duration)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connections."""
        request_context = self._build_request_context(websocket)

        try:
            handler = self._websocket_handler
            if not handler:
                self.logger.error("No WebSocket handler defined")
                await websocket.close(code=1011)
                return

            self.logger.debug("WebSocket connection established")
            await handler(websocket, request_context)

        except WebSocketDisconnect:
            self.logger.debug("WebSocket disconnected")
        except Exception:
            self.logger.exception("WebSocket handler failed")
            try:
                await websocket.close(code=1011)
            except Exception:
                pass

    def _handle_task_action(self, payload: dict) -> Optional[JSONResponse]:
        """Handle task management actions if present in payload."""
        action = payload.get("_agent_core_app_action")
        if not action:
            return None

        self.logger.debug("Processing debug action: %s", action)

        try:
            actions = {
                TASK_ACTION_PING_STATUS: lambda: JSONResponse(
                    {
                        "status": self.get_current_ping_status().value,
                        "time_of_last_update": int(self._last_status_update_time),
                    }
                ),
                TASK_ACTION_JOB_STATUS: lambda: JSONResponse(self.get_async_task_info()),
                TASK_ACTION_FORCE_HEALTHY: lambda: (
                    self.force_ping_status(PingStatus.HEALTHY),
                    self.logger.info("Ping status forced to Healthy"),
                    JSONResponse({"forced_status": "Healthy"}),
                )[2],
                TASK_ACTION_FORCE_BUSY: lambda: (
                    self.force_ping_status(PingStatus.HEALTHY_BUSY),
                    self.logger.info("Ping status forced to HealthyBusy"),
                    JSONResponse({"forced_status": "HealthyBusy"}),
                )[2],
                TASK_ACTION_CLEAR_FORCED_STATUS: lambda: (
                    self.clear_forced_ping_status(),
                    self.logger.info("Forced ping status cleared"),
                    JSONResponse({"forced_status": "Cleared"}),
                )[2],
            }

            if action in actions:
                response = actions[action]()
                self.logger.debug("Debug action '%s' completed successfully", action)
                return response

            self.logger.warning("Unknown debug action requested: %s", action)
            return JSONResponse({"error": f"Unknown action: {action}"}, status_code=400)

        except Exception as e:
            self.logger.exception("Debug action '%s' failed", action)
            return JSONResponse({"error": "Debug action failed", "details": str(e)}, status_code=500)

    async def _stream_with_error_handling(self, generator):
        """Wrap async generator to handle errors and convert to SSE format."""
        try:
            async for value in generator:
                yield self._convert_to_sse(value)
        except Exception as e:
            self.logger.exception("Error in async streaming")
            error_event = {
                "error": str(e),
                "error_type": type(e).__name__,
                "message": "An error occurred during streaming",
            }
            yield self._convert_to_sse(error_event)

    def _sync_stream_with_error_handling(self, generator):
        """Wrap sync generator to handle errors and convert to SSE format."""
        try:
            for value in generator:
                yield self._convert_to_sse(value)
        except Exception as e:
            self.logger.exception("Error in sync streaming")
            error_event = {
                "error": str(e),
                "error_type": type(e).__name__,
                "message": "An error occurred during streaming",
            }
            yield self._convert_to_sse(error_event)
