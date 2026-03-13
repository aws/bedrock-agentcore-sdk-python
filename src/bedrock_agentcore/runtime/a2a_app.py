"""Bedrock AgentCore A2A application implementation.

Provides a Starlette-based web server for A2A (Agent-to-Agent) protocol communication.
"""

import inspect
import json
import os
import time
from collections.abc import Sequence
from typing import Any, Callable, Dict, Optional

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
from .base_app import _BaseAgentCoreApp, _BaseRequestContextFormatter
from .utils import convert_complex_objects


class A2ARequestContextFormatter(_BaseRequestContextFormatter):
    """Formatter including request and session IDs for A2A applications."""

    extra_fields = {"protocol": "A2A"}


class BedrockAgentCoreA2AApp(_BaseAgentCoreApp):
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

    _default_port = A2A_DEFAULT_PORT

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

        routes = [
            Route("/", self._handle_jsonrpc, methods=["POST"]),
            Route("/.well-known/agent-card.json", self._handle_agent_card, methods=["GET"]),
            Route("/ping", self._handle_ping, methods=["GET"]),
        ]
        super().__init__(
            routes=routes,
            debug=debug,
            lifespan=lifespan,
            middleware=middleware,
            logger_name="bedrock_agentcore.a2a_app",
            log_formatter=A2ARequestContextFormatter(),
        )

    def _get_runtime_url(self, request=None) -> Optional[str]:
        """Get the runtime URL from environment or current request.

        Returns:
            The runtime URL if set, None otherwise.
        """
        runtime_url = os.environ.get("AGENTCORE_RUNTIME_URL")
        if runtime_url:
            return runtime_url

        if request is not None and getattr(request, "base_url", None):
            return str(request.base_url)

        return None

    async def _handle_jsonrpc(self, request):
        """Handle JSON-RPC 2.0 requests at root endpoint."""
        request_context = self._build_request_context(request)
        start_time = time.time()
        body = None

        try:
            body = await request.json()
            if not isinstance(body, dict):
                return self._jsonrpc_error_response(
                    None,
                    JsonRpcErrorCode.INVALID_REQUEST,
                    "Invalid request object",
                )

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
            response = JsonRpcResponse.success(jsonrpc_request.id, self._convert_to_serializable(result))
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
                body.get("id") if body is not None else None,
                JsonRpcErrorCode.INTERNAL_ERROR,
                "Internal error",
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
                "Internal error",
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
                "Internal error",
            )
            yield self._to_sse(error_response.to_dict())

    def _to_sse(self, data: Any) -> bytes:
        """Convert data to SSE format."""
        json_string = self._safe_serialize_to_json_string(data)
        return f"data: {json_string}\n\n".encode("utf-8")

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert A2A helper models and common Python objects to JSON-safe payloads."""
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return self._convert_to_serializable(obj.to_dict())

        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(value) for value in obj]

        if isinstance(obj, set):
            return [self._convert_to_serializable(value) for value in obj]

        return convert_complex_objects(obj)

    def _safe_serialize_to_json_string(self, obj: Any) -> str:
        """Safely serialize streaming payloads to JSON, with A2A model support."""
        try:
            return json.dumps(obj, ensure_ascii=False)
        except (TypeError, ValueError, UnicodeEncodeError):
            try:
                return json.dumps(self._convert_to_serializable(obj), ensure_ascii=False)
            except Exception:
                return json.dumps(str(obj), ensure_ascii=False)

    def _handle_agent_card(self, request):
        """Handle GET /.well-known/agent-card.json endpoint."""
        try:
            runtime_url = self._get_runtime_url(request)
            card_dict = self.agent_card.to_dict(url=runtime_url)

            self.logger.debug("Serving Agent Card: %s", self.agent_card.name)
            return JSONResponse(card_dict)
        except Exception as e:
            self.logger.exception("Failed to serve Agent Card")
            return JSONResponse({"error": "Internal error"}, status_code=500)

    def run(self, port: Optional[int] = None, host: Optional[str] = None, **kwargs):
        """Start the Bedrock AgentCore A2A server.

        Args:
            port: Port to serve on, defaults to 9000 (A2A standard)
            host: Host to bind to, auto-detected if None
            **kwargs: Additional arguments passed to uvicorn.run()
        """
        if port is None:
            port = self._default_port
        self.logger.info("Starting A2A server on port %d", port)
        super().run(port=port, host=host, **kwargs)
