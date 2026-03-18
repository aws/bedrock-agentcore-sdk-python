"""AG-UI protocol support for Bedrock AgentCore Runtime.

Provides a focused Starlette app that handles RunAgentInput parsing,
EventEncoder streaming over SSE (POST /invocations) or WebSocket (/ws),
health checks, and Docker host detection.

The AG-UI contract specifies SSE and WebSocket as alternative transports
for the same AG-UI event stream. A single ``entrypoint`` handler is wired
to both endpoints automatically.
"""

import inspect
import logging
import time
import uuid
from typing import Any, Callable, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

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

logger = logging.getLogger(__name__)


def _check_ag_ui_sdk() -> None:
    """Raise ImportError with install instructions if ag-ui-protocol is missing."""
    try:
        import ag_ui  # noqa: F401
    except ImportError:
        raise ImportError(
            "ag-ui-protocol is required for AG-UI protocol support. "
            'Install it with: pip install "bedrock-agentcore[ag-ui]"'
        ) from None


class AGUIApp(Starlette):
    """Bedrock AgentCore AG-UI application.

    Exposes the same agent handler over two alternative transports
    per the AG-UI contract:

    - ``POST /invocations`` — SSE (unidirectional server→client streaming)
    - ``/ws`` — WebSocket (bidirectional, same AG-UI events)
    - ``GET /ping`` — health check
    """

    def __init__(
        self,
        debug: bool = False,
        lifespan: Any = None,
        middleware: Any = None,
    ):
        """Initialize AG-UI application.

        Args:
            debug: Enable debug mode (default: False).
            lifespan: Optional lifespan context manager.
            middleware: Optional sequence of Starlette Middleware.
        """
        _check_ag_ui_sdk()

        self._handler: Optional[Callable] = None
        self._ping_handler: Optional[Callable] = None
        self._last_status_update_time: float = time.time()

        routes = [
            Route("/invocations", self._handle_invocation, methods=["POST"]),
            Route("/ping", self._handle_ping, methods=["GET"]),
            WebSocketRoute("/ws", self._handle_websocket),
        ]
        super().__init__(routes=routes, debug=debug, lifespan=lifespan, middleware=middleware)

    def entrypoint(self, agent_or_func: Any) -> Any:
        """Register the agent handler for both SSE and WebSocket transports.

        Accepts either:
        - An object with a ``.run()`` method (framework agents)
        - A callable / async generator function (custom agents, decorator form)

        The registered handler is served on both ``POST /invocations`` (SSE)
        and ``/ws`` (WebSocket).

        Args:
            agent_or_func: The agent or function to register.

        Returns:
            The original argument (so it works as a decorator).
        """
        if hasattr(agent_or_func, "run") and callable(agent_or_func.run):
            self._handler = agent_or_func.run
        else:
            self._handler = agent_or_func
        return agent_or_func

    def ping(self, func: Callable) -> Callable:
        """Register a custom ping handler (decorator).

        Args:
            func: A callable returning a ``PingStatus``.

        Returns:
            The original function.
        """
        self._ping_handler = func
        return func

    def run(self, port: int = 8080, host: Optional[str] = None, **kwargs: Any) -> None:
        """Start the AG-UI server.

        Args:
            port: Port to serve on (default 8080).
            host: Host to bind to; auto-detected if None.
            **kwargs: Additional arguments forwarded to ``uvicorn.run()``.
        """
        import os

        import uvicorn

        if host is None:
            if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
                host = "0.0.0.0"  # nosec B104 - Container needs this to expose the port
            else:
                host = "127.0.0.1"

        uvicorn_params: dict[str, Any] = {
            "host": host,
            "port": port,
            "log_level": "info",
        }
        uvicorn_params.update(kwargs)

        uvicorn.run(self, **uvicorn_params)

    # -- shared helpers -------------------------------------------------------

    def _build_request_context(self, request: Request | WebSocket) -> RequestContext:
        """Extract Bedrock headers and build a RequestContext."""
        headers = request.headers

        request_id = headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        session_id = headers.get(SESSION_HEADER)
        BedrockAgentCoreContext.set_request_context(request_id, session_id)

        workload_access_token = headers.get(ACCESS_TOKEN_HEADER)
        if workload_access_token:
            BedrockAgentCoreContext.set_workload_access_token(workload_access_token)

        oauth2_callback_url = headers.get(OAUTH2_CALLBACK_URL_HEADER)
        if oauth2_callback_url:
            BedrockAgentCoreContext.set_oauth2_callback_url(oauth2_callback_url)

        request_headers: dict[str, str] = {}
        authorization_header = headers.get(AUTHORIZATION_HEADER)
        if authorization_header is not None:
            request_headers[AUTHORIZATION_HEADER] = authorization_header
        for header_name, header_value in headers.items():
            if header_name.lower().startswith(CUSTOM_HEADER_PREFIX.lower()):
                request_headers[header_name] = header_value
        if request_headers:
            BedrockAgentCoreContext.set_request_headers(request_headers)

        return RequestContext(
            session_id=session_id,
            request_headers=BedrockAgentCoreContext.get_request_headers(),
            request=request,
        )

    @staticmethod
    def _takes_context(handler: Callable) -> bool:
        """Check whether the handler accepts a second ``context`` parameter."""
        try:
            params = list(inspect.signature(handler).parameters.keys())
            return len(params) >= 2 and params[1] == "context"
        except Exception:
            return False

    # -- SSE transport (POST /invocations) ------------------------------------

    async def _handle_invocation(self, request: Request) -> Any:
        """Handle POST /invocations — parse RunAgentInput, stream AG-UI events via SSE.

        Returns HTTP 400 for malformed JSON or invalid RunAgentInput (the
        stream hasn't started yet, so a proper HTTP error is appropriate).
        Errors that occur *during* streaming are emitted as a ``RunErrorEvent``
        on the open SSE connection, per the AG-UI spec.
        """
        from ag_ui.core import RunAgentInput, RunErrorEvent
        from ag_ui.encoder import EventEncoder

        request_context = self._build_request_context(request)

        if self._handler is None:
            return JSONResponse({"error": "No entrypoint defined"}, status_code=500)

        try:
            payload = await request.json()
        except Exception as e:
            return JSONResponse({"error": "Invalid JSON", "details": str(e)}, status_code=400)

        try:
            run_input = RunAgentInput(**payload)
        except Exception as e:
            return JSONResponse({"error": "Invalid RunAgentInput", "details": str(e)}, status_code=400)

        accept_header = request.headers.get("accept")
        encoder = EventEncoder(accept=accept_header)
        takes_context = self._takes_context(self._handler)

        async def event_generator():
            try:
                args = (run_input, request_context) if takes_context else (run_input,)
                async for event in self._handler(*args):
                    yield encoder.encode(event)
            except Exception as e:
                logger.exception("Error during AG-UI event streaming")
                yield encoder.encode(RunErrorEvent(message=str(e), code="INTERNAL_ERROR"))

        return StreamingResponse(event_generator(), media_type=encoder.get_content_type())

    # -- WebSocket transport (/ws) --------------------------------------------

    async def _handle_websocket(self, websocket: WebSocket) -> None:
        """Handle /ws — same agent handler, WebSocket transport.

        Protocol: client connects → sends a JSON message (``RunAgentInput``)
        → server streams AG-UI events as text frames → closes on completion.
        Errors during streaming are sent as a ``RunErrorEvent`` before close.
        """
        from ag_ui.core import RunAgentInput, RunErrorEvent
        from ag_ui.encoder import EventEncoder

        await websocket.accept()
        request_context = self._build_request_context(websocket)

        if self._handler is None:
            logger.error("No entrypoint defined")
            await websocket.close(code=1011, reason="No entrypoint defined")
            return

        try:
            payload = await websocket.receive_json()
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected before sending input")
            return
        except Exception as e:
            logger.warning("Invalid WebSocket message: %s", e)
            await websocket.close(code=1003, reason="Invalid JSON")
            return

        try:
            run_input = RunAgentInput(**payload)
        except Exception as e:
            logger.warning("Invalid RunAgentInput over WebSocket: %s", e)
            await websocket.close(code=1003, reason=f"Invalid RunAgentInput: {e}")
            return

        encoder = EventEncoder()
        takes_context = self._takes_context(self._handler)

        try:
            args = (run_input, request_context) if takes_context else (run_input,)
            async for event in self._handler(*args):
                await websocket.send_text(encoder.encode(event))
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected during streaming")
            return
        except Exception as e:
            logger.exception("Error during AG-UI WebSocket streaming")
            try:
                await websocket.send_text(encoder.encode(RunErrorEvent(message=str(e), code="INTERNAL_ERROR")))
            except Exception:
                pass

        try:
            await websocket.close()
        except Exception:
            pass

    # -- ping -----------------------------------------------------------------

    def _handle_ping(self, request: Request) -> JSONResponse:
        """Handle GET /ping — health check."""
        try:
            if self._ping_handler is not None:
                status = self._ping_handler()
            else:
                status = PingStatus.HEALTHY
            self._last_status_update_time = time.time()
        except Exception:
            logger.exception("Custom ping handler failed, falling back to Healthy")
            status = PingStatus.HEALTHY
        return JSONResponse({"status": status.value, "time_of_last_update": int(self._last_status_update_time)})


def build_ag_ui_app(
    agent: Any,
    *,
    ping_handler: Optional[Callable[[], PingStatus]] = None,
) -> AGUIApp:
    """Build a Starlette app wired for AG-UI protocol with Bedrock extras.

    The returned app serves the agent on both ``POST /invocations`` (SSE) and
    ``/ws`` (WebSocket).

    Args:
        agent: An agent object with ``.run()`` or an async generator callable.
        ping_handler: Optional callback returning a ``PingStatus``.

    Returns:
        An ``AGUIApp`` instance (not started).
    """
    app = AGUIApp()
    app.entrypoint(agent)
    if ping_handler is not None:
        app.ping(ping_handler)
    return app


def serve_ag_ui(
    agent: Any,
    *,
    port: int = 8080,
    host: Optional[str] = None,
    ping_handler: Optional[Callable[[], PingStatus]] = None,
    **kwargs: Any,
) -> None:
    """Start a Bedrock-compatible AG-UI server.

    Args:
        agent: An agent object with ``.run()`` or an async generator callable.
        port: Port to serve on (default 8080).
        host: Host to bind to; auto-detected if ``None``.
        ping_handler: Optional callback returning a ``PingStatus``.
        **kwargs: Additional arguments forwarded to ``uvicorn.run()``.
    """
    app = build_ag_ui_app(agent, ping_handler=ping_handler)
    app.run(port=port, host=host, **kwargs)
