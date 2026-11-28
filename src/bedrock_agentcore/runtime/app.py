"""Bedrock AgentCore base implementation.

Provides a Starlette-based web server that wraps user functions as HTTP endpoints.
"""

import asyncio
import contextlib
import contextvars
import functools
import inspect
import json
import logging
import os
import queue
import socket
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, NamedTuple, Optional

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import BaseRoute, Route, Router, WebSocketRoute
from starlette.types import ASGIApp, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..config_bundle.baggage import _extract_baggage, _parse_config_bundle_baggage
from ..config_bundle.bundle import ConfigBundleRef
from ..config_bundle.client import ConfigBundleClient
from .context import BedrockAgentCoreContext, RequestContext
from .models import (
    _AUTHORIZATION_HEADER_LOWER,
    ACCESS_TOKEN_HEADER,
    AUTHORIZATION_HEADER,
    BAGGAGE_KEY_EXPERIMENT_ARN,
    BAGGAGE_KEY_EXPERIMENT_VARIANT,
    IDENTITY_WAT_HEADER,
    OAUTH2_CALLBACK_URL_HEADER,
    REQUEST_ID_HEADER,
    SESSION_HEADER,
    TASK_ACTION_CLEAR_FORCED_STATUS,
    TASK_ACTION_FORCE_BUSY,
    TASK_ACTION_FORCE_HEALTHY,
    TASK_ACTION_JOB_STATUS,
    TASK_ACTION_PING_STATUS,
    WARMUP_TYPE_HEADER,
    HookConfig,
    PingStatus,
    WarmupType,
    is_forwardable_header,
)
from .tracing import _ensure_baggage_processor_registered
from .utils import convert_complex_objects

# Sentinel so we only parse OTEL_RESOURCE_ATTRIBUTES once per process.
_UNRESOLVED = object()
_runtime_arn_cache: object = _UNRESOLVED
_runtime_arn_lock: threading.Lock = threading.Lock()


def _parse_runtime_arn() -> Optional[str]:
    """Return the runtime ARN for this process, derived from OTEL_RESOURCE_ATTRIBUTES.

    Reads the ``cloud.resource_id`` attribute, which OTEL sets to either a
    runtime ARN or a runtime-endpoint ARN.
    Runtime-endpoint ARNs are normalised to a plain runtime ARN by stripping
    the ``/runtime-endpoint/...`` suffix.

    The result is cached after the first call — the env var does not change
    during the process lifetime.

    Returns ``None`` when the env var is absent or ``cloud.resource_id`` is
    not present.
    """
    global _runtime_arn_cache
    if _runtime_arn_cache is not _UNRESOLVED:
        return _runtime_arn_cache  # type: ignore[return-value]

    with _runtime_arn_lock:
        if _runtime_arn_cache is not _UNRESOLVED:
            return _runtime_arn_cache  # type: ignore[return-value]

        result: Optional[str] = None
        otel_attrs = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
        for attr in otel_attrs.split(","):
            attr = attr.strip()
            if not attr.startswith("cloud.resource_id="):
                continue
            arn = attr[len("cloud.resource_id=") :]
            # Normalise runtime-endpoint ARN → runtime ARN.
            if "/runtime-endpoint/" in arn:
                arn = arn.split("/runtime-endpoint/")[0]
            result = arn
            break

        _runtime_arn_cache = result
        return result


def _is_async_callable(obj: Any) -> bool:
    """Check if obj is async-callable, unwrapping functools.partial."""
    while isinstance(obj, functools.partial):
        obj = obj.func
    return asyncio.iscoroutinefunction(obj) or (callable(obj) and asyncio.iscoroutinefunction(obj.__call__))


def _is_async_gen_callable(obj: Any) -> bool:
    """Check if obj is an async generator function, unwrapping functools.partial."""
    while isinstance(obj, functools.partial):
        obj = obj.func
    return inspect.isasyncgenfunction(obj) or (callable(obj) and inspect.isasyncgenfunction(obj.__call__))


def _is_generator_callable(obj: Any) -> bool:
    """Check if obj is a generator or async generator function, unwrapping functools.partial.

    A callable object counts when its ``__call__`` is one.
    """
    while isinstance(obj, functools.partial):
        obj = obj.func
    obj = obj.__call__ if callable(obj) and not inspect.isroutine(obj) else obj
    return inspect.isgeneratorfunction(obj) or inspect.isasyncgenfunction(obj)


def _callable_name(obj: Any) -> str:
    """Name a callable for error messages, unwrapping functools.partial."""
    while isinstance(obj, functools.partial):
        obj = obj.func
    return getattr(obj, "__qualname__", None) or type(obj).__qualname__


def _restore_context(ctx: contextvars.Context) -> None:
    """Restore context variables from a snapshot (Django asgiref pattern)."""
    for var, value in ctx.items():
        try:
            if var.get() != value:
                var.set(value)
        except LookupError:
            var.set(value)


class RequestContextFormatter(logging.Formatter):
    """Formatter including request and session IDs."""

    def format(self, record):
        """Format log record as AWS Lambda JSON."""
        import json
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
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


def _add_route(route: Route, routes: list[BaseRoute], where: str = "by the app") -> None:
    """Append route to routes, raising ValueError if one of its (method, path) pairs is already served.

    Only HTTP routes count: a WebSocketRoute serves WebSocket connections (a GET upgrade), never a POST.
    ``where`` names the routes in the error, e.g. ``"on port 9000"``.
    """
    served = {(method, r.path) for r in routes if isinstance(r, Route) for method in r.methods or ()}
    clashes = sorted({(method, route.path) for method in route.methods or ()} & served)
    if clashes:
        raise ValueError(f"Route {', '.join(m for m, _ in clashes)} {route.path} is already served {where}")
    routes.append(route)


_HOOK_HOST = "127.0.0.1"  # the runtime calls hooks on loopback
_HOOK_PORTS_NEED_RUN = (
    "Hook ports are only served by app.run(); start the agent with app.run(), or leave HookConfig.port unset"
)


class _HookPortServer(uvicorn.Server):
    """Serves one hook port on the main server's event loop, started and stopped by the app's lifespan."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        """Leave SIGINT and SIGTERM to the main server; its lifespan shutdown stops this server."""
        yield


class _RunningHookServer(NamedTuple):
    """A hook port's server and the task serving it."""

    server: _HookPortServer
    task: asyncio.Task


def _bind_hook_socket(port: int) -> socket.socket:
    """Bind a hook port's socket on loopback, raising a RuntimeError that names the port if it can't be bound."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((_HOOK_HOST, port))
    except OSError as e:
        sock.close()
        raise RuntimeError(
            f"Can't serve the hook on HookConfig.port {port}: binding {_HOOK_HOST}:{port} failed: {e.strerror or e}"
        ) from e
    sock.set_inheritable(True)
    return sock


@dataclass
class _Hook:
    """One lifecycle hook: where it's served, its handlers, and the lock that keeps its calls from overlapping."""

    config: HookConfig
    handlers: Dict[Any, Callable] = field(default_factory=dict)  # warmup: WarmupType -> handler
    # One lock per hook, so its calls never overlap. A waiting call awaits the lock rather than
    # blocking, so the server keeps answering /ping.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # used only by this hook's route


class BedrockAgentCoreApp(Starlette):
    """Bedrock AgentCore application class that extends Starlette for AI agent deployment."""

    def __init__(
        self,
        debug: bool = False,
        lifespan: Optional[Lifespan] = None,
        middleware: Sequence[Middleware] | None = None,
        *,
        warmup_hook: Optional[HookConfig] = None,
    ):
        """Initialize Bedrock AgentCore application.

        Args:
            debug: Enable debug actions for task management (default: False)
            lifespan: Optional lifespan for startup/shutdown, preferably an async context manager factory
                (``@contextlib.asynccontextmanager``); any style Starlette accepts works. When a hook has a
                ``port``, it runs nested inside the app's own lifespan, which starts the hook servers before
                it and stops them after it.
            middleware: Optional sequence of Starlette Middleware objects (or Middleware(...) entries). They
                apply to the serving port, not to a hook served on its own port.
            warmup_hook: Where to serve the build warmup hook (``POST``), registered with ``@app.warmup``.
                It's served on ``warmup_hook.port`` when set, otherwise on the serving port. A separate port
                needs ``app.run()``: started another way, the app fails its startup, or answers 503 if the
                lifespan is off. ``None`` (the default) adds no warmup route.

        Raises:
            ValueError: If ``warmup_hook.path`` clashes with an existing ``POST`` route on its port, such as
                ``/invocations`` on the serving port.
        """
        self.handlers: Dict[str, Callable] = {}
        self._ping_handler: Optional[Callable] = None
        self._websocket_handler: Optional[Callable] = None
        self._active_tasks: Dict[int, Dict[str, Any]] = {}
        self._task_counter_lock: threading.Lock = threading.Lock()
        self._forced_ping_status: Optional[PingStatus] = None
        self._last_status_update_time: float = time.time()
        self._worker_loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_loop_lock: threading.Lock = threading.Lock()
        self._hooks: Dict[str, _Hook] = {}
        if warmup_hook is not None:
            self._hooks["warmup"] = _Hook(warmup_hook)
        self.warmup_hook = warmup_hook
        # Routes of hooks with their own port, by port. app.run() serves each port with its own small app,
        # except the serving port, whose routes it moves onto this app.
        self._hook_port_routes: Dict[int, list[BaseRoute]] = {}
        self._started_by_run = False  # set by run(); hook ports are only served then
        self._hook_servers_running = False

        # Config bundle client — created lazily on first use.
        # _resolve_bundle_config is wrapped with lru_cache(maxsize=30), keyed by
        # ConfigBundleRef. The API is called at most once per unique ref across all
        # requests on this app instance (one process = one microVM = one app instance).
        self._config_client: Optional[ConfigBundleClient] = None
        self._config_client_lock: threading.Lock = threading.Lock()
        self._resolve_bundle_config = functools.lru_cache(maxsize=30)(  # type: ignore[method-assign]
            self._resolve_bundle_config
        )

        routes: list[BaseRoute] = [
            Route("/invocations", self._handle_invocation, methods=["POST"]),
            Route("/ping", self._handle_ping, methods=["GET"]),
            WebSocketRoute("/ws", self._handle_websocket),
        ]
        if warmup_hook is not None:
            self._add_hook_route(warmup_hook, Route(warmup_hook.path, self._handle_warmup, methods=["POST"]), routes)
        if self._hook_port_routes:
            lifespan = self._lifespan_with_hook_servers(lifespan)
        super().__init__(routes=routes, lifespan=lifespan, middleware=middleware)
        self.debug = debug  # Set after super().__init__ to avoid override

        self.logger = logging.getLogger("bedrock_agentcore.app")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = RequestContextFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

        # Register early so the ASGI entry span (POST /invocations) gets stamped.
        # In the managed runtime ADOT sets up the TracerProvider before __init__ runs,
        # so this call lands on the real provider rather than the no-op default.
        _ensure_baggage_processor_registered()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve the app, or answer 503 if a hook has its own port but the hook servers aren't running.

        That happens when the server skips the app's lifespan (e.g. ``uvicorn agent:app --lifespan off``),
        which is what starts them.
        """
        if scope["type"] == "http" and self._hook_port_routes and not self._hook_servers_running:
            await JSONResponse({"error": _HOOK_PORTS_NEED_RUN}, status_code=503)(scope, receive, send)
            return
        await super().__call__(scope, receive, send)

    def _add_hook_route(self, config: HookConfig, route: Route, main_routes: list[BaseRoute]) -> None:
        """Add a hook's route to the main app's routes, or, if the hook has a port, to that port's routes.

        Either way it's checked against the routes already served there. A hook on a port that turns out to
        be the serving port is checked against the main routes by ``run()``, which moves it there.
        """
        if config.port is None:
            _add_route(route, main_routes)
            return
        _add_route(route, self._hook_port_routes.setdefault(config.port, []), where=f"on port {config.port}")

    def _lifespan_with_hook_servers(self, user_lifespan: Optional[Lifespan]) -> Lifespan:
        """Wrap the user's lifespan in one that starts the hook servers before it and stops them after it."""
        # Starlette's Router turns every lifespan style it accepts into an async context manager factory.
        user_context = None if user_lifespan is None else Router(lifespan=user_lifespan).lifespan_context

        @contextlib.asynccontextmanager
        async def lifespan(app: ASGIApp):
            # Raising here makes Starlette report lifespan.startup.failed, so the server exits.
            if not self._started_by_run:
                raise RuntimeError(_HOOK_PORTS_NEED_RUN)
            hook_servers = await self._start_hook_servers()
            try:
                if user_context is None:
                    yield None
                else:
                    async with user_context(app) as state:
                        yield state
            finally:
                await self._stop_hook_servers(hook_servers)

        return lifespan

    async def _start_hook_servers(self) -> list[_RunningHookServer]:
        """Start one server per hook port on this event loop, and return once every hook port listens.

        The lifespan runs before the main server binds the serving port, so hook ports listen first.
        """
        sockets: list[socket.socket] = []
        try:
            for port in self._hook_port_routes:
                sockets.append(_bind_hook_socket(port))
        except Exception:
            for sock in sockets:
                sock.close()
            raise

        hook_servers: list[_RunningHookServer] = []
        for (port, routes), sock in zip(self._hook_port_routes.items(), sockets, strict=True):
            config = uvicorn.Config(
                Starlette(routes=routes),
                host=_HOOK_HOST,
                port=port,
                lifespan="off",
                log_config=None,  # keep the main server's logging config
                access_log=True,  # uvicorn.access is set up (or silenced) by the main server
            )
            server = _HookPortServer(config)
            hook_servers.append(_RunningHookServer(server, asyncio.create_task(server.serve(sockets=[sock]))))

        while not all(hook_server.server.started for hook_server in hook_servers):
            if any(hook_server.task.done() for hook_server in hook_servers):
                await self._stop_hook_servers(hook_servers)
                for sock in sockets:
                    sock.close()
                raise RuntimeError(f"A hook server failed to start on ports {list(self._hook_port_routes)}")
            await asyncio.sleep(0.01)
        self._hook_servers_running = True
        for port, routes in self._hook_port_routes.items():
            paths = ", ".join(f"POST {route.path}" for route in routes if isinstance(route, Route))
            self.logger.info("Hook server listening on http://%s:%d (%s)", _HOOK_HOST, port, paths)
        return hook_servers

    async def _stop_hook_servers(self, hook_servers: list[_RunningHookServer]) -> None:
        """Stop the hook servers and wait for them to finish their in-flight requests."""
        self._hook_servers_running = False
        for hook_server in hook_servers:
            hook_server.server.should_exit = True
        results = await asyncio.gather(*(hook_server.task for hook_server in hook_servers), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                self.logger.error("Hook server failed: %r", result)
        if hook_servers:
            self.logger.info("Hook servers stopped")

    def entrypoint(self, func: Callable) -> Callable:
        """Decorator to register a function as the main entrypoint.

        Invocation payloads are passed to the registered function unchanged. Applications should validate input
        before forwarding it to an agent framework.

        Args:
            func: The function to register as entrypoint

        Returns:
            The decorated function with added serve method
        """
        self.handlers["main"] = func
        try:
            func.run = lambda port=8080, host=None: self.run(port, host)
        except AttributeError:
            pass
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

    def warmup(self, type: WarmupType | str) -> Callable[[Callable], Callable]:
        """Decorator to register a build warmup handler for one warmup type.

        AgentCore Runtime calls the build warmup hook (``POST`` to ``warmup_hook.path``) twice while it prepares
        the agent during ``CreateAgentRuntime`` and ``UpdateAgentRuntime``: first ``memory``, then ``disk``.
        Register a handler for zero, one or both types. A type without a handler is answered with 200, so
        it has nothing to do, and so is a type this SDK version doesn't know; only a missing or empty type
        header is answered with 400. The handler takes no arguments and may be sync or async, but not a
        generator (no ``yield``): sync handlers run in a worker thread and async handlers on a separate event
        loop, so the server keeps answering ``/ping`` while they run. The response is sent after the handler
        returns. An exception in the handler returns 500, which fails the create or update, and so does a
        handler that returns a generator or an awaitable it didn't await. Handlers never run concurrently with
        each other in one process.

        See https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-lifecycle-hooks.html

        Args:
            type: The warmup type to handle, a ``WarmupType`` or its string value (``"memory"`` or ``"disk"``)

        Returns:
            A decorator that registers the function and returns it unchanged. The decorator raises
            ``ValueError`` if the function is a generator function (``def`` or ``async def`` with ``yield``,
            also through ``functools.partial`` or a callable object's ``__call__``), since its body would
            never run.

        Raises:
            ValueError: If the app has no ``warmup_hook``, ``type`` isn't a valid warmup type, or a handler is
                already registered for it.

        Example:
            app = BedrockAgentCoreApp(warmup_hook=HookConfig(path="/warmup"))

            @app.warmup(WarmupType.MEMORY)
            def load_model():
                global MODEL
                MODEL = load("/opt/model.bin")
        """
        if "warmup" not in self._hooks:
            raise ValueError(
                "@app.warmup needs BedrockAgentCoreApp(warmup_hook=HookConfig(path=...)) matching "
                "hooksConfiguration.build.warmup.action.httpPost.path"
            )
        warmup_type = WarmupType(type)
        handlers = self._hooks["warmup"].handlers
        if warmup_type in handlers:
            raise ValueError(f"A {warmup_type.value} warmup handler is already registered")

        def register(func: Callable) -> Callable:
            if _is_generator_callable(func):
                raise ValueError(
                    f"warmup handler {_callable_name(func)} is a generator function: its body would never run. "
                    "Remove the `yield`."
                )
            handlers[warmup_type] = func
            return func

        return register

    def async_task(self, func: Callable) -> Callable:
        """Decorator to track async tasks for ping status.

        When a function is decorated with @async_task, it will:
        - Set ping status to HEALTHY_BUSY while running
        - Revert to HEALTHY when complete
        """
        if not _is_async_callable(func):
            raise ValueError("@async_task can only be applied to async functions")

        async def wrapper(*args, **kwargs):
            task_id = self.add_async_task(func.__name__)

            try:
                self.logger.debug("Starting async task: %s", func.__name__)
                start_time = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                self.logger.info("Async task completed: %s (%.3fs)", func.__name__, duration)
                return result
            except Exception:
                duration = time.time() - start_time
                self.logger.exception("Async task failed: %s (%.3fs)", func.__name__, duration)
                raise
            finally:
                self.complete_async_task(task_id)

        wrapper.__name__ = func.__name__
        return wrapper

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

    def force_ping_status(self, status: PingStatus):
        """Force ping status to a specific value."""
        self._forced_ping_status = status

    def clear_forced_ping_status(self):
        """Clear forced status and resume automatic."""
        self._forced_ping_status = None

    def get_async_task_info(self) -> Dict[str, Any]:
        """Get info about running async tasks."""
        running_jobs = []
        for t in self._active_tasks.values():
            try:
                running_jobs.append(
                    {"name": t.get("name", "unknown"), "duration": time.time() - t.get("start_time", time.time())}
                )
            except Exception as e:
                self.logger.warning("Caught exception, continuing...: %s", e)
                continue

        return {"active_count": len(self._active_tasks), "running_jobs": running_jobs}

    def add_async_task(self, name: str, metadata: Optional[Dict] = None) -> int:
        """Register an async task for interactive health tracking.

        This method provides granular control over async task lifecycle,
        allowing developers to interactively start tracking tasks for health monitoring.
        Use this when you need precise control over when tasks begin and end.

        Args:
            name: Human-readable task name for monitoring
            metadata: Optional additional task metadata

        Returns:
            Task ID for tracking and completion

        Example:
            task_id = app.add_async_task("file_processing", {"file": "data.csv"})
            # ... do background work ...
            app.complete_async_task(task_id)
        """
        with self._task_counter_lock:
            task_id = hash(str(uuid.uuid4()))  # Generate truly unique hash-based ID

            # Register task start with same structure as @async_task decorator
            task_info = {"name": name, "start_time": time.time()}
            if metadata:
                task_info["metadata"] = metadata

            self._active_tasks[task_id] = task_info

        self.logger.info("Async task started: %s (ID: %s)", name, task_id)
        return task_id

    def complete_async_task(self, task_id: int) -> bool:
        """Mark an async task as complete for interactive health tracking.

        This method provides granular control over async task lifecycle,
        allowing developers to interactively complete tasks for health monitoring.
        Call this when your background work finishes.

        Args:
            task_id: Task ID returned from add_async_task

        Returns:
            True if task was found and completed, False otherwise

        Example:
            task_id = app.add_async_task("file_processing")
            # ... do background work ...
            completed = app.complete_async_task(task_id)
        """
        with self._task_counter_lock:
            task_info = self._active_tasks.pop(task_id, None)
            if task_info:
                task_name = task_info.get("name", "unknown")
                duration = time.time() - task_info.get("start_time", time.time())

                self.logger.info("Async task completed: %s (ID: %s, Duration: %.2fs)", task_name, task_id, duration)
                return True
            else:
                self.logger.warning("Attempted to complete unknown task ID: %s", task_id)
                return False

    def _build_request_context(self, request) -> RequestContext:
        """Build request context and setup all context variables."""
        try:
            headers = request.headers
            request_id = headers.get(REQUEST_ID_HEADER)
            if not request_id:
                request_id = str(uuid.uuid4())

            session_id = headers.get(SESSION_HEADER)
            BedrockAgentCoreContext.set_request_context(request_id, session_id)

            agent_identity_token = headers.get(IDENTITY_WAT_HEADER) or headers.get(ACCESS_TOKEN_HEADER)
            if agent_identity_token:
                BedrockAgentCoreContext.set_workload_access_token(agent_identity_token)

            oauth2_callback_url = headers.get(OAUTH2_CALLBACK_URL_HEADER)
            if oauth2_callback_url:
                BedrockAgentCoreContext.set_oauth2_callback_url(oauth2_callback_url)

            # Collect forwardable request headers.
            # Authorization is normalised to a canonical key regardless of wire casing
            # (HTTP/2 always lowercases headers; HTTP/1.1 may preserve mixed case).
            # All other headers are checked against the runtime header allowlist rules.
            request_headers = {}

            for header_name, header_value in headers.items():
                if header_name.lower() == _AUTHORIZATION_HEADER_LOWER:
                    request_headers[AUTHORIZATION_HEADER] = header_value
                elif is_forwardable_header(header_name):
                    request_headers[header_name] = header_value

            # Set in context if any headers were found
            if request_headers:
                BedrockAgentCoreContext.set_request_headers(request_headers)

            # Parse baggage once; reuse for both config bundle and routing experiment.
            all_baggage: dict = {}
            bundle_ref = None
            try:
                all_baggage = _extract_baggage(headers)
                bundle_ref = _parse_config_bundle_baggage(all_baggage)
            except Exception as e:
                self.logger.warning(
                    "Failed to parse baggage: %s: %s — raw baggage: %r",
                    type(e).__name__,
                    e,
                    headers.get("baggage", ""),
                )

            if bundle_ref is not None:
                self.logger.info("Received config bundle ref: %s", bundle_ref.bundle_id)
                BedrockAgentCoreContext.set_config_bundle_ref(bundle_ref)
                BedrockAgentCoreContext._set_bundle_loader(
                    fetcher=lambda: self._resolve_bundle_config(bundle_ref),
                )
            else:
                self.logger.debug("No config bundle ref found in request baggage")
                BedrockAgentCoreContext.set_config_bundle_ref(None)
                BedrockAgentCoreContext._clear_bundle_loader()

            experiment_arn = next(iter(all_baggage.get(BAGGAGE_KEY_EXPERIMENT_ARN, [])), None)
            experiment_variant = next(iter(all_baggage.get(BAGGAGE_KEY_EXPERIMENT_VARIANT, [])), None)
            BedrockAgentCoreContext.set_routing_experiment(experiment_arn, experiment_variant)
            # Re-registers if the TracerProvider was replaced after __init__ ran
            # (e.g. a framework calling set_tracer_provider during first-request setup).
            _ensure_baggage_processor_registered()

            # Get the headers from context to pass to RequestContext
            req_headers = BedrockAgentCoreContext.get_request_headers()

            return RequestContext(
                session_id=session_id,
                request_headers=req_headers,
                request=request,  # Pass through the Starlette request object
            )
        except Exception as e:
            self.logger.warning("Failed to build request context: %s: %s", type(e).__name__, e)
            request_id = str(uuid.uuid4())
            BedrockAgentCoreContext.set_request_context(request_id, None)
            return RequestContext(session_id=None, request=None)

    def _get_config_client(self) -> ConfigBundleClient:
        """Return the config client, creating it lazily once per process."""
        if self._config_client is None:
            with self._config_client_lock:
                if self._config_client is None:
                    self._config_client = ConfigBundleClient()
        return self._config_client

    def _resolve_bundle_config(self, ref: ConfigBundleRef) -> Dict[str, Any]:
        """Fetch bundle from API and return this runtime's config section.

        Manages client lifecycle, API call, runtime ARN filtering.
        Called by _DeferredBundleConfig.get() on a cache miss — at most once per
        unique (bundle_id, bundle_version) across all requests.
        """
        self.logger.debug("Fetching config bundle %r version %r", ref.bundle_id, ref.bundle_version)
        try:
            response = self._get_config_client().get_configuration_bundle_version(
                bundleId=ref.bundle_id, versionId=ref.bundle_version
            )
        except Exception as e:
            self.logger.error(
                "Failed to fetch config bundle %r version %r: %s: %s",
                ref.bundle_id,
                ref.bundle_version,
                type(e).__name__,
                e,
            )
            raise

        components = response.get("components", {})
        runtime_arn = _parse_runtime_arn()
        if runtime_arn is None:
            self.logger.warning("OTEL_RESOURCE_ATTRIBUTES not set — cannot select config component")
            return {}

        component = components.get(runtime_arn)
        if component is None:
            self.logger.warning(
                "Runtime ARN %r not found in bundle %r — available: %s",
                runtime_arn,
                ref.bundle_id,
                list(components.keys()),
            )
            return {}

        return component.get("configuration", {})

    def _takes_context(self, handler: Callable) -> bool:
        try:
            params = list(inspect.signature(handler).parameters.keys())
            return len(params) >= 2 and params[1] == "context"
        except Exception:
            return False

    async def _handle_invocation(self, request):
        request_context = self._build_request_context(request)

        start_time = time.time()

        try:
            payload = await request.json()
        except json.JSONDecodeError as e:
            duration = time.time() - start_time
            self.logger.warning("Invalid JSON in request (%.3fs): %s", duration, e)
            return JSONResponse({"error": "Invalid JSON", "details": str(e)}, status_code=400)
        except UnicodeDecodeError as e:
            duration = time.time() - start_time
            self.logger.warning("Invalid encoding in request (%.3fs): %s", duration, e)
            return JSONResponse({"error": "Invalid encoding", "details": str(e)}, status_code=400)

        try:
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

            # If handler returned a Starlette Response directly, pass it through.
            # This lets handlers control status codes (e.g. JSONResponse(data, status_code=404)).
            if isinstance(result, Response):
                status = getattr(result, "status_code", 200)
                # Log at warning level for error responses so operators can distinguish
                # intentional error responses from successful invocations in logs.
                if status >= 400:
                    self.logger.warning("Invocation returned HTTP %d (%.3fs)", status, duration)
                else:
                    self.logger.info("Invocation completed successfully (%.3fs)", duration)
                return result

            self.logger.info("Invocation completed successfully (%.3fs)", duration)
            # Use safe serialization for consistency with streaming paths
            safe_json_string = self._safe_serialize_to_json_string(result)
            return Response(safe_json_string, media_type="application/json")
        except HTTPException as e:
            duration = time.time() - start_time
            # Use error level for 5xx to match the generic Exception handler's severity,
            # since server errors warrant the same urgency regardless of how they're raised.
            # Use warning for 4xx since those are intentional client-error responses.
            if e.status_code >= 500:
                self.logger.error("HTTP %d (%.3fs): %s", e.status_code, duration, e.detail)
            else:
                self.logger.warning("HTTP %d (%.3fs): %s", e.status_code, duration, e.detail)
            return JSONResponse({"error": e.detail}, status_code=e.status_code)
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception("Invocation failed (%.3fs)", duration)
            return JSONResponse({"error": str(e)}, status_code=500)

    def _handle_ping(self, request):
        try:
            status = self.get_current_ping_status()
            self.logger.debug("Ping request - status: %s", status.value)
            return JSONResponse({"status": status.value, "time_of_last_update": int(self._last_status_update_time)})
        except Exception:
            self.logger.exception("Ping endpoint failed")
            return JSONResponse({"status": PingStatus.HEALTHY.value, "time_of_last_update": int(time.time())})

    async def _handle_warmup(self, request):
        raw_type = request.headers.get(WARMUP_TYPE_HEADER, "").strip().lower()
        if not raw_type:
            self.logger.warning("Warmup request without a %s header", WARMUP_TYPE_HEADER)
            return JSONResponse({"error": f"{WARMUP_TYPE_HEADER} header is missing or empty"}, status_code=400)
        try:
            warmup_type = WarmupType(raw_type)
        except ValueError:
            # A type this SDK version doesn't know, perhaps added by the service later: there's nothing to do,
            # and failing it would fail the create or update of every agent on this version.
            self.logger.info("Warmup (%s): unknown type, nothing to do", raw_type)
            return JSONResponse({"status": "ok", "type": raw_type, "result": "skipped"})

        hook = self._hooks["warmup"]
        handler = hook.handlers.get(warmup_type)
        if handler is None:
            self.logger.info("Warmup (%s): no handler registered, nothing to do", warmup_type.value)
            return JSONResponse({"status": "ok", "type": warmup_type.value, "result": "skipped"})

        async with hook.lock:
            start_time = time.time()
            try:
                result = await self._dispatch_handler(handler, ())
                if inspect.isgenerator(result) or inspect.isasyncgen(result):
                    # Never iterated, so the body never ran; close it so it doesn't linger.
                    if inspect.isgenerator(result):
                        result.close()
                    else:
                        await result.aclose()
                    raise TypeError("warmup handler returned a generator, so its body never ran; remove the `yield`")
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()  # never awaited on purpose; close it so Python doesn't warn
                    raise TypeError("warmup handler returned an awaitable; declare it with async def")
            except Exception as e:
                self.logger.exception("Warmup (%s) failed (%.3fs)", warmup_type.value, time.time() - start_time)
                return JSONResponse({"error": str(e), "type": warmup_type.value}, status_code=500)
            self.logger.info("Warmup (%s) completed (%.3fs)", warmup_type.value, time.time() - start_time)
        return JSONResponse({"status": "ok", "type": warmup_type.value, "result": "completed"})

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

    def run(self, port: int = 8080, host: Optional[str] = None, **kwargs):
        """Start the Bedrock AgentCore server.

        A hook with its own ``HookConfig.port`` is served on that port by a separate server in this process,
        started from the app's lifespan before the serving port opens, and stopped after the main server
        shuts down. A hook whose port equals ``port`` is served on the main server.

        Args:
            port: Port to serve on, defaults to 8080
            host: Host to bind to, auto-detected if None. Hook ports bind to 127.0.0.1 only.
            **kwargs: Additional arguments passed to uvicorn.run()

        Raises:
            ValueError: If a hook has a ``port`` and ``lifespan="off"`` is passed, since the lifespan starts the
                hook servers, or if a hook on ``port`` clashes with a route of the main server.
        """
        import os

        if host is None:
            if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
                host = "0.0.0.0"  # nosec B104 - Docker needs this to expose the port
            else:
                host = "127.0.0.1"

        # Set default uvicorn parameters, allow kwargs to override
        uvicorn_params = {
            "host": host,
            "port": port,
            "access_log": self.debug,
            "log_level": "info" if self.debug else "warning",
        }
        uvicorn_params.update(kwargs)

        if (
            any(hook.config.port is not None for hook in self._hooks.values())
            and uvicorn_params.get("lifespan") == "off"
        ):
            raise ValueError(
                "app.run(lifespan=\"off\") can't serve HookConfig.port: the app's lifespan starts the hook servers"
            )
        # The serving port is only known here, so a hook set to that port couldn't be placed in __init__.
        # Serve it from the main app (a second server can't bind the same port); this is its first check
        # against the main routes.
        for route in self._hook_port_routes.pop(port, []):
            _add_route(route, self.router.routes)
        self._started_by_run = True

        uvicorn.run(self, **uvicorn_params)

    def _ensure_worker_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily create and start a dedicated worker event loop in a background thread.

        The worker loop isolates async handler execution from the main event loop,
        ensuring that blocking async handlers do not prevent /ping from responding.
        """
        if self._worker_loop is not None and self._worker_loop.is_running():
            return self._worker_loop
        with self._worker_loop_lock:
            if self._worker_loop is None or not self._worker_loop.is_running():
                ready = threading.Event()
                self._worker_thread = threading.Thread(
                    target=self._run_worker_loop,
                    args=(ready,),
                    daemon=True,
                    name="agentcore-worker-loop",
                )
                self._worker_thread.start()
                if not ready.wait(timeout=10):
                    raise RuntimeError("agentcore-worker-loop failed to start")
        return self._worker_loop

    def _run_worker_loop(self, ready: threading.Event) -> None:
        """Entry point for the worker loop background thread.

        The event loop is created here (inside the worker thread) rather than in
        the parent thread to avoid conflicts with OpenTelemetry's threading
        instrumentation, which propagates context from the parent thread and can
        cause ``RuntimeError: Cannot run the event loop while another loop is
        running``.
        """
        # Clear any running-loop state that leaked from the parent thread
        # (e.g. via OpenTelemetry's threading instrumentation context propagation).
        # Without this, run_forever() raises RuntimeError because
        # asyncio._get_running_loop() still returns the parent's loop.
        asyncio._set_running_loop(None)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._worker_loop = loop
        loop.call_soon(ready.set)
        loop.run_forever()

    @staticmethod
    async def _run_with_context(coro: Any, ctx: contextvars.Context) -> Any:
        """Run a coroutine after restoring context variables from a snapshot."""
        _restore_context(ctx)
        return await coro

    def _async_gen_to_sync_gen(self, async_gen: Any, ctx: contextvars.Context) -> Any:
        """Bridge an async generator through the worker loop as a sync generator.

        The async generator is iterated on the worker loop. Chunks are sent to
        a thread-safe queue and yielded synchronously. Starlette's StreamingResponse
        iterates this sync generator via iterate_in_threadpool, so the main event
        loop is never blocked.

        The producer must never block the shared worker loop, or a full queue
        would freeze it and starve every other session. When the consumer stops
        (SSE client disconnect), the producer task is cancelled so it is torn
        down even while parked awaiting a slow downstream, and a stop event
        keeps a queue-full producer from spinning before the cancel lands.
        """
        worker_loop = self._ensure_worker_loop()
        q: queue.Queue = queue.Queue(maxsize=100)
        _DONE = object()
        stop = threading.Event()
        task_cell: dict[str, asyncio.Task] = {}

        async def _produce() -> None:
            _restore_context(ctx)
            try:
                async for chunk in async_gen:
                    await _offer((True, chunk))
                    if stop.is_set():
                        return
                await _offer((True, _DONE))
            except asyncio.CancelledError:
                # Consumer disconnected; nothing reads the queue anymore. Let the
                # cancellation unwind so the finally below still closes the source.
                raise
            except BaseException as e:
                await _offer((False, e))
            finally:
                # Ensure the source generator releases its resources if the
                # consumer disconnected mid-stream.
                aclose = getattr(async_gen, "aclose", None)
                if aclose is not None:
                    with contextlib.suppress(Exception):
                        await aclose()

        async def _offer(item: Any) -> None:
            """Put an item without ever blocking the worker loop.

            Retries with a short async sleep while the queue is full so the loop
            stays free to run other coroutines. Bails out if the consumer stopped.
            """
            while not stop.is_set():
                try:
                    q.put_nowait(item)
                    return
                except queue.Full:
                    await asyncio.sleep(0.01)

        def _start() -> None:
            task_cell["task"] = worker_loop.create_task(_produce())

        worker_loop.call_soon_threadsafe(_start)

        try:
            while True:
                ok, value = q.get()
                if not ok:
                    raise value
                if value is _DONE:
                    break
                yield value
        finally:
            # Consumer is done or the client disconnected (GeneratorExit). Set the
            # stop flag (unblocks a queue-full producer that hasn't started its
            # await yet) and cancel the task so a producer parked in a mid-chunk
            # await is torn down immediately instead of leaking until it returns.
            stop.set()

            def _cancel() -> None:
                task = task_cell.get("task")
                if task is not None and not task.done():
                    task.cancel()

            worker_loop.call_soon_threadsafe(_cancel)

    async def _invoke_handler(self, handler: Callable, request_context: Any, takes_context: bool, payload: Any) -> Any:
        """Build the handler arguments, ``(payload[, context])``, and call it through ``_dispatch_handler``."""
        args = (payload, request_context) if takes_context else (payload,)
        return await self._dispatch_handler(handler, args)

    async def _dispatch_handler(self, handler: Callable, args: tuple) -> Any:
        """Call ``handler(*args)`` based on the handler type.

        - Async generator functions: bridged through the worker loop as a sync generator
        - Regular async functions: run on the dedicated worker event loop
        - Sync functions (including sync generators): run in the thread pool

        This ensures the main event loop stays responsive for /ping health checks
        regardless of whether handlers contain blocking operations.
        """
        try:
            ctx = contextvars.copy_context()

            if _is_async_gen_callable(handler):
                return self._async_gen_to_sync_gen(handler(*args), ctx)
            elif _is_async_callable(handler):
                worker_loop = self._ensure_worker_loop()
                future = asyncio.run_coroutine_threadsafe(self._run_with_context(handler(*args), ctx), worker_loop)
                result = await asyncio.wrap_future(future)
                if inspect.isasyncgen(result):
                    return self._async_gen_to_sync_gen(result, ctx)
                return result
            else:
                return await run_in_threadpool(ctx.run, handler, *args)
        except Exception:
            handler_name = getattr(handler, "__name__", "unknown")
            self.logger.debug("Handler '%s' execution failed", handler_name)
            raise

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

    def _safe_serialize_to_json_string(self, obj):
        """Safely serialize object directly to JSON string with progressive fallback handling.

        This method eliminates double JSON encoding by returning the JSON string directly,
        avoiding the test-then-encode pattern that leads to redundant json.dumps() calls.
        Used by both streaming and non-streaming responses for consistent behavior.

        Returns:
            str: JSON string representation of the object
        """
        try:
            # First attempt: direct JSON serialization with Unicode support
            return json.dumps(obj, ensure_ascii=False)
        except (TypeError, ValueError, UnicodeEncodeError):
            try:
                # Second attempt: convert to serializable dictionaries, then JSON encode the dictionaries
                converted_obj = convert_complex_objects(obj)
                return json.dumps(converted_obj, ensure_ascii=False)
            except Exception:
                try:
                    # Third attempt: convert to string, then JSON encode the string
                    return json.dumps(str(obj), ensure_ascii=False)
                except Exception as e:
                    # Final fallback: JSON encode error object with ASCII fallback for problematic Unicode
                    self.logger.warning("Failed to serialize object: %s: %s", type(e).__name__, e)
                    error_obj = {"error": "Serialization failed", "original_type": type(obj).__name__}
                    return json.dumps(error_obj, ensure_ascii=False)

    def _convert_to_sse(self, obj) -> bytes:
        """Convert object to Server-Sent Events format using safe serialization.

        Args:
            obj: Object to convert to SSE format

        Returns:
            bytes: SSE-formatted data ready for streaming
        """
        json_string = self._safe_serialize_to_json_string(obj)
        sse_data = f"data: {json_string}\n\n"
        return sse_data.encode("utf-8")

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
