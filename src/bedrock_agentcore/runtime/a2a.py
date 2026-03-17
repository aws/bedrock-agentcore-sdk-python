"""A2A protocol support for Bedrock AgentCore Runtime.

Provides Bedrock-specific glue around the official a2a-sdk, handling header
extraction, health checks, and Docker host detection.
"""

import logging
import time
import uuid
from typing import Any, Callable, Optional

from .context import BedrockAgentCoreContext
from .models import (
    ACCESS_TOKEN_HEADER,
    AGENTCORE_RUNTIME_URL_ENV,
    AUTHORIZATION_HEADER,
    CUSTOM_HEADER_PREFIX,
    OAUTH2_CALLBACK_URL_HEADER,
    REQUEST_ID_HEADER,
    SESSION_HEADER,
    PingStatus,
)

logger = logging.getLogger(__name__)


def _check_a2a_sdk() -> None:
    """Raise ImportError with install instructions if a2a-sdk is missing."""
    try:
        import a2a  # noqa: F401
    except ImportError:
        raise ImportError(
            'a2a-sdk is required for A2A protocol support. Install it with: pip install "bedrock-agentcore[a2a]"'
        ) from None


def _build_agent_card(executor: Any, url: str) -> Any:
    """Build an AgentCard by introspecting a StrandsA2AExecutor.

    Extracts name/description from ``executor.agent``. Falls back to generic
    defaults for other executors.
    """
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    name = "agent"
    description = "A Bedrock AgentCore agent"

    agent = getattr(executor, "agent", None)
    if agent is not None:
        name = getattr(agent, "name", None) or name
        description = getattr(agent, "description", None) or description

    return AgentCard(
        name=name,
        description=description,
        url=url,
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=True),
        skills=[AgentSkill(id="main", name=name, description=description, tags=["main"])],
        default_input_modes=["text"],
        default_output_modes=["text"],
    )


def build_runtime_url(agent_arn: str, region: Optional[str] = None) -> str:
    """Build the Bedrock AgentCore runtime invocation URL from an agent ARN.

    Args:
        agent_arn: The agent runtime ARN, e.g.
            ``arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-agent-abc123``.
        region: AWS region override. If ``None``, extracted from the ARN.

    Returns:
        The full invocation URL with the ARN properly URL-encoded.
    """
    from urllib.parse import quote

    if region is None:
        # ARN format: arn:aws:bedrock-agentcore:<region>:<account>:runtime/<id>
        parts = agent_arn.split(":")
        if len(parts) >= 4:
            region = parts[3]
        else:
            raise ValueError(f"Cannot extract region from ARN: {agent_arn}")

    encoded_arn = quote(agent_arn, safe="")
    return f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{encoded_arn}/invocations"


class BedrockCallContextBuilder:
    """Extracts Bedrock runtime headers and propagates them into BedrockAgentCoreContext.

    Implements the a2a-sdk CallContextBuilder ABC so the A2A server
    automatically calls ``build()`` on every incoming request.
    """

    def build(self, request: Any) -> Any:
        """Build a ServerCallContext from a Starlette Request.

        Args:
            request: A Starlette Request object.

        Returns:
            A ServerCallContext with Bedrock headers stored in ``state``.
        """
        from a2a.server.context import ServerCallContext

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

        state = {
            "request_id": request_id,
            "session_id": session_id,
        }
        if workload_access_token:
            state["workload_access_token"] = workload_access_token
        if oauth2_callback_url:
            state["oauth2_callback_url"] = oauth2_callback_url

        return ServerCallContext(state=state)


# Register as a virtual subclass so isinstance checks pass without
# requiring a2a-sdk to be importable at class-definition time.
try:
    from a2a.server.apps import CallContextBuilder

    CallContextBuilder.register(BedrockCallContextBuilder)
except Exception:  # pragma: no cover
    pass


def build_a2a_app(
    executor: Any,
    agent_card: Any = None,
    *,
    task_store: Any = None,
    context_builder: Any = None,
    ping_handler: Optional[Callable[[], PingStatus]] = None,
) -> Any:
    """Build a Starlette app wired for A2A protocol with Bedrock extras.

    Args:
        executor: An ``AgentExecutor`` that implements the agent logic.
        agent_card: Optional ``a2a.types.AgentCard`` describing the agent.
            If ``None``, one is built automatically by introspecting the executor.
        task_store: Optional ``TaskStore``; defaults to ``InMemoryTaskStore``.
        context_builder: Optional ``CallContextBuilder``; defaults to
            ``BedrockCallContextBuilder``.
        ping_handler: Optional callback returning a ``PingStatus``.

    Returns:
        A Starlette application.
    """
    import os

    _check_a2a_sdk()

    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    runtime_url = os.environ.get(AGENTCORE_RUNTIME_URL_ENV, "http://localhost:9000/")

    if agent_card is None:
        agent_card = _build_agent_card(executor, runtime_url)
    elif os.environ.get(AGENTCORE_RUNTIME_URL_ENV):
        agent_card.url = runtime_url

    if task_store is None:
        task_store = InMemoryTaskStore()
    if context_builder is None:
        context_builder = BedrockCallContextBuilder()

    http_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=http_handler,
        context_builder=context_builder,
    )

    last_status_update_time = time.time()

    def _handle_ping(request: Any) -> JSONResponse:
        nonlocal last_status_update_time
        try:
            if ping_handler is not None:
                status = ping_handler()
            else:
                status = PingStatus.HEALTHY
            last_status_update_time = time.time()
        except Exception:
            logger.exception("Custom ping handler failed, falling back to Healthy")
            status = PingStatus.HEALTHY
        return JSONResponse({"status": status.value, "time_of_last_update": int(last_status_update_time)})

    # Build the Starlette app with /ping included upfront, then add A2A routes,
    # so we don't depend on mutating app.routes after build().
    app = Starlette(routes=[Route("/ping", _handle_ping, methods=["GET"])])
    a2a_app.add_routes_to_app(app)

    return app


def serve_a2a(
    executor: Any,
    agent_card: Any = None,
    *,
    port: int = 9000,
    host: Optional[str] = None,
    task_store: Any = None,
    context_builder: Any = None,
    ping_handler: Optional[Callable[[], PingStatus]] = None,
    **kwargs: Any,
) -> None:
    """Start a Bedrock-compatible A2A server.

    Args:
        executor: An ``AgentExecutor`` that implements the agent logic.
        agent_card: Optional ``a2a.types.AgentCard`` describing the agent.
            If ``None``, one is built automatically by introspecting the executor.
        port: Port to serve on (default 9000).
        host: Host to bind to; auto-detected if ``None``.
        task_store: Optional ``TaskStore``; defaults to ``InMemoryTaskStore``.
        context_builder: Optional ``CallContextBuilder``; defaults to
            ``BedrockCallContextBuilder``.
        ping_handler: Optional callback returning a ``PingStatus``.
        **kwargs: Additional arguments forwarded to ``uvicorn.run()``.
    """
    import os

    import uvicorn

    app = build_a2a_app(
        executor,
        agent_card,
        task_store=task_store,
        context_builder=context_builder,
        ping_handler=ping_handler,
    )

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

    uvicorn.run(app, **uvicorn_params)
