"""Configuration for AgentCore Payments integrations (Strands and LangGraph)."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .handlers import PaymentResponseHandler


@dataclass
class AgentCorePaymentsPluginConfig:
    """Configuration for AgentCore Payments integrations.

    This unified config is used by both the Strands plugin and LangGraph middleware.

    Attributes:
        payment_manager_arn: ARN of the payment manager service.
        user_id: User ID for payment processing. Required for SigV4 auth.
            Optional for bearer token auth (JWT identifies the user).
            When set with bearer auth, propagated via X-Amzn-Bedrock-AgentCore-Payments-User-Id header.
        payment_instrument_id: Optional payment instrument ID for the user.
            Can be set later via update_payment_instrument_id().
        payment_session_id: Optional payment session ID for the transaction.
            Can be set later via update_payment_session_id().
        payment_connector_id: Payment connector ID (optional).
        region: AWS region for the payment manager.
        network_preferences_config: Ordered list of network CAIP2 identifiers.
        auto_payment: Whether to automatically process 402 responses. Default True.
        agent_name: Agent name propagated via HTTP header on data-plane calls.
        bearer_token: Static JWT for OAuth/CUSTOM_JWT auth. Mutually exclusive with token_provider.
        token_provider: Callable returning a fresh JWT. Mutually exclusive with bearer_token.
        payment_tool_allowlist: Tool names eligible for payment processing. None = all tools.
        provide_http_request: Whether the integration registers its built-in http_request tool.
        post_payment_retry_delay_seconds: Delay after signing before retry. Default 3.0s.
        max_interrupt_retries: Maximum number of interrupt retries per tool use (Strands only).
            Defaults to 5. Set to 0 to disable interrupt retries entirely.
        custom_handlers: Custom PaymentResponseHandler instances keyed by tool name.
            Takes precedence over the built-in handler registry during resolution.
            A custom handler's extract_* methods receive the raw ToolMessage.content
            (a str or a list of content blocks), not the middleware's internal wrapped
            shape. The built-in handlers (GenericPaymentHandler, HttpRequestPaymentHandler,
            MCPRequestPaymentHandler) expect a different, normalized shape, so passing one
            of them directly as a custom handler will not detect 402s; subclass
            PaymentResponseHandler (or wrap a built-in) and parse the raw content instead.
        auto_session: Whether to auto-create a payment session on first 402 if
            payment_session_id is not set. Default False.
        auto_session_budget: Budget for auto-created sessions (USD). Default "1.00".
        auto_session_expiry_minutes: Expiry time for auto-created sessions. Default 60.
        on_payment_error: Optional callback invoked when a payment exception occurs.
            Receives PaymentErrorContext, returns ErrorResolution.RETRY or .PROPAGATE.
            When None (default), errors produce deterministic ToolMessages directly.
        max_error_retries: Maximum times the error callback can return RETRY per tool call.
            Default 3. Set to 0 to disable the callback entirely.
    """

    payment_manager_arn: str
    user_id: Optional[str] = None
    payment_instrument_id: Optional[str] = None
    payment_session_id: Optional[str] = None
    payment_connector_id: Optional[str] = None
    region: Optional[str] = None
    network_preferences_config: Optional[List[str]] = None
    auto_payment: bool = True
    agent_name: Optional[str] = None
    bearer_token: Optional[str] = None
    token_provider: Optional[Callable[[], str]] = None
    payment_tool_allowlist: Optional[List[str]] = None
    provide_http_request: bool = True
    post_payment_retry_delay_seconds: float = 3.0
    max_interrupt_retries: int = 5
    custom_handlers: Optional[Dict[str, Any]] = field(default=None)
    auto_session: bool = False
    auto_session_budget: str = "1.00"
    auto_session_expiry_minutes: int = 60
    on_payment_error: Optional[Callable] = None
    max_error_retries: int = 3

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.payment_manager_arn:
            raise ValueError("payment_manager_arn is required")
        if not self.payment_manager_arn.startswith("arn:"):
            raise ValueError(f"Invalid ARN format: {self.payment_manager_arn}")

        if self.bearer_token is not None and self.token_provider is not None:
            raise ValueError("bearer_token and token_provider are mutually exclusive")
        if self.bearer_token is not None and not isinstance(self.bearer_token, str):
            raise ValueError(f"bearer_token must be a string, got {type(self.bearer_token).__name__}")
        if self.token_provider is not None and not callable(self.token_provider):
            raise ValueError(f"token_provider must be callable, got {type(self.token_provider).__name__}")

        if not self.user_id and self.bearer_token is None and self.token_provider is None:
            raise ValueError("user_id is required for SigV4 auth (when bearer_token/token_provider not set)")
        if self.user_id is not None and self.user_id and not self.user_id.strip():
            raise ValueError("user_id cannot be whitespace-only")

        if not isinstance(self.auto_payment, bool):
            raise ValueError(f"auto_payment must be a boolean, got {type(self.auto_payment).__name__}")
        if not isinstance(self.provide_http_request, bool):
            raise ValueError(f"provide_http_request must be a boolean, got {type(self.provide_http_request).__name__}")

        if self.payment_tool_allowlist is not None:
            if not isinstance(self.payment_tool_allowlist, list):
                raise ValueError("payment_tool_allowlist must be a list of tool name strings")
            if not all(isinstance(t, str) for t in self.payment_tool_allowlist):
                raise ValueError("All entries in payment_tool_allowlist must be strings")

        if not isinstance(self.post_payment_retry_delay_seconds, (int, float)) or isinstance(
            self.post_payment_retry_delay_seconds, bool
        ):
            raise ValueError(
                "post_payment_retry_delay_seconds must be a number, got "
                f"{type(self.post_payment_retry_delay_seconds).__name__}"
            )
        if self.post_payment_retry_delay_seconds < 0:
            raise ValueError(
                f"post_payment_retry_delay_seconds must be >= 0, got {self.post_payment_retry_delay_seconds}"
            )

        if self.custom_handlers is not None:
            if not isinstance(self.custom_handlers, dict):
                raise ValueError(
                    "custom_handlers must be a dict mapping tool names to PaymentResponseHandler instances"
                )
            if not all(isinstance(k, str) for k in self.custom_handlers):
                raise ValueError("All keys in custom_handlers must be strings")
            if not all(isinstance(v, PaymentResponseHandler) for v in self.custom_handlers.values()):
                raise ValueError("All values in custom_handlers must be PaymentResponseHandler instances")

        if self.on_payment_error is not None and not callable(self.on_payment_error):
            raise ValueError(f"on_payment_error must be callable, got {type(self.on_payment_error).__name__}")

        if not isinstance(self.max_error_retries, int) or isinstance(self.max_error_retries, bool):
            raise ValueError(f"max_error_retries must be an int, got {type(self.max_error_retries).__name__}")
        if self.max_error_retries < 0:
            raise ValueError(f"max_error_retries must be >= 0, got {self.max_error_retries}")

    def update_payment_session_id(self, payment_session_id: str) -> None:
        """Update the payment session ID.

        Args:
            payment_session_id: New payment session ID for the transaction.
        """
        if not payment_session_id:
            raise ValueError("payment_session_id cannot be empty")
        self.payment_session_id = payment_session_id

    def update_payment_instrument_id(self, payment_instrument_id: str) -> None:
        """Update the payment instrument ID.

        Args:
            payment_instrument_id: New payment instrument ID for the user.
        """
        if not payment_instrument_id:
            raise ValueError("payment_instrument_id cannot be empty")
        self.payment_instrument_id = payment_instrument_id

    def add_to_allowlist(self, *tool_names: str) -> None:
        """Add tool names to the payment allowlist.

        Creates the allowlist if it doesn't exist yet (switching from "all tools"
        to explicit allowlist mode).

        Args:
            tool_names: One or more tool names to add.
        """
        if self.payment_tool_allowlist is None:
            self.payment_tool_allowlist = []
        for name in tool_names:
            if not isinstance(name, str):
                raise ValueError(f"Tool name must be a string, got {type(name).__name__}")
            if name not in self.payment_tool_allowlist:
                self.payment_tool_allowlist.append(name)

    def remove_from_allowlist(self, *tool_names: str) -> None:
        """Remove tool names from the payment allowlist.

        If the allowlist becomes empty, sets it to None (all tools eligible).

        Args:
            tool_names: One or more tool names to remove.
        """
        if self.payment_tool_allowlist is None:
            return
        for name in tool_names:
            if name in self.payment_tool_allowlist:
                self.payment_tool_allowlist.remove(name)
        if not self.payment_tool_allowlist:
            self.payment_tool_allowlist = None


# Backward-compatible alias for LangGraph imports
AgentCorePaymentsConfig = AgentCorePaymentsPluginConfig
