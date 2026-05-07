"""Configuration for AgentCorePaymentsPlugin."""

from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class AgentCorePaymentsPluginConfig:
    """Configuration for AgentCorePaymentsPlugin.

    Attributes:
        payment_manager_arn: ARN of the payment manager service
        region: AWS region for the payment manager
        user_id: User ID for payment processing. Required for SigV4 auth.
            Optional for bearer token auth (JWT identifies the user).
            When set with bearer auth, propagated via X-Amzn-Bedrock-AgentCore-Payments-User-Id header.
        payment_instrument_id: Optional payment instrument ID for the user.
            Can be set later via update_payment_instrument_id().
        payment_session_id: Optional payment session ID for the transaction.
            Can be set later via update_payment_session_id().
        network_preferences_config: Optional list of network CAIP2 identifiers
            in order of preference. If not provided, defaults to the system default.
        auto_payment: Whether to automatically process 402 payment requirements.
            Defaults to True to maintain existing behavior.
        max_interrupt_retries: Maximum number of interrupt retries per tool use.
            Defaults to 5. Set to 0 to disable interrupt retries entirely (no interrupts will be raised).
        agent_name: Optional agent name to propagate via the
            X-Amzn-Bedrock-AgentCore-Payments-Agent-Name HTTP header on every
            AgentCore payments data-plane API call. When set, the header is automatically injected
            by PaymentManager and propagated for Payments.
        bearer_token: Optional static JWT bearer token for OAuth/CUSTOM_JWT authentication.
            When set, PaymentManager uses Bearer token auth instead of SigV4.
            Mutually exclusive with token_provider.
        token_provider: Optional callable that returns a fresh JWT bearer token string.
            Called before each request to support token refresh.
            Mutually exclusive with bearer_token.
        payment_tool_allowlist: Optional list of tool names that are eligible for
            automatic X402 payment processing. When None (default), all tools are
            eligible (preserving existing behavior). When set, only tool calls whose
            name appears in this list will trigger payment processing; all others are
            skipped.
    """

    payment_manager_arn: str
    user_id: Optional[str] = None
    payment_instrument_id: Optional[str] = None
    payment_session_id: Optional[str] = None
    payment_connector_id: Optional[str] = None
    region: Optional[str] = None
    network_preferences_config: Optional[list[str]] = None
    auto_payment: bool = True
    max_interrupt_retries: int = 5
    agent_name: Optional[str] = None
    bearer_token: Optional[str] = None
    token_provider: Optional[Callable[[], str]] = None
    payment_tool_allowlist: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.payment_manager_arn:
            raise ValueError("payment_manager_arn is required")

        if not self.payment_manager_arn.startswith("arn:"):
            raise ValueError(f"Invalid ARN format: {self.payment_manager_arn}")

        if self.bearer_token is not None and not isinstance(self.bearer_token, str):
            raise ValueError(f"bearer_token must be a string, got {type(self.bearer_token).__name__}")

        if self.token_provider is not None and not callable(self.token_provider):
            raise ValueError(f"token_provider must be callable, got {type(self.token_provider).__name__}")

        if self.user_id is not None and self.user_id and not self.user_id.strip():
            raise ValueError("user_id cannot be whitespace-only")

        if not self.user_id and self.bearer_token is None and self.token_provider is None:
            raise ValueError("user_id is required for SigV4 auth (when bearer_token/token_provider not set)")

        if not isinstance(self.auto_payment, bool):
            raise ValueError(f"auto_payment must be a boolean, got {type(self.auto_payment).__name__}")

        if self.bearer_token is not None and self.token_provider is not None:
            raise ValueError("bearer_token and token_provider are mutually exclusive. Provide only one.")

        if self.payment_tool_allowlist is not None:
            if not isinstance(self.payment_tool_allowlist, list):
                raise ValueError("payment_tool_allowlist must be a list of tool name strings")
            if not all(isinstance(t, str) for t in self.payment_tool_allowlist):
                raise ValueError("All entries in payment_tool_allowlist must be strings")

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
