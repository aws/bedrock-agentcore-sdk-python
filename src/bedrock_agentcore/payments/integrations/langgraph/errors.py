"""Error callback types for AgentCorePaymentsMiddleware."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..config import AgentCorePaymentsConfig


class ErrorResolution(Enum):
    """Return value from on_payment_error callback."""

    RETRY = "retry"
    PROPAGATE = "propagate"


@dataclass
class PaymentErrorContext:
    """Context passed to the on_payment_error callback.

    The developer can inspect the exception, mutate `config` to fix the issue
    (e.g., set payment_instrument_id), and return ErrorResolution.RETRY.

    Attributes:
        exception: The exception instance that triggered the callback.
        exception_type: String name of the exception class.
        exception_message: str(exception).
        tool_name: Name of the tool that triggered the 402.
        tool_args: The tool call arguments dict.
        payment_required_request: The 402 payload dict (may be None if error before extraction).
        config: Mutable reference to AgentCorePaymentsConfig.
        retry_count: How many times we've already retried via the callback.
    """

    exception: Exception
    exception_type: str
    exception_message: str
    tool_name: str
    tool_args: Dict[str, Any]
    payment_required_request: Optional[Dict[str, Any]]
    config: "AgentCorePaymentsConfig"
    retry_count: int
