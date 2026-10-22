"""Shared validation for payments SDK inputs."""

import functools
import inspect
import re
from typing import Any, Optional

_PERMIT2_ALLOWANCE_LIMIT_PATTERN = re.compile(r"[0-9]{1,78}\Z")


def validate_permit2_allowance_limit(value: Optional[str]) -> None:
    """Validate a Permit2 allowance against the service model constraints."""
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(
            f"permit2_allowance_limit must be a string in the asset's smallest denomination, got {type(value).__name__}"
        )
    if not _PERMIT2_ALLOWANCE_LIMIT_PATTERN.fullmatch(value) or int(value) <= 0:
        raise ValueError(f"permit2_allowance_limit must be a positive ASCII integer of 1-78 digits, got {value!r}")


def is_async_callback(callback: Any) -> bool:
    """True if the callback ultimately resolves to an async coroutine function.

    Covers the wrappings a plain ``inspect.iscoroutinefunction`` can miss: callable
    objects with an ``async def __call__``, ``functools.partial`` (whose async-ness is
    not reliably visible through the partial across Python versions), and partials of
    either. ``callable()`` cannot distinguish a sync callback from an async one, so it is
    never a safe gate on a money path: an unawaited coroutine object is truthy.
    """
    target = callback
    while isinstance(target, functools.partial):
        target = target.func
    if inspect.iscoroutinefunction(target):
        return True
    call = getattr(target, "__call__", None)  # noqa: B004
    return inspect.iscoroutinefunction(call)


class PreSignPolicyRejected(Exception):
    """Raised when ``pre_sign_policy`` blocks a payment before signing.

    Carries no sensitive detail; callers translate it into the integration's own
    payment error type so the signing path fails closed.
    """


def apply_pre_sign_policy(policy: Optional[Any], payment_required_request: Any) -> None:
    """Enforce ``pre_sign_policy`` at a signing chokepoint. Fail-closed.

    Shared by the Strands plugin and the LangGraph middleware so both paths enforce the
    gate with identical semantics. The policy must return ``True`` exactly to allow
    signing; any other value blocks (so a policy that returns a reason dict, a status
    string, or forgets to return at all blocks rather than silently approving). Raising
    from the policy also blocks.

    Args:
        policy: The configured ``pre_sign_policy`` callback, or ``None`` (no-op).
        payment_required_request: The 402 request handed to the signer (carries ``payTo``
            and amount). Must not be ``None`` — the caller is responsible for never
            invoking signing with an unresolved request.

    Raises:
        PreSignPolicyRejected: If the policy returns anything other than ``True``, raises,
            or the request is ``None``.
    """
    if policy is None:
        return
    if payment_required_request is None:
        raise PreSignPolicyRejected("pre_sign_policy cannot evaluate an unresolved (None) payment request.")
    try:
        allowed = policy(payment_required_request)
    except Exception as exc:  # fail-closed: a policy error blocks signing
        raise PreSignPolicyRejected(f"pre_sign_policy raised {type(exc).__name__}.") from exc
    if allowed is not True:
        raise PreSignPolicyRejected("pre_sign_policy did not return True; blocking before signing.")
