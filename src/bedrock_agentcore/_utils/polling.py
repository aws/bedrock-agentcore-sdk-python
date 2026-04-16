"""Shared polling helpers for SDK clients."""

import logging
import time
from typing import Any, Callable, Dict, Optional, Set

from .config import WaitConfig

logger = logging.getLogger(__name__)


def wait_until(
    poll_fn: Callable[[], Dict[str, Any]],
    target: str,
    failed: Set[str],
    wait_config: Optional[WaitConfig] = None,
    error_field: str = "statusReasons",
) -> Dict[str, Any]:
    """Poll until a resource reaches the target status.

    Args:
        poll_fn: Zero-arg callable that returns the resource's current state.
        target: The status to wait for (e.g. "ACTIVE", "READY").
        failed: Statuses that indicate terminal failure.
        wait_config: Optional WaitConfig for polling behavior.
        error_field: Response field containing error details.

    Returns:
        Full response when target status is reached.

    Raises:
        RuntimeError: If the resource reaches a failed status.
        TimeoutError: If target status is not reached within max_wait.
    """
    wait = wait_config or WaitConfig()
    start_time = time.time()
    while time.time() - start_time < wait.max_wait:
        resp = poll_fn()
        status = resp.get("status")
        if status == target:
            return resp
        if status in failed:
            reason = resp.get(error_field, "Unknown")
            raise RuntimeError("Reached %s: %s" % (status, reason))
        time.sleep(wait.poll_interval)
    raise TimeoutError("Did not reach %s within %d seconds" % (target, wait.max_wait))
