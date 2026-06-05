"""Reconnection configuration for ShellSession."""

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Union

_DEFAULT_MAX_RETRIES = 5
_DEFAULT_BASE_DELAY = 1.0
_DEFAULT_MAX_DELAY = 15.0
_DEFAULT_METADATA_TIMEOUT = 10.0
_DEFAULT_RECONNECT_WINDOW = 900.0  # ~15 min — matches server-side KARP idle timeout
_DEFAULT_OUTER_LOOP_DELAY = 30.0  # wait between inner loop exhaustion and next outer attempt


@dataclass
class ReconnectConfig:
    """Configuration for automatic reconnection on WebSocket disconnect.

    When provided to ``open_shell``, ``ShellSession`` will automatically
    reconnect using the same ``shell_id`` after an unexpected
    disconnect.  The shell process on the VM keeps running while detached, and
    up to 256 KB of buffered output is replayed when the new WebSocket attaches.

    Attributes:
        max_retries: Maximum number of reconnect attempts per inner loop before
            pausing and starting a fresh inner loop. The inner
            loop is bounded at 5.  Use ``reconnect_window=None`` for unlimited
            overall retries across outer loop cycles.
        base_delay: Initial backoff delay in seconds.  Doubles on each attempt
            up to ``max_delay``.
        max_delay: Upper bound on backoff delay in seconds. The inner loop
            caps at 15s (sequence: 1s, 2s, 4s, 8s, 15s).
        reconnect_window: Total seconds to keep retrying after a disconnect
            before giving up entirely.  Matches the server-side ~15 min
            reconnection window (KARP idle timeout).  Set to ``None`` to retry
            indefinitely.
        outer_loop_delay: Seconds to wait between inner loop exhaustion and the
            next outer retry cycle.
        on_reconnect: Optional async or sync callback invoked after each
            successful reconnect.  Receives ``reconnected: bool`` — ``True``
            means the existing PTY was reattached (buffered output will follow
            as STDOUT frames); ``False`` means a fresh shell was started.

    Example — log reconnects and flush buffered output to a file:
        async def on_reconnect(reconnected: bool) -> None:
            if reconnected:
                print("Reattached to existing PTY — replaying buffered output")
            else:
                print("New shell started")

        config = ReconnectConfig(reconnect_window=None, on_reconnect=on_reconnect)  # None = unlimited
        async with client.open_shell(arn, reconnect_config=config) as shell:
            async for frame in shell:
                ...
    """

    max_retries: int = _DEFAULT_MAX_RETRIES
    base_delay: float = _DEFAULT_BASE_DELAY
    max_delay: float = _DEFAULT_MAX_DELAY
    reconnect_window: Optional[float] = _DEFAULT_RECONNECT_WINDOW
    outer_loop_delay: float = _DEFAULT_OUTER_LOOP_DELAY
    on_reconnect: Optional[Callable[[bool], Union[Awaitable[None], None]]] = field(default=None, repr=False)
