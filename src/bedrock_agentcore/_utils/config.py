"""Shared configuration dataclasses for SDK clients."""

from dataclasses import dataclass


@dataclass
class WaitConfig:
    """Configuration for *_and_wait polling methods.

    Args:
        max_wait: Maximum seconds to wait. Default: 300. Must be >= 1.
        poll_interval: Seconds between status checks. Default: 10. Must be >= 1.
    """

    max_wait: int = 300
    poll_interval: int = 10

    def __post_init__(self):
        if self.max_wait < 1:
            raise ValueError("max_wait must be at least 1")
        if self.poll_interval < 1:
            raise ValueError("poll_interval must be at least 1")
