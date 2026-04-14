"""Shared configuration dataclasses for SDK clients."""

from dataclasses import dataclass


@dataclass
class ListConfig:
    """Configuration for list_all_* auto-paginating methods.

    Args:
        total_items: Maximum number of items to return. Default: 100.
    """

    total_items: int = 100


@dataclass
class WaitConfig:
    """Configuration for *_and_wait polling methods.

    Args:
        max_wait: Maximum seconds to wait. Default: 300.
        poll_interval: Seconds between status checks. Default: 10.
    """

    max_wait: int = 300
    poll_interval: int = 10
