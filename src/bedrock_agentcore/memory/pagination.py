"""Pagination utilities for AgentCore Memory SDK."""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def paginate_turns(
    list_events_fn: Callable[..., Dict[str, Any]],
    base_params: Dict[str, Any],
    k: int,
    max_results: Optional[int],
    user_role_value: str,
    wrap_message: Callable[[Dict[str, Any]], Any],
) -> List[List[Any]]:
    """Paginate through events to collect k conversation turns.

    Args:
        list_events_fn: Function to call for listing events (boto3 client method)
        base_params: Base parameters for the list_events call (memoryId, actorId, sessionId)
        k: Number of turns to collect
        max_results: Maximum events to fetch total (None for auto-pagination until k turns found)
        user_role_value: The string value representing USER role (e.g., "USER")
        wrap_message: Function to wrap each message (e.g., identity or EventMessage constructor)

    Returns:
        List of turns, where each turn is a list of messages
    """
    turns: List[List[Any]] = []
    current_turn: List[Any] = []
    next_token = None
    total_fetched = 0

    while len(turns) < k:
        if max_results is not None:
            remaining = max_results - total_fetched
            if remaining <= 0:
                break
            batch_size = min(100, remaining)
        else:
            batch_size = 100

        params = {**base_params, "maxResults": batch_size, "includePayloads": True}
        if next_token:
            params["nextToken"] = next_token

        response = list_events_fn(**params)
        events = response.get("events", [])

        if not events:
            break

        total_fetched += len(events)

        for event in events:
            if len(turns) >= k:
                break
            for payload_item in event.get("payload", []):
                if "conversational" in payload_item:
                    role = payload_item["conversational"].get("role")

                    if role == user_role_value and current_turn:
                        turns.append(current_turn)
                        current_turn = []

                    current_turn.append(wrap_message(payload_item["conversational"]))

        next_token = response.get("nextToken")
        if not next_token:
            break

    # Don't forget the last turn
    if current_turn and len(turns) < k:
        turns.append(current_turn)

    return turns[:k]
