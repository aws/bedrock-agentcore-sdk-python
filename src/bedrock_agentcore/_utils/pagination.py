"""Shared pagination helpers for SDK clients."""

from typing import Any, Dict, List, Optional

from .config import ListConfig


def list_all(
    client: Any,
    method_name: str,
    result_key: str,
    list_config: Optional[ListConfig] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Generic auto-paginating list helper.

    Calls the specified boto3 list method, follows nextToken pagination,
    and caps results at list_config.total_items.

    Args:
        client: The boto3 client to call the method on.
        method_name: Name of the boto3 list method to call.
        result_key: Key in the response containing the items list.
        list_config: Optional ListConfig to control max items returned (default: 100).
        **kwargs: Additional arguments forwarded to the boto3 call.

    Returns:
        List of items up to list_config.total_items.
    """
    config = list_config or ListConfig()
    all_items: List[Dict[str, Any]] = []
    kwargs.pop("nextToken", None)
    method = getattr(client, method_name)
    while len(all_items) < config.total_items:
        response = method(**kwargs)
        all_items.extend(response.get(result_key, []))
        if not response.get("nextToken"):
            break
        kwargs["nextToken"] = response["nextToken"]
    return all_items[: config.total_items]
