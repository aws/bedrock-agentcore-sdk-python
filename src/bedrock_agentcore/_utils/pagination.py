"""Reusable pagination utility for fetching and converting paginated results."""

from typing import Any, Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_PAGE_SIZE = 100


def paginate_for_n_results(
    fetch_page: Callable[[dict[str, Any]], tuple[list[R], str | None]],
    initial_params: dict[str, Any],
    converter: Callable[[list[R]], list[T]],
    target_count: int,
) -> list[T]:
    """Paginate an arbitrary API, converting accumulated results after each page.

    The full accumulated list is re-converted after each page rather than converting
    per-page, because some converters (e.g. events_to_messages) iterate the input in
    reverse — converting per-page would produce incorrect ordering.

    Args:
        fetch_page: Takes params dict, returns (items, next_token). next_token is None when exhausted.
        initial_params: Base params for the first call. "nextToken" is added for subsequent pages.
        converter: Converts accumulated raw items to desired output type.
        target_count: Stop after collecting this many converted items.
    """
    all_items: list[R] = []
    next_token: str | None = None

    while True:
        params = {**initial_params}
        if next_token is not None:
            params["nextToken"] = next_token

        items, next_token = fetch_page(params)
        all_items.extend(items)

        converted = converter(all_items)
        if len(converted) >= target_count or next_token is None:
            return converted[:target_count]
