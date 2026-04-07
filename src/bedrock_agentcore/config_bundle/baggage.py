"""Parse W3C baggage headers for configuration bundle references."""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from .bundle import ConfigBundleRef

BAGGAGE_HEADER = "baggage"

logger = logging.getLogger(__name__)


def _extract_baggage(headers: Any) -> Dict[str, List[str]]:
    """Extract all W3C baggage entries from request headers into a multi-value dict.

    Args:
        headers: A Starlette ``Headers`` object or a list of ``(name, value)``
            tuples.  Must preserve duplicate header names as separate entries so
            that multiple ``baggage`` headers are each processed independently.
            A plain ``dict`` is not suitable — it can only hold one ``baggage``
            entry and will silently drop the rest.

    Returns:
        A dict mapping each baggage key to a list of its decoded values in the
        order they were encountered.  A key that appears in more than one
        ``baggage`` header, or more than once within a single header value,
        accumulates one entry per occurrence.

    Notes:
        - Header name matching is case-insensitive (``Baggage`` == ``baggage``).
        - Per-entry properties (the ``;property=value`` suffix) are stripped
          before the value is returned.
        - Values are percent-decoded (``%XX`` → character).
        - Entries with no ``=`` sign, an empty key, or an empty value are skipped.
    """
    result: Dict[str, List[str]] = {}
    items = headers.items() if hasattr(headers, "items") else headers
    for key, value in items:
        if key.lower() != BAGGAGE_HEADER:
            continue
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                logger.warning("Skipping malformed baggage entry (no '='): %r", item)
                continue
            entry_key, _, entry_value = item.partition("=")
            entry_key = entry_key.strip()
            if not entry_key:
                logger.warning("Skipping baggage entry with empty key in: %r", item)
                continue
            decoded_value = unquote(entry_value.split(";")[0].strip())
            if not decoded_value:
                logger.warning("Skipping baggage entry with empty value for key %r", entry_key)
                continue
            result.setdefault(entry_key, []).append(decoded_value)
    return result


def _parse_config_bundle_baggage(all_baggage: Dict[str, List[str]]) -> Optional[ConfigBundleRef]:
    """Build a ``ConfigBundleRef`` from extracted baggage entries, or ``None`` if absent.

    Expects ``all_baggage`` to have been produced by :func:`_extract_baggage`.
    The two keys used are:

    - ``aws.agentcore.configbundle_arn`` — full ARN of the configuration bundle
    - ``aws.agentcore.configbundle_version`` — version ID of the bundle

    Only a single bundle is supported::

        baggage: aws.agentcore.configbundle_arn=<arn>,aws.agentcore.configbundle_version=<version>

    If multiple values are present for either key, only the first is used and a
    warning is logged.

    Args:
        all_baggage: Multi-value baggage dict from :func:`_extract_baggage`.

    Returns:
        A ``ConfigBundleRef`` when both keys are present and valid, otherwise ``None``.
    """
    arns = all_baggage.get("aws.agentcore.configbundle_arn", [])
    versions = all_baggage.get("aws.agentcore.configbundle_version", [])

    if not arns or not versions:
        return None

    if len(arns) > 1 or len(versions) > 1:
        logger.warning("Multiple config bundle ARNs/versions found in baggage — only the first will be used")

    try:
        return ConfigBundleRef(bundle_arn=arns[0], bundle_version=versions[0])
    except ValueError as e:
        logger.warning("Skipping invalid config bundle ref (arn=%r, version=%r): %s", arns[0], versions[0], e)
        return None
