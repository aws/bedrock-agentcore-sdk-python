"""Client for fetching configuration bundle versions from the AgentCore control plane."""

import logging
import threading
from typing import Optional

import boto3

from .._utils.endpoints import DEFAULT_REGION, get_control_plane_endpoint

logger = logging.getLogger(__name__)

_ALLOWED_OPERATIONS = frozenset({
    "create_configuration_bundle",
    "delete_configuration_bundle",
    "get_configuration_bundle",
    "get_configuration_bundle_version",
    "list_configuration_bundle_versions",
    "list_configuration_bundles",
    "update_configuration_bundle",
})


class ConfigBundleClient:
    """Client for AgentCore configuration bundle operations.

    Wraps the ``bedrock-agentcore-control`` boto3 client and forwards all method
    calls to it via ``__getattr__``, so any boto3 operation (e.g.
    ``get_configuration_bundle_version``, ``list_configuration_bundles``) is
    available without explicit definitions.

    Intended to be created once at application startup and reused across requests.
    The underlying boto3 client is created lazily on first use so that agents
    which never receive config bundle baggage incur no startup overhead.
    """

    def __init__(self, region_name: Optional[str] = None, boto3_session: Optional[boto3.Session] = None):
        """Initialise the client with an optional region and boto3 session."""
        self._region = region_name or DEFAULT_REGION
        self._boto3_session = boto3_session
        self._client = None
        self._client_lock = threading.Lock()

    def _get_client(self):
        # Use __dict__ directly to avoid triggering __getattr__ if _client is
        # not yet set (e.g. during unpickling before __init__ completes).
        if self.__dict__.get("_client") is None:
            with self._client_lock:
                if self.__dict__.get("_client") is None:
                    session = self._boto3_session or boto3.Session()
                    self._client = session.client(
                        "bedrock-agentcore-control",
                        region_name=self._region,
                        endpoint_url=get_control_plane_endpoint(self._region),
                    )
        return self._client

    def __getattr__(self, name: str):
        """Forward configuration bundle method calls to the underlying boto3 client.

        Only operations in ``_ALLOWED_OPERATIONS`` are exposed. Attempts to call
        any other operation raise ``AttributeError``.

        Uses ``object.__getattribute__`` to access ``_get_client`` so that if Python
        looks up dunder attributes during unpickling or deepcopy before instance
        attributes are initialised, this method does not recurse into itself.
        """
        if name not in _ALLOWED_OPERATIONS:
            raise AttributeError(f"'{type(self).__name__}' does not expose operation '{name}'")
        return getattr(object.__getattribute__(self, "_get_client")(), name)
