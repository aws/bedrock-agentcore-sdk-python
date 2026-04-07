"""Client for fetching configuration bundle versions from the AgentCore control plane."""

import logging
import os
import threading
from typing import Optional

import boto3

from .._utils.endpoints import DEFAULT_REGION, get_control_plane_endpoint

logger = logging.getLogger(__name__)

# Path to the bundled service model that extends the installed botocore model
# with configuration bundle operations.
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _boto3_session_with_model(boto3_session: Optional[boto3.Session]) -> boto3.Session:
    """Return a boto3 Session whose botocore loader also searches our bundled data dir.

    TODO: This is a temporary workaround needed to run in the AgentCore runtime until
    the bedrock-agentcore-control service model ships in an official boto3/botocore
    release. Once boto3 includes the model natively this function and the bundled
    data directory can be removed.
    """
    if boto3_session is None:
        boto3_session = boto3.Session()
    core_session = boto3_session._session  # botocore.session.Session
    loader = core_session.get_component("data_loader")
    # Prepend our data dir so our model takes precedence over the installed one.
    # Guard against inserting twice if multiple ConfigBundleClient instances are created.
    if _DATA_DIR not in loader.search_paths:
        loader.search_paths.insert(0, _DATA_DIR)
    return boto3_session


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
                    session = _boto3_session_with_model(self._boto3_session)
                    self._client = session.client(
                        "bedrock-agentcore-control",
                        region_name=self._region,
                        endpoint_url=get_control_plane_endpoint(self._region),
                    )
        return self._client

    def __getattr__(self, name: str):
        """Forward method calls to the underlying boto3 bedrock-agentcore-control client.

        Enables access to any boto3 method (e.g. ``create_configuration_bundle``,
        ``list_configuration_bundles``) without explicitly defining them here.

        Uses ``object.__getattribute__`` to access ``_get_client`` so that if Python
        looks up dunder attributes during unpickling or deepcopy before instance
        attributes are initialised, this method does not recurse into itself.
        """
        return getattr(object.__getattribute__(self, "_get_client")(), name)
