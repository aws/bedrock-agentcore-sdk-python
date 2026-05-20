"""DatasetClient for managing evaluation datasets."""

import logging
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config

from bedrock_agentcore._utils.config import WaitConfig
from bedrock_agentcore._utils.polling import wait_until, wait_until_deleted
from bedrock_agentcore._utils.snake_case import accept_snake_case_kwargs, convert_kwargs
from bedrock_agentcore._utils.user_agent import build_user_agent_suffix

logger = logging.getLogger(__name__)


class DatasetClient:
    """Client for managing evaluation datasets.

    Provides pass-through access to all dataset management APIs on the
    bedrock-agentcore-control service, plus *_and_wait helpers for async operations.

    Example::

        client = DatasetClient(region_name="us-west-2")

        # Create a dataset and wait for ACTIVE
        dataset = client.create_dataset_and_wait(
            datasetName="my-dataset",
            schemaType="AGENTCORE_EVALUATION_PREDEFINED_V1",
            source={"inlineExamples": {"examples": [...]}},
        )

        # Pass-through to any dataset API
        client.list_datasets(maxResults=10)
        client.add_dataset_examples(datasetId="ds-123", examples=[...])
    """

    _ALLOWED_CP_METHODS = {
        # Dataset CRUD
        "create_dataset",
        "get_dataset",
        "list_datasets",
        "update_dataset",
        "delete_dataset",
        # Version management
        "create_dataset_version",
        "list_dataset_versions",
        # Examples management
        "add_dataset_examples",
        "update_dataset_examples",
        "delete_dataset_examples",
        "list_dataset_examples",
    }

    def __init__(
        self,
        region_name: Optional[str] = None,
        integration_source: Optional[str] = None,
        boto3_session: Optional[boto3.Session] = None,
    ):
        """Initialize the DatasetClient.

        Args:
            region_name: AWS region. Falls back to boto3 session region or us-west-2.
            integration_source: Optional integration framework identifier for telemetry.
            boto3_session: Optional boto3 Session. If not provided, a default is created.
        """
        session = boto3_session if boto3_session else boto3.Session()
        self.region_name = region_name or session.region_name or "us-west-2"
        self.integration_source = integration_source

        user_agent_extra = build_user_agent_suffix(integration_source)
        client_config = Config(user_agent_extra=user_agent_extra)

        self._cp_client = session.client(
            "bedrock-agentcore-control",
            region_name=self.region_name,
            config=client_config,
        )

        logger.info("Initialized DatasetClient in region %s", self.region_name)

    def __getattr__(self, name: str):
        """Dynamically forward allowlisted method calls to the boto3 client."""
        if "_cp_client" not in self.__dict__:
            raise AttributeError(name)

        if name in self._ALLOWED_CP_METHODS and hasattr(self._cp_client, name):
            method = getattr(self._cp_client, name)
            logger.debug("Forwarding method '%s' to _cp_client", name)
            return accept_snake_case_kwargs(method)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
            f"Method not found on control plane client. "
            f"Available methods can be found in the boto3 documentation for "
            f"'bedrock-agentcore-control' service."
        )

    # *_and_wait methods
    # -------------------------------------------------------------------------

    def create_dataset_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a dataset and wait for it to reach ACTIVE status.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the create_dataset API.

        Returns:
            Dataset details when ACTIVE.

        Raises:
            RuntimeError: If the dataset reaches CREATE_FAILED status.
            TimeoutError: If the dataset doesn't become ACTIVE within max_wait.
        """
        response = self._cp_client.create_dataset(**convert_kwargs(kwargs))
        dataset_id = response["datasetId"]
        return wait_until(
            lambda: self._cp_client.get_dataset(datasetId=dataset_id),
            "ACTIVE",
            {"CREATE_FAILED"},
            wait_config,
        )

    def delete_dataset_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Delete a dataset (or a single version) and wait for completion.

        - Full delete (no ``datasetVersion``): polls until ``GetDataset``
          raises ``ResourceNotFoundException``. Fails on ``DELETE_FAILED``.
        - Version-specific delete (``datasetVersion`` provided): the dataset
          itself is not removed. Polls ``GetDataset`` until status returns to
          ``ACTIVE``. Fails on ``UPDATE_FAILED``. Returns the dataset details.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the delete_dataset API.

        Raises:
            RuntimeError: On ``DELETE_FAILED`` or ``UPDATE_FAILED``.
            TimeoutError: If the operation does not finish within ``max_wait``.
        """
        converted = convert_kwargs(kwargs)
        response = self._cp_client.delete_dataset(**converted)
        dataset_id = response["datasetId"]

        if "datasetVersion" in converted:
            return wait_until(
                lambda: self._cp_client.get_dataset(datasetId=dataset_id),
                "ACTIVE",
                {"UPDATE_FAILED"},
                wait_config,
            )

        wait_until_deleted(
            lambda: self._cp_client.get_dataset(datasetId=dataset_id),
            failed={"DELETE_FAILED"},
            wait_config=wait_config,
        )
        return None

    def create_dataset_version_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a dataset version and wait for the dataset to reach ACTIVE status.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the create_dataset_version API.

        Returns:
            Dataset details when ACTIVE.

        Raises:
            RuntimeError: If the dataset reaches UPDATE_FAILED status.
            TimeoutError: If the dataset doesn't become ACTIVE within max_wait.
        """
        response = self._cp_client.create_dataset_version(**convert_kwargs(kwargs))
        dataset_id = response["datasetId"]
        return wait_until(
            lambda: self._cp_client.get_dataset(datasetId=dataset_id),
            "ACTIVE",
            {"UPDATE_FAILED"},
            wait_config,
        )

    def add_examples_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add examples to a dataset and wait for ACTIVE status.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the add_dataset_examples API.

        Returns:
            Dataset details when ACTIVE.

        Raises:
            RuntimeError: If the dataset reaches UPDATE_FAILED status.
            TimeoutError: If the dataset doesn't become ACTIVE within max_wait.
        """
        response = self._cp_client.add_dataset_examples(**convert_kwargs(kwargs))
        dataset_id = response["datasetId"]
        return wait_until(
            lambda: self._cp_client.get_dataset(datasetId=dataset_id),
            "ACTIVE",
            {"UPDATE_FAILED"},
            wait_config,
        )

    def update_examples_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Update examples in a dataset and wait for ACTIVE status.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the update_dataset_examples API.

        Returns:
            Dataset details when ACTIVE.

        Raises:
            RuntimeError: If the dataset reaches UPDATE_FAILED status.
            TimeoutError: If the dataset doesn't become ACTIVE within max_wait.
        """
        response = self._cp_client.update_dataset_examples(**convert_kwargs(kwargs))
        dataset_id = response["datasetId"]
        return wait_until(
            lambda: self._cp_client.get_dataset(datasetId=dataset_id),
            "ACTIVE",
            {"UPDATE_FAILED"},
            wait_config,
        )

    def delete_examples_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Delete examples from a dataset and wait for ACTIVE status.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the delete_dataset_examples API.

        Returns:
            Dataset details when ACTIVE.

        Raises:
            RuntimeError: If the dataset reaches UPDATE_FAILED status.
            TimeoutError: If the dataset doesn't become ACTIVE within max_wait.
        """
        response = self._cp_client.delete_dataset_examples(**convert_kwargs(kwargs))
        dataset_id = response["datasetId"]
        return wait_until(
            lambda: self._cp_client.get_dataset(datasetId=dataset_id),
            "ACTIVE",
            {"UPDATE_FAILED"},
            wait_config,
        )
