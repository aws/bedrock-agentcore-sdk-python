"""AgentCore Dataset SDK - Client for Dataset Management operations."""

import logging
import uuid
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config

from .._utils.config import WaitConfig
from .._utils.polling import wait_until, wait_until_deleted
from .._utils.snake_case import accept_snake_case_kwargs, convert_kwargs
from .._utils.user_agent import build_user_agent_suffix

logger = logging.getLogger(__name__)

_DATASET_FAILED_STATUSES = {"FAILED", "UPDATE_UNSUCCESSFUL"}


class DatasetClient:
    """Client for Bedrock AgentCore Dataset Management operations.

    Provides access to dataset CRUD operations.
    Allowlisted boto3 methods can be called directly on this client.
    Parameters accept both camelCase and snake_case (auto-converted).

    Example::

        client = DatasetClient(region_name="us-west-2")

        # Pass-through to boto3 control plane client
        dataset = client.create_dataset(
            name="my-dataset",
            roleArn="arn:aws:iam::123456789:role/dataset-role",
        )
    """

    _ALLOWED_CP_METHODS = {
        "create_dataset",
        "get_dataset",
        "list_datasets",
        "update_dataset",
        "delete_dataset",
        "get_paginator",
        "get_waiter",
    }

    def __init__(
        self,
        region_name: Optional[str] = None,
        integration_source: Optional[str] = None,
        boto3_session: Optional[boto3.Session] = None,
    ):
        """Initialize the Dataset client.

        Args:
            region_name: AWS region name. If not provided, uses the session's region or "us-west-2".
            integration_source: Optional integration source for user-agent telemetry.
            boto3_session: Optional boto3 Session to use. If not provided, a default session
                          is created. Useful for named profiles or custom credentials.
        """
        session = boto3_session if boto3_session else boto3.Session()
        self.region_name = region_name or session.region_name or "us-west-2"
        self.integration_source = integration_source

        user_agent_extra = build_user_agent_suffix(integration_source)
        client_config = Config(user_agent_extra=user_agent_extra)

        self.cp_client = session.client(
            "bedrock-agentcore-control", region_name=self.region_name, config=client_config
        )

        logger.info("Initialized DatasetClient for region: %s", self.cp_client.meta.region_name)

    # Pass-through
    # -------------------------------------------------------------------------
    def __getattr__(self, name: str):
        """Dynamically forward allowlisted method calls to the control plane boto3 client."""
        if name in self._ALLOWED_CP_METHODS and hasattr(self.cp_client, name):
            method = getattr(self.cp_client, name)
            logger.debug("Forwarding method '%s' to cp_client", name)
            return accept_snake_case_kwargs(method)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
            f"Method not found on cp_client. "
            f"Available methods can be found in the boto3 documentation for "
            f"'bedrock-agentcore-control' service."
        )

    # *_and_wait methods
    # -------------------------------------------------------------------------
    def create_dataset_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Create a dataset and wait for it to reach READY status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the create_dataset API.

        Returns:
            Dataset details when READY.

        Raises:
            RuntimeError: If the dataset reaches a failed state.
            TimeoutError: If the dataset doesn't become READY within max_wait.
        """
        params = convert_kwargs(kwargs)
        params.setdefault("clientToken", str(uuid.uuid4()))
        response = self.cp_client.create_dataset(**params)
        dataset_id = response["datasetId"]
        return wait_until(
            lambda: self.cp_client.get_dataset(datasetIdentifier=dataset_id)["dataset"],
            "READY",
            _DATASET_FAILED_STATUSES,
            wait_config,
        )

    def update_dataset_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Update a dataset and wait for it to reach READY status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the update_dataset API.

        Returns:
            Dataset details when READY.

        Raises:
            RuntimeError: If the dataset reaches a failed state.
            TimeoutError: If the dataset doesn't become READY within max_wait.
        """
        response = self.cp_client.update_dataset(**convert_kwargs(kwargs))
        dataset_id = response["datasetId"]
        return wait_until(
            lambda: self.cp_client.get_dataset(datasetIdentifier=dataset_id)["dataset"],
            "READY",
            _DATASET_FAILED_STATUSES,
            wait_config,
        )

    def delete_dataset_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> None:
        """Delete a dataset and wait for deletion to complete.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the delete_dataset API.

        Raises:
            TimeoutError: If the dataset isn't deleted within max_wait.
        """
        response = self.cp_client.delete_dataset(**convert_kwargs(kwargs))
        dataset_id = response.get("datasetId")
        if not dataset_id:
            raise ValueError("delete_dataset response did not include a 'datasetId'; cannot poll for deletion.")
        wait_until_deleted(
            lambda: self.cp_client.get_dataset(datasetIdentifier=dataset_id),
            wait_config=wait_config,
        )
