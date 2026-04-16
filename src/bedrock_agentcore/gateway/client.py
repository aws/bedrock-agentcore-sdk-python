"""AgentCore Gateway SDK - Client for MCP gateway and target operations."""

import logging
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config

from .._utils.config import ListConfig, WaitConfig
from .._utils.pagination import list_all
from .._utils.polling import wait_until, wait_until_deleted
from .._utils.snake_case import accept_snake_case_kwargs
from .._utils.user_agent import build_user_agent_suffix

logger = logging.getLogger(__name__)

_GATEWAY_FAILED_STATUSES = {"FAILED", "UPDATE_UNSUCCESSFUL"}
_TARGET_FAILED_STATUSES = {"FAILED", "UPDATE_UNSUCCESSFUL", "SYNCHRONIZE_UNSUCCESSFUL"}


class GatewayClient:
    """Client for Bedrock AgentCore Gateway operations.

    Provides access to gateway and gateway target CRUD operations.
    Allowlisted boto3 methods can be called directly on this client.
    Parameters accept both camelCase and snake_case (auto-converted).

    Example::

        client = GatewayClient(region_name="us-west-2")

        # Pass-through to boto3 control plane client
        gateway = client.create_gateway(
            name="my-gateway",
            roleArn="arn:aws:iam::123456789:role/gateway-role",
            protocolType="MCP",
        )
    """

    _ALLOWED_CP_METHODS = {
        # Gateway CRUD
        "create_gateway",
        "get_gateway",
        "list_gateways",
        "update_gateway",
        "delete_gateway",
        # Gateway target CRUD
        "create_gateway_target",
        "get_gateway_target",
        "list_gateway_targets",
        "update_gateway_target",
        "delete_gateway_target",
    }

    def __init__(
        self,
        region_name: Optional[str] = None,
        integration_source: Optional[str] = None,
        boto3_session: Optional[boto3.Session] = None,
    ):
        """Initialize the Gateway client.

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

        self.cp_client = session.client("bedrock-agentcore-control", region_name=self.region_name, config=client_config)

        logger.info("Initialized GatewayClient for region: %s", self.cp_client.meta.region_name)

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

    # list_all_* methods
    # -------------------------------------------------------------------------
    def list_all_gateways(self, list_config: Optional[ListConfig] = None, **kwargs) -> List[Dict[str, Any]]:
        """List all gateways with automatic pagination.

        Args:
            list_config: Optional ListConfig to control max items returned (default: 100).
            **kwargs: Additional arguments forwarded to the list_gateways API.

        Returns:
            List of gateway summaries.
        """
        return list_all(self.cp_client, "list_gateways", "items", list_config, **kwargs)

    def list_all_gateway_targets(self, list_config: Optional[ListConfig] = None, **kwargs) -> List[Dict[str, Any]]:
        """List all gateway targets with automatic pagination.

        Args:
            list_config: Optional ListConfig to control max items returned (default: 100).
            **kwargs: Additional arguments forwarded to the list_gateway_targets API.
                Must include gatewayIdentifier.

        Returns:
            List of gateway target summaries.
        """
        return list_all(self.cp_client, "list_gateway_targets", "items", list_config, **kwargs)

    # *_and_wait methods
    # -------------------------------------------------------------------------
    def create_gateway_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Create a gateway and wait for it to reach READY status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the create_gateway API.

        Returns:
            Gateway details when READY.

        Raises:
            RuntimeError: If the gateway reaches a failed state.
            TimeoutError: If the gateway doesn't become READY within max_wait.
        """
        response = self.cp_client.create_gateway(**kwargs)
        gw_id = response["gatewayId"]
        return wait_until(
            lambda: self.cp_client.get_gateway(gatewayIdentifier=gw_id),
            "READY",
            _GATEWAY_FAILED_STATUSES,
            wait_config,
        )

    def update_gateway_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Update a gateway and wait for it to reach READY status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the update_gateway API.

        Returns:
            Gateway details when READY.

        Raises:
            RuntimeError: If the gateway reaches a failed state.
            TimeoutError: If the gateway doesn't become READY within max_wait.
        """
        response = self.cp_client.update_gateway(**kwargs)
        gw_id = response["gatewayId"]
        return wait_until(
            lambda: self.cp_client.get_gateway(gatewayIdentifier=gw_id),
            "READY",
            _GATEWAY_FAILED_STATUSES,
            wait_config,
        )

    def create_gateway_target_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Create a gateway target and wait for it to reach READY status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the create_gateway_target API.
                Must include gatewayIdentifier.

        Returns:
            Gateway target details when READY.

        Raises:
            RuntimeError: If the target reaches a failed state.
            TimeoutError: If the target doesn't become READY within max_wait.
        """
        response = self.cp_client.create_gateway_target(**kwargs)
        gw_arn = response["gatewayArn"]
        target_id = response["targetId"]
        return wait_until(
            lambda: self.cp_client.get_gateway_target(
                gatewayIdentifier=gw_arn,
                targetId=target_id,
            ),
            "READY",
            _TARGET_FAILED_STATUSES,
            wait_config,
        )

    def update_gateway_target_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Update a gateway target and wait for it to reach READY status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the update_gateway_target API.
                Must include gatewayIdentifier and targetId.

        Returns:
            Gateway target details when READY.

        Raises:
            RuntimeError: If the target reaches a failed state.
            TimeoutError: If the target doesn't become READY within max_wait.
        """
        response = self.cp_client.update_gateway_target(**kwargs)
        gw_arn = response["gatewayArn"]
        target_id = response["targetId"]
        return wait_until(
            lambda: self.cp_client.get_gateway_target(
                gatewayIdentifier=gw_arn,
                targetId=target_id,
            ),
            "READY",
            _TARGET_FAILED_STATUSES,
            wait_config,
        )

    def delete_gateway_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> None:
        """Delete a gateway and wait for deletion to complete.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the delete_gateway API.

        Raises:
            TimeoutError: If the gateway isn't deleted within max_wait.
        """
        response = self.cp_client.delete_gateway(**kwargs)
        gw_id = response["gatewayId"]
        wait_until_deleted(
            lambda: self.cp_client.get_gateway(gatewayIdentifier=gw_id),
            wait_config=wait_config,
        )

    def delete_gateway_target_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> None:
        """Delete a gateway target and wait for deletion to complete.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the delete_gateway_target API.

        Raises:
            TimeoutError: If the target isn't deleted within max_wait.
        """
        response = self.cp_client.delete_gateway_target(**kwargs)
        gw_arn = response["gatewayArn"]
        target_id = response["targetId"]
        wait_until_deleted(
            lambda: self.cp_client.get_gateway_target(
                gatewayIdentifier=gw_arn,
                targetId=target_id,
            ),
            wait_config=wait_config,
        )

    # Name-based lookup
    # -------------------------------------------------------------------------
    def get_gateway_by_name(self, name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Look up a gateway by name.

        Paginates through all gateways and returns the full resource details
        for the first match. Returns None if no gateway with that name exists.

        Args:
            name: The gateway name to search for.
            **kwargs: Additional arguments forwarded to the list_gateways API.

        Returns:
            Gateway details from get_gateway, or None if not found.
        """
        for gw in list_all(self.cp_client, "list_gateways", "items", **kwargs):
            if gw.get("name") == name:
                return self.cp_client.get_gateway(gatewayIdentifier=gw["gatewayId"])
        return None

    def get_gateway_target_by_name(self, gateway_identifier: str, name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Look up a gateway target by name.

        Paginates through all targets for the given gateway and returns the
        full resource details for the first match. Returns None if not found.

        Args:
            gateway_identifier: Gateway ID or ARN.
            name: The target name to search for.
            **kwargs: Additional arguments forwarded to the list_gateway_targets API.

        Returns:
            Gateway target details from get_gateway_target, or None if not found.
        """
        kwargs["gatewayIdentifier"] = gateway_identifier
        for target in list_all(self.cp_client, "list_gateway_targets", "items", **kwargs):
            if target.get("name") == name:
                return self.cp_client.get_gateway_target(
                    gatewayIdentifier=gateway_identifier, targetId=target["targetId"]
                )
        return None
