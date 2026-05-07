"""AgentCore Payments SDK - PaymentClient.

This module provides a low-level SDK for integrating payment management into Bedrock AgentCore.
It enables direct communication with payment control plane APIs for managing payment managers
and payment connectors.

The PaymentsClient provides both direct boto3 method forwarding and access to control plane operations.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, TypedDict, Union

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from bedrock_agentcore._utils.endpoints import get_control_plane_endpoint
from bedrock_agentcore._utils.user_agent import build_user_agent_suffix
from bedrock_agentcore.services.identity import IdentityClient

logger = logging.getLogger(__name__)


class CoinbaseCdpConfigurationInput(TypedDict, total=False):
    """Configuration for Coinbase CDP credential provider.

    Attributes:
        api_key_id: The API key ID from Coinbase Developer Platform
        api_key_secret: The API key secret from Coinbase Developer Platform
        wallet_secret: The wallet secret from Coinbase Developer Platform
    """

    api_key_id: str
    api_key_secret: str
    wallet_secret: str


class CoinbaseCdpCredentials(TypedDict):
    """Coinbase CDP specific credentials.

    Attributes:
        api_key_id: The API key ID from Coinbase CDP
        api_key_secret: The API key secret from Coinbase CDP
        wallet_secret: The wallet secret from Coinbase CDP
    """

    api_key_id: str
    api_key_secret: str
    wallet_secret: str


class StripePrivyConfigurationInput(TypedDict, total=False):
    """Configuration for Stripe + Privy credential provider.

    Attributes:
        app_id: The Privy application ID
        app_secret: The Privy application secret
        authorization_private_key: The private key used for authorization signing
        authorization_id: The authorization identifier
    """

    app_id: str
    app_secret: str
    authorization_private_key: str
    authorization_id: str


class StripePrivyCredentials(TypedDict):
    """Stripe + Privy specific credentials.

    Attributes:
        app_id: The Privy application ID
        app_secret: The Privy application secret
        authorization_private_key: The private key used for authorization signing
        authorization_id: The authorization identifier
    """

    app_id: str
    app_secret: str
    authorization_private_key: str
    authorization_id: str


# Union type for vendor-specific credentials
CredentialProviderCredentials = Union[CoinbaseCdpCredentials, StripePrivyCredentials]


class ConnectorCredentialProviderConfig(TypedDict, total=False):
    """Configuration for credential provider used by payment connector.

    Attributes:
        name: Unique name for the credential provider
        credential_provider_vendor: The vendor type (CoinbaseCDP or StripePrivy)
        credentials: Vendor-specific credentials (CoinbaseCdpCredentials or StripePrivyCredentials)
    """

    name: str
    credential_provider_vendor: str
    credentials: CredentialProviderCredentials


class PaymentConnectorConfig(TypedDict, total=False):
    """Configuration for payment connector with credential provider.

    Attributes:
        name: Unique name for the payment connector
        description: Optional description for the payment connector
        payment_credential_provider_config: Credential provider configuration containing:
            - name: Unique name for the credential provider
            - credential_provider_vendor: Vendor type (CoinbaseCDP or StripePrivy)
            - credentials: Vendor-specific credentials (CoinbaseCdpCredentials or StripePrivyCredentials)
    """

    name: str
    description: str
    payment_credential_provider_config: ConnectorCredentialProviderConfig


class PaymentClient:
    """Low-level control plane client for payment operations.

    Provides direct boto3 method forwarding for control plane operations.
    """

    # Allowed control plane methods (forwarded to bedrock-agentcore-control client)
    _ALLOWED_PAYMENTS_CP_METHODS = {
        "create_payment_manager",
        "get_payment_manager",
        "list_payment_managers",
        "update_payment_manager",
        "delete_payment_manager",
        "create_payment_connector",
        "get_payment_connector",
        "list_payment_connectors",
        "update_payment_connector",
        "delete_payment_connector",
    }

    @staticmethod
    def _is_not_blank(value: Optional[str]) -> bool:
        """Check if a parameter value is not blank (not None and not empty string).

        Args:
            value: The parameter value to validate

        Returns:
            True if the value is not blank, False otherwise
        """
        return value is not None and value != ""

    @staticmethod
    def _safe_error_message(e: Exception) -> str:
        """Extract a safe error message that won't leak credentials.

        For ClientError, returns the error code and sanitized message.
        For other exceptions, returns only the exception type to avoid
        leaking credential data that may appear in str(e).

        Args:
            e: The exception to extract a safe message from

        Returns:
            A sanitized error string safe for logging
        """
        if isinstance(e, ClientError):
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", "Unknown error")
            return f"{error_code}: {error_message}"
        if isinstance(e, ValueError):
            return f"ValueError: {str(e)}"
        return f"{type(e).__name__}: (details redacted for security)"

    @staticmethod
    def _build_provider_config_input(
        payment_credential_provider_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build provider configuration input based on vendor type.

        Args:
            payment_credential_provider_config: Credential provider configuration containing:
                - credential_provider_vendor: The vendor type (e.g., CoinbaseCDP, StripePrivy)
                - credentials: Vendor-specific credentials

        Returns:
            Dictionary with the appropriate configuration structure for the vendor

        Raises:
            ValueError: If required credential fields are missing or None

        Example:
            For CoinbaseCDP vendor:
            {
                "coinbaseCdpConfiguration": {
                    "apiKeyId": "...",
                    "apiKeySecret": "...",
                    "walletSecret": "..."
                }
            }

            For StripePrivy vendor:
            {
                "stripePrivyConfiguration": {
                    "appId": "...",
                    "appSecret": "...",
                    "authorizationPrivateKey": "...",
                    "authorizationId": "..."
                }
            }
        """
        vendor = payment_credential_provider_config.get("credential_provider_vendor")
        if not vendor:
            raise ValueError("credential_provider_vendor is required")

        credentials: CredentialProviderCredentials = payment_credential_provider_config.get("credentials", {})  # type: ignore

        if vendor == "CoinbaseCDP":
            required_fields = ["api_key_id", "api_key_secret", "wallet_secret"]
            missing = [f for f in required_fields if not credentials.get(f)]
            if missing:
                raise ValueError(f"Missing required CoinbaseCDP credential fields: {', '.join(missing)}")
            return {
                "coinbaseCdpConfiguration": {
                    "apiKeyId": credentials["api_key_id"],
                    "apiKeySecret": credentials["api_key_secret"],
                    "walletSecret": credentials["wallet_secret"],
                }
            }
        elif vendor == "StripePrivy":
            required_fields = ["app_id", "app_secret", "authorization_private_key", "authorization_id"]
            missing = [f for f in required_fields if not credentials.get(f)]
            if missing:
                raise ValueError(f"Missing required StripePrivy credential fields: {', '.join(missing)}")
            return {
                "stripePrivyConfiguration": {
                    "appId": credentials["app_id"],
                    "appSecret": credentials["app_secret"],
                    "authorizationPrivateKey": credentials["authorization_private_key"],
                    "authorizationId": credentials["authorization_id"],
                }
            }
        else:
            raise ValueError(
                f"Unsupported credential_provider_vendor: '{vendor}'. Supported vendors are: CoinbaseCDP, StripePrivy"
            )

    def __init__(
        self,
        region_name: Optional[str] = None,
        integration_source: Optional[str] = None,
    ) -> None:
        """Initialize the Payments control plane client.

        Args:
            region_name: AWS region name. Defaults to boto3 session region or us-west-2
            integration_source: Optional identifier for tracking integration source in telemetry

        """
        self.region_name = region_name or boto3.Session().region_name or "us-west-2"
        self.integration_source = integration_source

        # Build config with user-agent for telemetry
        user_agent_extra = build_user_agent_suffix(integration_source)
        client_config = Config(user_agent_extra=user_agent_extra)

        # Control plane operations are available through bedrock-agentcore-control service
        self.payments_cp_client = boto3.client(
            "bedrock-agentcore-control",
            region_name=self.region_name,
            endpoint_url=get_control_plane_endpoint(self.region_name),
            config=client_config,
        )

        # Initialize identity client for credential provider operations
        self.identity_client = IdentityClient(region=self.region_name)

        logger.info(
            "Initialized PaymentClient for control plane: %s",
            self.payments_cp_client.meta.region_name,
        )

    def __getattr__(self, name: str):
        """Dynamically forward method calls to the control plane boto3 client.

        This method enables access to all boto3 client methods without explicitly
        defining them. Methods are looked up in the following order:
        1. payments_cp_client (bedrock-agentcore-control) - for control plane operations

        Args:
            name: The method name being accessed

        Returns:
            A callable method from the control plane boto3 client

        Raises:
            AttributeError: If the method doesn't exist on the control plane client

        Example:
            # Access any boto3 method directly
            client = PaymentClient()

            # These calls are forwarded to the control plane boto3 client
            response = client.create_payment_manager(...)
            response = client.get_payment_connector(...)
        """
        if name in self._ALLOWED_PAYMENTS_CP_METHODS and hasattr(self.payments_cp_client, name):
            method = getattr(self.payments_cp_client, name)
            logger.debug("Forwarding method '%s' to payments_cp_client", name)
            return method

        # Method not found on control plane client
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
            f"Method not found on payments_cp_client. "
            f"Available methods can be found in the boto3 documentation for "
            f"'bedrock-agentcore-control' service."
        )

    def _wait_for_status(
        self,
        get_method,
        resource_id: str,
        target_status: str,
        max_wait: int = 300,
        poll_interval: int = 10,
        **get_kwargs,
    ) -> Dict[str, Any]:
        """Wait for a resource to reach a target status.

        Args:
            get_method: The get method to call (e.g., get_payment_manager)
            resource_id: ID of the resource to check
            target_status: Status to wait for
            max_wait: Maximum seconds to wait
            poll_interval: Seconds between checks
            **get_kwargs: Additional kwargs for the get method

        Returns:
            The resource details when target status is reached

        Raises:
            TimeoutError: If max_wait is exceeded
            ClientError: If the resource reaches a failed status
        """
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"Timeout waiting for resource {resource_id} to reach {target_status} status")

            try:
                # Call get_method with resource_id as first positional arg and any additional kwargs
                if get_kwargs:
                    response = get_method(resource_id, **get_kwargs)
                else:
                    response = get_method(resource_id)
                status = response.get("status")

                if status == target_status:
                    logger.info("Resource %s reached %s status", resource_id, target_status)
                    return response

                if status in ["CREATE_FAILED", "UPDATE_FAILED", "DELETE_FAILED"]:
                    raise ClientError(
                        {"Error": {"Code": "ResourceFailed", "Message": f"Resource reached {status} status"}},
                        "GetResource",
                    )

                logger.debug("Resource %s status: %s (elapsed: %.1fs)", resource_id, status, elapsed)
                time.sleep(poll_interval)

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")
                if error_code == "ResourceNotFoundException":
                    logger.debug("Resource %s not found yet (elapsed: %.1fs)", resource_id, elapsed)
                    time.sleep(poll_interval)
                else:
                    raise

    def create_payment_manager(
        self,
        name: str,
        role_arn: str,
        authorizer_type: str = "AWS_IAM",
        description: Optional[str] = None,
        authorizer_configuration: Optional[Dict[str, Any]] = None,
        client_token: Optional[str] = None,
        wait_for_ready: bool = False,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Create a payment manager resource.

        Args:
            name: Name of the payment manager
            role_arn: IAM role ARN for payment manager authorization
            authorizer_type: Authorization type (default: AWS_IAM)
            description: Optional description
            authorizer_configuration: Optional authorizer configuration
            client_token: Optional idempotency token. If not provided, a UUID will be generated.
            wait_for_ready: Whether to wait for manager to reach READY status
            max_wait: Maximum seconds to wait if wait_for_ready is True
            poll_interval: Seconds between checks if wait_for_ready is True

        Returns:
            Dictionary with paymentManagerArn, paymentManagerId, and status

        Raises:
            ClientError: If creation fails
            TimeoutError: If wait_for_ready is True and max_wait is exceeded
        """
        if client_token is None:
            client_token = str(uuid.uuid4())

        try:
            logger.info("Creating payment manager: %s with role %s", name, role_arn)
            params = {
                "name": name,
                "roleArn": role_arn,
                "authorizerType": authorizer_type,
                "clientToken": client_token,
            }

            if self._is_not_blank(description):
                params["description"] = description

            if authorizer_configuration:
                params["authorizerConfiguration"] = authorizer_configuration

            response = self.payments_cp_client.create_payment_manager(**params)

            manager_arn = response.get("paymentManagerArn")
            manager_id = response.get("paymentManagerId")
            status = response.get("status")

            logger.info("Payment manager created: %s (status: %s)", manager_arn, status)

            if wait_for_ready:
                logger.info("Waiting for payment manager %s to reach READY status", manager_id)
                response = self._wait_for_status(
                    self.get_payment_manager,
                    manager_id,
                    "READY",
                    max_wait=max_wait,
                    poll_interval=poll_interval,
                )
                status = response.get("status")

            return {
                "paymentManagerArn": manager_arn,
                "paymentManagerId": manager_id,
                "status": status,
            }

        except ClientError as e:
            logger.error("Failed to create payment manager: %s", e)
            raise

    def get_payment_manager(self, payment_manager_id: str) -> Dict[str, Any]:
        """Retrieve payment manager details.

        Args:
            payment_manager_id: ID of the payment manager

        Returns:
            Dictionary with payment manager configuration

        Raises:
            ClientError: If retrieval fails
        """
        try:
            logger.info("Retrieving payment manager: %s", payment_manager_id)
            response = self.payments_cp_client.get_payment_manager(paymentManagerId=payment_manager_id)

            return {
                "paymentManagerId": response.get("paymentManagerId"),
                "paymentManagerArn": response.get("paymentManagerArn"),
                "name": response.get("name"),
                "description": response.get("description"),
                "status": response.get("status"),
                "createdAt": response.get("createdAt"),
                "updatedAt": response.get("updatedAt"),
            }

        except ClientError as e:
            logger.error("Failed to get payment manager: %s", e)
            raise

    def list_payment_managers(self, max_results: int = 100, next_token: Optional[str] = None) -> Dict[str, Any]:
        """List all payment managers with pagination support.

        Args:
            max_results: Maximum number of results to return (default: 100)
            next_token: Token for pagination to retrieve the next set of results

        Returns:
            Dictionary containing:
                - paymentManagers: List of payment manager configurations
                - nextToken: Token for retrieving the next page (if more results exist)

        Raises:
            ClientError: If listing fails
        """
        try:
            logger.info("Listing payment managers with max_results=%s, next_token=%s", max_results, next_token)
            params = {"maxResults": max_results}
            if self._is_not_blank(next_token):
                params["nextToken"] = next_token

            response = self.payments_cp_client.list_payment_managers(**params)

            managers = []
            for manager in response.get("paymentManagers", []):
                managers.append(
                    {
                        "paymentManagerId": manager.get("paymentManagerId"),
                        "paymentManagerArn": manager.get("paymentManagerArn"),
                        "name": manager.get("name"),
                        "description": manager.get("description"),
                        "status": manager.get("status"),
                        "createdAt": manager.get("createdAt"),
                        "updatedAt": manager.get("updatedAt"),
                    }
                )

            logger.info("Retrieved %s payment managers", len(managers))
            return {
                "paymentManagers": managers,
                "nextToken": response.get("nextToken"),
            }

        except ClientError as e:
            logger.error("Failed to list payment managers: %s", e)
            raise

    def update_payment_manager(
        self,
        payment_manager_id: str,
        description: Optional[str] = None,
        authorizer_type: Optional[str] = None,
        authorizer_configuration: Optional[Dict[str, Any]] = None,
        role_arn: Optional[str] = None,
        client_token: Optional[str] = None,
        wait_for_ready: bool = False,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Update a payment manager.

        Args:
            payment_manager_id: ID of the payment manager to update
            description: Optional new description
            authorizer_type: Optional authorizer type (CUSTOM_JWT or AWS_IAM)
            authorizer_configuration: Optional authorizer configuration
            role_arn: Optional IAM role ARN for the payment manager
            client_token: Optional idempotency token. If not provided, a UUID will be generated.
            wait_for_ready: Whether to wait for manager to reach READY status
            max_wait: Maximum seconds to wait if wait_for_ready is True
            poll_interval: Seconds between checks if wait_for_ready is True

        Returns:
            Dictionary with updated manager details

        Raises:
            ClientError: If update fails
            TimeoutError: If wait_for_ready is True and max_wait is exceeded
        """
        if client_token is None:
            client_token = str(uuid.uuid4())

        try:
            logger.info("Updating payment manager: %s", payment_manager_id)
            params = {
                "paymentManagerId": payment_manager_id,
                "clientToken": client_token,
            }

            if self._is_not_blank(description):
                params["description"] = description

            if authorizer_type:
                params["authorizerType"] = authorizer_type

            if authorizer_configuration:
                params["authorizerConfiguration"] = authorizer_configuration

            if self._is_not_blank(role_arn):
                params["roleArn"] = role_arn

            response = self.payments_cp_client.update_payment_manager(**params)

            status = response.get("status")

            if wait_for_ready:
                logger.info("Waiting for payment manager %s to reach READY status", payment_manager_id)
                response = self._wait_for_status(
                    self.get_payment_manager,
                    payment_manager_id,
                    "READY",
                    max_wait=max_wait,
                    poll_interval=poll_interval,
                )
                status = response.get("status")

            result = {
                "paymentManagerId": response.get("paymentManagerId"),
                "paymentManagerArn": response.get("paymentManagerArn"),
                "name": response.get("name"),
                "description": response.get("description"),
                "status": status,
                "updatedAt": response.get("updatedAt"),
            }

            return result

        except ClientError as e:
            logger.error("Failed to update payment manager: %s", e)
            raise

    def delete_payment_manager(
        self,
        payment_manager_id: str,
    ) -> Dict[str, Any]:
        """Delete a payment manager.

        Args:
            payment_manager_id: ID of the payment manager to delete

        Returns:
            Dictionary with deletion status

        Raises:
            ClientError: If deletion fails
        """
        try:
            logger.info("Deleting payment manager: %s", payment_manager_id)
            response = self.payments_cp_client.delete_payment_manager(
                paymentManagerId=payment_manager_id,
            )

            logger.info("Initiated deletion of payment manager: %s", payment_manager_id)

            return {
                "paymentManagerId": payment_manager_id,
                "status": response.get("status", "DELETED"),
            }

        except ClientError as e:
            logger.error("Failed to delete payment manager: %s", e)
            raise

    def create_payment_connector(
        self,
        payment_manager_id: str,
        name: str,
        connector_type: str,
        credential_provider_configurations: List[Dict[str, Any]],
        description: Optional[str] = None,
        client_token: Optional[str] = None,
        wait_for_ready: bool = False,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Create a payment connector for a provider.

        Args:
            payment_manager_id: ID of the payment manager
            name: Name of the connector
            connector_type: Connector type (e.g., CoinbaseCDP)
            credential_provider_configurations: List of credential provider configurations.
                Each config should be a dict with provider name as key and credential config as value.
                Example: [{"coinbaseCDP": {"credentialProviderArn": "arn:..."}}]
            description: Optional description
            client_token: Optional idempotency token. If not provided, a UUID will be generated.
            wait_for_ready: Whether to wait for connector to reach READY status
            max_wait: Maximum seconds to wait if wait_for_ready is True
            poll_interval: Seconds between checks if wait_for_ready is True

        Returns:
            Dictionary with paymentConnectorId and status

        Raises:
            ClientError: If creation fails
            TimeoutError: If wait_for_ready is True and max_wait is exceeded
        """
        if client_token is None:
            client_token = str(uuid.uuid4())

        try:
            logger.info(
                "Creating payment connector: %s for manager %s",
                name,
                payment_manager_id,
            )

            params = {
                "paymentManagerId": payment_manager_id,
                "name": name,
                "type": connector_type,
                "credentialProviderConfigurations": credential_provider_configurations,
                "clientToken": client_token,
            }

            if self._is_not_blank(description):
                params["description"] = description

            response = self.payments_cp_client.create_payment_connector(**params)

            payment_connector_id = response.get("paymentConnectorId")
            status = response.get("status")

            logger.info("Payment connector created: %s (status: %s)", payment_connector_id, status)

            if wait_for_ready:
                logger.info("Waiting for payment connector %s to reach READY status", payment_connector_id)

                # Create a wrapper function that calls get_payment_connector with the correct arguments
                def get_connector_status(conn_id):
                    return self.get_payment_connector(payment_manager_id, conn_id)

                response = self._wait_for_status(
                    get_connector_status,
                    payment_connector_id,
                    "READY",
                    max_wait=max_wait,
                    poll_interval=poll_interval,
                )
                status = response.get("status")

            return {
                "paymentConnectorId": payment_connector_id,
                "status": status,
            }

        except ClientError as e:
            logger.error("Failed to create payment connector: %s", e)
            raise

    def get_payment_connector(self, payment_manager_id: str, payment_connector_id: str) -> Dict[str, Any]:
        """Retrieve payment connector details.

        Args:
            payment_manager_id: ID of the payment manager
            payment_connector_id: ID of the connector

        Returns:
            Dictionary with payment connector configuration

        Raises:
            ClientError: If retrieval fails
        """
        try:
            logger.info("Retrieving payment connector: %s for manager %s", payment_connector_id, payment_manager_id)
            response = self.payments_cp_client.get_payment_connector(
                paymentManagerId=payment_manager_id, paymentConnectorId=payment_connector_id
            )

            return {
                "paymentConnectorId": response.get("paymentConnectorId"),
                "paymentManagerId": response.get("paymentManagerId"),
                "name": response.get("name"),
                "description": response.get("description"),
                "providerType": response.get("type"),
                "status": response.get("status"),
                "createdAt": response.get("createdAt"),
                "updatedAt": response.get("lastUpdatedAt"),
            }

        except ClientError as e:
            logger.error("Failed to get payment connector: %s", e)
            raise

    def list_payment_connectors(
        self, payment_manager_id: str, max_results: int = 100, next_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """List connectors for a payment manager with pagination support.

        Args:
            payment_manager_id: ID of the payment manager
            max_results: Maximum number of results to return (default: 100)
            next_token: Token for pagination to retrieve the next set of results

        Returns:
            Dictionary containing:
                - paymentConnectors: List of payment connector configurations
                - nextToken: Token for retrieving the next page (if more results exist)

        Raises:
            ClientError: If listing fails
        """
        try:
            logger.info(
                "Listing payment connectors for manager %s with max_results=%s, next_token=%s",
                payment_manager_id,
                max_results,
                next_token,
            )
            params = {"paymentManagerId": payment_manager_id, "maxResults": max_results}
            if self._is_not_blank(next_token):
                params["nextToken"] = next_token

            response = self.payments_cp_client.list_payment_connectors(**params)

            connectors = []
            for connector in response.get("paymentConnectors", []):
                connectors.append(
                    {
                        "paymentConnectorId": connector.get("paymentConnectorId"),
                        "paymentManagerId": connector.get("paymentManagerId"),
                        "name": connector.get("name"),
                        "description": connector.get("description"),
                        "providerType": connector.get("type"),
                        "status": connector.get("status"),
                        "createdAt": connector.get("createdAt"),
                        "updatedAt": connector.get("lastUpdatedAt"),
                    }
                )

            logger.info("Retrieved %s payment connectors", len(connectors))
            return {
                "paymentConnectors": connectors,
                "nextToken": response.get("nextToken"),
            }

        except ClientError as e:
            logger.error("Failed to list payment connectors: %s", e)
            raise

    def delete_payment_connector(
        self,
        payment_manager_id: str,
        payment_connector_id: str,
    ) -> Dict[str, Any]:
        """Delete a payment connector.

        Args:
            payment_manager_id: ID of the payment manager
            payment_connector_id: ID of the connector to delete

        Returns:
            Dictionary with deletion status

        Raises:
            ClientError: If deletion fails
        """
        try:
            logger.info("Deleting payment connector: %s for manager %s", payment_connector_id, payment_manager_id)
            response = self.payments_cp_client.delete_payment_connector(
                paymentManagerId=payment_manager_id,
                paymentConnectorId=payment_connector_id,
            )

            logger.info("Initiated deletion of payment connector: %s", payment_connector_id)

            return {
                "paymentConnectorId": payment_connector_id,
                "status": response.get("status", "DELETED"),
            }

        except ClientError as e:
            logger.error("Failed to delete payment connector: %s", e)
            raise

    def update_payment_connector(
        self,
        payment_manager_id: str,
        payment_connector_id: str,
        description: Optional[str] = None,
        connector_type: Optional[str] = None,
        credential_provider_configurations: Optional[List[Dict[str, Any]]] = None,
        client_token: Optional[str] = None,
        wait_for_ready: bool = False,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Update a payment connector.

        Args:
            payment_manager_id: ID of the payment manager
            payment_connector_id: ID of the connector to update
            description: Optional new description
            connector_type: Optional connector type (e.g., CoinbaseCDP)
            credential_provider_configurations: Optional list of credential provider configurations.
                Each config should be a dict with provider name as key and credential config as value.
                Example: [{"coinbaseCDP": {"credentialProviderArn": "arn:..."}}]
            client_token: Optional idempotency token. If not provided, a UUID will be generated.
            wait_for_ready: Whether to wait for connector to reach READY status
            max_wait: Maximum seconds to wait if wait_for_ready is True
            poll_interval: Seconds between checks if wait_for_ready is True

        Returns:
            Dictionary with updated connector details

        Raises:
            ClientError: If update fails
            TimeoutError: If wait_for_ready is True and max_wait is exceeded
        """
        if client_token is None:
            client_token = str(uuid.uuid4())

        try:
            logger.info("Updating payment connector: %s for manager %s", payment_connector_id, payment_manager_id)
            params = {
                "paymentManagerId": payment_manager_id,
                "paymentConnectorId": payment_connector_id,
                "clientToken": client_token,
            }

            if self._is_not_blank(description):
                params["description"] = description

            if connector_type:
                params["type"] = connector_type

            if credential_provider_configurations:
                params["credentialProviderConfigurations"] = credential_provider_configurations

            response = self.payments_cp_client.update_payment_connector(**params)

            status = response.get("status")

            if wait_for_ready:
                logger.info("Waiting for payment connector %s to reach READY status", payment_connector_id)

                def get_connector_status(conn_id):
                    return self.get_payment_connector(payment_manager_id, conn_id)

                response = self._wait_for_status(
                    get_connector_status,
                    payment_connector_id,
                    "READY",
                    max_wait=max_wait,
                    poll_interval=poll_interval,
                )

                status = response.get("status")

            result = {
                "paymentConnectorId": response.get("paymentConnectorId"),
                "paymentManagerId": response.get("paymentManagerId"),
                "name": response.get("name"),
                "description": response.get("description"),
                "providerType": response.get("type"),
                "status": status,
                "updatedAt": response.get("lastUpdatedAt"),
            }

            return result

        except ClientError as e:
            logger.error("Failed to update payment connector: %s", e)
            raise

    def create_payment_manager_with_connector(
        self,
        payment_manager_name: str,
        payment_manager_description: Optional[str],
        authorizer_type: str,
        role_arn: str,
        payment_connector_config: PaymentConnectorConfig,
        wait_for_ready: bool = False,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Create a payment manager with connector and credential provider in one operation.

        This method orchestrates the creation of three interdependent resources:
        1. Payment Credential Provider (via IdentityClient) - stores vendor credentials
        2. Payment Manager (via PaymentClient) - manages payment operations
        3. Payment Connector (via PaymentClient) - connects to payment provider

        Client tokens are generated internally for each resource creation call to ensure idempotency.
        If any step fails, the method automatically rolls back previously created resources.

        Args:
            payment_manager_name: Name of the payment manager
            payment_manager_description: Optional description for payment manager
            authorizer_type: Authorization type (default: AWS_IAM)
            role_arn: IAM role ARN for payment manager authorization
            payment_connector_config: Configuration for payment connector including:
                - name: Unique name for the payment connector
                - description: Optional description for the payment connector
                - payment_credential_provider_config: Credential provider configuration with:
                    - name: Unique name for the credential provider
                    - credential_provider_vendor: Vendor type (e.g., CoinbaseCDP, StripePrivy)
                    - credentials: Vendor-specific credentials
                      (CoinbaseCdpCredentials or StripePrivyCredentials)
            wait_for_ready: Whether to wait for resources to reach READY status
            max_wait: Maximum seconds to wait if wait_for_ready is True
            poll_interval: Seconds between checks if wait_for_ready is True

        Returns:
            Dictionary containing consolidated response with:
                - paymentManager: Payment manager details (ARN, ID, status)
                - paymentConnector: Payment connector details (ID, status)
                - credentialProvider: Credential provider details (ARN, name)

        Raises:
            ValueError: If required parameters are missing or invalid
            ClientError: If any API call fails (with automatic rollback)

        Example:
            ```python
            from bedrock_agentcore.payments.client import PaymentClient

            payment_client = PaymentClient(region_name="us-east-1")

            response = payment_client.create_payment_manager_with_connector(
                payment_manager_name="CDPPaymentManager",
                payment_manager_description="Coinbase Payment Manager",
                authorizer_type="AWS_IAM",
                role_arn="arn:aws:iam::123456789012:role/BedrockAgentCoreFullAccess",
                payment_connector_config={
                    "name": "coinbase-connector",
                    "description": "Coinbase CDP Connector",
                    "payment_credential_provider_config": {
                        "name": "coinbase-provider-name",
                        "credential_provider_vendor": "CoinbaseCDP",
                        "credentials": {
                            "api_key_id": "your-api-key-id",
                            "api_key_secret": "your-api-key-secret",
                            "wallet_secret": "your-wallet-secret",
                        },
                    },
                },
                wait_for_ready=True,
            )

            manager_arn = response["paymentManager"]["paymentManagerArn"]
            payment_connector_id = response["paymentConnector"]["paymentConnectorId"]
            provider_arn = response["credentialProvider"]["credentialProviderArn"]
            ```
        """
        mgr_client_token = str(uuid.uuid4())
        connector_client_token = str(uuid.uuid4())

        # Extract credential provider config
        payment_credential_provider_config = payment_connector_config.get("payment_credential_provider_config", {})

        # Track created resources for rollback
        created_resources = {
            "credential_provider_name": None,
            "payment_manager_id": None,
            "payment_connector_id": None,
        }

        try:
            # Step 1: Create Payment Credential Provider via IdentityClient
            logger.info("Step 1: Creating payment credential provider: %s", payment_credential_provider_config["name"])

            # Build provider configuration based on vendor type
            provider_config_input = self._build_provider_config_input(payment_credential_provider_config)

            credential_provider_response = self.identity_client.create_payment_credential_provider(
                name=payment_credential_provider_config["name"],
                credential_provider_vendor=payment_credential_provider_config["credential_provider_vendor"],
                provider_configuration_input=provider_config_input,
            )
            created_resources["credential_provider_name"] = payment_credential_provider_config["name"]
            credential_provider_arn = credential_provider_response.get("credentialProviderArn")
            logger.info("Successfully created credential provider: %s", credential_provider_arn)

            # Step 2: Create Payment Manager
            logger.info("Step 2: Creating payment manager: %s", payment_manager_name)

            manager_response = self.create_payment_manager(
                name=payment_manager_name,
                role_arn=role_arn,
                authorizer_type=authorizer_type,
                description=payment_manager_description,
                client_token=mgr_client_token,
                wait_for_ready=wait_for_ready,
                max_wait=max_wait,
                poll_interval=poll_interval,
            )

            payment_manager_id = manager_response.get("paymentManagerId")
            payment_manager_arn = manager_response.get("paymentManagerArn")
            created_resources["payment_manager_id"] = payment_manager_id

            logger.info("Successfully created payment manager: %s", payment_manager_arn)

            # Step 3: Create Payment Connector
            logger.info("Step 3: Creating payment connector: %s", payment_connector_config["name"])

            # Build credential provider configurations in the expected format
            vendor = payment_credential_provider_config["credential_provider_vendor"]
            if vendor == "CoinbaseCDP":
                credential_provider_configs = [{"coinbaseCDP": {"credentialProviderArn": credential_provider_arn}}]
            elif vendor == "StripePrivy":
                credential_provider_configs = [{"stripePrivy": {"credentialProviderArn": credential_provider_arn}}]
            else:
                raise ValueError(
                    f"Unsupported credential_provider_vendor: '{vendor}'. "
                    f"Supported vendors are: CoinbaseCDP, StripePrivy"
                )

            connector_response = self.create_payment_connector(
                payment_manager_id=payment_manager_id,
                name=payment_connector_config["name"],
                connector_type=payment_credential_provider_config["credential_provider_vendor"],
                credential_provider_configurations=credential_provider_configs,
                description=payment_connector_config.get("description"),
                client_token=connector_client_token,
                wait_for_ready=wait_for_ready,
                max_wait=max_wait,
                poll_interval=poll_interval,
            )

            payment_connector_id = connector_response.get("paymentConnectorId")
            created_resources["payment_connector_id"] = payment_connector_id

            logger.info("Successfully created payment connector: %s", payment_connector_id)

            # Return consolidated response
            logger.info("Successfully completed payment manager with connector creation")
            return {
                "paymentManager": {
                    "paymentManagerArn": payment_manager_arn,
                    "paymentManagerId": payment_manager_id,
                    "name": payment_manager_name,
                    "description": payment_manager_description,
                    "status": manager_response.get("status"),
                },
                "paymentConnector": {
                    "paymentConnectorId": payment_connector_id,
                    "paymentManagerId": payment_manager_id,
                    "name": payment_connector_config["name"],
                    "description": payment_connector_config.get("description"),
                    "providerType": payment_credential_provider_config["credential_provider_vendor"],
                    "status": connector_response.get("status"),
                },
                "credentialProvider": {
                    "credentialProviderArn": credential_provider_arn,
                    "name": payment_credential_provider_config["name"],
                    "credentialProviderVendor": payment_credential_provider_config["credential_provider_vendor"],
                },
            }

        except Exception as e:
            safe_error = self._safe_error_message(e)
            logger.error("Error during payment manager with connector creation: %s", safe_error)
            logger.info("Initiating rollback of created resources...")

            # Rollback: Delete created resources in reverse order
            rollback_errors = []

            # Rollback Payment Connector
            if created_resources["payment_connector_id"]:
                try:
                    logger.info("Rolling back payment connector: %s", created_resources["payment_connector_id"])
                    self.delete_payment_connector(
                        payment_manager_id=created_resources["payment_manager_id"],
                        payment_connector_id=created_resources["payment_connector_id"],
                    )
                    logger.info("Successfully rolled back payment connector")
                except Exception as rollback_error:
                    error_msg = f"Failed to rollback connector: {self._safe_error_message(rollback_error)}"
                    logger.error(error_msg)
                    rollback_errors.append(error_msg)

            # Rollback Payment Manager
            if created_resources["payment_manager_id"]:
                try:
                    logger.info("Rolling back payment manager: %s", created_resources["payment_manager_id"])
                    self.delete_payment_manager(payment_manager_id=created_resources["payment_manager_id"])
                    logger.info("Successfully rolled back payment manager")
                except Exception as rollback_error:
                    error_msg = f"Failed to rollback manager: {self._safe_error_message(rollback_error)}"
                    logger.error(error_msg)
                    rollback_errors.append(error_msg)

            # Rollback Credential Provider
            if created_resources["credential_provider_name"]:
                try:
                    logger.info("Rolling back credential provider: %s", created_resources["credential_provider_name"])
                    self.identity_client.delete_payment_credential_provider(
                        name=created_resources["credential_provider_name"]
                    )
                    logger.info("Successfully rolled back credential provider")
                except Exception as rollback_error:
                    error_msg = f"Failed to rollback credential provider: {self._safe_error_message(rollback_error)}"
                    logger.error(error_msg)
                    rollback_errors.append(error_msg)

            # Raise error with rollback information
            if rollback_errors:
                rollback_summary = "\n".join(rollback_errors)
                error_message = (
                    f"Failed to create payment manager with connector. "
                    f"Original error: {safe_error}\n"
                    f"Rollback errors:\n{rollback_summary}"
                )
                logger.error(error_message)
                raise ClientError(
                    {
                        "Error": {
                            "Code": "PaymentManagerCreationFailed",
                            "Message": error_message,
                        }
                    },
                    "CreatePaymentManagerWithConnector",
                ) from e
            else:
                logger.info("Rollback completed successfully")
                raise ClientError(
                    {
                        "Error": {
                            "Code": "PaymentManagerCreationFailed",
                            "Message": f"Failed to create payment manager with connector: {safe_error}",
                        }
                    },
                    "CreatePaymentManagerWithConnector",
                ) from e
