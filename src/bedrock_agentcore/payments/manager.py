"""PaymentManager class for managing payment operations."""

import base64
import binascii
import json
import logging
import os
import sys
import uuid
from typing import Any, Callable, Dict, Optional

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

from bedrock_agentcore._utils.endpoints import get_data_plane_endpoint
from bedrock_agentcore._utils.user_agent import build_user_agent_suffix

logger = logging.getLogger(__name__)


class PaymentError(Exception):
    """Base exception for payment operations."""

    pass


class PaymentInstrumentNotFound(PaymentError):
    """Raised when a payment instrument is not found."""

    pass


class PaymentSessionNotFound(PaymentError):
    """Raised when a payment session is not found."""

    pass


class InvalidPaymentInstrument(PaymentError):
    """Raised when a payment instrument is invalid or inactive."""

    pass


class InsufficientBudget(PaymentError):
    """Raised when payment amount exceeds remaining budget."""

    pass


class PaymentSessionExpired(PaymentError):
    """Raised when attempting to use an expired payment session."""


class PaymentInstrumentConfigurationRequired(PaymentError):
    """Raised when payment_instrument_id is not set on the plugin config."""


class PaymentSessionConfigurationRequired(PaymentError):
    """Raised when payment_session_id is not set on the plugin config."""

    pass


class PaymentManager:
    """Manages payment operations through a simplified interface.

    The PaymentManager provides a high-level wrapper around AgentCorePayment operations, simplifying
    payment operations by managing the paymentManagerArn internally. It provides a clean interface
    for payment instrument creation, payment session management, and payment processing.

    Key Capabilities:
        - **Payment Instrument Management**: Create and manage payment instruments without
          repeatedly passing the manager ARN
        - **Payment Session Management**: Create payment sessions with automatic ARN injection
        - **Payment Processing**: Process payments with automatic payment instrument validation
        - **Method Forwarding**: Access PaymentClient methods directly when needed

    Usage Patterns:
        1. **Create Payment Instrument**: Store a payment method for a user
        2. **Create Payment Session**: Establish a time-bounded payment context
        3. **Process Payment**: Execute a payment with automatic validation

    Example:
        ```python
        # Initialize manager
        manager = PaymentManager(
            payment_manager_arn="arn:aws:bedrock-agentcore:us-east-1:123456789012:payment-manager/pm-123",
            region_name="us-east-1"
        )

        # Create a payment instrument
        instrument_response = manager.create_payment_instrument(
            payment_connector_id="connector-456",
            payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
            payment_instrument_details={"embeddedCryptoWallet": {"network": "ETHEREUM",
                "linkedAccounts": [{"email": {"emailAddress": "user@example.com"}}]}},
            user_id="user-123",
        )

        # Create a payment session
        session_response = manager.create_payment_session(
            expiry_time_in_minutes=60,
            user_id="user-123",
            limits={"maxSpendAmount": {"value": "100.00", "currency": "USD"}},
        )

        # Process a payment
        payment_response = manager.process_payment(
            payment_session_id=session_response["paymentSessionId"],
            payment_instrument_id=instrument_response["paymentInstrumentId"],
            payment_type="CRYPTO_X402",
            payment_input={"cryptoX402": {
                "version": "1",
                "payload": {
                    "scheme": "exact",
                    "network": "base-sepolia",
                    "maxAmountRequired": "5000",
                    "resource": "https://premiousEndpoint",
                    "description": "Premium AI joke generation",
                    "mimeType": "application/json",
                    "payTo": "0x6813749E1eB9E0001A44C2684695FE8AD676cdD9",
                    "maxTimeoutSeconds": 300,
                    "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF71",
                    "outputSchema": {"input": {"type": "http", "method": "GET", "discoverable": True}},
                    "extra": {"name": "USDC", "version": "2"},
                },
            }},
            user_id="user-123",
        )
        ```

    Thread Safety:
        This class is not thread-safe. Create separate instances for concurrent operations.

    AWS Permissions Required:
        - bedrock-agentcore:CreatePaymentInstrument
        - bedrock-agentcore:GetPaymentInstrument
        - bedrock-agentcore:CreatePaymentSession
        - bedrock-agentcore:ProcessPayment
    """

    # Allowed data plane methods (forwarded to bedrock-agentcore client)
    _ALLOWED_PAYMENTS_DP_METHODS = {
        "create_payment_instrument",
        "get_payment_instrument",
        "get_payment_instrument_balance",
        "list_payment_instruments",
        "delete_payment_instrument",
        "create_payment_session",
        "get_payment_session",
        "list_payment_sessions",
        "delete_payment_session",
        "process_payment",
    }

    def __init__(
        self,
        payment_manager_arn: str,
        region_name: Optional[str] = None,
        boto3_session: Optional[boto3.Session] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
        agent_name: Optional[str] = None,
        bearer_token: Optional[str] = None,
        token_provider: Optional[Callable[[], str]] = None,
    ):
        """Initialize a PaymentManager instance.

        Args:
            payment_manager_arn: The ARN of the payment manager instance. Must be a non-empty string.
            region_name: AWS region for the bedrock-agentcore client. If not provided,
                        will use the region from boto3_session or default session.
            boto3_session: Optional boto3 Session to use. If provided and region_name
                          parameter is also specified, validation will ensure they match.
            boto_client_config: Optional boto3 client configuration. If provided, will be
                              merged with default configuration including user agent.
            agent_name: Optional agent name to propagate via the
                X-Amzn-Bedrock-AgentCore-Payments-Agent-Name HTTP header on every
                data-plane API call.
            bearer_token: Optional static JWT bearer token for OAuth/CUSTOM_JWT authentication.
                         When set, requests use Bearer token auth instead of SigV4.
                         Mutually exclusive with token_provider.
            token_provider: Optional callable that returns a fresh JWT bearer token string.
                           Called before each request to support token refresh.
                           Mutually exclusive with bearer_token.

        Raises:
            ValueError: If payment_manager_arn is invalid, region_name conflicts with boto3_session region,
                       configuration parameters are inconsistent, or both bearer_token and token_provider
                       are provided.
        """
        if not payment_manager_arn or not isinstance(payment_manager_arn, str):
            raise ValueError(
                f"payment_manager_arn is required and must be a non-empty string. Received: {payment_manager_arn!r}"
            )

        if bearer_token is not None and token_provider is not None:
            raise ValueError("bearer_token and token_provider are mutually exclusive. Provide only one.")

        if bearer_token is not None:
            if not isinstance(bearer_token, str) or not bearer_token.strip():
                raise ValueError("bearer_token must be a non-empty string.")
            if any(c in bearer_token for c in ("\r", "\n", "\x00")):
                raise ValueError("bearer_token must not contain newlines or null bytes.")

        if token_provider is not None and not callable(token_provider):
            raise ValueError("token_provider must be callable.")

        # Store payment manager ARN
        self._payment_manager_arn: str = payment_manager_arn
        self._agent_name: Optional[str] = agent_name
        self._bearer_token: Optional[str] = bearer_token
        self._token_provider: Optional[Callable[[], str]] = token_provider

        # Setup session and validate region consistency
        self.region_name = self._validate_and_resolve_region(region_name, boto3_session)
        session = boto3_session if boto3_session else boto3.Session()

        # Configure and create boto3 client
        client_config = self._build_client_config(boto_client_config)
        self._payment_client = session.client(
            "bedrock-agentcore",
            region_name=self.region_name,
            config=client_config,
            endpoint_url=get_data_plane_endpoint(self.region_name),
        )

        # Register event handler to inject agent name header on every data-plane call
        if self._agent_name:
            self._payment_client.meta.events.register(
                "before-sign.bedrock-agentcore.*",
                self._add_agent_name_header,
            )

        # Configure bearer token auth if provided (overrides default SigV4)
        # Uses before-send (after signing) so the Bearer header replaces the SigV4 Authorization header
        if bearer_token is not None or token_provider is not None:
            self._payment_client.meta.events.register(
                "before-send.bedrock-agentcore.*",
                self._inject_bearer_token,
            )

        logger.debug(
            "PaymentManager initialized with ARN: %s in region: %s (agent_name: %s, auth: %s)",
            self._payment_manager_arn,
            self.region_name,
            self._agent_name or "not set",
            "bearer" if self._is_bearer_auth else "sigv4",
        )

    def _inject_bearer_token(self, request, **kwargs) -> None:
        """Inject Bearer token into the request, replacing SigV4 authorization.

        For token_provider, calls the provider to get a fresh token.
        For static bearer_token, uses the stored value.
        Note: userId is NOT injected as a header — for CUSTOM_JWT auth, the service
        derives userId from the JWT 'sub' claim.
        """
        if self._token_provider:
            try:
                token = self._token_provider()
            except Exception as e:
                raise PaymentError(f"Token provider failed: {e}") from e
            if not token or not isinstance(token, str) or not token.strip():
                raise PaymentError("Token provider returned an empty or invalid token.")
            if any(c in token for c in ("\r", "\n", "\x00")):
                raise PaymentError("Token provider returned a token containing newlines or null bytes.")
        else:
            token = self._bearer_token

        request.headers["Authorization"] = f"Bearer {token}"

    def _add_agent_name_header(self, request, **kwargs):
        """Inject the agent name HTTP header into every outgoing data-plane request.

        This is registered as a boto3 event handler on ``before-sign`` so the
        header is present before the request is signed.

        Args:
            request: The ``AWSPreparedRequest`` about to be sent.
            **kwargs: Additional event keyword arguments (ignored).
        """
        request.headers["X-Amzn-Bedrock-AgentCore-Payments-Agent-Name"] = self._agent_name

    @property
    def _is_bearer_auth(self) -> bool:
        """Check if bearer token auth is configured."""
        return self._bearer_token is not None or self._token_provider is not None

    def _validate_and_resolve_region(self, region_name: Optional[str], session: Optional[boto3.Session]) -> str:
        """Validate region consistency and resolve the final region to use.

        Args:
            region_name: Explicitly provided region name
            session: Optional Boto3 session instance

        Returns:
            The resolved region name to use

        Raises:
            ValueError: If region_name conflicts with session region
        """
        session_region = session.region_name if session else None

        # Validate region consistency if both are provided
        if region_name and session and session_region and (region_name != session_region):
            raise ValueError(
                f"Region mismatch: provided region_name '{region_name}' does not match "
                f"boto3_session region '{session_region}'. Please ensure both "
                f"parameters specify the same region or omit the region_name parameter "
                f"to use the session's region."
            )

        return (
            region_name or session_region or os.environ.get("AWS_REGION") or boto3.Session().region_name or "us-west-2"
        )

    def _build_client_config(self, boto_client_config: Optional[BotocoreConfig]) -> BotocoreConfig:
        """Build the final boto3 client configuration with SDK user agent.

        Args:
            boto_client_config: Optional user-provided client configuration

        Returns:
            Final client configuration with SDK user agent
        """
        user_agent_extra = build_user_agent_suffix()

        if boto_client_config:
            existing_user_agent = getattr(boto_client_config, "user_agent_extra", None)
            if existing_user_agent:
                new_user_agent = f"{existing_user_agent} {user_agent_extra}"
            else:
                new_user_agent = user_agent_extra
            return boto_client_config.merge(BotocoreConfig(user_agent_extra=new_user_agent))
        else:
            return BotocoreConfig(user_agent_extra=user_agent_extra)

    def create_payment_instrument(
        self,
        payment_connector_id: str,
        payment_instrument_type: str,
        payment_instrument_details: Dict[str, Any],
        user_id: Optional[str] = None,
        client_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a payment instrument for a user.

        Creates a new payment instrument (e.g., crypto wallet) associated with a user.
        The paymentManagerArn is automatically injected from the manager's configuration.

        Args:
            payment_connector_id: ID of the payment connector to use
            payment_instrument_type: Type of payment instrument (e.g., EMBEDDED_CRYPTO_WALLET)
            payment_instrument_details: Details of the payment instrument (e.g., embeddedCryptoWallet)
            user_id: Unique identifier for the user (optional, omitted for bearer auth)
            client_token: Optional idempotency token

        Returns:
            Dictionary containing paymentInstrumentId and other instrument details

        Raises:
            PaymentError: If validation fails or API call fails

        Example:
            ```python
            response = manager.create_payment_instrument(
                user_id="user-123",
                payment_connector_id="connector-456",
                payment_instrument_type="EMBEDDED_CRYPTO_WALLET",
                payment_instrument_details={"embeddedCryptoWallet": {"network": "ETHEREUM"}}
            )
            instrument_id = response["paymentInstrumentId"]
            ```
        """
        user_id = user_id.strip() if user_id else None

        if client_token is None:
            client_token = str(uuid.uuid4())

        try:
            logger.info("Creating payment instrument for user %s", user_id)
            params = {
                **({"userId": user_id} if user_id and not self._is_bearer_auth else {}),
                "paymentManagerArn": self._payment_manager_arn,
                "paymentConnectorId": payment_connector_id,
                "paymentInstrumentType": payment_instrument_type,
                "paymentInstrumentDetails": payment_instrument_details,
                "clientToken": client_token,
            }

            result = self._payment_client.create_payment_instrument(**params)
            logger.info("Successfully created instrument for user %s", user_id)
            # Unwrap the nested paymentInstrument response
            return result.get("paymentInstrument", result)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ValidationException":
                logger.error("Validation error creating payment instrument: %s", message)
                raise PaymentError(f"Validation error: {message}") from e

            logger.error("Failed to create payment instrument: %s", str(e))
            raise PaymentError(f"Failed to create payment instrument: {str(e)}") from e

    def get_payment_instrument(
        self,
        payment_instrument_id: str,
        user_id: Optional[str] = None,
        payment_connector_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve payment instrument details.

        Args:
            payment_instrument_id: Unique identifier for the instrument
            user_id: Unique identifier for the user (optional, omitted for bearer auth)
            payment_connector_id: ID of the payment connector (optional)

        Returns:
            Dictionary containing instrument details

        Raises:
            PaymentInstrumentNotFound: If instrument not found
            PaymentError: If API call fails
        """
        user_id = user_id.strip() if user_id else None

        try:
            logger.info("Retrieving payment instrument %s for user %s", payment_instrument_id, user_id)
            params = {
                **({"userId": user_id} if user_id and not self._is_bearer_auth else {}),
                "paymentManagerArn": self._payment_manager_arn,
                "paymentInstrumentId": payment_instrument_id,
            }
            if payment_connector_id is not None:
                params["paymentConnectorId"] = payment_connector_id

            result = self._payment_client.get_payment_instrument(**params)
            logger.info("Successfully retrieved instrument %s", payment_instrument_id)
            # Unwrap the nested paymentInstrument response
            return result.get("paymentInstrument", result)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ResourceNotFoundException" or "not found" in message.lower():
                logger.error("Instrument not found: %s", payment_instrument_id)
                raise PaymentInstrumentNotFound(f"Instrument not found: {payment_instrument_id}") from e

            logger.error("Failed to get payment instrument: %s", str(e))
            raise PaymentError(f"Failed to get payment instrument: {str(e)}") from e

    def list_payment_instruments(
        self,
        user_id: Optional[str] = None,
        payment_connector_id: Optional[str] = None,
        max_results: int = 100,
        next_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List payment instruments for a user.

        Args:
            user_id: Unique identifier for the user (optional, omitted for bearer auth)
            payment_connector_id: Optional ID of the payment connector to filter by
            max_results: Maximum number of results to return (default 100)
            next_token: Token for pagination

        Returns:
            Dictionary containing list of instruments and next_token if more results exist

        Raises:
            PaymentError: If API call fails
        """
        user_id = user_id.strip() if user_id else None

        try:
            logger.info("Listing payment instruments for user %s", user_id)
            params = {
                **({"userId": user_id} if user_id and not self._is_bearer_auth else {}),
                "paymentManagerArn": self._payment_manager_arn,
                "maxResults": max_results,
            }

            if payment_connector_id:
                params["paymentConnectorId"] = payment_connector_id

            if next_token:
                params["nextToken"] = next_token

            result = self._payment_client.list_payment_instruments(**params)
            # Unwrap the nested paymentInstruments response
            instruments = result.get("paymentInstruments", result.get("instruments", []))
            logger.info("Retrieved %d instruments for user %s", len(instruments), user_id)
            response = {"paymentInstruments": instruments}
            if "nextToken" in result:
                response["nextToken"] = result["nextToken"]
            return response

        except ClientError as e:
            logger.error("Failed to list payment instruments: %s", str(e))
            raise PaymentError(f"Failed to list payment instruments: {str(e)}") from e

    def get_payment_instrument_balance(
        self,
        payment_connector_id: str,
        payment_instrument_id: str,
        chain: str,
        token: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get the token balance for a payment instrument on a specific chain.

        Args:
            payment_connector_id: ID of the payment connector
            payment_instrument_id: Unique identifier for the instrument
            chain: Blockchain chain to query (e.g., "BASE_SEPOLIA", "SOLANA_DEVNET")
            token: Token to query balance for (e.g., "USDC")
            user_id: Unique identifier for the user (optional, omitted for bearer auth)

        Returns:
            Dictionary containing paymentInstrumentId and tokenBalance

        Raises:
            PaymentInstrumentNotFound: If instrument not found
            PaymentError: If API call fails
        """
        user_id = user_id.strip() if user_id else None

        try:
            logger.info(
                "Getting balance for instrument %s on chain %s",
                payment_instrument_id,
                chain,
            )
            params = {
                **({"userId": user_id} if user_id and not self._is_bearer_auth else {}),
                "paymentManagerArn": self._payment_manager_arn,
                "paymentConnectorId": payment_connector_id,
                "paymentInstrumentId": payment_instrument_id,
                "chain": chain,
                "token": token,
            }

            result = self._payment_client.get_payment_instrument_balance(**params)
            logger.info("Successfully retrieved balance for instrument %s", payment_instrument_id)
            return result

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ResourceNotFoundException" or "not found" in message.lower():
                raise PaymentInstrumentNotFound(f"Instrument not found: {payment_instrument_id}") from e

            logger.error("Failed to get instrument balance: %s", str(e))
            raise PaymentError(f"Failed to get instrument balance: {str(e)}") from e

    def create_payment_session(
        self,
        expiry_time_in_minutes: int,
        user_id: Optional[str] = None,
        limits: Optional[dict] = None,
        client_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a payment session with spending limits.

        Args:
            expiry_time_in_minutes: Session expiry time in minutes (15-480)
            user_id: Unique identifier for the user (optional, omitted for bearer auth)
            limits: Optional spending limits dict with maxSpendAmount structure
            client_token: Optional idempotency token

        Returns:
            Dictionary containing paymentSessionId and other session details

        Raises:
            PaymentError: If validation fails or API call fails
        """
        user_id = user_id.strip() if user_id else None

        if client_token is None:
            client_token = str(uuid.uuid4())

        try:
            logger.info("Creating payment session for user %s with session limits %s", user_id, limits)
            params = {
                **({"userId": user_id} if user_id and not self._is_bearer_auth else {}),
                "paymentManagerArn": self._payment_manager_arn,
                "expiryTimeInMinutes": expiry_time_in_minutes,
                "clientToken": client_token,
            }

            if limits is not None:
                params["limits"] = limits

            result = self._payment_client.create_payment_session(**params)
            logger.info("Successfully created session for user %s", user_id)
            # Unwrap the nested paymentSession response
            return result.get("paymentSession", result)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ValidationException":
                if "expiry" in message.lower() or "duration" in message.lower():
                    logger.error("Invalid expiry_time_in_minutes: %s", message)
                    raise PaymentError(f"Invalid expiry_time_in_minutes: {message}") from e

            logger.error("Failed to create payment session: %s", str(e))
            raise PaymentError(f"Failed to create payment session: {str(e)}") from e

    def get_payment_session(
        self,
        payment_session_id: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve payment session details.

        Args:
            payment_session_id: Unique identifier for the session
            user_id: Unique identifier for the user (optional, omitted for bearer auth)

        Returns:
            Dictionary containing session details including remaining_amount and spent_amount

        Raises:
            PaymentSessionNotFound: If session not found
            PaymentError: If API call fails
        """
        user_id = user_id.strip() if user_id else None

        try:
            logger.info("Retrieving payment session %s for user %s", payment_session_id, user_id)
            params = {
                **({"userId": user_id} if user_id and not self._is_bearer_auth else {}),
                "paymentManagerArn": self._payment_manager_arn,
                "paymentSessionId": payment_session_id,
            }

            result = self._payment_client.get_payment_session(**params)
            logger.info("Successfully retrieved session %s", payment_session_id)
            # Unwrap the nested paymentSession response
            return result.get("paymentSession", result)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ResourceNotFoundException" or "not found" in message.lower():
                logger.error("Session not found: %s", payment_session_id)
                raise PaymentSessionNotFound(f"Session not found: {payment_session_id}") from e

            if error_code == "AccessDeniedException" or "unauthorized" in message.lower():
                logger.error("Unauthorized access to session %s", payment_session_id)
                raise PaymentError(f"Unauthorized access to session: {payment_session_id}") from e

            logger.error("Failed to get payment session: %s", str(e))
            raise PaymentError(f"Failed to get payment session: {str(e)}") from e

    def list_payment_sessions(
        self,
        user_id: Optional[str] = None,
        max_results: int = 100,
        next_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List payment sessions for a user.

        Args:
            user_id: Unique identifier for the user (optional, omitted for bearer auth)
            max_results: Maximum number of results to return (default 100)
            next_token: Token for pagination

        Returns:
            Dictionary containing list of sessions and next_token if more results exist

        Raises:
            PaymentError: If API call fails
        """
        user_id = user_id.strip() if user_id else None

        try:
            logger.info("Listing payment sessions for user %s", user_id)
            params = {
                **({"userId": user_id} if user_id and not self._is_bearer_auth else {}),
                "paymentManagerArn": self._payment_manager_arn,
                "maxResults": max_results,
            }

            if next_token:
                params["nextToken"] = next_token

            result = self._payment_client.list_payment_sessions(**params)
            # Unwrap the nested paymentSessions response
            sessions = result.get("paymentSessions", result.get("sessions", []))
            logger.info("Retrieved %d sessions for user %s", len(sessions), user_id)
            response = {"paymentSessions": sessions}
            if "nextToken" in result:
                response["nextToken"] = result["nextToken"]
            return response

        except ClientError as e:
            logger.error("Failed to list payment sessions: %s", str(e))
            raise PaymentError(f"Failed to list payment sessions: {str(e)}") from e

    def delete_payment_session(
        self,
        payment_session_id: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a payment session.

        Permanently removes a payment session record (hard delete). Once deleted,
        the session can no longer be used for payment processing.

        Deleting a non-existent or already-deleted session returns PaymentSessionNotFound.

        Args:
            payment_session_id: Unique identifier for the session to delete
            user_id: Unique identifier for the user (optional, omitted for bearer auth)

        Returns:
            Dictionary containing deletion status: {"status": "DELETED"}

        Raises:
            PaymentSessionNotFound: If session not found or already deleted
            PaymentError: If API call fails

        Example:
            ```python
            result = manager.delete_payment_session(
                payment_session_id="payment-session-abc123",
                user_id="user-123",
            )
            # result: {"status": "DELETED"}
            ```
        """
        user_id = user_id.strip() if user_id else None

        try:
            logger.info("Deleting payment session %s for user %s", payment_session_id, user_id)
            params = {
                **({"userId": user_id} if user_id and not self._is_bearer_auth else {}),
                "paymentManagerArn": self._payment_manager_arn,
                "paymentSessionId": payment_session_id,
            }

            result = self._payment_client.delete_payment_session(**params)
            logger.info("Successfully deleted payment session %s", payment_session_id)
            return result

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ResourceNotFoundException" or "not found" in message.lower():
                logger.error("Session not found: %s", payment_session_id)
                raise PaymentSessionNotFound(f"Session not found: {payment_session_id}") from e

            logger.error("Failed to delete payment session: %s", str(e))
            raise PaymentError(f"Failed to delete payment session: {str(e)}") from e

    def delete_payment_instrument(
        self,
        payment_instrument_id: str,
        payment_connector_id: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a payment instrument.

        Marks a payment instrument as deleted (soft delete). The record is preserved
        for audit and compliance purposes but is excluded from normal list and get operations.

        Deleting an already-deleted or non-existent instrument returns PaymentInstrumentNotFound.

        Args:
            payment_instrument_id: Unique identifier for the instrument to delete
            payment_connector_id: ID of the payment connector (required)
            user_id: Unique identifier for the user (optional, omitted for bearer auth)

        Returns:
            Dictionary containing deletion status: {"status": "DELETED"}

        Raises:
            PaymentInstrumentNotFound: If instrument not found or already deleted
            PaymentError: If API call fails

        Example:
            ```python
            result = manager.delete_payment_instrument(
                payment_instrument_id="payment-instrument-xyz789",
                payment_connector_id="connector-456",
                user_id="user-123",
            )
            # result: {"status": "DELETED"}
            ```
        """
        user_id = user_id.strip() if user_id else None

        try:
            logger.info("Deleting payment instrument %s for user %s", payment_instrument_id, user_id)
            params = {
                **({"userId": user_id} if user_id and not self._is_bearer_auth else {}),
                "paymentManagerArn": self._payment_manager_arn,
                "paymentConnectorId": payment_connector_id,
                "paymentInstrumentId": payment_instrument_id,
            }

            result = self._payment_client.delete_payment_instrument(**params)
            logger.info("Successfully deleted payment instrument %s", payment_instrument_id)
            return result

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ResourceNotFoundException" or "not found" in message.lower():
                logger.error("Instrument not found: %s", payment_instrument_id)
                raise PaymentInstrumentNotFound(f"Instrument not found: {payment_instrument_id}") from e

            logger.error("Failed to delete payment instrument: %s", str(e))
            raise PaymentError(f"Failed to delete payment instrument: {str(e)}") from e

    def process_payment(
        self,
        payment_session_id: str,
        payment_instrument_id: str,
        payment_type: str,
        payment_input: Dict[str, Any],
        user_id: Optional[str] = None,
        client_token: Optional[str] = None,
        payment_connector_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a payment transaction.

        Args:
            payment_session_id: Unique identifier for the payment session
            payment_instrument_id: Unique identifier for the payment instrument
            payment_type: Type of payment being processed (e.g., CRYPTO_X402)
            payment_input: Payment input details specific to the payment type
            user_id: Unique identifier for the user (optional, omitted for bearer auth)
            client_token: Optional idempotency token for request uniqueness
            payment_connector_id: Optional payment connector ID to route the payment

        Returns:
            Dictionary containing processPaymentId and transaction details

        Raises:
            PaymentInstrumentNotFound: If payment instrument not found
            InsufficientBudget: If payment amount exceeds remaining budget
            PaymentSessionExpired: If payment session has expired
            InvalidPaymentInstrument: If payment instrument is invalid or inactive
            PaymentError: If API call fails
        """
        user_id = user_id.strip() if user_id else None

        if client_token is None:
            client_token = str(uuid.uuid4())

        try:
            logger.info("Processing payment of type %s for user %s", payment_type, user_id)

            params = {
                **({"userId": user_id} if user_id and not self._is_bearer_auth else {}),
                "paymentManagerArn": self._payment_manager_arn,
                "paymentSessionId": payment_session_id,
                "paymentInstrumentId": payment_instrument_id,
                "paymentType": payment_type,
                "paymentInput": payment_input,
                "clientToken": client_token,
            }
            if payment_connector_id is not None:
                params["paymentConnectorId"] = payment_connector_id

            result = self._payment_client.process_payment(**params)
            logger.info("Successfully processed payment for user %s", user_id)
            # Unwrap the nested processPayment response
            return result.get("processPayment", result)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ValidationException":
                if "budget" in message.lower() or "insufficient" in message.lower():
                    logger.error("Insufficient budget: %s", message)
                    raise InsufficientBudget(f"Insufficient budget: {message}") from e
                if "expired" in message.lower():
                    logger.error("Session expired: %s", message)
                    raise PaymentSessionExpired(f"Session expired: {message}") from e
                if "instrument" in message.lower() or "inactive" in message.lower():
                    logger.error("Invalid instrument: %s", message)
                    raise InvalidPaymentInstrument(f"Invalid instrument: {message}") from e
                if "session not found" in message.lower():
                    logger.error("PaymentSession not found: %s", message)
                    raise PaymentSessionNotFound(f"Session not found or expired: {payment_session_id}") from e

            logger.error("Failed to process payment: %s", str(e))
            raise PaymentError(f"Failed to process payment: {str(e)}") from e

    def generate_payment_header(
        self,
        payment_instrument_id: str,
        payment_session_id: str,
        payment_required_request: Dict[str, Any],
        user_id: Optional[str] = None,
        network_preferences: Optional[list[str]] = None,
        client_token: Optional[str] = None,
        payment_connector_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate a payment header for 402 payment required request.

        This method orchestrates the complete payment header generation workflow:
        1. Validates input parameters
        2. Generates or validates client_token
        3. Retrieves payment instrument details
        4. Extracts payment requirement from 402 payment required request
        5. Selects appropriate blockchain network accept header. Here is the Selection process:
            1. Filter accepts to those matching the instrument's blockchain type
            2. Use provided network_preferences or default to NETWORK_PREFERENCES from constants
            3. Pick the first network from preferences that matches a filtered accept
            4. If no match found, return the first filtered accept
        6. Processes the payment transaction
        7. Builds the final payment header (v1 or v2 format)

        Args:
            payment_instrument_id: Unique identifier for the payment instrument
            payment_session_id: Unique identifier for the payment session
            payment_required_request: Dictionary containing 402 response with statusCode, headers, and body
            user_id: Unique identifier for the user (optional, omitted for bearer auth)
            network_preferences: Optional list of network identifiers in order of preference.
                If not provided, defaults to NETWORK_PREFERENCES from constants.
            client_token: Optional unique token for idempotency. If not provided, a new one is generated.
            payment_connector_id: Optional payment connector ID to pass to process_payment.

        Returns:
            Dictionary with header name and value (e.g., {"X-PAYMENT": "base64..."} or
            {"PAYMENT-SIGNATURE": "base64..."}) for X402 payment required request

        Raises:
            PaymentError: For validation or processing failures
            PaymentInstrumentNotFound: If instrument not found
            PaymentSessionNotFound: If session not found
            PaymentSessionExpired: If session has expired
            InsufficientBudget: If payment amount exceeds budget

        Example:
            ```python
            header = manager.generate_payment_header(
                user_id="user-123",
                payment_instrument_id="instrument-456",
                payment_session_id="session-789",
                payment_required_request={
                    "statusCode": 402,
                    "headers": {"..."},
                    "body": {...}
                },
                client_token="optional-token-123",
                network_preferences=["solana-mainnet", "eip155:8453"]
            )
            # Returns: {"X-PAYMENT": "base64..."} or {"PAYMENT-SIGNATURE": "base64..."}
            ```
        """
        user_id = user_id.strip() if user_id else None

        logger.info(
            "Generating payment header for user %s with instrument %s and session %s",
            user_id,
            payment_instrument_id,
            payment_session_id,
        )

        try:
            # Step 1: Validate input parameters (including client_token)
            self._validate_input_parameters(
                user_id,
                payment_instrument_id,
                payment_session_id,
                payment_required_request,
            )
            logger.debug("Input validation passed")

            # Step 2: Check statusCode == 402
            status_code = payment_required_request.get("statusCode")
            logger.debug("Checking 402 status code: %s", status_code)
            if status_code != 402:
                raise PaymentError(
                    f"402 Status Validation: Invalid status code - Expected statusCode 402, got {status_code}"
                )
            logger.debug("Status code validation passed")

            # Step 3: Generate client_token if not provided
            if client_token is None:
                client_token = str(uuid.uuid4())
                logger.debug("Generated new client_token: %s", client_token[:8] + "...")
            else:
                # Validate client_token is a string and not empty
                if not isinstance(client_token, str):
                    raise PaymentError("client_token is invalid - must be a string")
                if not client_token.strip():
                    raise PaymentError("client_token is invalid - cannot be empty")
                logger.debug("Using provided client_token: %s", client_token[:8] + "...")

            # Step 4: Extract X.402 payload and detect version.
            # Will have another method for MPP or any other payments protocols
            x402_payload, x402_version = self._extract_x402_payload(payment_required_request)
            logger.debug("Extracted X.402 payload version %d", x402_version)

            # Step 5: Retrieve payment instrument and extract network
            instrument = self.get_payment_instrument(
                user_id=user_id,
                payment_instrument_id=payment_instrument_id,
            )
            logger.debug("Retrieved instrument: %s", instrument)

            # Extract network from nested structure: paymentInstrumentDetails.embeddedCryptoWallet.network
            network = None
            if "paymentInstrumentDetails" in instrument:
                details = instrument.get("paymentInstrumentDetails", {})
                if "embeddedCryptoWallet" in details:
                    network = details.get("embeddedCryptoWallet", {}).get("network")

            if not network:
                raise PaymentError(
                    "Instrument Retrieval: Missing network information - "
                    "instrument details do not contain network information at "
                    "paymentInstrumentDetails.embeddedCryptoWallet.network"
                )
            logger.debug("Retrieved instrument with network: %s", network)

            # Step 6: Validate instrument network and select matching accept
            selected_accept = self._select_accept_for_instrument_network(x402_payload, network, network_preferences)
            logger.debug("Selected accept for instrument network: %s", network)

            # Step 7: Process payment
            logger.debug("Processing payment with type CRYPTO_X402")
            payment_input = {
                "cryptoX402": {
                    "version": str(x402_version),
                    "payload": selected_accept,
                }
            }

            process_payment_params = {
                "user_id": user_id,
                "payment_session_id": payment_session_id,
                "payment_instrument_id": payment_instrument_id,
                "payment_type": "CRYPTO_X402",
                "payment_input": payment_input,
                "client_token": client_token,
            }
            if payment_connector_id is not None:
                process_payment_params["payment_connector_id"] = payment_connector_id

            payment_result = self.process_payment(**process_payment_params)
            logger.debug("Payment processed successfully")

            # Extract cryptoX402 proof from payment result
            crypto_x402_proof = payment_result.get("paymentOutput", {}).get("cryptoX402", {})
            if not crypto_x402_proof:
                raise PaymentError(
                    "Payment Processing: Missing cryptoX402 in payment output - "
                    "payment result does not contain cryptoX402 proof"
                )
            logger.debug("Extracted cryptoX402 proof from payment result")

            # Step 8: Build payment header
            payment_header = self._build_payment_header(x402_version, x402_payload, selected_accept, crypto_x402_proof)
            logger.info("Successfully generated payment header for user %s", user_id)

            return payment_header

        except PaymentError:
            logger.error("Payment header generation failed: %s", str(sys.exc_info()[1]))
            raise
        except Exception as e:
            logger.error("Unexpected error during payment header generation: %s", str(e))
            raise PaymentError(f"Unexpected error: {str(e)}") from e

    def __getattr__(self, name: str):
        """Dynamically forward method calls to the PaymentClient.

        This method enables access to allowed PaymentClient methods without explicitly
        defining them. Methods are looked up on the PaymentClient instance only if they
        are in the allowed list.

        Args:
            name: The method name being accessed

        Returns:
            A callable method from the PaymentClient

        Raises:
            AttributeError: If the method doesn't exist on PaymentClient or is not allowed

        Example:
            ```python
            # Access allowed PaymentClient methods directly
            manager = PaymentManager(config)

            # These calls are forwarded to the PaymentClient
            instruments = manager.list_payment_instruments(...)
            ```
        """
        if name in self._ALLOWED_PAYMENTS_DP_METHODS and hasattr(self._payment_client, name):
            method = getattr(self._payment_client, name)
            return method

        # Method not found on client or not in allowed list
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
            f"Method not found on _payment_client or not in allowed methods. "
            f"Available methods can be found in the boto3 documentation for "
            f"'bedrock-agentcore' services."
        )

    # Network mappings for blockchain identification (normalized to lowercase)
    _ETHEREUM_NETWORKS = {
        n.lower()
        for n in {
            "eip155:8453",  # Base mainnet (low fees)
            "eip155:1",  # Ethereum mainnet
            "base",
            "eip155:42161",  # Arbitrum One
            "eip155:10",  # Optimism
            "ethereum",
            "sepolia",
            "base-sepolia",
            "eip155:84532",  # Base Sepolia (testnet)
            "eip155:11155111",  # Base Sepolia (Test)
        }
    }

    _SOLANA_NETWORKS = {
        n.lower()
        for n in {
            "solana",  # Generic Solana identifier
            "solana-mainnet",  # Solana Mainnet (simplified identifier)
            "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",  # Mainnet genesis hash (32 chars, CAIP-2)
            "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d",  # Mainnet full genesis hash (44 chars)
            "solana-devnet",  # Solana Devnet (simplified identifier)
            "solana:EtWTRABZaYq6iMfeYKouRu166VU2xqa1",  # Devnet genesis hash (32 chars, CAIP-2)
            "solana:EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG",  # Devnet full genesis hash (44 chars)
            "solana-testnet",  # Solana Testnet (simplified identifier)
            "solana:4uhcVJyU9pJkvQyS88uRDiswHXSCkY3z",  # Testnet genesis hash (32 chars, CAIP-2)
            "solana:4uhcVJyU9pJkvQyS88uRDiswHXSCkY3zQawwpjk2NsNY",  # Testnet full genesis hash (44 chars)
        }
    }

    def _validate_input_parameters(
        self,
        user_id: str,
        instrument_id: str,
        session_id: str,
        payment_required_request: Dict[str, Any],
    ) -> None:
        """Validate all input parameters for generatePaymentHeader.

        Args:
            user_id: User identifier to validate
            instrument_id: Instrument identifier to validate
            session_id: Session identifier to validate
            payment_required_request: X.402 response dictionary to validate
            client_token: Optional client token to validate

        Raises:
            PaymentError: If any parameter is invalid
        """
        if not self._is_bearer_auth:
            if not user_id or not isinstance(user_id, str) or not user_id.strip():
                raise PaymentError("Input Validation: user_id is empty - user_id must be a non-empty string")

        if not instrument_id or not isinstance(instrument_id, str) or not instrument_id.strip():
            raise PaymentError("Input Validation: instrument_id is empty - instrument_id must be a non-empty string")

        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise PaymentError("Input Validation: session_id is empty - session_id must be a non-empty string")

        if not isinstance(payment_required_request, dict) or not payment_required_request:
            raise PaymentError(
                "Input Validation: payment_required_request is invalid - "
                "payment_required_request must be a non-empty dictionary"
            )

        # Validate required fields in payment_required_request
        required_fields = {"statusCode", "headers", "body"}
        if not all(field in payment_required_request for field in required_fields):
            raise PaymentError(
                "Input Validation: 402 payment required request is missing required fields - "
                "402 payment required request must contain statusCode, headers, and body"
            )

    def _extract_x402_payload(self, payment_required_request: Dict[str, Any]) -> tuple:
        """Extract X.402 payload from payment_required_request and detect version.

        Args:
            payment_required_request: X.402 response dictionary

        Returns:
            Tuple of (x402_payload, x402_version)

        Raises:
            PaymentError: If extraction or validation fails
        """
        try:
            # Try to detect version from headers first (v2)
            headers = payment_required_request.get("headers", {})
            payment_required_header = None

            # Check for "payment-required" header (case-insensitive)
            for key, value in headers.items():
                if key.lower() == "payment-required":
                    payment_required_header = value
                    break

            if payment_required_header is not None:
                if not payment_required_header:
                    raise PaymentError("X.402 Extraction: payment-required header is present but empty")
                # v2: Decode base64 header
                try:
                    decoded = base64.b64decode(payment_required_header)
                    x402_payload = json.loads(decoded)

                    # Validate that decoded payload is a dictionary
                    if not isinstance(x402_payload, dict):
                        raise PaymentError(
                            f"X.402 Extraction: v2 payload decoded to {type(x402_payload).__name__}, "
                            f"expected a JSON object"
                        )

                    # Require x402Version field
                    if "x402Version" not in x402_payload:
                        raise PaymentError(
                            "X.402 Extraction: Missing x402Version - x402Payload must contain x402Version field"
                        )
                    try:
                        x402_version = int(x402_payload["x402Version"])
                    except (ValueError, TypeError) as ve:
                        raise PaymentError(
                            f"X.402 Extraction: Invalid x402Version '{x402_payload['x402Version']}' - "
                            f"must be an integer"
                        ) from ve
                except (ValueError, json.JSONDecodeError, binascii.Error) as e:
                    raise PaymentError(
                        f"X.402 Extraction: Failed to decode v2 payload - "
                        f"payment-required header contains invalid base64 or JSON: {str(e)}"
                    ) from e
            else:
                # v1: Extract from body
                body = payment_required_request.get("body")
                if isinstance(body, str):
                    try:
                        x402_payload = json.loads(body)
                        # Validate that decoded payload is a dictionary
                        if not isinstance(x402_payload, dict):
                            raise PaymentError(
                                f"X.402 Extraction: v1 payload decoded to {type(x402_payload).__name__}, "
                                f"expected a JSON object"
                            )
                    except json.JSONDecodeError as e:
                        raise PaymentError(
                            f"X.402 Extraction: Failed to parse v1 payload from body - "
                            f"body contains invalid JSON: {str(e)}"
                        ) from e
                elif isinstance(body, dict):
                    x402_payload = body
                else:
                    raise PaymentError(
                        "X.402 Extraction: Invalid body format - body must be a JSON string or dictionary"
                    )

                # Require x402Version field
                if "x402Version" not in x402_payload:
                    raise PaymentError(
                        "X.402 Extraction: Missing x402Version - x402Payload must contain x402Version field"
                    )
                try:
                    x402_version = int(x402_payload["x402Version"])
                except (ValueError, TypeError) as ve:
                    raise PaymentError(
                        f"X.402 Extraction: Invalid x402Version '{x402_payload['x402Version']}' - must be an integer"
                    ) from ve

            # Validate required fields
            required_fields = {"x402Version", "accepts"}
            missing_fields = required_fields - set(x402_payload.keys())
            if missing_fields:
                raise PaymentError(
                    f"X.402 Validation: Missing required fields - "
                    f"x402Payload must contain {', '.join(sorted(required_fields))}, "
                    f"but missing: {', '.join(sorted(missing_fields))}"
                )

            # Validate accepts is a list
            if not isinstance(x402_payload.get("accepts"), list):
                raise PaymentError("X.402 Validation: Invalid accepts field - accepts must be a list of accept headers")

            logger.debug("Successfully extracted X.402 payload version %d", x402_version)
            return x402_payload, x402_version

        except PaymentError:
            raise
        except Exception as e:
            raise PaymentError(f"X.402 Extraction: Unexpected error - {str(e)}") from e

    def _determine_blockchain_type(self, network: str) -> str:
        """Determine blockchain type from network identifier.

        Args:
            network: Network identifier from instrument (ETHEREUM or SOLANA)

        Returns:
            Blockchain type: "ethereum" or "solana"

        Raises:
            PaymentError: If network is not supported
        """
        network_upper = network.upper()
        if network_upper == "ETHEREUM":
            return "ethereum"
        elif network_upper == "SOLANA":
            return "solana"
        else:
            raise PaymentError(
                f"Instrument Network: Unsupported network - instrument network '{network}' is not supported. "
                f"Supported networks are ETHEREUM and SOLANA."
            )

    def _select_accept_for_instrument_network(
        self,
        x402_payload: Dict[str, Any],
        instrument_network: str,
        network_preferences: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Select appropriate accept header based on instrument network and preferences.

        Selection process:
        1. Filter accepts to those matching the instrument's blockchain type
        2. Use provided network_preferences or default to NETWORK_PREFERENCES from constants
        3. Pick the first network from preferences that matches a filtered accept
        4. If no match found, return the first filtered accept

        Args:
            x402_payload: Extracted X.402 payload
            instrument_network: Instrument network type (ETHEREUM or SOLANA)
            network_preferences: Optional list of network identifiers in order of preference.
                If not provided, defaults to NETWORK_PREFERENCES from constants.

        Returns:
            Selected accept header

        Raises:
            PaymentError: If no matching accept found for instrument network
        """
        from bedrock_agentcore.payments.constants import NETWORK_PREFERENCES

        # Determine blockchain type from instrument network
        blockchain_type = self._determine_blockchain_type(instrument_network)

        # Get the appropriate network set based on blockchain type
        if blockchain_type == "ethereum":
            supported_networks = self._ETHEREUM_NETWORKS
        else:  # solana
            supported_networks = self._SOLANA_NETWORKS

        # Step 1: Filter accepts to those matching the instrument's blockchain type
        accepts = x402_payload.get("accepts", [])
        filtered_accepts = []
        for accept in accepts:
            accept_network = accept.get("network", "").lower()
            if accept_network in supported_networks:
                filtered_accepts.append(accept)

        if not filtered_accepts:
            raise PaymentError(
                f"Accept Selection: No matching accept - No accept header found for "
                f"instrument network '{instrument_network}' in X.402 payload. "
                f"Instrument does not support the network for header generation."
            )

        # Step 2: Use provided preferences or default
        preferences = network_preferences if network_preferences is not None else NETWORK_PREFERENCES

        # Step 3: Pick the first network from preferences that matches a filtered accept
        for preferred_network in preferences:
            for accept in filtered_accepts:
                accept_network = accept.get("network", "").lower()
                if accept_network == preferred_network.lower():
                    logger.debug(
                        "Selected accept for instrument network: %s using preference: %s",
                        instrument_network,
                        preferred_network,
                    )
                    return accept

        # Step 4: If no match found, return the first filtered accept
        logger.debug(
            "No preference match found, selecting first available accept for instrument network: %s",
            instrument_network,
        )
        return filtered_accepts[0]

    def _build_payment_header(
        self,
        x402_version: int,
        x402_payload: Dict[str, Any],
        selected_accept: Dict[str, Any],
        crypto_x402_proof: Dict[str, Any],
    ) -> Dict[str, str]:
        """Build the final payment header from cryptoX402 proof.

        Args:
            x402_version: X.402 version (1 or 2)
            x402_payload: Extracted X.402 payload
            selected_accept: Selected accept header
            crypto_x402_proof: CryptoX402 proof from payment result

        Returns:
            Dictionary with header name and encoded value
            (e.g., {"X-PAYMENT": "base64..."} or {"PAYMENT-SIGNATURE": "base64..."})

        Raises:
            PaymentError: If header building fails
        """
        try:
            if x402_version == 1:
                # v1: X-PAYMENT format
                x402_header = {
                    "x402Version": 1,
                    "scheme": selected_accept.get("scheme"),
                    "network": selected_accept.get("network"),
                    "payload": crypto_x402_proof.get("payload"),
                }
                header_json = json.dumps(x402_header)
                encoded = base64.b64encode(header_json.encode()).decode()
                return {"X-PAYMENT": encoded}

            elif x402_version == 2:
                # v2: PAYMENT-SIGNATURE format
                payment_signature = {
                    "x402Version": 2,
                    "resource": x402_payload.get("resource"),
                    "accepted": selected_accept,
                    "extension": x402_payload.get("extension", {}),
                    "payload": crypto_x402_proof.get("payload"),
                }
                header_json = json.dumps(payment_signature)
                encoded = base64.b64encode(header_json.encode()).decode()
                return {"PAYMENT-SIGNATURE": encoded}

            else:
                raise PaymentError(
                    f"Header Building: Unsupported X.402 version - "
                    f"x402Version {x402_version} is not supported. "
                    f"Supported versions: 1, 2"
                )

        except PaymentError:
            raise
        except Exception as e:
            raise PaymentError(f"Header Building: Encoding failed - {str(e)}") from e
