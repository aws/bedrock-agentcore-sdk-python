"""AgentCorePaymentsPlugin for Strands Agents framework."""

import logging
import uuid
from typing import Any, Dict, Optional

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.plugins import Plugin, hook
from strands.tools import tool

from bedrock_agentcore.payments.manager import (
    PaymentError,
    PaymentInstrumentConfigurationRequired,
    PaymentManager,
    PaymentSessionConfigurationRequired,
)

from ..config import AgentCorePaymentsPluginConfig
from ..handlers import get_payment_handler
from .tools import validate_required_params

logger = logging.getLogger(__name__)


class AgentCorePaymentsPlugin(Plugin):
    """Plugin for handling X402 payment requirements and providing payment tools in Strands Agents.

    This plugin provides three tools for querying payment information:
    - getPaymentInstrument: Retrieve details about a specific payment instrument
    - listPaymentInstruments: List all payment instruments for a user
    - getPaymentSession: Retrieve details about a specific payment session

    The plugin also intercepts tool calls and responses to handle HTTP 402 Payment Required
    responses by processing X402 payment requirements and retrying requests with
    appropriate payment credentials. Payment processing is controlled by the auto_payment
    configuration flag (default: True).

    Attributes:
        name: Plugin identifier ("agent-core-payments-plugin")
        MAX_PAYMENT_RETRIES: Maximum number of payment retry attempts per tool use (3)
    """

    name = "agent-core-payments-plugin"
    MAX_PAYMENT_RETRIES = 3  # Maximum number of payment retry attempts per tool use

    def __init__(self, config: AgentCorePaymentsPluginConfig):
        """Initialize the payment plugin.

        Args:
            config: Configuration for the payment plugin

        Raises:
            ValueError: If config is invalid
        """
        super().__init__()
        self.config = config
        self.payment_manager: Optional[PaymentManager] = None
        logger.info("Initialized AgentCorePaymentsPlugin")

    def init_agent(self, agent) -> None:
        """Initialize plugin with agent.

        This method initializes the PaymentManager with the configured ARN and region.

        Args:
            agent: The Strands Agent instance

        Raises:
            RuntimeError: If PaymentManager initialization fails
        """
        logger.info(
            "Initializing AgentCorePaymentsPlugin with agent - ARN: %s, Region: %s",
            self.config.payment_manager_arn,
            self.config.region or "default",
        )

        try:
            # Initialize PaymentManager
            self.payment_manager = PaymentManager(
                payment_manager_arn=self.config.payment_manager_arn,
                region_name=self.config.region,
                agent_name=self.config.agent_name,
                bearer_token=self.config.bearer_token,
                token_provider=self.config.token_provider,
            )
            logger.info("PaymentManager initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize PaymentManager: %s", str(e))
            raise RuntimeError(f"Failed to initialize PaymentManager: {str(e)}") from e

    @hook
    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Handle before tool call event.

        This checks for any stored payment failures from the previous tool call
        and raises an interrupt to notify the agent.

        Args:
            event: The before tool call event
        """
        logger.debug("BeforeToolCallEvent: tool=%s", event.tool_use.get("name", "unknown"))

        # Check for any stored payment failures from previous tool calls
        for key, value in list(event.invocation_state.items()):
            if key.startswith("payment_failure_"):
                # Found a payment failure from a previous tool call
                failure_info = value
                tool_use_id = failure_info.get("toolUseId", "unknown")

                # Check interrupt retry limit using agent.state
                if self._check_interrupt_retry_limit(event.agent, tool_use_id):
                    logger.warning(
                        "Interrupt retry limit (%d) reached for tool %s, skipping interrupt",
                        self.config.max_interrupt_retries,
                        tool_use_id,
                    )
                    del event.invocation_state[key]
                    return

                self._increment_interrupt_retry_count(event.agent, tool_use_id)

                interrupt_name = f"payment-failure-{tool_use_id}" + str(uuid.uuid4())
                interrupt_reason = failure_info

                logger.info(
                    "Raising payment failure interrupt from stored state: %s and interrupt_reason: %s",
                    interrupt_name,
                    interrupt_reason,
                )
                event.interrupt(interrupt_name, reason=interrupt_reason)

                # Remove the stored failure after raising interrupt
                del event.invocation_state[key]
                return

    @hook
    def after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Handle after tool call event.

        This is where we intercept 402 responses and process payment requirements.
        Payment processing is controlled by the auto_payment configuration flag.

        Args:
            event: The after tool call event
        """
        logger.debug("AfterToolCallEvent: tool=%s", event.tool_use.get("name", "unknown"))

        # Check if auto_payment is disabled
        if not self.config.auto_payment:
            logger.debug(
                "auto_payment is disabled, skipping X.402 payment processing for tool: %s",
                event.tool_use.get("name", "unknown"),
            )
            return

        # Check if tool is in the payment allowlist
        if self.config.payment_tool_allowlist is not None:
            tool_name = event.tool_use.get("name", "unknown")
            if tool_name not in self.config.payment_tool_allowlist:
                logger.debug(
                    "Tool '%s' is not in payment_tool_allowlist, skipping payment processing",
                    tool_name,
                )
                return

        # Check if payment retry limit has been reached
        if self._check_payment_retry_limit(event):
            logger.warning("Payment processing retry limit has been reached. Processing skipped.")
            return

        # Check if response is a 402 Payment Required
        if not hasattr(event, "result") or event.result is None:
            return

        logger.debug("event.result: %s", event.result)

        # Note: get_payment_handler always returns a handler (tool-specific or generic fallback).
        # The generic handler will attempt to extract payment information from any tool result
        # that contains a PAYMENT_REQUIRED marker or HTTP-like response structure.
        # When payment_tool_allowlist is set, only allowlisted tools reach this point.
        tool_name = event.tool_use.get("name", "unknown")
        tool_input = event.tool_use.get("input", {})
        handler = get_payment_handler(tool_name, tool_input)

        try:
            # Extract status code from the result using the handler
            status_code = handler.extract_status_code(event.result)

            if status_code != 402:
                logger.debug("Response status code is %s, not 402, no payment processing needed.", status_code)
                return

            logger.info("Detected 402 Payment Required response from tool: %s", event.tool_use.get("name", "unknown"))

            # Increment retry count in invocation state
            self._increment_payment_retry_count(event)

            # Build payment_required_request dict using handler methods
            headers = handler.extract_headers(event.result)
            body = handler.extract_body(event.result)
            payment_required_request = {
                "statusCode": status_code,
                "headers": headers or {},
                "body": body or {},
            }

            # If we already retried with payment credentials and still got a 402,
            # this is a post-payment failure (e.g., insufficient balance, invalid signature).
            # Propagate as an interrupt instead of retrying again to avoid infinite loops.
            if self._is_post_payment_failure(event, body):
                logger.warning(
                    "Received 402 after payment retry for tool %s — treating as payment failure",
                    event.tool_use.get("name", "unknown"),
                )
                error_msg = self._extract_payment_error_message(body)
                self._store_payment_failure_state(event, PaymentError(f"Payment failed after retry: {error_msg}"))
                return

            # Validate tool input before processing payment
            tool_input = event.tool_use.get("input", {})
            if not handler.validate_tool_input(tool_input):
                logger.error("Tool input validation failed, cannot apply payment header")
                self._store_payment_failure_state(event, Exception("Tool input validation failed"))
                return

            # Process payment through PaymentManager.generate_payment_header
            payment_header_dict = self._process_payment_required_request(payment_required_request)

            # Apply payment header to tool input using the handler
            if not handler.apply_payment_header(tool_input, payment_header_dict):
                logger.error("Failed to apply payment header to tool input")
                self._store_payment_failure_state(event, Exception("Failed to apply payment header"))
                return

            # Set retry flag to re-execute the tool with payment credentials.
            # Do NOT reset the payment retry counter here — it must persist across
            # retries so that _is_post_payment_failure and _check_payment_retry_limit
            # can detect repeated 402s and break the loop.
            event.retry = True
            self._reset_interrupt_retry_count(event)
            logger.info("Set retry flag to re-execute tool with payment credentials")

        except (PaymentInstrumentConfigurationRequired, PaymentSessionConfigurationRequired) as e:
            logger.error("Payment configuration error (not retryable): %s", str(e))
            self._store_payment_failure_state(event, e)
            return
        except PaymentError as e:
            logger.error("Payment processing failed: %s", str(e))
            self._store_payment_failure_state(event, e)
            return
        except Exception as e:
            logger.error("Unexpected error during payment processing: %s", str(e))
            self._store_payment_failure_state(event, e)
            return

    def _check_payment_retry_limit(self, event: AfterToolCallEvent) -> bool:
        """Check if the payment retry limit has been reached for this tool use.

        Only checks the payment-specific retry counter (invocation_state).
        Interrupt retry limits are checked separately in before_tool_call and
        do not gate 402 payment processing.

        Args:
            event: The after tool call event

        Returns:
            True if the payment retry limit has been reached, False otherwise
        """
        tool_use_id = event.tool_use.get("toolUseId", "unknown")
        payment_retry_key = f"payment_retry_count_{tool_use_id}"
        retry_count = event.invocation_state.get(payment_retry_key, 0)

        if retry_count >= self.MAX_PAYMENT_RETRIES:
            logger.warning(
                "Tool use %s has reached maximum payment retry attempts (%d), not retrying",
                tool_use_id,
                self.MAX_PAYMENT_RETRIES,
            )
            return True

        return False

    def _increment_payment_retry_count(self, event: AfterToolCallEvent) -> None:
        """Increment the payment retry count for this tool use.

        Args:
            event: The after tool call event
        """
        tool_use_id = event.tool_use.get("toolUseId", "unknown")
        payment_retry_key = f"payment_retry_count_{tool_use_id}"
        retry_count = event.invocation_state.get(payment_retry_key, 0)

        event.invocation_state[payment_retry_key] = retry_count + 1
        logger.info(
            "Payment retry attempt %d/%d for tool use %s", retry_count + 1, self.MAX_PAYMENT_RETRIES, tool_use_id
        )

    def _is_post_payment_failure(self, event: AfterToolCallEvent, body: Optional[Dict[str, Any]]) -> bool:
        """Check if this 402 response is a failure after we already retried with payment credentials.

        A post-payment failure occurs when:
        1. We already sent a payment header (retry count > 0 before this increment), AND
        2. The 402 response body contains an error that is NOT the initial "payment required"
           (e.g., "invalid_exact_evm_insufficient_balance", "payment_rejected", etc.)

        This prevents infinite loops where the plugin keeps signing and retrying
        against a server that keeps rejecting the payment for non-retryable reasons.

        Args:
            event: The after tool call event
            body: The extracted response body (may be None)

        Returns:
            True if this is a post-payment failure that should be propagated as an interrupt
        """
        tool_use_id = event.tool_use.get("toolUseId", "unknown")
        payment_retry_key = f"payment_retry_count_{tool_use_id}"
        # retry count was already incremented before this check, so > 1 means
        # we already attempted at least one payment retry
        retry_count = event.invocation_state.get(payment_retry_key, 0)

        if retry_count <= 1:
            return False

        # If the body contains an error field that is NOT the initial "payment required",
        # this is a post-payment failure
        if body and isinstance(body, dict):
            error = body.get("error", "")
            if isinstance(error, str) and error.lower() not in ("", "payment required"):
                logger.info(
                    "Post-payment failure detected for tool %s: error=%s (retry_count=%d)",
                    tool_use_id,
                    error,
                    retry_count,
                )
                return True

        return False

    @staticmethod
    def _extract_payment_error_message(body: Optional[Dict[str, Any]]) -> str:
        """Extract a human-readable error message from a 402 response body.

        Args:
            body: The extracted response body (may be None)

        Returns:
            Error message string, or "unknown error" if not extractable
        """
        if body and isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, str) and error:
                return error
        return "unknown error"

    def _store_payment_failure_state(self, event: AfterToolCallEvent, exception: Exception) -> None:
        """Store payment failure information in invocation state for agent to handle.

        Args:
            event: The after tool call event
            exception: The exception that caused the payment failure
        """
        import time

        tool_use_id = event.tool_use.get("toolUseId", "unknown")
        tool_name = event.tool_use.get("name", "unknown")

        # Store payment failure state in invocation_state
        payment_failure_key = f"payment_failure_{tool_use_id}"
        event.invocation_state[payment_failure_key] = {
            "tool": tool_name,
            "toolUseId": tool_use_id,
            "exceptionType": type(exception).__name__,
            "exceptionMessage": str(exception),
            "retryAttempt": event.invocation_state.get(f"payment_retry_count_{tool_use_id}", 0),
            "maxRetries": self.MAX_PAYMENT_RETRIES,
            "timestamp": time.time(),
        }

        logger.info("Stored payment failure state for tool use %s: %s", tool_use_id, type(exception).__name__)

    def _check_interrupt_retry_limit(self, agent, tool_use_id: str) -> bool:
        """Check if interrupt retry limit has been reached for a tool use.

        Uses agent.state to persist the count across interrupt cycles.

        Args:
            agent: The Strands Agent instance.
            tool_use_id: The tool use ID to check.

        Returns:
            True if the limit has been reached, False otherwise.
        """
        if not agent or self.config.max_interrupt_retries <= 0:
            return True

        state_key = f"payment_interrupt_retry_{tool_use_id}"
        current_count = agent.state.get(state_key) or 0
        return current_count >= self.config.max_interrupt_retries

    def _increment_interrupt_retry_count(self, agent, tool_use_id: str) -> None:
        """Increment the interrupt retry count for a tool use in agent.state.

        Args:
            agent: The Strands Agent instance.
            tool_use_id: The tool use ID to increment the count for.
        """
        if not agent:
            return

        state_key = f"payment_interrupt_retry_{tool_use_id}"
        current_count = agent.state.get(state_key) or 0
        agent.state.set(state_key, current_count + 1)

    def _reset_interrupt_retry_count(self, event: AfterToolCallEvent) -> None:
        """Reset the interrupt retry count after successful payment processing.

        Args:
            event: The after tool call event.
        """
        agent = event.agent
        if not agent:
            return

        tool_use_id = event.tool_use.get("toolUseId", "unknown")
        state_key = f"payment_interrupt_retry_{tool_use_id}"
        agent.state.delete(state_key)

    def _process_payment_required_request(self, payment_required_request: Dict[str, Any]) -> Dict[str, str]:
        """Process 402 payment required request and generate payment header.

        Calls PaymentManager.generate_payment_header with the 402 payment required request
        and returns the payment header dictionary.

        Args:
            payment_required_request: Dictionary containing 402 payment requirements with statusCode, headers, and body

        Returns:
            Dictionary with payment header name and value (e.g., {"X-PAYMENT": "base64..."})

        Raises:
            PaymentError: If payment processing fails
        """
        if not self.payment_manager:
            raise PaymentError("PaymentManager not initialized")

        logger.debug("Processing 402 payment required request")

        if self.config.payment_instrument_id is None:
            raise PaymentInstrumentConfigurationRequired(
                "payment_instrument_id is required for x402 payments.\n"
                "Setup steps:\n"
                "1. Create instrument: PaymentManager.create_payment_instrument(connector_id, type, details, user_id)\n"
                "2. Fund wallet: https://faucet.circle.com/ (Base Sepolia, USDC, paste wallet address)\n"
                "3. Grant signing: visit the redirectUrl from step 1\n"
                "4. Pass instrument_id in invoke payload or plugin config"
            )

        if self.config.payment_session_id is None:
            raise PaymentSessionConfigurationRequired(
                "payment_session_id is required for x402 payments.\n"
                "Create a session: PaymentManager.create_payment_session(expiry_time_in_minutes, user_id, limits)\n"
                "Then pass session_id in invoke payload or plugin config.\n"
                "Tip: use 'agentcore invoke --payment-session-id <id>' or '--auto-session' from the CLI."
            )

        # Generate payment header using PaymentManager
        payment_header_dict = self.payment_manager.generate_payment_header(
            user_id=self.config.user_id,
            payment_instrument_id=self.config.payment_instrument_id,
            payment_session_id=self.config.payment_session_id,
            payment_required_request=payment_required_request,
            network_preferences=self.config.network_preferences_config,
            client_token=str(uuid.uuid4()),
            payment_connector_id=self.config.payment_connector_id,
        )

        logger.debug("Generated payment header: %s", list(payment_header_dict.keys()))
        return payment_header_dict

    @tool
    def get_payment_instrument(
        self,
        payment_instrument_id: Optional[str] = None,
        user_id: Optional[str] = None,
        payment_connector_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve details about a specific payment instrument.

        This tool allows agents to query payment instrument information at runtime,
        enabling dynamic payment workflows and decision-making based on instrument
        properties.

        Args:
            payment_instrument_id: Payment instrument identifier (optional, falls back to plugin config)
            user_id: User identifier (optional, falls back to plugin config)
            payment_connector_id: Payment connector identifier (optional)

        Returns:
            Dictionary containing payment instrument details with the following structure:
            {
                "paymentInstrumentId": str,
                "paymentInstrumentType": str,
                "paymentInstrumentDetails": dict,
                "status": str,
                ...other fields from PaymentManager response
            }
        """
        logger.info(
            "Executing getPaymentInstrument tool for user %s, instrument %s",
            user_id,
            payment_instrument_id,
        )

        try:
            # Ensure PaymentManager is initialized
            if not self.payment_manager:
                raise PaymentError("PaymentManager not initialized")

            resolved_instrument_id = (
                payment_instrument_id.strip() if payment_instrument_id else None
            ) or self.config.payment_instrument_id
            if not resolved_instrument_id:
                raise PaymentError(
                    "payment_instrument_id is not set. Provide it as a parameter or configure it in the plugin."
                )

            resolved_user_id = (user_id.strip() if user_id else None) or self.config.user_id

            # Call PaymentManager to get instrument details
            instrument_details = self.payment_manager.get_payment_instrument(
                user_id=resolved_user_id,
                payment_instrument_id=resolved_instrument_id,
                payment_connector_id=payment_connector_id,
            )

            logger.info("Successfully retrieved payment instrument %s", resolved_instrument_id)
            return instrument_details

        except Exception as e:
            logger.error(
                "Error executing getPaymentInstrument tool: %s - %s",
                type(e).__name__,
                str(e),
            )
            raise

    @tool
    def list_payment_instruments(
        self,
        user_id: Optional[str] = None,
        payment_connector_id: Optional[str] = None,
        max_results: int = 100,
        next_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List all payment instruments for a user.

        This tool allows agents to query and iterate through payment instruments,
        enabling dynamic selection and management of payment methods.

        Args:
            user_id: User identifier (optional, falls back to plugin config)
            payment_connector_id: Filter by payment connector identifier (optional)
            max_results: Maximum number of results to return (default 100)
            next_token: Pagination token for retrieving next page (optional)

        Returns:
            Dictionary containing list of instruments and optional pagination token:
            {
                "paymentInstruments": [
                    {
                        "paymentInstrumentId": str,
                        "paymentInstrumentType": str,
                        ...other instrument fields
                    },
                    ...
                ],
                "nextToken": str (optional, present if more results exist)
            }
        """
        logger.info(
            "Executing listPaymentInstruments tool for user %s (max_results=%d)",
            user_id,
            max_results,
        )

        try:
            # Validate parameters
            validation_error = validate_required_params(
                {},
                required=[],
                optional=["user_id", "payment_connector_id", "max_results", "next_token"],
            )
            if validation_error:
                logger.warning("Parameter validation failed for listPaymentInstruments: %s", validation_error)
                raise ValueError(validation_error["message"])

            # Ensure PaymentManager is initialized
            if not self.payment_manager:
                raise PaymentError("PaymentManager not initialized")

            resolved_user_id = (user_id.strip() if user_id else None) or self.config.user_id

            # Call PaymentManager to list instruments
            instruments_list = self.payment_manager.list_payment_instruments(
                user_id=resolved_user_id,
                payment_connector_id=payment_connector_id,
                max_results=max_results,
                next_token=next_token,
            )

            logger.info(
                "Successfully retrieved %d payment instruments for user %s",
                len(instruments_list.get("paymentInstruments", [])),
                user_id,
            )
            return instruments_list

        except Exception as e:
            logger.error(
                "Error executing listPaymentInstruments tool: %s - %s",
                type(e).__name__,
                str(e),
            )
            raise

    @tool
    def get_payment_instrument_balance(
        self,
        payment_instrument_id: str,
        chain: str = "BASE_SEPOLIA",
        token: str = "USDC",
        payment_connector_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get the token balance for a payment instrument on a specific blockchain.

        Args:
            payment_instrument_id: Payment instrument identifier
            chain: Blockchain chain to query (e.g., BASE_SEPOLIA, SOLANA_DEVNET)
            token: Token to query balance for (e.g., USDC)
            payment_connector_id: Payment connector identifier (optional, falls back to plugin config)
            user_id: User identifier (optional, falls back to plugin config)

        Returns:
            Dictionary containing balance information:
            {
                "paymentInstrumentId": str,
                "tokenBalance": {
                    "amount": str,
                    "chain": str,
                    "decimals": int,
                    "network": str,
                    "token": str
                }
            }
        """
        resolved_user_id = (user_id.strip() if user_id else None) or self.config.user_id
        resolved_connector_id = payment_connector_id or self.config.payment_connector_id

        logger.info("Executing getPaymentInstrumentBalance for instrument %s on %s", payment_instrument_id, chain)

        try:
            if not self.payment_manager:
                raise PaymentError("PaymentManager not initialized")

            result = self.payment_manager.get_payment_instrument_balance(
                payment_connector_id=resolved_connector_id,
                payment_instrument_id=payment_instrument_id,
                chain=chain,
                token=token,
                user_id=resolved_user_id,
            )
            return result

        except Exception as e:
            logger.error("Error executing getPaymentInstrumentBalance: %s", str(e))
            raise

    @tool
    def get_payment_session(
        self, payment_session_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve details about a specific payment session.

        This tool allows agents to query payment session information at runtime,
        enabling dynamic tracking of payment budgets and session status.

        Args:
            payment_session_id: Payment session identifier (optional, falls back to plugin config)
            user_id: User identifier (optional, falls back to plugin config)

        Returns:
            Dictionary containing payment session details with the following structure:
            {
                "paymentSessionId": str,
                "paymentManagerArn": str,
                "userId": str,
                "availableLimits": {
                    "availableSpendAmount": {
                        "value": str,
                        "currency": str
                    },
                    "updatedAt": str
                },
                "limits": {
                    "maxSpendAmount": {
                        "value": str,
                        "currency": str
                    }
                },
                "expiryTimeInMinutes": int,
                "createdAt": str,
                "updatedAt": str
            }
        """
        logger.info(
            "Executing getPaymentSession tool for user %s, session %s",
            user_id,
            payment_session_id,
        )

        try:
            # Ensure PaymentManager is initialized
            if not self.payment_manager:
                raise PaymentError("PaymentManager not initialized")

            resolved_session_id = (
                payment_session_id.strip() if payment_session_id else None
            ) or self.config.payment_session_id
            if not resolved_session_id:
                raise PaymentError(
                    "payment_session_id is not set. Provide it as a parameter or configure it in the plugin."
                )

            resolved_user_id = (user_id.strip() if user_id else None) or self.config.user_id

            # Call PaymentManager to get session details
            session_details = self.payment_manager.get_payment_session(
                user_id=resolved_user_id,
                payment_session_id=resolved_session_id,
            )

            logger.info("Successfully retrieved payment session %s", resolved_session_id)
            return session_details

        except Exception as e:
            logger.error(
                "Error executing getPaymentSession tool: %s - %s",
                type(e).__name__,
                str(e),
            )
            raise
