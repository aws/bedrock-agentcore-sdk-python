"""Shared deterministic error messages for payment exceptions.

These messages are designed to be shown to LLMs via tool results. They instruct
the model not to retry and to inform the user of the specific issue.

Used by: LangGraph middleware, and available for any future plugin integration.
"""

from typing import Dict, Type

from bedrock_agentcore.payments.manager import (
    InsufficientBudget,
    PaymentError,
    PaymentInstrumentConfigurationRequired,
    PaymentInstrumentNotFound,
    PaymentSessionConfigurationRequired,
    PaymentSessionExpired,
    PaymentSessionNotFound,
)

# Maps exception types to deterministic, LLM-instructive messages.
PAYMENT_ERROR_MESSAGES: Dict[Type[Exception], str] = {
    PaymentInstrumentConfigurationRequired: (
        "No payment instrument configured. Do not retry this call. "
        "Inform the user they need to configure a payment instrument before making paid requests."
    ),
    PaymentSessionConfigurationRequired: (
        "No payment session configured. Do not retry this call. "
        "Inform the user they need to create a payment session before making paid requests."
    ),
    PaymentInstrumentNotFound: (
        "Payment instrument not found. Do not retry this call. "
        "Inform the user their payment instrument ID is invalid or has been deleted."
    ),
    PaymentSessionNotFound: (
        "Payment session not found. Do not retry this call. "
        "Inform the user their payment session ID is invalid or has expired."
    ),
    PaymentSessionExpired: (
        "Payment session has expired. Do not retry this call. "
        "Inform the user they need to create a new payment session."
    ),
    InsufficientBudget: (
        "Insufficient budget. The payment amount exceeds the remaining session limit. "
        "Do not retry this call. Inform the user they need to increase their session budget "
        "or create a new session with higher limits."
    ),
}


def get_payment_error_message(exception: Exception) -> str:
    """Get the deterministic error message for a payment exception.

    Looks up the exception type in the message map. Falls back to a generic
    message that includes the exception string for unrecognized types.

    Args:
        exception: The payment exception.

    Returns:
        Human/LLM-readable error message string.
    """
    msg = PAYMENT_ERROR_MESSAGES.get(type(exception))
    if msg is not None:
        return msg
    if isinstance(exception, PaymentError):
        return (
            f"Payment processing failed ({exception}). "
            "Do not retry this call. Inform the user that payment could not be completed."
        )
    return (
        f"An unexpected error occurred during payment processing ({exception}). "
        "Do not retry this call. Inform the user that payment could not be completed."
    )
