"""Runtime injection of MPP shapes into the bedrock-agentcore service model.

The MPP additions to ``ProcessPayment`` (the ``mpp`` members of the ``PaymentInput`` /
``PaymentOutput`` unions and the ``MPP`` value on the ``PaymentType`` enum) reached the
service before they reached a public botocore release. Without them, botocore rejects an
MPP payment **client-side**, before the request is ever signed::

    Unknown parameter in paymentInput: "mpp", must be one of: cryptoX402

This module adds those shapes to the in-memory service model of a single boto3 client.
It is idempotent and self-limiting: once a botocore release ships the shapes natively,
:func:`patch_process_payment_model` detects them and does nothing. It never writes to
the installed botocore package, and it only ever adds shapes — no existing member,
enum value, or constraint is modified, so the x402 path is unaffected.

The shape definitions mirror the service's Smithy model.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Shape names as they appear in the bedrock-agentcore service-2.json model.
_PAYMENT_TYPE = "PaymentType"
_PAYMENT_INPUT = "PaymentInput"
_PAYMENT_OUTPUT = "PaymentOutput"
_MPP_INPUT = "MppPaymentInput"
_MPP_OUTPUT = "MppPaymentOutput"
_MPP_CREDENTIAL = "MppPaymentCredential"
_MPP_VERSION = "MppVersion"
_MPP_HEADER = "WwwAuthenticateHeader"
_MPP_HEADER_LIST = "WwwAuthenticateHeaderList"
_MPP_SELECTED_ID = "MppSelectedPaymentId"

_MPP_ENUM_VALUE = "MPP"


def _mpp_shapes() -> Dict[str, Dict[str, Any]]:
    """Build the MPP shape definitions to merge into the service model.

    Mirrors AmazonBedrockAgentCorePaymentsDataPlaneModel's processPayment.smithy.

    Returns:
        Mapping of shape name to its JSON model definition.
    """
    return {
        _MPP_VERSION: {
            "type": "string",
            "min": 1,
            "max": 10,
            "pattern": "^[0-9]+$",
            "documentation": "<p>Protocol version identifier — bare numeric string.</p>",
        },
        _MPP_HEADER: {
            "type": "string",
            "min": 1,
            "max": 16384,
            "documentation": "<p>A raw WWW-Authenticate: Payment header value from a 402 response.</p>",
        },
        _MPP_HEADER_LIST: {
            "type": "list",
            "member": {"shape": _MPP_HEADER},
            "min": 1,
            "max": 1,
            "documentation": "<p>The raw WWW-Authenticate header value to fulfill.</p>",
        },
        _MPP_SELECTED_ID: {
            "type": "string",
            "min": 1,
            "max": 512,
            "documentation": "<p>The id of the challenge that was paid.</p>",
        },
        _MPP_CREDENTIAL: {
            "type": "string",
            "min": 1,
            "max": 32768,
            "sensitive": True,
            "documentation": "<p>Ready-to-send Authorization: Payment header value.</p>",
        },
        _MPP_INPUT: {
            "type": "structure",
            "required": ["version", "wwwAuthenticateHeaders"],
            "members": {
                "version": {
                    "shape": _MPP_VERSION,
                    "documentation": "<p>The MPP protocol version.</p>",
                },
                "wwwAuthenticateHeaders": {
                    "shape": _MPP_HEADER_LIST,
                    "documentation": "<p>The raw WWW-Authenticate: Payment header value(s), verbatim.</p>",
                },
                "buyerPaysGasFees": {
                    "shape": "Boolean",
                    "documentation": (
                        "<p>Authorizes ACP to sign a payment whose blockchain network (gas) fees are "
                        "charged to the buyer's own wallet, on top of the payment amount.</p>"
                    ),
                },
            },
            "documentation": "<p>The input for an MPP payment.</p>",
        },
        _MPP_OUTPUT: {
            "type": "structure",
            "required": ["version", "selectedPaymentId", "paymentCredential"],
            "members": {
                "version": {
                    "shape": _MPP_VERSION,
                    "documentation": "<p>The MPP protocol version.</p>",
                },
                "selectedPaymentId": {
                    "shape": _MPP_SELECTED_ID,
                    "documentation": "<p>The id of the challenge that was paid.</p>",
                },
                "paymentCredential": {
                    "shape": _MPP_CREDENTIAL,
                    "documentation": "<p>Ready-to-send Authorization header value.</p>",
                },
            },
            "documentation": "<p>The output from an MPP payment.</p>",
        },
    }


def _model_supports_mpp(shapes: Dict[str, Any]) -> bool:
    """Check whether the model already exposes the MPP union members natively."""
    payment_input = shapes.get(_PAYMENT_INPUT) or {}
    return "mpp" in (payment_input.get("members") or {})


def patch_process_payment_model(client: Any) -> bool:
    """Add the MPP shapes to *client*'s service model if botocore lacks them.

    Mutates the ``service-2.json`` dict backing this client's service model, then
    clears the cached shape objects so botocore rebuilds them on next use. Only this
    client's model object is affected; the on-disk botocore data files are untouched.

    Args:
        client: A boto3 ``bedrock-agentcore`` client.

    Returns:
        True if shapes were injected, False if the model already supported MPP or the
        model could not be patched.
    """
    try:
        service_model = client.meta.service_model
        shapes = service_model._shape_resolver._shape_map
    except AttributeError:
        # botocore internals moved. MPP will fail loudly at call time with a clear
        # ParamValidationError rather than being silently mis-sent, so degrade quietly.
        logger.debug("MPP: could not access the service model shape map; skipping model patch")
        return False

    if _model_supports_mpp(shapes):
        logger.debug("MPP: botocore model already supports MPP; no patch needed")
        return False

    for name, definition in _mpp_shapes().items():
        shapes.setdefault(name, definition)

    # Add the union members that reference the new shapes.
    payment_input = shapes.get(_PAYMENT_INPUT)
    if isinstance(payment_input, dict) and isinstance(payment_input.get("members"), dict):
        payment_input["members"].setdefault(
            "mpp",
            {"shape": _MPP_INPUT, "documentation": "<p>Input for an MPP payment.</p>"},
        )

    payment_output = shapes.get(_PAYMENT_OUTPUT)
    if isinstance(payment_output, dict) and isinstance(payment_output.get("members"), dict):
        payment_output["members"].setdefault(
            "mpp",
            {"shape": _MPP_OUTPUT, "documentation": "<p>Output from an MPP payment.</p>"},
        )

    # Extend the paymentType enum. botocore does not validate enum values on
    # serialization, but keeping the model accurate avoids surprising introspection.
    payment_type = shapes.get(_PAYMENT_TYPE)
    if isinstance(payment_type, dict) and isinstance(payment_type.get("enum"), list):
        if _MPP_ENUM_VALUE not in payment_type["enum"]:
            payment_type["enum"].append(_MPP_ENUM_VALUE)

    # Drop cached Shape instances so the resolver rebuilds them from the updated map.
    try:
        client.meta.service_model._shape_resolver._shape_cache.clear()
    except AttributeError:
        logger.debug("MPP: no shape cache to clear")

    logger.debug("MPP: injected MPP shapes into the bedrock-agentcore service model")
    return True
