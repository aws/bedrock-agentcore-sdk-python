"""MPP (Machine Payments Protocol) challenge parsing and selection.

MPP is an open protocol for machine-to-machine payments (https://mpp.dev). A server
that requires payment answers with ``402 Payment Required`` and one or more
``WWW-Authenticate: Payment ...`` challenges. The client picks a challenge it can
satisfy, pays it, and retries the request with an ``Authorization: Payment <token>``
credential.

The AgentCore Payments ``ProcessPayment`` API fulfills exactly one challenge per call
(``WwwAuthenticateHeaderList`` is constrained to a single entry), so this module
provides the selection logic that reduces a list of advertised challenges to the one
the caller's payment instrument can pay — mirroring how x402 accept headers are
narrowed by network preference.

Only the ``charge`` intent is implemented, across the ``evm``, ``tempo`` and ``solana``
payment methods.

Parsing is deliberately lenient about unknown auth-params: per the MPP specification,
"Unknown parameters must be ignored by clients". The raw header value is always
preserved verbatim so the exact base64url ``request``/``opaque`` bytes that the
challenge HMAC binds to are forwarded to the service unchanged.
"""

import base64
import binascii
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .constants import (
    MPP_AUTH_SCHEME,
    MPP_METHOD_BLOCKCHAIN,
    MPP_SOLANA_DEFAULT_NETWORK,
    MPP_SOLANA_NETWORK_ALIASES,
    MPP_SUPPORTED_INTENT,
    NETWORK_PREFERENCES,
)

logger = logging.getLogger(__name__)

# Header names that may carry MPP challenges on a 402 response.
_WWW_AUTHENTICATE = "www-authenticate"
_REQUIRED_CHALLENGE_FIELDS = ("id", "realm", "method", "intent", "request")
_BASE64URL_PATTERN = re.compile(r"[A-Za-z0-9_-]+\Z")


class MppChallengeSelectionError(Exception):
    """Raised when no advertised MPP challenge can be satisfied.

    Defined here rather than in ``manager`` to keep this module import-cycle free;
    ``PaymentManager`` re-raises it as a ``PaymentError``.
    """


def _split_outside_quotes(value: str, delimiter: str = ",") -> List[str]:
    """Split *value* on *delimiter*, ignoring delimiters inside quoted strings.

    RFC 9110 auth-param values may be quoted strings containing commas, so a naive
    ``str.split(",")`` would corrupt them.

    Args:
        value: The string to split.
        delimiter: Single character to split on.

    Returns:
        List of parts with surrounding whitespace stripped. Empty parts are dropped.
    """
    parts: List[str] = []
    current: List[str] = []
    in_quotes = False
    escaped = False

    for char in value:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\" and in_quotes:
            current.append(char)
            escaped = True
            continue
        if char == '"':
            in_quotes = not in_quotes
            current.append(char)
            continue
        if char == delimiter and not in_quotes:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)

    parts.append("".join(current).strip())
    return [p for p in parts if p]


def _unquote(value: str) -> str:
    """Remove surrounding double quotes from an auth-param value and unescape it."""
    value = value.strip()
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        inner = value[1:-1]
        return inner.replace('\\"', '"').replace("\\\\", "\\")
    return value


def _parse_auth_param(part: str) -> Optional[tuple]:
    """Parse a single ``key=value`` auth-param.

    Args:
        part: The auth-param text, e.g. ``id="aB3cDeF4"``.

    Returns:
        ``(lowercased_key, unquoted_value)``, or None if *part* is not a ``key=value`` pair.
    """
    if "=" not in part:
        return None
    key, _, raw_value = part.partition("=")
    key = key.strip().lower()
    if not key:
        return None
    return key, _unquote(raw_value)


def _starts_new_auth_scheme(part: str) -> bool:
    """Check whether *part* begins a new (non-Payment) auth-scheme.

    A ``WWW-Authenticate`` field may list several challenges across different schemes,
    separated by the same commas that separate auth-params. Per RFC 9110 a challenge
    opens with a bare scheme token, optionally followed by its first auth-param —
    ``Bearer`` or ``Bearer realm="x"`` — whereas an auth-param belonging to the current
    challenge is just ``key=value``. The distinguishing feature is whitespace before
    the ``=``: a scheme token precedes its first param, so the text left of ``=``
    contains a space.

    Args:
        part: One comma-separated segment of the header value.

    Returns:
        True if *part* opens a scheme other than ``Payment``.
    """
    stripped = part.strip()
    if not stripped:
        return False

    head = stripped.partition("=")[0]
    # `key=value` -> no whitespace in the key. `Scheme key=value` -> whitespace.
    # A bare token with no `=` at all is also a scheme (e.g. a valueless `Negotiate`).
    if "=" in stripped and not head.strip().count(" "):
        return False

    token = stripped.split()[0]
    return token.lower() != MPP_AUTH_SCHEME.lower()


def parse_www_authenticate(header_value: str) -> List[Dict[str, Any]]:
    """Parse ``WWW-Authenticate`` header value(s) into MPP challenges.

    A single header field value may advertise more than one challenge, and servers
    commonly emit one ``WWW-Authenticate`` line per payment option. Both forms are
    handled: the value is split on the ``Payment`` auth-scheme token, and each
    resulting group is parsed into auth-params.

    Non-``Payment`` schemes (e.g. ``Bearer``) are ignored.

    Args:
        header_value: Raw header value, possibly containing several challenges.

    Returns:
        List of valid challenge dicts. Each contains the parsed auth-params (``id``,
        ``realm``, ``method``, ``intent``, ``request``, ``expires``, ``opaque``, plus
        any unknown params) and a ``raw`` key holding the verbatim single-challenge
        header value to forward to ProcessPayment.
    """
    if not header_value or not isinstance(header_value, str):
        return []

    challenges: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    current_parts: List[str] = []
    scheme_prefix = MPP_AUTH_SCHEME.lower() + " "

    def _flush() -> None:
        if current is not None and current_parts:
            current["raw"] = f"{MPP_AUTH_SCHEME} " + ", ".join(current_parts)
            if _is_valid_challenge(current):
                challenges.append(current)
            else:
                logger.debug("MPP: discarded malformed challenge id=%s", current.get("id"))

    for part in _split_outside_quotes(header_value):
        lowered = part.lower()
        # A part that begins with the scheme name starts a new challenge.
        if lowered.startswith(scheme_prefix) or lowered == MPP_AUTH_SCHEME.lower():
            _flush()
            current = {}
            current_parts = []
            remainder = part[len(MPP_AUTH_SCHEME) :].strip()
            if remainder:
                parsed = _parse_auth_param(remainder)
                if parsed:
                    current[parsed[0]] = parsed[1]
                    current_parts.append(remainder)
            continue

        # A part that opens some other auth-scheme (e.g. `Bearer realm="x"`) ends the
        # current challenge. Without this, that scheme's auth-params would be absorbed
        # into the MPP challenge — and into its `raw` value, corrupting the exact bytes
        # the challenge HMAC binds to.
        if _starts_new_auth_scheme(part):
            _flush()
            current = None
            current_parts = []
            continue

        if current is None:
            # Auth-params belonging to a non-Payment scheme, or a malformed value.
            continue

        parsed = _parse_auth_param(part)
        if parsed:
            current[parsed[0]] = parsed[1]
            current_parts.append(part)

    _flush()
    return challenges


def extract_challenges(payment_required_request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect all MPP challenges from a 402 payment required request.

    Scans headers case-insensitively for ``WWW-Authenticate``. A header value may be a
    single string or a list of strings (some HTTP clients expose repeated headers as a
    list); both are supported.

    Args:
        payment_required_request: Dict with ``statusCode``, ``headers`` and ``body``.

    Returns:
        List of parsed challenge dicts, in the order advertised by the server.
    """
    headers = payment_required_request.get("headers") or {}
    if not isinstance(headers, dict):
        return []

    challenges: List[Dict[str, Any]] = []
    for key, value in headers.items():
        if not isinstance(key, str) or key.lower() != _WWW_AUTHENTICATE:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for entry in values:
            challenges.extend(parse_www_authenticate(entry))

    return challenges


def is_mpp_payment_required(payment_required_request: Dict[str, Any]) -> bool:
    """Check whether a 402 response advertises at least one MPP ``Payment`` challenge.

    Used to route a 402 to the MPP code path instead of the x402 one.

    Args:
        payment_required_request: Dict with ``statusCode``, ``headers`` and ``body``.

    Returns:
        True if a parseable ``WWW-Authenticate: Payment`` challenge is present.
    """
    if not isinstance(payment_required_request, dict):
        return False
    return bool(extract_challenges(payment_required_request))


def decode_challenge_request(challenge: Dict[str, Any]) -> Dict[str, Any]:
    """Decode a challenge's ``request`` auth-param into its JSON object.

    The ``request`` param is JCS-canonicalized JSON, base64url-encoded without padding.

    Args:
        challenge: A parsed challenge dict.

    Returns:
        The decoded request object, or an empty dict if absent or undecodable.
        Decoding never raises — selection treats an undecodable request as an
        unusable challenge rather than failing the whole call.
    """
    encoded = challenge.get("request")
    if not encoded or not isinstance(encoded, str) or not _BASE64URL_PATTERN.fullmatch(encoded):
        return {}

    # base64url without padding — restore padding before decoding.
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        decoded = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
        request = json.loads(decoded)
    except (binascii.Error, ValueError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug("MPP: failed to decode challenge request for id=%s: %s", challenge.get("id"), e)
        return {}

    if not isinstance(request, dict):
        logger.debug("MPP: challenge request for id=%s decoded to a non-object", challenge.get("id"))
        return {}
    return request


def _is_valid_challenge(challenge: Dict[str, Any]) -> bool:
    """Return whether a parsed challenge satisfies the core MPP requirements."""
    for field in _REQUIRED_CHALLENGE_FIELDS:
        value = challenge.get(field)
        if not isinstance(value, str) or not value.strip():
            return False

    method = challenge["method"]
    if not method.isascii() or not method.isalpha() or method.lower() != method:
        return False

    return bool(decode_challenge_request(challenge))


def challenge_network(challenge: Dict[str, Any]) -> Optional[str]:
    """Derive the network identifier for a challenge, for preference ordering.

    EVM-family challenges carry ``methodDetails.chainId``, which maps to the CAIP-2
    style ``eip155:<chainId>`` identifiers used in ``NETWORK_PREFERENCES``. Solana
    challenges carry an optional ``methodDetails.network`` (mainnet/devnet/localnet),
    defaulting to mainnet.

    Args:
        challenge: A parsed challenge dict.

    Returns:
        Lowercased network identifier, or None if it cannot be determined.
    """
    method = (challenge.get("method") or "").lower()
    request = decode_challenge_request(challenge)
    method_details = request.get("methodDetails")
    if not isinstance(method_details, dict):
        method_details = {}

    if MPP_METHOD_BLOCKCHAIN.get(method) == "ETHEREUM":
        chain_id = method_details.get("chainId")
        if isinstance(chain_id, bool) or chain_id is None:
            return None
        try:
            return f"eip155:{int(chain_id)}"
        except (TypeError, ValueError):
            return None

    if MPP_METHOD_BLOCKCHAIN.get(method) == "SOLANA":
        network = method_details.get("network")
        if not isinstance(network, str) or not network.strip():
            network = MPP_SOLANA_DEFAULT_NETWORK
        return MPP_SOLANA_NETWORK_ALIASES.get(network.strip().lower())

    return None


def _parse_expires(value: Any) -> Optional[datetime]:
    """Parse an RFC 3339 ``expires`` auth-param into a timezone-aware datetime."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    # datetime.fromisoformat on Python 3.10 does not accept a trailing "Z".
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        logger.debug("MPP: unparseable expires value %r", value)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def is_expired(challenge: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    """Check whether a challenge's ``expires`` auth-param is in the past.

    A challenge with no ``expires`` param, or an unparseable one, is treated as not
    expired — the service is the authority on validity, and discarding a challenge we
    merely failed to parse would be worse than attempting it.

    Args:
        challenge: A parsed challenge dict.
        now: Reference time. Defaults to the current UTC time.

    Returns:
        True if the challenge has definitively expired.
    """
    expires = _parse_expires(challenge.get("expires"))
    if expires is None:
        return False
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return expires <= reference


def select_challenge(
    challenges: List[Dict[str, Any]],
    instrument_network: str,
    network_preferences: Optional[List[str]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Select the one challenge to pay from those the server advertised.

    Selection process (mirrors x402 accept selection):

    1. Drop challenges whose ``intent`` is not ``charge`` — only charge is implemented.
    2. Drop challenges that have definitively expired.
    3. Keep only challenges whose ``method`` the instrument's blockchain can satisfy
       (``evm``/``tempo`` require an ETHEREUM instrument, ``solana`` a SOLANA one).
    4. Order the survivors by ``network_preferences`` (default ``NETWORK_PREFERENCES``),
       deriving each challenge's network from its decoded ``request``.
    5. Tiebreak on the soonest expiry, then on server order.

    Args:
        challenges: Parsed challenges, in the order the server advertised them.
        instrument_network: The instrument's network (``ETHEREUM`` or ``SOLANA``).
        network_preferences: Optional network identifiers, most preferred first.
            Defaults to ``NETWORK_PREFERENCES``.
        now: Reference time for expiry checks. Defaults to the current UTC time.

    Returns:
        The selected challenge dict.

    Raises:
        MppChallengeSelectionError: If no advertised challenge can be satisfied.
    """
    challenges = [challenge for challenge in challenges if _is_valid_challenge(challenge)]
    if not challenges:
        raise MppChallengeSelectionError(
            "MPP Challenge Selection: No challenges - the 402 response contained no "
            "valid 'WWW-Authenticate: Payment' challenge."
        )

    blockchain = (instrument_network or "").strip().upper()
    if blockchain not in {"ETHEREUM", "SOLANA"}:
        raise MppChallengeSelectionError(
            f"MPP Challenge Selection: Unsupported instrument network '{instrument_network}'. "
            f"Supported networks are ETHEREUM and SOLANA."
        )

    # Step 1 & 2: intent and expiry filters.
    reference = now or datetime.now(timezone.utc)
    candidates = []
    skipped_intents: List[str] = []
    expired_count = 0
    for challenge in challenges:
        intent = challenge["intent"].strip().lower()
        if intent != MPP_SUPPORTED_INTENT:
            skipped_intents.append(intent)
            continue
        if is_expired(challenge, reference):
            expired_count += 1
            continue
        candidates.append(challenge)

    if not candidates:
        intent_detail = f": {', '.join(sorted(set(skipped_intents)))}" if skipped_intents else ""
        raise MppChallengeSelectionError(
            f"MPP Challenge Selection: No usable challenge - all {len(challenges)} advertised "
            f"challenge(s) were rejected ({expired_count} expired, "
            f"{len(skipped_intents)} with unsupported intent{intent_detail}). "
            f"Only the '{MPP_SUPPORTED_INTENT}' intent is supported."
        )

    # Step 3: keep only methods the instrument can satisfy.
    supported = []
    seen_methods: List[str] = []
    for challenge in candidates:
        method = (challenge.get("method") or "").strip().lower()
        if method:
            seen_methods.append(method)
        if MPP_METHOD_BLOCKCHAIN.get(method) == blockchain:
            supported.append(challenge)

    if not supported:
        raise MppChallengeSelectionError(
            f"MPP Challenge Selection: No matching challenge - no advertised payment method "
            f"can be satisfied by an instrument on network '{blockchain}'. "
            f"Advertised methods: {', '.join(sorted(set(seen_methods))) or 'none'}. "
            f"Supported methods: {', '.join(sorted(MPP_METHOD_BLOCKCHAIN))}."
        )

    # Step 4: order by network preference.
    preferences = network_preferences if network_preferences is not None else NETWORK_PREFERENCES
    normalized_preferences = [p.lower() for p in preferences]

    def sort_key(indexed):
        index, challenge = indexed
        network = challenge_network(challenge)
        try:
            rank = normalized_preferences.index(network) if network else len(normalized_preferences)
        except ValueError:
            rank = len(normalized_preferences)
        # Step 5: soonest expiry first, then server order. Challenges without an
        # expiry sort last among equals so bounded offers are taken first.
        expires = _parse_expires(challenge.get("expires"))
        has_expiry = 0 if expires is not None else 1
        expiry_ts = expires.timestamp() if expires is not None else 0.0
        return (rank, has_expiry, expiry_ts, index)

    selected = min(enumerate(supported), key=sort_key)[1]
    logger.debug(
        "MPP: selected challenge id=%s method=%s network=%s from %d candidate(s)",
        selected.get("id"),
        selected.get("method"),
        challenge_network(selected),
        len(supported),
    )
    return selected
