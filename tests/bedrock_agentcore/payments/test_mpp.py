"""Unit tests for MPP (Machine Payments Protocol) challenge parsing and selection."""

import base64
import json
from datetime import datetime, timedelta, timezone

import pytest

from bedrock_agentcore.payments.mpp import (
    MppChallengeSelectionError,
    challenge_network,
    decode_challenge_request,
    extract_challenges,
    is_expired,
    is_mpp_payment_required,
    parse_www_authenticate,
    select_challenge,
)


def b64url(obj) -> str:
    """Encode a dict as base64url JSON without padding, as MPP requires."""
    raw = json.dumps(obj, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def evm_request(chain_id=84532, amount="1000000"):
    """Build an evm-charge request object per draft-evm-charge-00."""
    return {
        "amount": amount,
        "currency": "0x036CbD53842c5426634e7929541eC2318f3dCF71",
        "recipient": "0x742d35Cc6634C0532925a3b844Bc9e7595f8fE00",
        "methodDetails": {"chainId": chain_id},
    }


def solana_request(network="devnet", amount="1000000"):
    """Build a solana-charge request object per draft-solana-charge-00."""
    details = {}
    if network is not None:
        details["network"] = network
    return {
        "amount": amount,
        "currency": "sol",
        "recipient": "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
        "methodDetails": details,
    }


def challenge_header(challenge_id, method, request_obj, intent="charge", expires=None, realm="api.example.com"):
    """Build a single 'Payment ...' WWW-Authenticate challenge value."""
    parts = [
        f'id="{challenge_id}"',
        f'realm="{realm}"',
        f'method="{method}"',
        f'intent="{intent}"',
        f'request="{b64url(request_obj)}"',
    ]
    if expires:
        parts.append(f'expires="{expires}"')
    return "Payment " + ", ".join(parts)


FUTURE = "2099-04-01T12:05:00Z"
PAST = "2020-04-01T12:05:00Z"


class TestParseWwwAuthenticate:
    """Tests for parsing WWW-Authenticate header values into challenges."""

    def test_parses_single_challenge_auth_params(self):
        header = challenge_header("aB3cDeF4gHiJkLmN", "evm", evm_request(), expires=FUTURE)

        challenges = parse_www_authenticate(header)

        assert len(challenges) == 1
        c = challenges[0]
        assert c["id"] == "aB3cDeF4gHiJkLmN"
        assert c["realm"] == "api.example.com"
        assert c["method"] == "evm"
        assert c["intent"] == "charge"
        assert c["expires"] == FUTURE
        assert c["request"]

    def test_parses_multiple_challenges_in_one_header_value(self):
        header = ", ".join(
            [
                challenge_header("evm-1", "evm", evm_request()),
                challenge_header("sol-1", "solana", solana_request()),
            ]
        )

        challenges = parse_www_authenticate(header)

        assert [c["id"] for c in challenges] == ["evm-1", "sol-1"]
        assert [c["method"] for c in challenges] == ["evm", "solana"]

    def test_preserves_raw_header_verbatim_per_challenge(self):
        """The raw value must round-trip so the challenge HMAC stays valid."""
        first = challenge_header("evm-1", "evm", evm_request())
        second = challenge_header("sol-1", "solana", solana_request())

        challenges = parse_www_authenticate(f"{first}, {second}")

        # Each challenge's raw value contains only its own auth-params.
        assert "evm-1" in challenges[0]["raw"]
        assert "sol-1" not in challenges[0]["raw"]
        assert "sol-1" in challenges[1]["raw"]
        assert "evm-1" not in challenges[1]["raw"]
        # The base64url request bytes survive unchanged.
        assert b64url(evm_request()) in challenges[0]["raw"]

    def test_ignores_non_payment_schemes(self):
        header = 'Bearer realm="example", ' + challenge_header("evm-1", "evm", evm_request())

        challenges = parse_www_authenticate(header)

        assert len(challenges) == 1
        assert challenges[0]["id"] == "evm-1"

    def test_bearer_only_yields_no_challenges(self):
        assert parse_www_authenticate('Bearer realm="example", error="invalid_token"') == []

    def test_scheme_after_payment_does_not_leak_into_challenge(self):
        """A trailing scheme's params must not be absorbed into the MPP challenge.

        Leaking them would also corrupt the ``raw`` value forwarded to the service,
        breaking the challenge HMAC.
        """
        header = challenge_header("evm-1", "evm", evm_request()) + ', Bearer realm="api", error="invalid_token"'

        challenges = parse_www_authenticate(header)

        assert len(challenges) == 1
        assert "realm" in challenges[0]  # the challenge's own realm, not Bearer's
        assert challenges[0]["realm"] == "api.example.com"
        assert "error" not in challenges[0]
        assert "Bearer" not in challenges[0]["raw"]
        assert "invalid_token" not in challenges[0]["raw"]

    @pytest.mark.parametrize("other_scheme", ['Bearer realm="api"', 'Basic realm="corp"', "Negotiate"])
    def test_trailing_schemes_are_excluded_from_raw(self, other_scheme):
        header = f"{challenge_header('evm-1', 'evm', evm_request())}, {other_scheme}"

        challenges = parse_www_authenticate(header)

        assert len(challenges) == 1
        assert challenges[0]["raw"] == challenge_header("evm-1", "evm", evm_request())

    def test_foreign_scheme_between_two_payment_challenges(self):
        header = ", ".join(
            [
                challenge_header("evm-1", "evm", evm_request()),
                'Bearer realm="api"',
                challenge_header("sol-1", "solana", solana_request()),
            ]
        )

        challenges = parse_www_authenticate(header)

        assert [c["id"] for c in challenges] == ["evm-1", "sol-1"]
        assert all("Bearer" not in c["raw"] for c in challenges)

    def test_quoted_value_containing_spaces_is_not_mistaken_for_a_scheme(self):
        """`description="fast access now"` is an auth-param, not a new scheme."""
        header = challenge_header("x", "evm", evm_request()) + ', description="fast access now"'

        challenges = parse_www_authenticate(header)

        assert len(challenges) == 1
        assert challenges[0]["description"] == "fast access now"

    def test_scheme_name_is_case_insensitive(self):
        header = challenge_header("x", "evm", evm_request()).replace("Payment ", "payment ", 1)
        challenges = parse_www_authenticate(header)

        assert len(challenges) == 1
        assert challenges[0]["id"] == "x"

    def test_auth_param_keys_are_lowercased(self):
        header = (
            challenge_header("x", "evm", evm_request())
            .replace('id="x"', 'ID="x"')
            .replace('method="evm"', 'Method="evm"')
        )
        challenges = parse_www_authenticate(header)

        assert challenges[0]["id"] == "x"
        assert challenges[0]["method"] == "evm"

    def test_commas_inside_quoted_values_are_not_split(self):
        header = challenge_header("x", "evm", evm_request()) + ', description="Premium, fast access"'

        challenges = parse_www_authenticate(header)

        assert len(challenges) == 1
        assert challenges[0]["description"] == "Premium, fast access"

    def test_escaped_quote_inside_quoted_value(self):
        header = challenge_header("x", "evm", evm_request()) + ', description="a \\"quoted\\" word"'

        challenges = parse_www_authenticate(header)

        assert challenges[0]["description"] == 'a "quoted" word'

    def test_unknown_auth_params_are_retained_not_rejected(self):
        """The spec requires clients to ignore unknown params, not fail on them."""
        header = challenge_header("x", "evm", evm_request()) + ', futureParam="v", opaque="o"'

        challenges = parse_www_authenticate(header)

        assert len(challenges) == 1
        assert challenges[0]["futureparam"] == "v"
        assert challenges[0]["opaque"] == "o"

    def test_unquoted_values_are_accepted(self):
        challenges = parse_www_authenticate(
            f"Payment id=abc, realm=api.example.com, method=evm, intent=charge, request={b64url(evm_request())}"
        )

        assert challenges[0]["id"] == "abc"
        assert challenges[0]["method"] == "evm"

    @pytest.mark.parametrize("value", ["", None, 123, [], {}])
    def test_invalid_input_returns_empty_list(self, value):
        assert parse_www_authenticate(value) == []

    def test_bare_payment_scheme_without_params_is_dropped(self):
        assert parse_www_authenticate("Payment") == []

    @pytest.mark.parametrize("missing_field", ["id", "realm", "method", "intent", "request"])
    def test_missing_required_field_is_rejected(self, missing_field):
        values = {
            "id": "x",
            "realm": "api.example.com",
            "method": "evm",
            "intent": "charge",
            "request": b64url(evm_request()),
        }
        del values[missing_field]
        header = "Payment " + ", ".join(f'{name}="{value}"' for name, value in values.items())

        assert parse_www_authenticate(header) == []

    @pytest.mark.parametrize(
        "encoded_request",
        ["!!!not-base64!!!", b64url("not json"), b64url(["not", "object"])],
    )
    def test_invalid_request_is_rejected(self, encoded_request):
        header = f'Payment id="x", realm="api.example.com", method="evm", intent="charge", request="{encoded_request}"'

        assert parse_www_authenticate(header) == []

    def test_invalid_sibling_does_not_suppress_valid_challenge(self):
        invalid = 'Payment method="evm"'
        valid = challenge_header("valid", "evm", evm_request())

        assert [challenge["id"] for challenge in parse_www_authenticate(f"{invalid}, {valid}")] == ["valid"]


class TestExtractChallenges:
    """Tests for pulling challenges out of a 402 payment required request."""

    def test_extracts_from_canonical_header_name(self):
        req = {
            "statusCode": 402,
            "headers": {"WWW-Authenticate": challenge_header("evm-1", "evm", evm_request())},
            "body": {},
        }

        assert [c["id"] for c in extract_challenges(req)] == ["evm-1"]

    @pytest.mark.parametrize("header_name", ["www-authenticate", "WWW-Authenticate", "Www-Authenticate"])
    def test_header_lookup_is_case_insensitive(self, header_name):
        req = {
            "statusCode": 402,
            "headers": {header_name: challenge_header("evm-1", "evm", evm_request())},
            "body": {},
        }

        assert len(extract_challenges(req)) == 1

    def test_extracts_from_list_valued_header(self):
        """Some HTTP clients expose repeated headers as a list."""
        req = {
            "statusCode": 402,
            "headers": {
                "www-authenticate": [
                    challenge_header("evm-1", "evm", evm_request()),
                    challenge_header("sol-1", "solana", solana_request()),
                ]
            },
            "body": {},
        }

        assert [c["id"] for c in extract_challenges(req)] == ["evm-1", "sol-1"]

    def test_no_www_authenticate_header_yields_nothing(self):
        req = {"statusCode": 402, "headers": {"content-type": "application/json"}, "body": {}}

        assert extract_challenges(req) == []

    @pytest.mark.parametrize("headers", [None, "not-a-dict", 42, []])
    def test_malformed_headers_yield_nothing(self, headers):
        assert extract_challenges({"statusCode": 402, "headers": headers, "body": {}}) == []


class TestIsMppPaymentRequired:
    """Tests for protocol detection used to route 402s to the MPP path."""

    def test_true_for_mpp_challenge(self):
        req = {
            "statusCode": 402,
            "headers": {"WWW-Authenticate": challenge_header("evm-1", "evm", evm_request())},
            "body": {},
        }

        assert is_mpp_payment_required(req) is True

    def test_false_for_x402_body_payload(self):
        """An x402 402 must not be misrouted to MPP."""
        req = {
            "statusCode": 402,
            "headers": {"content-type": "application/json"},
            "body": {"x402Version": 1, "accepts": [{"network": "base-sepolia"}]},
        }

        assert is_mpp_payment_required(req) is False

    def test_false_for_x402_v2_payment_required_header(self):
        req = {
            "statusCode": 402,
            "headers": {"payment-required": "eyJ4NDAyVmVyc2lvbiI6Mn0="},
            "body": {},
        }

        assert is_mpp_payment_required(req) is False

    def test_false_for_bearer_challenge(self):
        req = {"statusCode": 402, "headers": {"WWW-Authenticate": 'Bearer realm="x"'}, "body": {}}

        assert is_mpp_payment_required(req) is False

    @pytest.mark.parametrize("value", [None, "string", 42, []])
    def test_non_dict_input_is_false(self, value):
        assert is_mpp_payment_required(value) is False


class TestDecodeChallengeRequest:
    """Tests for decoding the base64url JCS-JSON request auth-param."""

    def test_decodes_evm_request(self):
        challenge = parse_www_authenticate(challenge_header("evm-1", "evm", evm_request(chain_id=8453)))[0]

        request = decode_challenge_request(challenge)

        assert request["amount"] == "1000000"
        assert request["methodDetails"]["chainId"] == 8453

    def test_decodes_unpadded_base64url(self):
        """MPP encodes base64url without padding; decoding must restore it."""
        payload = {"amount": "1", "currency": "sol", "recipient": "r"}
        encoded = b64url(payload)
        assert not encoded.endswith("=")

        challenge = parse_www_authenticate(challenge_header("x", "solana", payload))[0]

        assert decode_challenge_request(challenge) == payload

    @pytest.mark.parametrize("bad_request", ["!!!not-base64!!!", base64.urlsafe_b64encode(b"not json").decode()])
    def test_undecodable_request_returns_empty_dict(self, bad_request):
        challenge = {"id": "x", "request": bad_request}

        assert decode_challenge_request(challenge) == {}

    def test_missing_request_returns_empty_dict(self):
        assert decode_challenge_request({"id": "x"}) == {}

    def test_non_object_json_returns_empty_dict(self):
        encoded = base64.urlsafe_b64encode(b'["a","list"]').decode().rstrip("=")

        assert decode_challenge_request({"id": "x", "request": encoded}) == {}


class TestChallengeNetwork:
    """Tests for deriving a preference-orderable network identifier."""

    @pytest.mark.parametrize(
        "chain_id,expected",
        [(8453, "eip155:8453"), (1, "eip155:1"), (84532, "eip155:84532"), (4326, "eip155:4326")],
    )
    def test_evm_chain_id_maps_to_eip155(self, chain_id, expected):
        challenge = parse_www_authenticate(challenge_header("x", "evm", evm_request(chain_id=chain_id)))[0]

        assert challenge_network(challenge) == expected

    def test_tempo_uses_evm_chain_id_mapping(self):
        challenge = parse_www_authenticate(challenge_header("x", "tempo", evm_request(chain_id=4326)))[0]

        assert challenge_network(challenge) == "eip155:4326"

    def test_evm_string_chain_id_is_coerced(self):
        request = evm_request()
        request["methodDetails"]["chainId"] = "8453"
        challenge = parse_www_authenticate(challenge_header("x", "evm", request))[0]

        assert challenge_network(challenge) == "eip155:8453"

    @pytest.mark.parametrize(
        "network,expected",
        [
            ("mainnet", "solana-mainnet"),
            ("devnet", "solana-devnet"),
            ("testnet", "solana-testnet"),
            ("MAINNET", "solana-mainnet"),
        ],
    )
    def test_solana_network_maps_to_identifier(self, network, expected):
        challenge = parse_www_authenticate(challenge_header("x", "solana", solana_request(network=network)))[0]

        assert challenge_network(challenge) == expected

    def test_localnet_is_not_aliased_to_a_public_network(self):
        """localnet is a local RPC/Surfpool environment, not Solana testnet.

        Aliasing it would let a local-only challenge satisfy a solana-testnet preference.
        """
        challenge = parse_www_authenticate(challenge_header("x", "solana", solana_request(network="localnet")))[0]

        assert challenge_network(challenge) is None

    def test_payable_devnet_challenge_outranks_localnet(self):
        """The regression this guards: a local-only challenge must not beat a payable one."""
        header = ", ".join(
            [
                challenge_header("local-1", "solana", solana_request(network="localnet")),
                challenge_header("dev-1", "solana", solana_request(network="devnet")),
            ]
        )
        challenges = parse_www_authenticate(header)

        selected = select_challenge(challenges, instrument_network="SOLANA")

        assert selected["id"] == "dev-1"

    def test_localnet_remains_selectable_when_it_is_the_only_option(self):
        """Unranked, not unusable — a local-only challenge is still payable if alone."""
        challenges = parse_www_authenticate(challenge_header("local-1", "solana", solana_request(network="localnet")))

        assert select_challenge(challenges, instrument_network="SOLANA")["id"] == "local-1"

    def test_solana_defaults_to_mainnet_when_network_absent(self):
        """Per draft-solana-charge-00, methodDetails.network is optional."""
        challenge = parse_www_authenticate(challenge_header("x", "solana", solana_request(network=None)))[0]

        assert challenge_network(challenge) == "solana-mainnet"

    def test_missing_chain_id_returns_none(self):
        request = evm_request()
        del request["methodDetails"]["chainId"]
        challenge = parse_www_authenticate(challenge_header("x", "evm", request))[0]

        assert challenge_network(challenge) is None

    def test_boolean_chain_id_returns_none(self):
        """bool is an int subclass; chainId=True must not become eip155:1."""
        request = evm_request()
        request["methodDetails"]["chainId"] = True
        challenge = parse_www_authenticate(challenge_header("x", "evm", request))[0]

        assert challenge_network(challenge) is None

    def test_unknown_method_returns_none(self):
        challenge = parse_www_authenticate(challenge_header("x", "bitcoin", evm_request()))[0]

        assert challenge_network(challenge) is None


class TestIsExpired:
    """Tests for challenge expiry evaluation."""

    def test_past_expires_is_expired(self):
        challenge = {"id": "x", "expires": PAST}

        assert is_expired(challenge) is True

    def test_future_expires_is_not_expired(self):
        challenge = {"id": "x", "expires": FUTURE}

        assert is_expired(challenge) is False

    def test_missing_expires_is_not_expired(self):
        assert is_expired({"id": "x"}) is False

    def test_unparseable_expires_is_not_expired(self):
        """Prefer attempting a challenge over discarding one we failed to parse."""
        assert is_expired({"id": "x", "expires": "not-a-date"}) is False

    def test_expiry_exactly_now_is_expired(self):
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)

        assert is_expired({"expires": "2026-04-01T12:00:00Z"}, now=now) is True

    def test_offset_timezone_is_honoured(self):
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)

        # 13:30+02:00 == 11:30Z, already past.
        assert is_expired({"expires": "2026-04-01T13:30:00+02:00"}, now=now) is True
        # 15:30+02:00 == 13:30Z, still ahead.
        assert is_expired({"expires": "2026-04-01T15:30:00+02:00"}, now=now) is False

    def test_naive_datetime_reference_is_treated_as_utc(self):
        naive_now = datetime(2026, 4, 1, 12, 0, 0)

        assert is_expired({"expires": "2026-04-01T11:00:00Z"}, now=naive_now) is True


class TestSelectChallenge:
    """Tests for the challenge selection algorithm."""

    def _parse(self, *headers):
        return parse_www_authenticate(", ".join(headers))

    def test_selects_solana_challenge_for_solana_instrument(self):
        challenges = self._parse(
            challenge_header("evm-1", "evm", evm_request()),
            challenge_header("sol-1", "solana", solana_request()),
        )

        selected = select_challenge(challenges, instrument_network="SOLANA")

        assert selected["id"] == "sol-1"

    def test_selects_evm_challenge_for_ethereum_instrument(self):
        challenges = self._parse(
            challenge_header("sol-1", "solana", solana_request()),
            challenge_header("evm-1", "evm", evm_request()),
        )

        selected = select_challenge(challenges, instrument_network="ETHEREUM")

        assert selected["id"] == "evm-1"

    def test_tempo_is_satisfied_by_ethereum_instrument(self):
        challenges = self._parse(challenge_header("tempo-1", "tempo", evm_request(chain_id=4326)))

        selected = select_challenge(challenges, instrument_network="ETHEREUM")

        assert selected["id"] == "tempo-1"

    def test_instrument_network_is_case_insensitive(self):
        challenges = self._parse(challenge_header("sol-1", "solana", solana_request()))

        assert select_challenge(challenges, instrument_network="solana")["id"] == "sol-1"

    def test_network_preference_order_decides_among_same_family(self):
        """Base mainnet (eip155:8453) outranks Ethereum mainnet in NETWORK_PREFERENCES."""
        challenges = self._parse(
            challenge_header("eth-mainnet", "evm", evm_request(chain_id=1)),
            challenge_header("base", "evm", evm_request(chain_id=8453)),
        )

        selected = select_challenge(challenges, instrument_network="ETHEREUM")

        assert selected["id"] == "base"

    def test_explicit_network_preferences_override_default(self):
        challenges = self._parse(
            challenge_header("base", "evm", evm_request(chain_id=8453)),
            challenge_header("eth-mainnet", "evm", evm_request(chain_id=1)),
        )

        selected = select_challenge(
            challenges,
            instrument_network="ETHEREUM",
            network_preferences=["eip155:1", "eip155:8453"],
        )

        assert selected["id"] == "eth-mainnet"

    def test_unsupported_intent_is_filtered_out(self):
        challenges = self._parse(
            challenge_header("sub-1", "evm", evm_request(), intent="subscription"),
            challenge_header("charge-1", "evm", evm_request()),
        )

        selected = select_challenge(challenges, instrument_network="ETHEREUM")

        assert selected["id"] == "charge-1"

    def test_all_non_charge_intents_raises(self):
        challenges = self._parse(
            challenge_header("s-1", "evm", evm_request(), intent="session"),
            challenge_header("s-2", "evm", evm_request(), intent="subscription"),
        )

        with pytest.raises(MppChallengeSelectionError, match="charge"):
            select_challenge(challenges, instrument_network="ETHEREUM")

    def test_absent_intent_is_rejected(self):
        header = f'Payment id="x", realm="api.example.com", method="evm", request="{b64url(evm_request())}"'

        assert parse_www_authenticate(header) == []

    def test_expired_challenge_is_filtered_out(self):
        challenges = self._parse(
            challenge_header("old", "evm", evm_request(), expires=PAST),
            challenge_header("fresh", "evm", evm_request(), expires=FUTURE),
        )

        selected = select_challenge(challenges, instrument_network="ETHEREUM")

        assert selected["id"] == "fresh"

    def test_all_expired_raises(self):
        challenges = self._parse(challenge_header("old", "evm", evm_request(), expires=PAST))

        with pytest.raises(MppChallengeSelectionError, match="expired"):
            select_challenge(challenges, instrument_network="ETHEREUM")

    def test_soonest_expiry_wins_when_network_rank_ties(self):
        soon = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
        later = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat().replace("+00:00", "Z")
        challenges = self._parse(
            challenge_header("later", "evm", evm_request(chain_id=8453), expires=later),
            challenge_header("soon", "evm", evm_request(chain_id=8453), expires=soon),
        )

        selected = select_challenge(challenges, instrument_network="ETHEREUM")

        assert selected["id"] == "soon"

    def test_bounded_offer_preferred_over_unbounded_on_tie(self):
        soon = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
        challenges = self._parse(
            challenge_header("no-expiry", "evm", evm_request(chain_id=8453)),
            challenge_header("bounded", "evm", evm_request(chain_id=8453), expires=soon),
        )

        selected = select_challenge(challenges, instrument_network="ETHEREUM")

        assert selected["id"] == "bounded"

    def test_server_order_breaks_full_tie(self):
        challenges = self._parse(
            challenge_header("first", "evm", evm_request(chain_id=8453)),
            challenge_header("second", "evm", evm_request(chain_id=8453)),
        )

        selected = select_challenge(challenges, instrument_network="ETHEREUM")

        assert selected["id"] == "first"

    def test_unknown_network_still_selectable_when_it_is_the_only_option(self):
        """A challenge on an unranked chain must not be silently unusable."""
        challenges = self._parse(challenge_header("exotic", "evm", evm_request(chain_id=999999)))

        selected = select_challenge(challenges, instrument_network="ETHEREUM")

        assert selected["id"] == "exotic"

    def test_ranked_network_beats_unranked(self):
        challenges = self._parse(
            challenge_header("exotic", "evm", evm_request(chain_id=999999)),
            challenge_header("base", "evm", evm_request(chain_id=8453)),
        )

        selected = select_challenge(challenges, instrument_network="ETHEREUM")

        assert selected["id"] == "base"

    def test_no_matching_method_raises_with_advertised_methods(self):
        challenges = self._parse(challenge_header("sol-1", "solana", solana_request()))

        with pytest.raises(MppChallengeSelectionError, match="solana"):
            select_challenge(challenges, instrument_network="ETHEREUM")

    def test_unknown_method_is_not_satisfiable(self):
        challenges = self._parse(challenge_header("btc-1", "bitcoin", evm_request()))

        with pytest.raises(MppChallengeSelectionError, match="No matching challenge"):
            select_challenge(challenges, instrument_network="ETHEREUM")

    def test_empty_challenge_list_raises(self):
        with pytest.raises(MppChallengeSelectionError, match="No challenges"):
            select_challenge([], instrument_network="ETHEREUM")

    @pytest.mark.parametrize("network", ["BITCOIN", "", None, "  "])
    def test_unsupported_instrument_network_raises(self, network):
        challenges = self._parse(challenge_header("evm-1", "evm", evm_request()))

        with pytest.raises(MppChallengeSelectionError, match="[Uu]nsupported instrument network"):
            select_challenge(challenges, instrument_network=network)

    def test_selected_challenge_retains_raw_header(self):
        challenges = self._parse(challenge_header("evm-1", "evm", evm_request()))

        selected = select_challenge(challenges, instrument_network="ETHEREUM")

        assert selected["raw"].startswith("Payment ")
        assert 'id="evm-1"' in selected["raw"]
