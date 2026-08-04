"""Unit tests for the MPP payment path on PaymentManager."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.payments.manager import PaymentError, PaymentManager

PAYMENT_MANAGER_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/pm-abc123"


def b64url(obj) -> str:
    """Encode a dict as base64url JSON without padding."""
    return base64.urlsafe_b64encode(json.dumps(obj, separators=(",", ":")).encode()).decode().rstrip("=")


def evm_request(chain_id=84532):
    return {
        "amount": "1000000",
        "currency": "0x036CbD53842c5426634e7929541eC2318f3dCF71",
        "recipient": "0x742d35Cc6634C0532925a3b844Bc9e7595f8fE00",
        "methodDetails": {"chainId": chain_id},
    }


def solana_request(network="devnet"):
    return {
        "amount": "1000000",
        "currency": "sol",
        "recipient": "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
        "methodDetails": {"network": network},
    }


def challenge_header(challenge_id, method, request_obj, intent="charge"):
    return (
        f'Payment id="{challenge_id}", realm="api.example.com", method="{method}", '
        f'intent="{intent}", request="{b64url(request_obj)}"'
    )


def mpp_402(*headers):
    """Build a 402 payment required request advertising the given challenges."""
    return {
        "statusCode": 402,
        "headers": {"WWW-Authenticate": ", ".join(headers)},
        "body": {},
    }


@pytest.fixture
def manager():
    """A PaymentManager with a mocked data-plane client."""
    with patch("boto3.Session") as mock_session:
        mock_session.return_value.region_name = "us-west-2"
        mgr = PaymentManager(payment_manager_arn=PAYMENT_MANAGER_ARN, region_name="us-west-2")
    mgr._payment_client = MagicMock()
    return mgr


def set_instrument_network(manager, network):
    manager._payment_client.get_payment_instrument.return_value = {
        "paymentInstrument": {
            "paymentInstrumentId": "pi-1",
            "paymentInstrumentDetails": {"embeddedCryptoWallet": {"network": network}},
        }
    }


def set_mpp_result(manager, selected_payment_id="evm-1", credential="eyJjaGFsbGVuZ2UiOnt9fQ", version="1"):
    manager._payment_client.process_payment.return_value = {
        "processPayment": {
            "processPaymentId": "pp-1",
            "status": "PROOF_GENERATED",
            "paymentOutput": {
                "mpp": {
                    "version": version,
                    "selectedPaymentId": selected_payment_id,
                    "paymentCredential": credential,
                }
            },
        }
    }


class TestMppRouting:
    """generate_payment_header must route by detected protocol."""

    def test_mpp_402_returns_authorization_header(self, manager):
        set_instrument_network(manager, "ETHEREUM")
        set_mpp_result(manager, credential="eyJhIjoxfQ")

        header = manager.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            user_id="user-1",
            payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
        )

        assert header == {"Authorization": "Payment eyJhIjoxfQ"}

    def test_mpp_sends_payment_type_mpp(self, manager):
        set_instrument_network(manager, "ETHEREUM")
        set_mpp_result(manager)

        manager.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            user_id="user-1",
            payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
        )

        kwargs = manager._payment_client.process_payment.call_args.kwargs
        assert kwargs["paymentType"] == "MPP"
        assert "cryptoX402" not in kwargs["paymentInput"]

    def test_raw_challenge_header_is_forwarded_verbatim(self, manager):
        """The challenge HMAC binds to the exact request bytes, so no re-encoding."""
        set_instrument_network(manager, "ETHEREUM")
        set_mpp_result(manager)
        raw = challenge_header("evm-1", "evm", evm_request())

        manager.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            user_id="user-1",
            payment_required_request=mpp_402(raw),
        )

        sent = manager._payment_client.process_payment.call_args.kwargs["paymentInput"]["mpp"]
        assert sent["wwwAuthenticateHeaders"] == [raw]
        assert b64url(evm_request()) in sent["wwwAuthenticateHeaders"][0]

    def test_exactly_one_challenge_is_sent(self, manager):
        """The service model constrains wwwAuthenticateHeaders to a single entry."""
        set_instrument_network(manager, "SOLANA")
        set_mpp_result(manager, selected_payment_id="sol-1")

        manager.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            user_id="user-1",
            payment_required_request=mpp_402(
                challenge_header("evm-1", "evm", evm_request()),
                challenge_header("sol-1", "solana", solana_request()),
            ),
        )

        sent = manager._payment_client.process_payment.call_args.kwargs["paymentInput"]["mpp"]
        assert len(sent["wwwAuthenticateHeaders"]) == 1
        assert 'id="sol-1"' in sent["wwwAuthenticateHeaders"][0]

    def test_mpp_version_is_numeric_string(self, manager):
        """The model constrains version to ^[0-9]+$."""
        set_instrument_network(manager, "ETHEREUM")
        set_mpp_result(manager)

        manager.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            user_id="user-1",
            payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
        )

        version = manager._payment_client.process_payment.call_args.kwargs["paymentInput"]["mpp"]["version"]
        assert version.isdigit()

    def test_x402_402_still_uses_crypto_x402_path(self, manager):
        """MPP detection must not regress the existing x402 flow."""
        set_instrument_network(manager, "ETHEREUM")
        manager._payment_client.process_payment.return_value = {
            "processPayment": {"paymentOutput": {"cryptoX402": {"payload": {"signature": "0xsig"}}}}
        }
        x402_request = {
            "statusCode": 402,
            "headers": {"content-type": "application/json"},
            "body": {
                "x402Version": 1,
                "accepts": [{"scheme": "exact", "network": "base-sepolia", "maxAmountRequired": "1000"}],
            },
        }

        header = manager.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            user_id="user-1",
            payment_required_request=x402_request,
        )

        assert "X-PAYMENT" in header
        assert manager._payment_client.process_payment.call_args.kwargs["paymentType"] == "CRYPTO_X402"

    def test_client_token_is_forwarded(self, manager):
        set_instrument_network(manager, "ETHEREUM")
        set_mpp_result(manager)

        manager.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            user_id="user-1",
            payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
            client_token="my-token-123",
        )

        assert manager._payment_client.process_payment.call_args.kwargs["clientToken"] == "my-token-123"

    def _mpp_input(self, manager):
        return manager._payment_client.process_payment.call_args.kwargs["paymentInput"]["mpp"]

    def _process_mpp(self, manager, **kwargs):
        return manager.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            user_id="user-1",
            payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
            **kwargs,
        )

    def test_buyer_pays_gas_fees_omitted_when_not_specified(self, manager):
        """Omitting the field must leave the service-side protocol default in place."""
        set_instrument_network(manager, "ETHEREUM")
        set_mpp_result(manager)

        self._process_mpp(manager)

        assert "buyerPaysGasFees" not in self._mpp_input(manager)

    def test_buyer_pays_gas_fees_true_is_forwarded(self, manager):
        set_instrument_network(manager, "ETHEREUM")
        set_mpp_result(manager)

        self._process_mpp(manager, buyer_pays_gas_fees=True)

        assert self._mpp_input(manager)["buyerPaysGasFees"] is True

    def test_buyer_pays_gas_fees_false_is_forwarded_explicitly(self, manager):
        """False is a deliberate refusal and must be sent, not treated as unset."""
        set_instrument_network(manager, "ETHEREUM")
        set_mpp_result(manager)

        self._process_mpp(manager, buyer_pays_gas_fees=False)

        assert self._mpp_input(manager)["buyerPaysGasFees"] is False

    @pytest.mark.parametrize("value,expected", [(1, True), (0, False)])
    def test_buyer_pays_gas_fees_is_coerced_to_bool(self, manager, value, expected):
        """The model types this as Boolean, so truthy inputs must not leak through as ints."""
        set_instrument_network(manager, "ETHEREUM")
        set_mpp_result(manager)

        self._process_mpp(manager, buyer_pays_gas_fees=value)

        forwarded = self._mpp_input(manager)["buyerPaysGasFees"]
        assert forwarded is expected
        assert isinstance(forwarded, bool)

    def test_buyer_pays_gas_fees_not_sent_on_x402_path(self, manager):
        """The field is MPP-only and must never appear in a cryptoX402 input."""
        set_instrument_network(manager, "ETHEREUM")
        manager._payment_client.process_payment.return_value = {
            "processPayment": {"paymentOutput": {"cryptoX402": {"payload": {"signature": "0xsig"}}}}
        }

        manager.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            user_id="user-1",
            payment_required_request={
                "statusCode": 402,
                "headers": {"content-type": "application/json"},
                "body": {"x402Version": 1, "accepts": [{"scheme": "exact", "network": "base-sepolia"}]},
            },
            buyer_pays_gas_fees=True,
        )

        payment_input = manager._payment_client.process_payment.call_args.kwargs["paymentInput"]
        assert "mpp" not in payment_input
        assert "buyerPaysGasFees" not in payment_input["cryptoX402"]

    def test_network_preferences_are_applied(self, manager):
        set_instrument_network(manager, "ETHEREUM")
        set_mpp_result(manager, selected_payment_id="eth-1")

        manager.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            user_id="user-1",
            payment_required_request=mpp_402(
                challenge_header("base-1", "evm", evm_request(chain_id=8453)),
                challenge_header("eth-1", "evm", evm_request(chain_id=1)),
            ),
            network_preferences=["eip155:1", "eip155:8453"],
        )

        sent = manager._payment_client.process_payment.call_args.kwargs["paymentInput"]["mpp"]
        assert 'id="eth-1"' in sent["wwwAuthenticateHeaders"][0]

    def test_non_402_status_code_is_rejected(self, manager):
        request = mpp_402(challenge_header("evm-1", "evm", evm_request()))
        request["statusCode"] = 200

        with pytest.raises(PaymentError, match="Expected statusCode 402"):
            manager.generate_payment_header(
                payment_instrument_id="pi-1",
                payment_session_id="ps-1",
                user_id="user-1",
                payment_required_request=request,
            )


class TestMppCredentialHandling:
    """Tests for turning the service's output into an Authorization header."""

    def test_scheme_prefix_is_added_when_absent(self, manager):
        set_instrument_network(manager, "ETHEREUM")
        set_mpp_result(manager, credential="eyJhIjoxfQ")

        header = manager.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            user_id="user-1",
            payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
        )

        assert header["Authorization"] == "Payment eyJhIjoxfQ"

    def test_existing_scheme_prefix_is_not_duplicated(self, manager):
        set_instrument_network(manager, "ETHEREUM")
        set_mpp_result(manager, credential="Payment eyJhIjoxfQ")

        header = manager.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            user_id="user-1",
            payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
        )

        assert header["Authorization"] == "Payment eyJhIjoxfQ"

    def test_missing_mpp_output_raises(self, manager):
        set_instrument_network(manager, "ETHEREUM")
        manager._payment_client.process_payment.return_value = {
            "processPayment": {"paymentOutput": {"cryptoX402": {"payload": {}}}}
        }

        with pytest.raises(PaymentError, match="Missing mpp in payment output"):
            manager.generate_payment_header(
                payment_instrument_id="pi-1",
                payment_session_id="ps-1",
                user_id="user-1",
                payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
            )

    @pytest.mark.parametrize(
        "payment_output",
        [
            None,
            {"mpp": None},
            {"mpp": "not-a-dict"},
            {"mpp": {}},
            {"mpp": []},
        ],
    )
    def test_malformed_payment_output_raises_actionable_error(self, manager, payment_output):
        """A null or non-object member must name the field, not leak an AttributeError."""
        set_instrument_network(manager, "ETHEREUM")
        manager._payment_client.process_payment.return_value = {"processPayment": {"paymentOutput": payment_output}}

        with pytest.raises(PaymentError, match="Missing mpp in payment output"):
            manager.generate_payment_header(
                payment_instrument_id="pi-1",
                payment_session_id="ps-1",
                user_id="user-1",
                payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
            )

    @pytest.mark.parametrize("credential", [None, "", "   ", 123])
    def test_missing_credential_raises(self, manager, credential):
        set_instrument_network(manager, "ETHEREUM")
        manager._payment_client.process_payment.return_value = {
            "processPayment": {
                "paymentOutput": {
                    "mpp": {
                        "version": "1",
                        "selectedPaymentId": "evm-1",
                        "paymentCredential": credential,
                    }
                }
            }
        }

        with pytest.raises(PaymentError, match="Missing paymentCredential"):
            manager.generate_payment_header(
                payment_instrument_id="pi-1",
                payment_session_id="ps-1",
                user_id="user-1",
                payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
            )

    def test_challenge_id_mismatch_raises(self, manager):
        """Attaching a credential for a challenge we did not select would fail upstream."""
        set_instrument_network(manager, "ETHEREUM")
        set_mpp_result(manager, selected_payment_id="some-other-id")

        with pytest.raises(PaymentError, match="Challenge mismatch"):
            manager.generate_payment_header(
                payment_instrument_id="pi-1",
                payment_session_id="ps-1",
                user_id="user-1",
                payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
            )

    def test_absent_selected_payment_id_is_tolerated(self, manager):
        set_instrument_network(manager, "ETHEREUM")
        manager._payment_client.process_payment.return_value = {
            "processPayment": {"paymentOutput": {"mpp": {"version": "1", "paymentCredential": "eyJhIjoxfQ"}}}
        }

        header = manager.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            user_id="user-1",
            payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
        )

        assert header["Authorization"] == "Payment eyJhIjoxfQ"


class TestMppSelectionErrors:
    """Selection failures must surface as PaymentError, not leak internal types."""

    def test_no_satisfiable_method_raises_payment_error(self, manager):
        set_instrument_network(manager, "ETHEREUM")

        with pytest.raises(PaymentError, match="No matching challenge"):
            manager.generate_payment_header(
                payment_instrument_id="pi-1",
                payment_session_id="ps-1",
                user_id="user-1",
                payment_required_request=mpp_402(challenge_header("sol-1", "solana", solana_request())),
            )

        manager._payment_client.process_payment.assert_not_called()

    def test_unsupported_instrument_network_raises_payment_error(self, manager):
        set_instrument_network(manager, "BITCOIN")

        with pytest.raises(PaymentError, match="[Uu]nsupported instrument network"):
            manager.generate_payment_header(
                payment_instrument_id="pi-1",
                payment_session_id="ps-1",
                user_id="user-1",
                payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
            )

    def test_instrument_without_network_raises(self, manager):
        manager._payment_client.get_payment_instrument.return_value = {
            "paymentInstrument": {"paymentInstrumentId": "pi-1", "paymentInstrumentDetails": {}}
        }

        with pytest.raises(PaymentError, match="Missing network information"):
            manager.generate_payment_header(
                payment_instrument_id="pi-1",
                payment_session_id="ps-1",
                user_id="user-1",
                payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
            )


class TestMppBearerAuth:
    """MPP must work under CUSTOM_JWT bearer auth, where userId is omitted."""

    def test_bearer_auth_omits_user_id(self):
        with patch("boto3.Session") as mock_session:
            mock_session.return_value.region_name = "us-west-2"
            mgr = PaymentManager(
                payment_manager_arn=PAYMENT_MANAGER_ARN,
                region_name="us-west-2",
                bearer_token="jwt-token",
            )
        mgr._payment_client = MagicMock()
        set_instrument_network(mgr, "ETHEREUM")
        set_mpp_result(mgr)

        header = mgr.generate_payment_header(
            payment_instrument_id="pi-1",
            payment_session_id="ps-1",
            payment_required_request=mpp_402(challenge_header("evm-1", "evm", evm_request())),
        )

        assert "Authorization" in header
        assert "userId" not in mgr._payment_client.process_payment.call_args.kwargs
