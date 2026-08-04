"""Integration tests for MPP (Machine Payments Protocol) payment processing.

These tests exercise the MPP path end to end against the real ProcessPayment API:
challenge selection, ``paymentType="MPP"`` submission, and credential minting.

SETUP INSTRUCTIONS:
===================

1. Set the following environment variables before running tests:

   # Required: AWS region
   export BEDROCK_TEST_REGION="us-west-2"

   # Required: Payment manager ARN (created via control plane)
   export TEST_PAYMENT_MANAGER_ARN="arn:aws:bedrock:us-west-2:123456789012:payment-manager/pm-123"

   # Required: Payment connector ID (created via control plane)
   export TEST_PAYMENT_CONNECTOR_ID="pc-123"

   # Required for live payment tests: a funded instrument and an open session.
   # Without these, only the offline selection/detection tests run.
   export TEST_PAYMENT_INSTRUMENT_ID="payment-instrument-..."
   export TEST_PAYMENT_SESSION_ID="payment-session-..."

   # Optional: User ID for testing (default: generated)
   export TEST_USER_ID="test-user"

   # Optional: an MPP-enabled resource that answers 402 with a
   # 'WWW-Authenticate: Payment' challenge. When set, test_live_402_challenge
   # fetches a real challenge instead of using a synthetic one.
   export TEST_MPP_RESOURCE_URL="https://api.example.com/api/paid"

2. Ensure AWS credentials are configured (see tests_integ/payments/README.md).

3. Run the tests:
   pytest tests_integ/payments/test_mpp_payment.py -v

SERVICE SIDE VERIFICATION:
==========================

Monitor service logs to verify:
- ProcessPayment invoked with paymentType=MPP
- The wwwAuthenticateHeaders value matches the challenge byte-for-byte
- MppPaymentOutput.selectedPaymentId echoes the submitted challenge id
- The minted credential is not written to logs (it is @sensitive)
"""

import base64
import json
import os
import uuid

import pytest

from bedrock_agentcore.payments.manager import PaymentError, PaymentManager
from bedrock_agentcore.payments.mpp import (
    MppChallengeSelectionError,
    extract_challenges,
    is_mpp_payment_required,
    select_challenge,
)

REQUIRES_MANAGER = pytest.mark.skipif(
    not os.environ.get("TEST_PAYMENT_MANAGER_ARN"),
    reason="TEST_PAYMENT_MANAGER_ARN environment variable not set",
)

REQUIRES_LIVE_PAYMENT = pytest.mark.skipif(
    not (
        os.environ.get("TEST_PAYMENT_MANAGER_ARN")
        and os.environ.get("TEST_PAYMENT_INSTRUMENT_ID")
        and os.environ.get("TEST_PAYMENT_SESSION_ID")
    ),
    reason="TEST_PAYMENT_MANAGER_ARN, TEST_PAYMENT_INSTRUMENT_ID and TEST_PAYMENT_SESSION_ID must be set",
)


def b64url(obj) -> str:
    """Encode a dict as base64url JSON without padding, as MPP requires."""
    return base64.urlsafe_b64encode(json.dumps(obj, separators=(",", ":")).encode()).decode().rstrip("=")


def evm_challenge(challenge_id="evm-1", chain_id=84532, amount="1000"):
    """Build an evm-charge challenge per draft-evm-charge-00."""
    request = {
        "amount": amount,
        "currency": "0x036CbD53842c5426634e7929541eC2318f3dCF71",
        "recipient": "0x742d35Cc6634C0532925a3b844Bc9e7595f8fE00",
        "methodDetails": {"chainId": chain_id},
    }
    return (
        f'Payment id="{challenge_id}", realm="api.example.com", method="evm", '
        f'intent="charge", request="{b64url(request)}"'
    )


def solana_challenge(challenge_id="sol-1", network="devnet", amount="1000"):
    """Build a solana-charge challenge per draft-solana-charge-00."""
    request = {
        "amount": amount,
        "currency": "sol",
        "recipient": "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
        "methodDetails": {"network": network},
    }
    return (
        f'Payment id="{challenge_id}", realm="api.example.com", method="solana", '
        f'intent="charge", request="{b64url(request)}"'
    )


def mpp_402(*challenges):
    """Build a 402 payment required request advertising the given challenges."""
    return {
        "statusCode": 402,
        "headers": {"WWW-Authenticate": ", ".join(challenges)},
        "body": {},
    }


@pytest.mark.integration
class TestMppModelSupport:
    """The SDK must be able to submit MPP regardless of the installed botocore version."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        default_arn = "arn:aws:bedrock:us-west-2:123456789012:payment-manager/pm-test"
        cls.payment_manager_arn = os.environ.get("TEST_PAYMENT_MANAGER_ARN", default_arn)

    def test_manager_client_accepts_mpp_payment_input(self):
        """A real client built by PaymentManager must model paymentInput.mpp."""
        manager = PaymentManager(payment_manager_arn=self.payment_manager_arn, region_name=self.region)

        shapes = manager._payment_client.meta.service_model._shape_resolver._shape_map

        assert "mpp" in shapes["PaymentInput"]["members"]
        assert "mpp" in shapes["PaymentOutput"]["members"]
        assert "MPP" in shapes["PaymentType"]["enum"]

    def test_crypto_x402_support_is_retained(self):
        """The MPP additions must not disturb the shipped x402 contract."""
        manager = PaymentManager(payment_manager_arn=self.payment_manager_arn, region_name=self.region)

        shapes = manager._payment_client.meta.service_model._shape_resolver._shape_map

        assert "cryptoX402" in shapes["PaymentInput"]["members"]
        assert "CRYPTO_X402" in shapes["PaymentType"]["enum"]


@pytest.mark.integration
class TestMppChallengeSelection:
    """Selection is pure logic, so it is verified without calling the service."""

    def test_detects_mpp_402(self):
        assert is_mpp_payment_required(mpp_402(evm_challenge())) is True

    def test_does_not_claim_x402_402(self):
        x402 = {
            "statusCode": 402,
            "headers": {"content-type": "application/json"},
            "body": {"x402Version": 1, "accepts": [{"network": "base-sepolia"}]},
        }

        assert is_mpp_payment_required(x402) is False

    def test_selects_challenge_matching_instrument_blockchain(self):
        challenges = extract_challenges(mpp_402(evm_challenge(), solana_challenge()))

        assert select_challenge(challenges, instrument_network="SOLANA")["id"] == "sol-1"
        assert select_challenge(challenges, instrument_network="ETHEREUM")["id"] == "evm-1"

    def test_selected_raw_header_round_trips_byte_for_byte(self):
        """The challenge HMAC binds to these exact bytes."""
        raw = evm_challenge()
        challenges = extract_challenges(mpp_402(raw))

        assert select_challenge(challenges, instrument_network="ETHEREUM")["raw"] == raw

    def test_unsatisfiable_challenge_raises(self):
        challenges = extract_challenges(mpp_402(solana_challenge()))

        with pytest.raises(MppChallengeSelectionError):
            select_challenge(challenges, instrument_network="ETHEREUM")


@pytest.mark.integration
class TestMppProcessPayment:
    """Tests that call the real ProcessPayment API with paymentType=MPP."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.payment_manager_arn = os.environ.get("TEST_PAYMENT_MANAGER_ARN")
        cls.user_id = os.environ.get("TEST_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")
        cls.instrument_id = os.environ.get("TEST_PAYMENT_INSTRUMENT_ID")
        cls.session_id = os.environ.get("TEST_PAYMENT_SESSION_ID")

        if cls.payment_manager_arn:
            cls.manager = PaymentManager(
                payment_manager_arn=cls.payment_manager_arn,
                region_name=cls.region,
            )
        else:
            cls.manager = None

    def _instrument_network(self):
        """Read the configured instrument's blockchain so tests pick a payable challenge."""
        instrument = self.manager.get_payment_instrument(
            payment_instrument_id=self.instrument_id,
            user_id=self.user_id,
        )
        return instrument["paymentInstrumentDetails"]["embeddedCryptoWallet"]["network"]

    @REQUIRES_LIVE_PAYMENT
    def test_process_payment_accepts_mpp_payment_type(self):
        """ProcessPayment must accept paymentType=MPP and an MppPaymentInput."""
        network = self._instrument_network()
        challenge = solana_challenge() if network.upper() == "SOLANA" else evm_challenge()

        payment_input = {
            "mpp": {
                "version": "1",
                "wwwAuthenticateHeaders": [challenge],
            }
        }

        # The synthetic challenge is not signed by a real merchant, so the service is
        # expected to reject it on verification. What this test proves is that the
        # request shape itself is accepted: a ValidationException naming 'mpp',
        # 'paymentType' or 'paymentInput' would mean the contract is wrong.
        try:
            response = self.manager.process_payment(
                user_id=self.user_id,
                payment_session_id=self.session_id,
                payment_instrument_id=self.instrument_id,
                payment_type="MPP",
                payment_input=payment_input,
                client_token=str(uuid.uuid4()),
            )
        except PaymentError as e:
            message = str(e).lower()
            for shape_hint in ("unknown parameter", "paymenttype", "paymentinput"):
                assert shape_hint not in message, f"MPP request shape was rejected: {e}"
            pytest.skip(f"Service rejected the synthetic challenge (expected without a live merchant): {e}")
        else:
            assert isinstance(response, dict)
            assert "mpp" in response.get("paymentOutput", {})

    @REQUIRES_LIVE_PAYMENT
    def test_generate_payment_header_routes_mpp_402_to_authorization(self):
        """An MPP 402 must yield an Authorization header, not an x402 one."""
        network = self._instrument_network()
        challenge = solana_challenge() if network.upper() == "SOLANA" else evm_challenge()

        try:
            header = self.manager.generate_payment_header(
                user_id=self.user_id,
                payment_instrument_id=self.instrument_id,
                payment_session_id=self.session_id,
                payment_required_request=mpp_402(challenge),
                client_token=str(uuid.uuid4()),
            )
        except PaymentError as e:
            pytest.skip(f"Service rejected the synthetic challenge (expected without a live merchant): {e}")

        assert set(header) == {"Authorization"}
        assert header["Authorization"].startswith("Payment ")
        assert "X-PAYMENT" not in header

    @REQUIRES_LIVE_PAYMENT
    def test_buyer_pays_gas_fees_is_accepted_by_the_service(self):
        """ProcessPayment must accept the optional buyerPaysGasFees flag."""
        network = self._instrument_network()
        challenge = solana_challenge() if network.upper() == "SOLANA" else evm_challenge()

        try:
            self.manager.process_payment(
                user_id=self.user_id,
                payment_session_id=self.session_id,
                payment_instrument_id=self.instrument_id,
                payment_type="MPP",
                payment_input={
                    "mpp": {
                        "version": "1",
                        "wwwAuthenticateHeaders": [challenge],
                        "buyerPaysGasFees": True,
                    }
                },
                client_token=str(uuid.uuid4()),
            )
        except PaymentError as e:
            message = str(e).lower()
            assert "buyerpaysgasfees" not in message, f"buyerPaysGasFees was rejected: {e}"
            assert "unknown parameter" not in message, f"MPP request shape was rejected: {e}"
            pytest.skip(f"Service rejected the synthetic challenge (expected without a live merchant): {e}")

    @REQUIRES_MANAGER
    def test_unsatisfiable_challenge_fails_before_calling_service(self):
        """Selection failures must not burn a ProcessPayment call or session budget."""
        instrument_id = self.instrument_id or "payment-instrument-unused"

        with pytest.raises(PaymentError, match="No matching challenge|Missing network information"):
            self.manager.generate_payment_header(
                user_id=self.user_id,
                payment_instrument_id=instrument_id,
                payment_session_id=self.session_id or "payment-session-unused",
                # Only a bitcoin challenge is advertised; no instrument can satisfy it.
                payment_required_request=mpp_402(
                    'Payment id="btc-1", realm="api.example.com", method="bitcoin", intent="charge"'
                ),
                client_token=str(uuid.uuid4()),
            )

    @REQUIRES_LIVE_PAYMENT
    def test_expired_challenge_is_rejected_locally(self):
        """An expired challenge must be filtered out before the service is called."""
        expired_request = b64url(
            {
                "amount": "1",
                "currency": "0xUSDC",
                "recipient": "0xabc",
                "methodDetails": {"chainId": 84532},
            }
        )
        expired = (
            'Payment id="old-1", realm="api.example.com", method="evm", '
            'intent="charge", expires="2020-01-01T00:00:00Z", '
            f'request="{expired_request}"'
        )

        with pytest.raises(PaymentError, match="expired|No usable challenge"):
            self.manager.generate_payment_header(
                user_id=self.user_id,
                payment_instrument_id=self.instrument_id,
                payment_session_id=self.session_id,
                payment_required_request=mpp_402(expired),
                client_token=str(uuid.uuid4()),
            )


@pytest.mark.integration
class TestMppLiveResource:
    """Tests against a real MPP-enabled endpoint, when one is configured."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.resource_url = os.environ.get("TEST_MPP_RESOURCE_URL")
        cls.payment_manager_arn = os.environ.get("TEST_PAYMENT_MANAGER_ARN")
        cls.user_id = os.environ.get("TEST_USER_ID", f"test-user-{uuid.uuid4().hex[:8]}")
        cls.instrument_id = os.environ.get("TEST_PAYMENT_INSTRUMENT_ID")
        cls.session_id = os.environ.get("TEST_PAYMENT_SESSION_ID")

    @pytest.mark.skipif(
        not os.environ.get("TEST_MPP_RESOURCE_URL"),
        reason="TEST_MPP_RESOURCE_URL environment variable not set",
    )
    def test_live_402_challenge_is_parseable(self):
        """A real server's 402 must produce at least one usable challenge."""
        import httpx

        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(self.resource_url)

        assert response.status_code == 402, f"Expected 402, got {response.status_code}"

        payment_required_request = {
            "statusCode": response.status_code,
            "headers": dict(response.headers),
            "body": {},
        }

        assert is_mpp_payment_required(payment_required_request), (
            f"No MPP challenge in response headers: {dict(response.headers)}"
        )

        challenges = extract_challenges(payment_required_request)
        assert challenges
        # Every challenge must carry the auth-params selection depends on.
        for challenge in challenges:
            assert challenge.get("id"), f"Challenge missing id: {challenge}"
            assert challenge.get("method"), f"Challenge missing method: {challenge}"
            assert challenge["raw"].startswith("Payment ")

    @pytest.mark.skipif(
        not (
            os.environ.get("TEST_MPP_RESOURCE_URL")
            and os.environ.get("TEST_PAYMENT_MANAGER_ARN")
            and os.environ.get("TEST_PAYMENT_INSTRUMENT_ID")
            and os.environ.get("TEST_PAYMENT_SESSION_ID")
        ),
        reason="TEST_MPP_RESOURCE_URL, TEST_PAYMENT_MANAGER_ARN, TEST_PAYMENT_INSTRUMENT_ID "
        "and TEST_PAYMENT_SESSION_ID must be set",
    )
    def test_end_to_end_pay_and_retry(self):
        """Full loop: 402 → select → pay → retry with Authorization → 200."""
        import httpx

        manager = PaymentManager(payment_manager_arn=self.payment_manager_arn, region_name=self.region)

        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            first = client.get(self.resource_url)
            assert first.status_code == 402

            header = manager.generate_payment_header(
                user_id=self.user_id,
                payment_instrument_id=self.instrument_id,
                payment_session_id=self.session_id,
                payment_required_request={
                    "statusCode": first.status_code,
                    "headers": dict(first.headers),
                    "body": {},
                },
                client_token=str(uuid.uuid4()),
            )
            assert "Authorization" in header

            retry = client.get(self.resource_url, headers=header)

        assert retry.status_code == 200, f"Retry after payment failed: {retry.status_code} {retry.text[:300]}"
