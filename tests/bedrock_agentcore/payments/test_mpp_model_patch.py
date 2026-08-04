"""Unit tests for the MPP service-model patch.

These tests exercise the patch against a real boto3 client so they fail loudly if a
future botocore release changes the internals the patch relies on, or ships MPP
natively (in which case the patch must become a no-op rather than double-apply).
"""

from unittest.mock import MagicMock

import botocore.session
import pytest
from botocore.exceptions import ParamValidationError

from bedrock_agentcore.payments._model_patch import patch_process_payment_model

PAYMENT_MANAGER_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:payment-manager/pm-abcdefghij"


def make_client():
    """Build a client whose service model is isolated from other clients.

    botocore caches loaded service models per-session, so clients from the same
    session share one shape map — patching one would leak into the others and make
    these tests order-dependent. A fresh botocore Session gives each client its own
    copy of the model.
    """
    return botocore.session.Session().create_client(
        "bedrock-agentcore",
        region_name="us-west-2",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


@pytest.fixture
def client():
    """A bedrock-agentcore client with an isolated service model and dummy credentials."""
    return make_client()


def shape_map(client):
    return client.meta.service_model._shape_resolver._shape_map


class TestPatchProcessPaymentModel:
    """Tests for injecting MPP shapes into the loaded service model."""

    def test_adds_mpp_member_to_payment_input_union(self, client):
        patch_process_payment_model(client)

        assert "mpp" in shape_map(client)["PaymentInput"]["members"]

    def test_adds_mpp_member_to_payment_output_union(self, client):
        patch_process_payment_model(client)

        assert "mpp" in shape_map(client)["PaymentOutput"]["members"]

    def test_adds_mpp_to_payment_type_enum(self, client):
        patch_process_payment_model(client)

        assert "MPP" in shape_map(client)["PaymentType"]["enum"]

    def test_preserves_existing_crypto_x402_member(self, client):
        patch_process_payment_model(client)

        assert "cryptoX402" in shape_map(client)["PaymentInput"]["members"]
        assert "CRYPTO_X402" in shape_map(client)["PaymentType"]["enum"]

    def test_is_idempotent(self, client):
        assert patch_process_payment_model(client) is True
        assert patch_process_payment_model(client) is False

        # The enum must not accumulate duplicates.
        assert shape_map(client)["PaymentType"]["enum"].count("MPP") == 1

    def test_returns_false_when_model_already_supports_mpp(self, client):
        """Once botocore ships MPP natively, the patch must do nothing."""
        shape_map(client)["PaymentInput"]["members"]["mpp"] = {"shape": "MppPaymentInput"}

        assert patch_process_payment_model(client) is False

    def test_returns_false_on_unexpected_botocore_internals(self):
        """A botocore refactor must degrade quietly, not raise at client init."""
        broken = MagicMock()
        del broken.meta.service_model._shape_resolver._shape_map

        assert patch_process_payment_model(broken) is False

    def test_injected_shapes_carry_the_modelled_constraints(self, client):
        """The shapes must mirror the service Smithy model.

        botocore enforces only some of these client-side (see TestParamValidation),
        but the model must still describe the service contract accurately.
        """
        patch_process_payment_model(client)
        shapes = shape_map(client)

        assert shapes["WwwAuthenticateHeaderList"]["min"] == 1
        assert shapes["WwwAuthenticateHeaderList"]["max"] == 1
        assert shapes["WwwAuthenticateHeader"]["max"] == 16384
        assert shapes["MppVersion"]["pattern"] == "^[0-9]+$"
        assert shapes["MppPaymentCredential"]["max"] == 32768
        assert shapes["MppSelectedPaymentId"]["max"] == 512

    def test_credential_shape_is_marked_sensitive(self, client):
        """The credential is bearer-like and must be kept out of logs and traces."""
        patch_process_payment_model(client)

        assert shape_map(client)["MppPaymentCredential"]["sensitive"] is True

    def test_buyer_pays_gas_fees_is_an_optional_boolean(self, client):
        patch_process_payment_model(client)
        shapes = shape_map(client)

        member = shapes["MppPaymentInput"]["members"]["buyerPaysGasFees"]
        assert shapes[member["shape"]]["type"] == "boolean"
        # Optional: the buyer declining gas fees is the protocol default.
        assert "buyerPaysGasFees" not in shapes["MppPaymentInput"]["required"]

    def test_required_members_match_the_model(self, client):
        patch_process_payment_model(client)
        shapes = shape_map(client)

        assert set(shapes["MppPaymentInput"]["required"]) == {"version", "wwwAuthenticateHeaders"}
        assert set(shapes["MppPaymentOutput"]["required"]) == {
            "version",
            "selectedPaymentId",
            "paymentCredential",
        }

    def test_does_not_mutate_installed_botocore_data_files(self, client):
        """The patch must only touch the in-memory model, never botocore on disk."""
        patch_process_payment_model(client)

        untouched = make_client()
        assert "mpp" not in shape_map(untouched)["PaymentInput"]["members"]


class TestParamValidation:
    """The patch must make botocore accept MPP params it would otherwise reject.

    Note: botocore's client-side validator enforces ``min`` length but not ``max``
    length or ``pattern`` — those constraints are enforced by the service. Tests here
    assert only what botocore actually checks; the modelled bounds are asserted
    against the shape map in TestPatchProcessPaymentModel instead.
    """

    VALID_ARGS = {
        "paymentManagerArn": PAYMENT_MANAGER_ARN,
        "paymentSessionId": "payment-session-" + "a" * 20,
        "paymentInstrumentId": "payment-instrument-" + "b" * 20,
        "clientToken": "c" * 33,
    }

    def test_mpp_input_is_rejected_before_patching(self, client):
        """Documents the problem the patch solves."""
        with pytest.raises(ParamValidationError, match='Unknown parameter in paymentInput: "mpp"'):
            client.process_payment(
                paymentType="MPP",
                paymentInput={"mpp": {"version": "1", "wwwAuthenticateHeaders": ['Payment id="a"']}},
                **self.VALID_ARGS,
            )

    def test_mpp_input_passes_validation_after_patching(self, client):
        patch_process_payment_model(client)

        # Validation now passes, so the call proceeds to signing/transport and fails
        # there instead — any exception except ParamValidationError means the params
        # were accepted client-side.
        with pytest.raises(Exception) as exc_info:
            client.process_payment(
                paymentType="MPP",
                paymentInput={"mpp": {"version": "1", "wwwAuthenticateHeaders": ['Payment id="a"']}},
                **self.VALID_ARGS,
            )

        assert not isinstance(exc_info.value, ParamValidationError)

    def test_empty_header_list_is_still_rejected(self, client):
        """The model constrains wwwAuthenticateHeaders to exactly one entry."""
        patch_process_payment_model(client)

        with pytest.raises(ParamValidationError):
            client.process_payment(
                paymentType="MPP",
                paymentInput={"mpp": {"version": "1", "wwwAuthenticateHeaders": []}},
                **self.VALID_ARGS,
            )

    def test_buyer_pays_gas_fees_passes_validation(self, client):
        patch_process_payment_model(client)

        with pytest.raises(Exception) as exc_info:
            client.process_payment(
                paymentType="MPP",
                paymentInput={
                    "mpp": {
                        "version": "1",
                        "wwwAuthenticateHeaders": ['Payment id="a"'],
                        "buyerPaysGasFees": True,
                    }
                },
                **self.VALID_ARGS,
            )

        assert not isinstance(exc_info.value, ParamValidationError)

    def test_non_boolean_buyer_pays_gas_fees_is_rejected(self, client):
        patch_process_payment_model(client)

        with pytest.raises(ParamValidationError):
            client.process_payment(
                paymentType="MPP",
                paymentInput={
                    "mpp": {
                        "version": "1",
                        "wwwAuthenticateHeaders": ['Payment id="a"'],
                        "buyerPaysGasFees": "yes",
                    }
                },
                **self.VALID_ARGS,
            )

    def test_x402_input_still_validates_after_patching(self, client):
        """The patch must not disturb the existing x402 path."""
        patch_process_payment_model(client)

        with pytest.raises(Exception) as exc_info:
            client.process_payment(
                paymentType="CRYPTO_X402",
                paymentInput={"cryptoX402": {"version": "1", "payload": {"scheme": "exact"}}},
                **self.VALID_ARGS,
            )

        assert not isinstance(exc_info.value, ParamValidationError)


class TestPaymentManagerAppliesPatch:
    """PaymentManager must patch its client at construction time."""

    def test_manager_client_supports_mpp_after_init(self):
        from unittest.mock import patch

        from bedrock_agentcore.payments.manager import PaymentManager

        real_client = make_client()
        assert "mpp" not in shape_map(real_client)["PaymentInput"]["members"]

        with patch("boto3.Session") as mock_session:
            mock_session.return_value.region_name = "us-west-2"
            mock_session.return_value.client.return_value = real_client
            PaymentManager(payment_manager_arn=PAYMENT_MANAGER_ARN, region_name="us-west-2")

        assert "mpp" in shape_map(real_client)["PaymentInput"]["members"]
