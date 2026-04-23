"""Integration tests for IdentityClient passthrough and __getattr__ methods."""

import os
import time

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.services.identity import IdentityClient


@pytest.mark.integration
class TestIdentityClientPassthrough:
    """Integration tests for IdentityClient passthrough via __getattr__.

    Tests read-only operations that don't require pre-existing resources.
    """

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.client = IdentityClient(region=cls.region)

    @pytest.mark.order(1)
    def test_list_oauth2_credential_providers_passthrough(self):
        response = self.client.list_oauth2_credential_providers()
        assert "credentialProviders" in response

    @pytest.mark.order(2)
    def test_list_api_key_credential_providers_passthrough(self):
        response = self.client.list_api_key_credential_providers()
        assert "credentialProviders" in response

    @pytest.mark.order(3)
    def test_list_oauth2_snake_case(self):
        response = self.client.list_oauth2_credential_providers(
            max_results=10,
        )
        assert "credentialProviders" in response

    @pytest.mark.order(4)
    def test_get_nonexistent_oauth2_provider(self):
        with pytest.raises(ClientError) as exc_info:
            self.client.get_oauth2_credential_provider(
                name="nonexistent-provider",
            )
        assert exc_info.value.response["Error"]["Code"] in (
            "ResourceNotFoundException",
            "AccessDeniedException",
        )

    @pytest.mark.order(5)
    def test_get_nonexistent_api_key_provider(self):
        with pytest.raises(ClientError) as exc_info:
            self.client.get_api_key_credential_provider(
                name="nonexistent-provider",
            )
        assert exc_info.value.response["Error"]["Code"] in (
            "ResourceNotFoundException",
            "AccessDeniedException",
        )

    @pytest.mark.order(6)
    def test_non_allowlisted_method_raises(self):
        with pytest.raises(AttributeError):
            self.client.not_a_real_method()


@pytest.mark.integration
class TestIdentityClientOauth2Crud:
    """Integration tests for OAuth2 credential provider CRUD via passthrough.

    Requires COGNITO_POOL_ID, COGNITO_CLIENT_ID, COGNITO_CLIENT_SECRET.
    """

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.pool_id = os.environ.get("COGNITO_POOL_ID")
        cls.client_id = os.environ.get("COGNITO_CLIENT_ID")
        cls.client_secret = os.environ.get("COGNITO_CLIENT_SECRET")
        if not all([cls.pool_id, cls.client_id, cls.client_secret]):
            pytest.skip("COGNITO_POOL_ID, COGNITO_CLIENT_ID, and COGNITO_CLIENT_SECRET must all be set")
        cls.client = IdentityClient(region=cls.region)
        cls.discovery_url = (
            f"https://cognito-idp.{cls.region}.amazonaws.com/{cls.pool_id}/.well-known/openid-configuration"
        )
        cls.provider_name = f"sdk-integ-{int(time.time())}"

    @classmethod
    def teardown_class(cls):
        try:
            cls.client.delete_oauth2_credential_provider(
                name=cls.provider_name,
            )
        except Exception as e:
            print(f"Teardown: {e}")

    @pytest.mark.order(10)
    def test_create_oauth2_credential_provider(self):
        self.client.create_oauth2_credential_provider(
            name=self.provider_name,
            credentialProviderVendor="CustomOauth2",
            oauth2ProviderConfigInput={
                "customOauth2ProviderConfig": {
                    "oauthDiscovery": {
                        "discoveryUrl": self.discovery_url,
                    },
                    "clientId": self.client_id,
                    "clientSecret": self.client_secret,
                }
            },
        )
        provider = self.client.get_oauth2_credential_provider(
            name=self.provider_name,
        )
        assert provider["name"] == self.provider_name

    @pytest.mark.order(11)
    def test_get_oauth2_provider_passthrough(self):
        provider = self.client.get_oauth2_credential_provider(
            name=self.provider_name,
        )
        assert provider["name"] == self.provider_name

    @pytest.mark.order(12)
    def test_delete_oauth2_credential_provider(self):
        self.client.delete_oauth2_credential_provider(
            name=self.provider_name,
        )
        # Provider may take a moment to delete
        import time

        time.sleep(5)
        with pytest.raises(ClientError):
            self.client.get_oauth2_credential_provider(
                name=self.provider_name,
            )
