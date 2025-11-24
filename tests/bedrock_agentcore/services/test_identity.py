"""Tests for Bedrock AgentCore Identity Client functionality."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bedrock_agentcore.services.identity import (
    DEFAULT_POLLING_INTERVAL_SECONDS,
    DEFAULT_POLLING_TIMEOUT_SECONDS,
    IdentityClient,
    UserIdIdentifier,
    UserTokenIdentifier,
    _DefaultApiTokenPoller,
)


class TestIdentityClient:
    """Test IdentityClient functionality."""

    def test_initialization(self):
        """Test IdentityClient initialization."""
        region = "us-east-1"

        with patch("boto3.client") as mock_boto_client:
            client = IdentityClient(region)

            assert client.region == region
            mock_boto_client.assert_called_with(
                "bedrock-agentcore",
                region_name=region,
                endpoint_url="https://bedrock-agentcore.us-east-1.amazonaws.com",
            )

    def test_create_oauth2_credential_provider(self):
        """Test OAuth2 credential provider creation."""
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client

            identity_client = IdentityClient(region)

            # Test data
            req = {"name": "test-provider", "clientId": "test-client"}
            expected_response = {"providerId": "test-provider-id"}
            mock_client.create_oauth2_credential_provider.return_value = expected_response

            result = identity_client.create_oauth2_credential_provider(req)

            assert result == expected_response
            mock_client.create_oauth2_credential_provider.assert_called_once_with(**req)

    def test_create_api_key_credential_provider(self):
        """Test API key credential provider creation."""
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client

            identity_client = IdentityClient(region)

            # Test data
            req = {"name": "test-api-provider", "apiKeyName": "test-key"}
            expected_response = {"providerId": "test-api-provider-id"}
            mock_client.create_api_key_credential_provider.return_value = expected_response

            result = identity_client.create_api_key_credential_provider(req)

            assert result == expected_response
            mock_client.create_api_key_credential_provider.assert_called_once_with(**req)

    @pytest.mark.asyncio
    async def test_get_token_direct_response(self):
        """Test get_token when token is returned directly."""
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client

            identity_client = IdentityClient(region)

            # Test data
            provider_name = "test-provider"
            scopes = ["read", "write"]
            agent_identity_token = "test-agent-token"
            expected_token = "test-access-token"

            mock_client.get_resource_oauth2_token.return_value = {"accessToken": expected_token}

            result = await identity_client.get_token(
                provider_name=provider_name, scopes=scopes, agent_identity_token=agent_identity_token, auth_flow="M2M"
            )

            assert result == expected_token
            mock_client.get_resource_oauth2_token.assert_called_once_with(
                resourceCredentialProviderName=provider_name,
                scopes=scopes,
                oauth2Flow="M2M",
                workloadIdentityToken=agent_identity_token,
            )

    @pytest.mark.asyncio
    async def test_get_token_with_auth_url_polling(self):
        """Test get_token with authorization URL and polling."""
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client

            identity_client = IdentityClient(region)

            # Test data
            provider_name = "test-provider"
            agent_identity_token = "test-agent-token"
            auth_url = "https://example.com/auth"
            expected_token = "test-access-token"
            session_uri = "https://example-federation-authorization-request/12345"

            # First call returns auth URL, subsequent calls return token
            mock_client.get_resource_oauth2_token.side_effect = [
                {"authorizationUrl": auth_url},
                {"accessToken": expected_token},
                {"sessionUri": session_uri},
            ]

            # Mock the token poller
            mock_poller = Mock()
            mock_poller.poll_for_token = AsyncMock(return_value=expected_token)

            with patch("bedrock_agentcore.services.identity._DefaultApiTokenPoller", return_value=mock_poller):
                result = await identity_client.get_token(
                    provider_name=provider_name, agent_identity_token=agent_identity_token, auth_flow="USER_FEDERATION"
                )

            assert result == expected_token
            mock_poller.poll_for_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_token_with_auth_url_and_callback(self):
        """Test get_token with authorization URL and callback function."""
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client

            identity_client = IdentityClient(region)

            # Test data
            provider_name = "test-provider"
            agent_identity_token = "test-agent-token"
            auth_url = "https://example.com/auth"
            expected_token = "test-access-token"

            # Mock callback function
            callback_called = False

            def on_auth_url(url):
                nonlocal callback_called
                callback_called = True
                assert url == auth_url

            mock_client.get_resource_oauth2_token.return_value = {"authorizationUrl": auth_url}

            # Mock the token poller
            mock_poller = Mock()
            mock_poller.poll_for_token = AsyncMock(return_value=expected_token)

            with patch("bedrock_agentcore.services.identity._DefaultApiTokenPoller", return_value=mock_poller):
                result = await identity_client.get_token(
                    provider_name=provider_name,
                    agent_identity_token=agent_identity_token,
                    auth_flow="USER_FEDERATION",
                    on_auth_url=on_auth_url,
                )

            assert result == expected_token
            assert callback_called
            mock_poller.poll_for_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_token_with_async_callback(self):
        """Test get_token with async authorization URL callback."""
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client

            identity_client = IdentityClient(region)

            # Test data
            provider_name = "test-provider"
            agent_identity_token = "test-agent-token"
            auth_url = "https://example.com/auth"
            expected_token = "test-access-token"

            # Mock async callback function
            callback_called = False

            async def on_auth_url(url):
                nonlocal callback_called
                callback_called = True
                assert url == auth_url

            mock_client.get_resource_oauth2_token.return_value = {"authorizationUrl": auth_url}

            # Mock the token poller
            mock_poller = Mock()
            mock_poller.poll_for_token = AsyncMock(return_value=expected_token)

            with patch("bedrock_agentcore.services.identity._DefaultApiTokenPoller", return_value=mock_poller):
                result = await identity_client.get_token(
                    provider_name=provider_name,
                    agent_identity_token=agent_identity_token,
                    auth_flow="USER_FEDERATION",
                    on_auth_url=on_auth_url,
                )

            assert result == expected_token
            assert callback_called
            mock_poller.poll_for_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_token_with_optional_parameters(self):
        """Test get_token with all optional parameters."""
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client

            identity_client = IdentityClient(region)

            # Test data
            provider_name = "test-provider"
            scopes = ["read", "write"]
            agent_identity_token = "test-agent-token"
            callback_url = "https://example.com/callback"
            force_authentication = True
            custom_state = "myAppCustomState"
            expected_token = "test-access-token"

            mock_client.get_resource_oauth2_token.return_value = {"accessToken": expected_token}

            result = await identity_client.get_token(
                provider_name=provider_name,
                scopes=scopes,
                agent_identity_token=agent_identity_token,
                auth_flow="USER_FEDERATION",
                callback_url=callback_url,
                force_authentication=force_authentication,
                custom_state=custom_state,
            )

            assert result == expected_token
            mock_client.get_resource_oauth2_token.assert_called_once_with(
                resourceCredentialProviderName=provider_name,
                scopes=scopes,
                oauth2Flow="USER_FEDERATION",
                workloadIdentityToken=agent_identity_token,
                resourceOauth2ReturnUrl=callback_url,
                forceAuthentication=force_authentication,
                customState=custom_state,
            )

    @pytest.mark.asyncio
    async def test_get_token_with_custom_token_poller(self):
        """Test get_token with custom token poller."""
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client

            identity_client = IdentityClient(region)

            # Test data
            provider_name = "test-provider"
            agent_identity_token = "test-agent-token"
            auth_url = "https://example.com/auth"
            expected_token = "test-access-token"
            force_authentication = True
            session_uri = "https://example-federation-authorization-request/12345"

            mock_client.get_resource_oauth2_token.return_value = {
                "authorizationUrl": auth_url,
                "sessionUri": session_uri,
            }

            # Mock custom token poller
            custom_poller = Mock()
            custom_poller.poll_for_token = AsyncMock(return_value=expected_token)

            result = await identity_client.get_token(
                provider_name=provider_name,
                agent_identity_token=agent_identity_token,
                auth_flow="USER_FEDERATION",
                token_poller=custom_poller,
                force_authentication=force_authentication,
            )

            assert result == expected_token
            custom_poller.poll_for_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_token_with_custom_parameters(self):
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client

            identity_client = IdentityClient(region)

            provider_name = "test-provider"
            scopes = ["read", "write"]
            agent_identity_token = "test-agent-token"
            custom_parameters = {"param1": "value1", "param2": "value2"}
            expected_token = "test-access-token"

            mock_client.get_resource_oauth2_token.return_value = {"accessToken": expected_token}

            result = await identity_client.get_token(
                provider_name=provider_name,
                scopes=scopes,
                agent_identity_token=agent_identity_token,
                auth_flow="USER_FEDERATION",
                custom_parameters=custom_parameters,
            )

            assert result == expected_token
            mock_client.get_resource_oauth2_token.assert_called_once_with(
                resourceCredentialProviderName=provider_name,
                scopes=scopes,
                oauth2Flow="USER_FEDERATION",
                workloadIdentityToken=agent_identity_token,
                customParameters=custom_parameters,
            )

    @pytest.mark.asyncio
    async def test_get_api_key_success(self):
        """Test successful API key retrieval."""
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_client = Mock()
            mock_boto_client.return_value = mock_client

            identity_client = IdentityClient(region)

            # Test data
            provider_name = "test-provider"
            agent_identity_token = "test-agent-token"
            expected_api_key = "test-api-key"

            mock_client.get_resource_api_key.return_value = {"apiKey": expected_api_key}

            result = await identity_client.get_api_key(
                provider_name=provider_name, agent_identity_token=agent_identity_token
            )

            assert result == expected_api_key
            mock_client.get_resource_api_key.assert_called_once_with(
                resourceCredentialProviderName=provider_name, workloadIdentityToken=agent_identity_token
            )

    def test_get_workload_access_token_with_user_token(self):
        """Test get_workload_access_token with user token."""
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_cp_client = Mock()
            mock_identity_client = Mock()
            mock_dp_client = Mock()
            mock_boto_client.side_effect = [mock_cp_client, mock_identity_client, mock_dp_client]

            identity_client = IdentityClient(region)

            # Test data
            workload_name = "test-workload"
            user_token = "test-user-jwt-token"
            user_id = "test-user-id"  # This should be ignored when user_token is provided
            expected_response = {"workloadAccessToken": "test-workload-token"}

            mock_dp_client.get_workload_access_token_for_jwt.return_value = expected_response

            result = identity_client.get_workload_access_token(workload_name, user_token=user_token, user_id=user_id)

            assert result == expected_response
            mock_dp_client.get_workload_access_token_for_jwt.assert_called_once_with(
                workloadName=workload_name, userToken=user_token
            )
            # Should not call the user_id version
            mock_dp_client.get_workload_access_token_for_user_id.assert_not_called()
            mock_dp_client.get_workload_access_token.assert_not_called()

    def test_get_workload_access_token_with_user_id(self):
        """Test get_workload_access_token with user ID."""
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_cp_client = Mock()
            mock_identity_client = Mock()
            mock_dp_client = Mock()
            mock_boto_client.side_effect = [mock_cp_client, mock_identity_client, mock_dp_client]

            identity_client = IdentityClient(region)

            # Test data
            workload_name = "test-workload"
            user_id = "test-user-id"
            expected_response = {"workloadAccessToken": "test-workload-token"}

            mock_dp_client.get_workload_access_token_for_user_id.return_value = expected_response

            result = identity_client.get_workload_access_token(workload_name, user_id=user_id)

            assert result == expected_response
            mock_dp_client.get_workload_access_token_for_user_id.assert_called_once_with(
                workloadName=workload_name, userId=user_id
            )
            # Should not call other versions
            mock_dp_client.get_workload_access_token_for_jwt.assert_not_called()
            mock_dp_client.get_workload_access_token.assert_not_called()

    def test_get_workload_access_token_without_user_info(self):
        """Test get_workload_access_token without user token or ID."""
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_cp_client = Mock()
            mock_identity_client = Mock()
            mock_dp_client = Mock()
            mock_boto_client.side_effect = [mock_cp_client, mock_identity_client, mock_dp_client]

            identity_client = IdentityClient(region)

            # Test data
            workload_name = "test-workload"
            expected_response = {"workloadAccessToken": "test-workload-token"}

            mock_dp_client.get_workload_access_token.return_value = expected_response

            result = identity_client.get_workload_access_token(workload_name)

            assert result == expected_response
            mock_dp_client.get_workload_access_token.assert_called_once_with(workloadName=workload_name)
            # Should not call user-specific versions
            mock_dp_client.get_workload_access_token_for_jwt.assert_not_called()
            mock_dp_client.get_workload_access_token_for_user_id.assert_not_called()

    def test_create_workload_identity(self):
        """Test create_workload_identity with and without name."""
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_cp_client = Mock()
            mock_identity_client = Mock()
            mock_dp_client = Mock()
            mock_boto_client.side_effect = [mock_cp_client, mock_identity_client, mock_dp_client]

            identity_client = IdentityClient(region)

            # Test with provided name
            custom_name = "my-custom-workload"
            expected_response = {"name": custom_name, "workloadIdentityId": "workload-123"}
            mock_identity_client.create_workload_identity.return_value = expected_response

            result = identity_client.create_workload_identity(name=custom_name)

            assert result == expected_response
            mock_identity_client.create_workload_identity.assert_called_with(
                name=custom_name, allowedResourceOauth2ReturnUrls=[]
            )

            # Test without provided name (auto-generated)
            mock_identity_client.reset_mock()
            expected_response_auto = {"name": "workload-abcd1234", "workloadIdentityId": "workload-456"}
            mock_identity_client.create_workload_identity.return_value = expected_response_auto

            with patch("uuid.uuid4") as mock_uuid:
                mock_uuid.return_value.hex = "abcd1234efgh5678"

                result = identity_client.create_workload_identity()

                assert result == expected_response_auto
                mock_identity_client.create_workload_identity.assert_called_with(
                    name="workload-abcd1234", allowedResourceOauth2ReturnUrls=[]
                )

    def test_update_workload_identity(self):
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_cp_client = Mock()
            mock_identity_client = Mock()
            mock_dp_client = Mock()
            mock_boto_client.side_effect = [mock_cp_client, mock_identity_client, mock_dp_client]

            identity_client = IdentityClient(region)

            workload_name = "test-workload"
            allowed_urls = ["https://unit-test.com/callback", "https://test.com/oauth"]
            expected_response = {"name": workload_name, "allowedResourceOauth2ReturnUrls": allowed_urls}

            mock_identity_client.update_workload_identity.return_value = expected_response

            result = identity_client.update_workload_identity(workload_name, allowed_urls)

            assert result == expected_response
            mock_identity_client.update_workload_identity.assert_called_once_with(
                name=workload_name, allowedResourceOauth2ReturnUrls=allowed_urls
            )

    def test_get_workload_identity(self):
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_cp_client = Mock()
            mock_identity_client = Mock()
            mock_dp_client = Mock()
            mock_boto_client.side_effect = [mock_cp_client, mock_identity_client, mock_dp_client]

            identity_client = IdentityClient(region)

            workload_name = "test-workload"
            allowed_urls = ["https://unit-test.com/callback", "https://test.com/oauth"]
            expected_response = {"name": workload_name, "allowedResourceOauth2ReturnUrls": allowed_urls}

            mock_cp_client.get_workload_identity.return_value = expected_response

            result = identity_client.get_workload_identity(workload_name)

            assert result == expected_response
            mock_cp_client.get_workload_identity.assert_called_once_with(name=workload_name)

    def test_complete_resource_token_auth_with_user_id(self):
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_cp_client = Mock()
            mock_identity_client = Mock()
            mock_dp_client = Mock()
            mock_boto_client.side_effect = [mock_cp_client, mock_identity_client, mock_dp_client]

            identity_client = IdentityClient(region)

            session_uri = "https://unit-test.com/session/123"
            user_id = "test-user-123"
            user_identifier = UserIdIdentifier(user_id=user_id)
            expected_response = {}

            mock_dp_client.complete_resource_token_auth.return_value = expected_response

            result = identity_client.complete_resource_token_auth(session_uri, user_identifier)

            assert result == expected_response
            mock_dp_client.complete_resource_token_auth.assert_called_once_with(
                userIdentifier={"userId": user_id}, sessionUri=session_uri
            )

    def test_complete_resource_token_auth_with_user_token(self):
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_cp_client = Mock()
            mock_identity_client = Mock()
            mock_dp_client = Mock()
            mock_boto_client.side_effect = [mock_cp_client, mock_identity_client, mock_dp_client]

            identity_client = IdentityClient(region)

            session_uri = "https://example.com/session/456"
            user_token = "user-unit-test-token"
            user_identifier = UserTokenIdentifier(user_token=user_token)
            expected_response = {}

            mock_dp_client.complete_resource_token_auth.return_value = expected_response

            result = identity_client.complete_resource_token_auth(session_uri, user_identifier)

            assert result == expected_response
            mock_dp_client.complete_resource_token_auth.assert_called_once_with(
                userIdentifier={"userToken": user_token}, sessionUri=session_uri
            )

    def test_complete_resource_token_auth_with_invalid_identifier(self):
        region = "us-west-2"

        with patch("boto3.client") as mock_boto_client:
            mock_cp_client = Mock()
            mock_identity_client = Mock()
            mock_dp_client = Mock()
            mock_boto_client.side_effect = [mock_cp_client, mock_identity_client, mock_dp_client]

            identity_client = IdentityClient(region)

            session_uri = "https://unit-test.com/session/789"
            invalid_identifier = "invalid-string"  # Not a UserIdIdentifier or UserTokenIdentifier

            with pytest.raises(ValueError, match="Unexpected UserIdentifier"):
                identity_client.complete_resource_token_auth(session_uri, invalid_identifier)  # type: ignore - unit test


class TestDefaultApiTokenPoller:
    """Test DefaultApiTokenPoller implementation."""

    def test_initialization(self):
        """Test DefaultApiTokenPoller initialization."""
        auth_url = "https://example.com/auth"
        mock_func = Mock()

        poller = _DefaultApiTokenPoller(auth_url, mock_func)

        assert poller.auth_url == auth_url
        assert poller.polling_func == mock_func

    @pytest.mark.asyncio
    async def test_poll_for_token_success_immediate(self):
        """Test successful token polling that returns immediately."""
        auth_url = "https://example.com/auth"
        expected_token = "test-token-123"
        mock_func = Mock(return_value=expected_token)

        poller = _DefaultApiTokenPoller(auth_url, mock_func)

        with patch("asyncio.sleep") as mock_sleep:
            token = await poller.poll_for_token()

            assert token == expected_token
            mock_func.assert_called_once()
            mock_sleep.assert_called_once_with(DEFAULT_POLLING_INTERVAL_SECONDS)

    @pytest.mark.asyncio
    async def test_poll_for_token_success_after_retries(self):
        """Test successful token polling after several retries."""
        auth_url = "https://example.com/auth"
        expected_token = "test-token-456"

        # Mock function returns None twice, then returns token
        mock_func = Mock(side_effect=[None, None, expected_token])

        poller = _DefaultApiTokenPoller(auth_url, mock_func)

        with patch("asyncio.sleep") as mock_sleep:
            token = await poller.poll_for_token()

            assert token == expected_token
            assert mock_func.call_count == 3
            assert mock_sleep.call_count == 3

    @pytest.mark.asyncio
    async def test_poll_for_token_timeout(self):
        """Test that polling times out after the configured timeout."""
        auth_url = "https://example.com/auth"
        mock_func = Mock(return_value=None)  # Always returns None

        poller = _DefaultApiTokenPoller(auth_url, mock_func)

        # Mock time.time to simulate timeout quickly
        start_time = 1000.0
        timeout_time = start_time + DEFAULT_POLLING_TIMEOUT_SECONDS + 1

        with patch("time.time", side_effect=[start_time, timeout_time]):
            with patch("asyncio.sleep"):
                with pytest.raises(asyncio.TimeoutError) as exc_info:
                    await poller.poll_for_token()

                assert "Polling timed out" in str(exc_info.value)
                assert f"{DEFAULT_POLLING_TIMEOUT_SECONDS} seconds" in str(exc_info.value)
