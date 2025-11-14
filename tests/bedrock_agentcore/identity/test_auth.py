"""Tests for Bedrock AgentCore authentication decorators and functions."""

import json
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bedrock_agentcore.identity.auth import (
    _get_region,
    _get_workload_access_token,
    _set_up_local_auth,
    requires_access_token,
    requires_api_key,
)


class TestRequiresAccessTokenDecorator:
    """Test the requires_access_token decorator."""

    @pytest.mark.asyncio
    async def test_async_function_decoration(self):
        """Test decorator with async function."""
        # Mock IdentityClient
        with patch("bedrock_agentcore.identity.auth.IdentityClient") as mock_identity_client_class:
            mock_client = Mock()
            mock_identity_client_class.return_value = mock_client

            # Mock _get_workload_access_token
            with patch(
                "bedrock_agentcore.identity.auth._get_workload_access_token", new_callable=AsyncMock
            ) as mock_get_agent_token:
                mock_get_agent_token.return_value = "test-agent-token"

                # Mock client.get_token
                mock_client.get_token = AsyncMock(return_value="test-access-token")

                # Mock _get_region
                with patch("bedrock_agentcore.identity.auth._get_region", return_value="us-west-2"):

                    @requires_access_token(provider_name="test-provider", scopes=["read", "write"], auth_flow="M2M")
                    async def test_async_func(param1, access_token=None):
                        return f"param1={param1}, token={access_token}"

                    result = await test_async_func("value1")

                    assert result == "param1=value1, token=test-access-token"
                    mock_client.get_token.assert_called_once_with(
                        provider_name="test-provider",
                        agent_identity_token="test-agent-token",
                        scopes=["read", "write"],
                        on_auth_url=None,
                        auth_flow="M2M",
                        callback_url=None,
                        force_authentication=False,
                        token_poller=None,
                        custom_state=None,
                        custom_parameters=None,
                    )

    def test_sync_function_decoration_no_running_loop(self):
        """Test decorator with sync function when no asyncio loop is running."""
        # Mock IdentityClient
        with patch("bedrock_agentcore.identity.auth.IdentityClient") as mock_identity_client_class:
            mock_client = Mock()
            mock_identity_client_class.return_value = mock_client

            # Mock _get_workload_access_token
            with patch(
                "bedrock_agentcore.identity.auth._get_workload_access_token", new_callable=AsyncMock
            ) as mock_get_agent_token:
                mock_get_agent_token.return_value = "test-agent-token"

                # Mock client.get_token
                mock_client.get_token = AsyncMock(return_value="test-access-token")

                # Mock _get_region
                with patch("bedrock_agentcore.identity.auth._get_region", return_value="us-west-2"):

                    @requires_access_token(provider_name="test-provider", scopes=["read"], auth_flow="USER_FEDERATION")
                    def test_sync_func(param1, access_token=None):
                        return f"param1={param1}, token={access_token}"

                    # Mock asyncio.get_running_loop to raise RuntimeError (no loop)
                    with patch("asyncio.get_running_loop", side_effect=RuntimeError("no running loop")):
                        with patch("asyncio.run") as mock_asyncio_run:
                            mock_asyncio_run.return_value = "test-access-token"

                            result = test_sync_func("value1")

                            assert result == "param1=value1, token=test-access-token"
                            mock_asyncio_run.assert_called_once()

    def test_sync_function_decoration_with_running_loop(self):
        """Test decorator with sync function when asyncio loop is running."""
        # Mock IdentityClient
        with patch("bedrock_agentcore.identity.auth.IdentityClient") as mock_identity_client_class:
            mock_client = Mock()
            mock_identity_client_class.return_value = mock_client

            # Mock _get_workload_access_token
            with patch(
                "bedrock_agentcore.identity.auth._get_workload_access_token", new_callable=AsyncMock
            ) as mock_get_agent_token:
                mock_get_agent_token.return_value = "test-agent-token"

                # Mock client.get_token
                mock_client.get_token = AsyncMock(return_value="test-access-token")

                # Mock _get_region
                with patch("bedrock_agentcore.identity.auth._get_region", return_value="us-west-2"):

                    @requires_access_token(provider_name="test-provider", scopes=["read"], auth_flow="M2M")
                    def test_sync_func(param1, access_token=None):
                        return f"param1={param1}, token={access_token}"

                    # Mock asyncio.get_running_loop to succeed (loop is running)
                    with patch("asyncio.get_running_loop"):
                        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
                            mock_executor = Mock()
                            mock_executor_class.return_value.__enter__.return_value = mock_executor

                            mock_future = Mock()
                            mock_future.result.return_value = "test-access-token"
                            mock_executor.submit.return_value = mock_future

                            result = test_sync_func("value1")

                            assert result == "param1=value1, token=test-access-token"
                            mock_executor.submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_parameter_name(self):
        """Test decorator with custom parameter name for token injection."""
        # Mock IdentityClient
        with patch("bedrock_agentcore.identity.auth.IdentityClient") as mock_identity_client_class:
            mock_client = Mock()
            mock_identity_client_class.return_value = mock_client

            # Mock _get_workload_access_token
            with patch(
                "bedrock_agentcore.identity.auth._get_workload_access_token", new_callable=AsyncMock
            ) as mock_get_agent_token:
                mock_get_agent_token.return_value = "test-agent-token"

                # Mock client.get_token
                mock_client.get_token = AsyncMock(return_value="test-access-token")

                # Mock _get_region
                with patch("bedrock_agentcore.identity.auth._get_region", return_value="us-west-2"):

                    @requires_access_token(
                        provider_name="test-provider", into="my_token", scopes=["read"], auth_flow="M2M"
                    )
                    async def test_func(param1, my_token=None):
                        return f"param1={param1}, token={my_token}"

                    result = await test_func("value1")

                    assert result == "param1=value1, token=test-access-token"
                    mock_client.get_token.assert_called_once_with(
                        provider_name="test-provider",
                        agent_identity_token="test-agent-token",
                        scopes=["read"],
                        on_auth_url=None,
                        auth_flow="M2M",
                        callback_url=None,
                        force_authentication=False,
                        token_poller=None,
                        custom_state=None,
                        custom_parameters=None,
                    )

    @pytest.mark.asyncio
    async def test_with_all_optional_parameters(self):
        """Test decorator with all optional parameters."""
        # Mock IdentityClient
        with patch("bedrock_agentcore.identity.auth.IdentityClient") as mock_identity_client_class:
            mock_client = Mock()
            mock_identity_client_class.return_value = mock_client

            # Mock _get_workload_access_token
            with patch(
                "bedrock_agentcore.identity.auth._get_workload_access_token", new_callable=AsyncMock
            ) as mock_get_agent_token:
                mock_get_agent_token.return_value = "test-agent-token"

                # Mock client.get_token
                mock_client.get_token = AsyncMock(return_value="test-access-token")

                # Mock _get_region
                with patch("bedrock_agentcore.identity.auth._get_region", return_value="us-west-2"):
                    # Mock callback
                    callback_called = False

                    def on_auth_url(url):
                        nonlocal callback_called
                        callback_called = True

                    # Mock token poller
                    mock_poller = Mock()

                    @requires_access_token(
                        provider_name="test-provider",
                        into="token",
                        scopes=["read", "write"],
                        on_auth_url=on_auth_url,
                        auth_flow="USER_FEDERATION",
                        callback_url="https://example.com/callback",
                        force_authentication=True,
                        token_poller=mock_poller,
                        custom_state="myAppState",
                        custom_parameters=None,
                    )
                    async def test_func(token=None):
                        return f"token={token}"

                    result = await test_func()

                    assert result == "token=test-access-token"
                    mock_client.get_token.assert_called_once_with(
                        provider_name="test-provider",
                        agent_identity_token="test-agent-token",
                        scopes=["read", "write"],
                        on_auth_url=on_auth_url,
                        auth_flow="USER_FEDERATION",
                        callback_url="https://example.com/callback",
                        force_authentication=True,
                        token_poller=mock_poller,
                        custom_state="myAppState",
                        custom_parameters=None,
                    )

    @pytest.mark.asyncio
    async def test_custom_parameters_passed_to_client(self):
        with patch("bedrock_agentcore.identity.auth.IdentityClient") as mock_identity_client_class:
            mock_client = Mock()
            mock_identity_client_class.return_value = mock_client

            with patch(
                "bedrock_agentcore.identity.auth._get_workload_access_token", new_callable=AsyncMock
            ) as mock_get_agent_token:
                mock_get_agent_token.return_value = "test-agent-token"
                mock_client.get_token = AsyncMock(return_value="test-access-token")

                with patch("bedrock_agentcore.identity.auth._get_region", return_value="us-west-2"):
                    custom_params = {"param1": "value1", "param2": "value2"}

                    @requires_access_token(
                        provider_name="test-provider",
                        scopes=["read"],
                        auth_flow="USER_FEDERATION",
                        custom_parameters=custom_params,
                    )
                    async def test_func(access_token=None):
                        return access_token

                    result = await test_func()

                    assert result == "test-access-token"
                    mock_client.get_token.assert_called_once_with(
                        provider_name="test-provider",
                        agent_identity_token="test-agent-token",
                        scopes=["read"],
                        auth_flow="USER_FEDERATION",
                        callback_url=None,
                        force_authentication=False,
                        token_poller=None,
                        custom_state=None,
                        on_auth_url=None,
                        custom_parameters=custom_params,
                    )


class TestRequiresApiKeyDecorator:
    """Test the requires_api_key decorator."""

    @pytest.mark.asyncio
    async def test_async_function_decoration(self):
        """Test decorator with async function."""
        # Mock IdentityClient
        with patch("bedrock_agentcore.identity.auth.IdentityClient") as mock_identity_client_class:
            mock_client = Mock()
            mock_identity_client_class.return_value = mock_client

            # Mock _get_workload_access_token
            with patch(
                "bedrock_agentcore.identity.auth._get_workload_access_token", new_callable=AsyncMock
            ) as mock_get_agent_token:
                mock_get_agent_token.return_value = "test-agent-token"

                # Mock client.get_api_key
                mock_client.get_api_key = AsyncMock(return_value="test-api-key")

                # Mock _get_region
                with patch("bedrock_agentcore.identity.auth._get_region", return_value="us-west-2"):

                    @requires_api_key(provider_name="test-provider")
                    async def test_async_func(param1, api_key=None):
                        return f"param1={param1}, key={api_key}"

                    result = await test_async_func("value1")

                    assert result == "param1=value1, key=test-api-key"
                    mock_client.get_api_key.assert_called_once_with(
                        provider_name="test-provider", agent_identity_token="test-agent-token"
                    )

    def test_sync_function_decoration_no_running_loop(self):
        """Test decorator with sync function when no asyncio loop is running."""
        # Mock IdentityClient
        with patch("bedrock_agentcore.identity.auth.IdentityClient") as mock_identity_client_class:
            mock_client = Mock()
            mock_identity_client_class.return_value = mock_client

            # Mock _get_workload_access_token
            with patch(
                "bedrock_agentcore.identity.auth._get_workload_access_token", new_callable=AsyncMock
            ) as mock_get_agent_token:
                mock_get_agent_token.return_value = "test-agent-token"

                # Mock client.get_api_key
                mock_client.get_api_key = AsyncMock(return_value="test-api-key")

                # Mock _get_region
                with patch("bedrock_agentcore.identity.auth._get_region", return_value="us-west-2"):

                    @requires_api_key(provider_name="test-provider", into="my_key")
                    def test_sync_func(param1, my_key=None):
                        return f"param1={param1}, key={my_key}"

                    # Mock asyncio.get_running_loop to raise RuntimeError (no loop)
                    with patch("asyncio.get_running_loop", side_effect=RuntimeError("no running loop")):
                        with patch("asyncio.run") as mock_asyncio_run:
                            mock_asyncio_run.return_value = "test-api-key"

                            result = test_sync_func("value1")

                            assert result == "param1=value1, key=test-api-key"

    def test_sync_function_decoration_with_running_loop(self):
        """Test decorator with sync function when asyncio loop is running."""
        # Mock IdentityClient
        with patch("bedrock_agentcore.identity.auth.IdentityClient") as mock_identity_client_class:
            mock_client = Mock()
            mock_identity_client_class.return_value = mock_client

            # Mock _get_workload_access_token
            with patch(
                "bedrock_agentcore.identity.auth._get_workload_access_token", new_callable=AsyncMock
            ) as mock_get_agent_token:
                mock_get_agent_token.return_value = "test-agent-token"

                # Mock client.get_api_key
                mock_client.get_api_key = AsyncMock(return_value="test-api-key")

                # Mock _get_region
                with patch("bedrock_agentcore.identity.auth._get_region", return_value="us-west-2"):

                    @requires_api_key(provider_name="test-provider")
                    def test_sync_func(param1, api_key=None):
                        return f"param1={param1}, key={api_key}"

                    # Mock asyncio.get_running_loop to succeed (loop is running)
                    with patch("asyncio.get_running_loop"):
                        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
                            mock_executor = Mock()
                            mock_executor_class.return_value.__enter__.return_value = mock_executor

                            mock_future = Mock()
                            mock_future.result.return_value = "test-api-key"
                            mock_executor.submit.return_value = mock_future

                            result = test_sync_func("value1")

                            assert result == "param1=value1, key=test-api-key"
                            mock_executor.submit.assert_called_once()


class TestSetUpLocalAuth:
    """Test _set_up_local_auth function."""

    @pytest.mark.asyncio
    async def test_existing_config(self, tmp_path):
        """Test when config file exists with both workload_identity_name and user_id."""
        config_content = {"workload_identity_name": "existing-workload-123", "user_id": "existing-user-456"}
        mock_client = Mock()
        mock_client.get_workload_access_token = Mock(return_value={"workloadAccessToken": "test-access-token-456"})

        # Create the config file in the temp directory
        config_file = tmp_path / ".agentcore.json"
        config_file.write_text(json.dumps(config_content))

        # Change to the temp directory for the test
        import os

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = await _set_up_local_auth(mock_client)

            # Should use existing workload identity and user_id
            assert result == "test-access-token-456"
            mock_client.create_workload_identity.assert_not_called()
            mock_client.get_workload_access_token.assert_called_once_with(
                "existing-workload-123", user_id="existing-user-456"
            )
        finally:
            os.chdir(original_dir)

    @pytest.mark.asyncio
    async def test_no_config(self, tmp_path):
        """Test when config file doesn't exist."""
        mock_client = Mock()
        mock_client.create_workload_identity = Mock(return_value={"name": "test-workload-123"})
        mock_client.get_workload_access_token = Mock(return_value={"workloadAccessToken": "test-access-token-456"})

        # Change to the temp directory for the test
        import os

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            with patch("uuid.uuid4") as mock_uuid:
                mock_uuid.return_value.hex = "abcd1234efgh5678"

                result = await _set_up_local_auth(mock_client)

                # Should create new workload identity and user_id
                assert result == "test-access-token-456"
                mock_client.create_workload_identity.assert_called_once()
                mock_client.get_workload_access_token.assert_called_once_with("test-workload-123", user_id="abcd1234")

                # Verify that the config file was created
                config_file = tmp_path / ".agentcore.json"
                assert config_file.exists()

                # Verify the config file content
                saved_config = json.loads(config_file.read_text())
                assert saved_config["workload_identity_name"] == "test-workload-123"
                assert saved_config["user_id"] == "abcd1234"
        finally:
            os.chdir(original_dir)


class TestGetRegion:
    """Test _get_region function."""

    def test_get_region_from_env_var(self):
        """Test getting region from AWS_REGION environment variable."""
        with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}):
            result = _get_region()
            assert result == "us-east-1"

    def test_get_region_from_config_file(self):
        """Test getting region from boto3 session when AWS_REGION is not set."""
        with patch.dict(os.environ, {}, clear=True):  # Clear AWS_REGION
            with patch("boto3.Session") as mock_session_class:
                mock_session = Mock()
                mock_session.region_name = "eu-west-1"
                mock_session_class.return_value = mock_session

                result = _get_region()
                assert result == "eu-west-1"


class TestGetWorkloadAccessToken:
    """Test _get_workload_access_token function."""

    @pytest.mark.asyncio
    async def test_token_from_context(self):
        """Test when workload access token is available from context."""
        mock_client = Mock()

        with patch(
            "bedrock_agentcore.identity.auth.BedrockAgentCoreContext.get_workload_access_token"
        ) as mock_get_token:
            mock_get_token.return_value = "context-token-123"

            result = await _get_workload_access_token(mock_client)

            assert result == "context-token-123"
            mock_get_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_context_local_dev(self):
        """Test when no context token and running in local dev environment."""
        mock_client = Mock()

        with patch(
            "bedrock_agentcore.identity.auth.BedrockAgentCoreContext.get_workload_access_token"
        ) as mock_get_token:
            mock_get_token.return_value = None

            with patch("os.getenv") as mock_getenv:
                mock_getenv.return_value = None  # Not in Docker

                with patch("bedrock_agentcore.identity.auth._set_up_local_auth", new_callable=AsyncMock) as mock_setup:
                    mock_setup.return_value = "local-dev-token-456"

                    result = await _get_workload_access_token(mock_client)

                    assert result == "local-dev-token-456"
                    mock_get_token.assert_called_once()
                    mock_setup.assert_called_once_with(mock_client)

    @pytest.mark.asyncio
    async def test_no_context_docker_container(self):
        """Test when no context token and running in Docker container."""
        mock_client = Mock()

        with patch(
            "bedrock_agentcore.identity.auth.BedrockAgentCoreContext.get_workload_access_token"
        ) as mock_get_token:
            mock_get_token.return_value = None

            with patch("os.getenv") as mock_getenv:
                mock_getenv.return_value = "1"  # In Docker container

                with pytest.raises(
                    ValueError, match="Workload access token has not been set.*X-Amzn-Bedrock-AgentCore-Runtime-User-Id"
                ):
                    await _get_workload_access_token(mock_client)

                mock_get_token.assert_called_once()
