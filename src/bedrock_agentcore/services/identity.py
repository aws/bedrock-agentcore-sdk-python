"""The main high-level client for the Bedrock AgentCore Identity service."""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional

import boto3

from bedrock_agentcore._utils.endpoints import get_control_plane_endpoint, get_data_plane_endpoint


class TokenPoller(ABC):
    """Abstract base class for token polling implementations."""

    @abstractmethod
    async def poll_for_token(self) -> str:
        """Poll for a token and return it when available."""
        raise NotImplementedError


# Default configuration for the polling mechanism
DEFAULT_POLLING_INTERVAL_SECONDS = 5
DEFAULT_POLLING_TIMEOUT_SECONDS = 600


class _DefaultApiTokenPoller(TokenPoller):
    """Default implementation of token polling."""

    def __init__(self, auth_url: str, func: Callable[[], str | None]):
        """Initialize the token poller with auth URL and polling function."""
        self.auth_url = auth_url
        self.polling_func = func
        self.logger = logging.getLogger("bedrock_agentcore.default_token_poller")
        self.logger.setLevel("INFO")
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())

    async def poll_for_token(self) -> str:
        """Poll for a token until it becomes available or timeout occurs."""
        start_time = time.time()
        while time.time() - start_time < DEFAULT_POLLING_TIMEOUT_SECONDS:
            await asyncio.sleep(DEFAULT_POLLING_INTERVAL_SECONDS)

            self.logger.info("Polling for token for authorization url: %s", self.auth_url)
            resp = self.polling_func()
            if resp is not None:
                self.logger.info("Token is ready")
                return resp

        raise asyncio.TimeoutError(
            f"Polling timed out after {DEFAULT_POLLING_TIMEOUT_SECONDS} seconds. "
            + "User may not have completed authorization."
        )


class IdentityClient:
    """A high-level client for Bedrock AgentCore Identity."""

    def __init__(self, region: str):
        """Initialize the identity client with the specified region."""
        self.region = region
        self.cp_client = boto3.client(
            "bedrock-agentcore-control", region_name=region, endpoint_url=get_control_plane_endpoint(region)
        )
        self.identity_client = boto3.client(
            "bedrock-agentcore-control", region_name=region, endpoint_url=get_data_plane_endpoint(region)
        )
        self.dp_client = boto3.client(
            "bedrock-agentcore", region_name=region, endpoint_url=get_data_plane_endpoint(region)
        )
        self.logger = logging.getLogger("bedrock_agentcore.identity_client")

    def create_oauth2_credential_provider(self, req):
        """Create an OAuth2 credential provider."""
        self.logger.info("Creating OAuth2 credential provider...")
        return self.cp_client.create_oauth2_credential_provider(**req)

    def create_api_key_credential_provider(self, req):
        """Create an API key credential provider."""
        self.logger.info("Creating API key credential provider...")
        return self.cp_client.create_api_key_credential_provider(**req)

    def get_workload_access_token(
        self, workload_name: str, user_token: Optional[str] = None, user_id: Optional[str] = None
    ) -> Dict:
        """Get a workload access token using workload name and optionally user token."""
        if user_token:
            if user_id is not None:
                self.logger.warning("Both user token and user id are supplied, using user token")
            self.logger.info("Getting workload access token for JWT...")
            resp = self.dp_client.get_workload_access_token_for_jwt(workloadName=workload_name, userToken=user_token)
        elif user_id:
            self.logger.info("Getting workload access token for user id...")
            resp = self.dp_client.get_workload_access_token_for_user_id(workloadName=workload_name, userId=user_id)
        else:
            self.logger.info("Getting workload access token...")
            resp = self.dp_client.get_workload_access_token(workloadName=workload_name)

        self.logger.info("Successfully retrieved workload access token")
        return resp

    def create_workload_identity(self, name: Optional[str] = None) -> Dict:
        """Create workload identity with optional name."""
        self.logger.info("Creating workload identity...")
        if not name:
            name = f"workload-{uuid.uuid4().hex[:8]}"
        return self.identity_client.create_workload_identity(name=name)

    async def get_token(
        self,
        *,
        provider_name: str,
        scopes: Optional[List[str]] = None,
        agent_identity_token: str,
        on_auth_url: Optional[Callable[[str], Any]] = None,
        auth_flow: Literal["M2M", "USER_FEDERATION"],
        callback_url: Optional[str] = None,
        force_authentication: bool = False,
        token_poller: Optional[TokenPoller] = None,
    ) -> str:
        """Get an OAuth2 access token for the specified provider.

        Args:
            provider_name: The credential provider name
            scopes: Optional list of OAuth2 scopes to request
            agent_identity_token: Agent identity token for authentication
            on_auth_url: Callback for handling authorization URLs
            auth_flow: Authentication flow type ("M2M" or "USER_FEDERATION")
            callback_url: OAuth2 callback URL (must be pre-registered)
            force_authentication: Force re-authentication even if token exists in the token vault
            token_poller: Custom token poller implementation

        Returns:
            The access token string

        Raises:
            RequiresUserConsentException: When user consent is needed
            Various other exceptions for error conditions
        """
        self.logger.info("Getting OAuth2 token...")

        # Build parameters
        req = {
            "resourceCredentialProviderName": provider_name,
            "scopes": scopes,
            "oauth2Flow": auth_flow,
            "workloadIdentityToken": agent_identity_token,
        }

        # Add optional parameters
        if callback_url:
            req["resourceOauth2ReturnUrl"] = callback_url
        if force_authentication:
            req["forceAuthentication"] = force_authentication

        response = self.dp_client.get_resource_oauth2_token(**req)

        # If we got a token directly, return it
        if "accessToken" in response:
            return response["accessToken"]

        # If we got an authorization URL, handle the OAuth flow
        if "authorizationUrl" in response:
            auth_url = response["authorizationUrl"]
            # Notify about the auth URL if callback provided
            if on_auth_url:
                if asyncio.iscoroutinefunction(on_auth_url):
                    await on_auth_url(auth_url)
                else:
                    on_auth_url(auth_url)

            # only the initial request should have force authentication
            if force_authentication:
                req["forceAuthentication"] = False

            # Poll for the token
            active_poller = token_poller or _DefaultApiTokenPoller(
                auth_url, lambda: self.dp_client.get_resource_oauth2_token(**req).get("accessToken", None)
            )
            return await active_poller.poll_for_token()

        raise RuntimeError("Identity service did not return a token or an authorization URL.")

    async def get_api_key(self, *, provider_name: str, agent_identity_token: str) -> str:
        """Programmatically retrieves an API key from the Identity service."""
        self.logger.info("Getting API key...")
        req = {"resourceCredentialProviderName": provider_name, "workloadIdentityToken": agent_identity_token}

        return self.dp_client.get_resource_api_key(**req)["apiKey"]
