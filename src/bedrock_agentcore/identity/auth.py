"""Authentication decorators and utilities for Bedrock AgentCore SDK."""

import asyncio
import contextvars
import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional

import boto3

from bedrock_agentcore.runtime import BedrockAgentCoreContext
from bedrock_agentcore.services.identity import IdentityClient, TokenPoller

logger = logging.getLogger("bedrock_agentcore.auth")
logger.setLevel("INFO")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def requires_access_token(
    *,
    provider_name: str,
    into: str = "access_token",
    scopes: List[str],
    on_auth_url: Optional[Callable[[str], Any]] = None,
    auth_flow: Literal["M2M", "USER_FEDERATION"],
    callback_url: Optional[str] = None,
    force_authentication: bool = False,
    token_poller: Optional[TokenPoller] = None,
    custom_state: Optional[str] = None,
    custom_parameters: Optional[Dict[str, str]] = None,
) -> Callable:
    """Decorator that fetches an OAuth2 access token before calling the decorated function.

    Args:
        provider_name: The credential provider name
        into: Parameter name to inject the token into
        scopes: OAuth2 scopes to request
        on_auth_url: Callback for handling authorization URLs
        auth_flow: Authentication flow type ("M2M" or "USER_FEDERATION")
        callback_url: OAuth2 callback URL
        force_authentication: Force re-authentication
        token_poller: Custom token poller implementation
        custom_state: A state that allows applications to verify the validity of callbacks to callback_url
        custom_parameters: A map of custom parameters to include in authorization request to the credential provider
                           Note: these parameters are in addition to standard OAuth 2.0 flow parameters

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        client = IdentityClient(_get_region())

        async def _get_token() -> str:
            """Common token fetching logic."""
            return await client.get_token(
                provider_name=provider_name,
                agent_identity_token=await _get_workload_access_token(client),
                scopes=scopes,
                on_auth_url=on_auth_url,
                auth_flow=auth_flow,
                callback_url=_get_oauth2_callback_url(callback_url),
                force_authentication=force_authentication,
                token_poller=token_poller,
                custom_state=custom_state,
                custom_parameters=custom_parameters,
            )

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs_func: Any) -> Any:
            token = await _get_token()
            kwargs_func[into] = token
            return await func(*args, **kwargs_func)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs_func: Any) -> Any:
            if _has_running_loop():
                # for async env, eg. runtime
                ctx = contextvars.copy_context()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(ctx.run, asyncio.run, _get_token())
                    token = future.result()
            else:
                # for sync env, eg. local dev
                token = asyncio.run(_get_token())

            kwargs_func[into] = token
            return func(*args, **kwargs_func)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def requires_api_key(*, provider_name: str, into: str = "api_key") -> Callable:
    """Decorator that fetches an API key before calling the decorated function.

    Args:
        provider_name: The credential provider name
        into: Parameter name to inject the API key into

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        client = IdentityClient(_get_region())

        async def _get_api_key():
            return await client.get_api_key(
                provider_name=provider_name,
                agent_identity_token=await _get_workload_access_token(client),
            )

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            api_key = await _get_api_key()
            kwargs[into] = api_key
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if _has_running_loop():
                # for async env, eg. runtime
                ctx = contextvars.copy_context()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(ctx.run, asyncio.run, _get_api_key())
                    api_key = future.result()
            else:
                # for sync env, eg. local dev
                api_key = asyncio.run(_get_api_key())

            kwargs[into] = api_key
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _get_oauth2_callback_url(user_provided_oauth2_callback_url: Optional[str]):
    if user_provided_oauth2_callback_url:
        return user_provided_oauth2_callback_url

    return BedrockAgentCoreContext.get_oauth2_callback_url()


async def _get_workload_access_token(client: IdentityClient) -> str:
    token = BedrockAgentCoreContext.get_workload_access_token()
    if token is not None:
        return token
    else:
        # workload access token context var was not set, so we should be running in a local dev environment
        if os.getenv("DOCKER_CONTAINER") == "1":
            raise ValueError(
                "Workload access token has not been set. If invoking agent runtime via SIGV4 inbound auth, "
                "please specify the X-Amzn-Bedrock-AgentCore-Runtime-User-Id header and retry. "
                "For details, see - https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-oauth.html"
            )

        return await _set_up_local_auth(client)


async def _set_up_local_auth(client: IdentityClient) -> str:
    import json
    import uuid
    from pathlib import Path

    config_path = Path(".agentcore.json")
    workload_identity_name = None
    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = json.load(file) or {}
        except Exception:
            print("Could not find existing workload identity and user id")

    workload_identity_name = config.get("workload_identity_name")
    if workload_identity_name:
        print(f"Found existing workload identity from {config_path.absolute()}: {workload_identity_name}")
    else:
        workload_identity_name = client.create_workload_identity()["name"]
        print("Created a workload identity")

    user_id = config.get("user_id")
    if user_id:
        print(f"Found existing user id from {config_path.absolute()}: {user_id}")
    else:
        user_id = uuid.uuid4().hex[:8]
        print("Created an user id")

    try:
        config = {"workload_identity_name": workload_identity_name, "user_id": user_id}
        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
    except Exception:
        print("Warning: could not write the created workload identity to file")

    return client.get_workload_access_token(workload_identity_name, user_id=user_id)["workloadAccessToken"]


def _get_region() -> str:
    region_env = os.getenv("AWS_REGION", None)
    if region_env is not None:
        return region_env

    return boto3.Session().region_name or "us-west-2"


def _has_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False
