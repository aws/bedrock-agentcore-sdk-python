"""Authentication decorators and utilities for Bedrock AgentCore SDK."""

import asyncio
import contextvars
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any, Callable, List, Literal, Optional

import boto3

from bedrock_agentcore.runtime import BedrockAgentCoreContext
from bedrock_agentcore.services.identity import IdentityClient, TokenPoller

logger = logging.getLogger("bedrock_agentcore.auth")
logger.setLevel("INFO")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

# Module-level cached executor for better resource management
_executor_pool: Optional[ThreadPoolExecutor] = None


def _get_thread_pool() -> ThreadPoolExecutor:
    """Get or create cached thread pool executor for better resource management."""
    global _executor_pool
    if _executor_pool is None:
        _executor_pool = ThreadPoolExecutor(
            max_workers=2, 
            thread_name_prefix="auth-bridge"
        )
    return _executor_pool


def _run_async_in_sync_context(async_func: Callable) -> Any:
    """Run async function in sync context with proper resource management."""
    if _has_running_loop():
        # Async environment - use thread pool (compatible with existing tests)
        ctx = contextvars.copy_context()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(ctx.run, asyncio.run, async_func())
            return future.result()
    else:
        # Sync environment - direct async run
        return asyncio.run(async_func())


def _create_credential_decorator(
    credential_fetcher: Callable[[IdentityClient], Any],
    into: str = "credential"
) -> Callable:
    """Base factory for creating credential decorators.
    
    This eliminates code duplication between requires_access_token and requires_api_key
    by providing a common pattern for credential injection decorators.
    
    Args:
        credential_fetcher: Async function that takes IdentityClient and returns credentials
        into: Parameter name to inject credential into
        
    Returns:
        Decorator function that handles both sync and async functions
    """
    
    def decorator(func: Callable) -> Callable:
        client = IdentityClient(_get_region())
        
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            credential = await credential_fetcher(client)
            kwargs[into] = credential
            return await func(*args, **kwargs)
        
        @wraps(func)  
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            credential = _run_async_in_sync_context(
                lambda: credential_fetcher(client)
            )
            kwargs[into] = credential
            return func(*args, **kwargs)
            
        # Return appropriate wrapper based on function type
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


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

    Returns:
        Decorator function
    """
    
    async def fetch_access_token(client: IdentityClient) -> str:
        """Fetch OAuth2 access token using the identity client."""
        return await client.get_token(
            provider_name=provider_name,
            agent_identity_token=await _get_workload_access_token(client),
            scopes=scopes,
            on_auth_url=on_auth_url,
            auth_flow=auth_flow,
            callback_url=callback_url,
            force_authentication=force_authentication,
            token_poller=token_poller,
        )
    
    return _create_credential_decorator(fetch_access_token, into)


def requires_api_key(*, provider_name: str, into: str = "api_key") -> Callable:
    """Decorator that fetches an API key before calling the decorated function.

    Args:
        provider_name: The credential provider name
        into: Parameter name to inject the API key into

    Returns:
        Decorator function
    """
    
    async def fetch_api_key(client: IdentityClient) -> str:
        """Fetch API key using the identity client."""
        return await client.get_api_key(
            provider_name=provider_name,
            agent_identity_token=await _get_workload_access_token(client),
        )
    
    return _create_credential_decorator(fetch_api_key, into)


async def _get_workload_access_token(client: IdentityClient) -> str:
    token = BedrockAgentCoreContext.get_workload_access_token()
    if token is not None:
        return token
    else:
        # workload access token context var was not set, so we should be running in a local dev environment
        if os.getenv("DOCKER_CONTAINER") == "1":
            raise ValueError("Workload access token has not been set.")

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
