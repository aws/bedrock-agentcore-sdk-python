"""Authentication decorators and utilities for Bedrock AgentCore SDK."""

import asyncio
import contextvars
import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional

import boto3
from botocore.exceptions import ClientError

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


def requires_iam_access_token(
    *,
    audience: List[str],
    signing_algorithm: str = "ES384",
    duration_seconds: int = 300,
    tags: Optional[List[Dict[str, str]]] = None,
    into: str = "access_token",
) -> Callable:
    """Decorator that fetches an AWS IAM JWT token before calling the decorated function.

    This decorator obtains a signed JWT from AWS STS using the GetWebIdentityToken API.
    The JWT can be used to authenticate with external services that support OIDC token
    validation. No client secrets are required - the token is signed by AWS.

    This is separate from @requires_access_token which uses AgentCore Identity for
    OAuth 2.0 flows. Use this decorator for M2M authentication with services that
    accept AWS-signed JWTs.

    Args:
        audience: List of intended token recipients (populates 'aud' claim in JWT).
                  Must match what the external service expects.
        signing_algorithm: Algorithm for signing the JWT.
                        'ES384' (default) or 'RS256'.
        duration_seconds: Token lifetime in seconds (60-3600, default 300).
        tags: Optional custom claims as [{'Key': str, 'Value': str}, ...].
              These are added to the JWT as additional claims.
        into: Parameter name to inject the token into (default: 'access_token').

    Returns:
        Decorator function that wraps the target function.

    Raises:
        ValueError: If parameters are invalid.
        RuntimeError: If AWS JWT federation is not enabled for the account.
        ClientError: If the STS API call fails.

    Example:
        @tool
        @requires_iam_access_token(
            audience=["https://api.example.com"],
            signing_algorithm="ES384",
            duration_seconds=300,
        )
        def call_external_api(query: str, *, access_token: str) -> str:
            '''Call external API with AWS JWT authentication.'''
            import requests
            response = requests.get(
                "https://api.example.com/data",
                headers={"Authorization": f"Bearer {access_token}"},
                params={"q": query},
            )
            return response.text

    Note:
        Before using this decorator, you must:
        1. Enable AWS IAM Outbound Web Identity Federation for your account
           (via `agentcore identity setup-aws-jwt` or IAM API)
        2. Ensure the execution role has `sts:GetWebIdentityToken` permission
        3. Configure the external service to trust your AWS account's issuer URL
    """
    # Validate parameters
    if not audience:
        raise ValueError("audience is required")
    if signing_algorithm not in ["ES384", "RS256"]:
        raise ValueError("signing_algorithm must be 'ES384' or 'RS256'")
    if not (60 <= duration_seconds <= 3600):
        raise ValueError("duration_seconds must be between 60 and 3600")

    logger = logging.getLogger(__name__)

    def _get_iam_jwt_token(region: str) -> str:
        """Get JWT from AWS STS - NO IdentityClient involved."""
        logger.info("Getting AWS IAM JWT token from STS...")
        sts_client = boto3.client("sts", region_name=region)

        params = {
            "Audience": audience,
            "SigningAlgorithm": signing_algorithm,
            "DurationSeconds": duration_seconds,
        }
        if tags:
            params["Tags"] = tags

        try:
            response = sts_client.get_web_identity_token(**params)
            logger.info("Successfully obtained AWS IAM JWT token")
            return response["WebIdentityToken"]
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ["FeatureDisabledException", "FeatureDisabled"]:
                raise RuntimeError("AWS IAM Outbound Web Identity Federation is not enabled.") from e
            logger.error("Failed to get AWS IAM JWT token: %s", str(e))
            raise

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs_func: Any) -> Any:
            region = _get_region()
            token = _get_iam_jwt_token(region)
            kwargs_func[into] = token
            return await func(*args, **kwargs_func)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs_func: Any) -> Any:
            region = _get_region()
            token = _get_iam_jwt_token(region)
            kwargs_func[into] = token
            return func(*args, **kwargs_func)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
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
