import asyncio

from bedrock_agentcore.identity.auth import requires_access_token, requires_api_key, requires_iam_access_token


@requires_access_token(
    provider_name="Google4",  # replace with your own credential provider name
    scopes=["https://www.googleapis.com/auth/userinfo.email"],
    auth_flow="USER_FEDERATION",
    on_auth_url=lambda x: print(x),
    force_authentication=True,
)
async def need_token_3LO_async(*, access_token: str):
    print(access_token)


@requires_access_token(
    provider_name="custom-provider-3",  # replace with your own credential provider name
    scopes=["default"],
    auth_flow="M2M",
)
async def need_token_2LO_async(*, access_token: str):
    print(f"received 2LO token for async func: {access_token}")


@requires_api_key(
    provider_name="test-api-key-provider"  # replace with your own credential provider name
)
async def need_api_key(*, api_key: str):
    print(f"received api key for async func: {api_key}")


# New AWS IAM JWT flow tests using the separate decorator
@requires_iam_access_token(
    audience=["https://api.example.com"],  # replace with your target service audience
    signing_algorithm="ES384",
    duration_seconds=300,
)
async def need_aws_jwt_token_async(*, access_token: str):
    """Test AWS IAM JWT token retrieval with async function."""
    print(f"received AWS IAM JWT token for async func: {access_token[:50]}...")


@requires_iam_access_token(
    audience=["https://api.example.com"],
    signing_algorithm="ES384",
    duration_seconds=300,
)
def need_aws_jwt_token_sync(*, access_token: str):
    """Test AWS IAM JWT token retrieval with sync function."""
    print(f"received AWS IAM JWT token for sync func: {access_token[:50]}...")


@requires_iam_access_token(
    audience=["https://api.example.com"],
    signing_algorithm="RS256",
    duration_seconds=600,
)
async def need_aws_jwt_token_rs256(*, access_token: str):
    """Test AWS IAM JWT token retrieval with RS256 algorithm."""
    print(f"received AWS IAM JWT token (RS256) for async func: {access_token[:50]}...")


@requires_iam_access_token(
    audience=["https://api1.example.com", "https://api2.example.com"],
    signing_algorithm="ES384",
    duration_seconds=300,
    tags=[
        {"Key": "environment", "Value": "test"},
        {"Key": "service", "Value": "integration-test"},
    ],
)
async def need_aws_jwt_token_with_tags(*, access_token: str):
    """Test AWS IAM JWT token retrieval with custom tags."""
    print(f"received AWS IAM JWT token with tags: {access_token[:50]}...")


@requires_iam_access_token(
    audience=["https://api.example.com"],
    into="jwt_token",  # Custom parameter name
)
async def need_aws_jwt_custom_param(*, jwt_token: str):
    """Test AWS IAM JWT token with custom parameter name."""
    print(f"received AWS IAM JWT token in custom param: {jwt_token[:50]}...")


if __name__ == "__main__":
    # OAuth flows (require credential providers to be set up)
    asyncio.run(need_api_key(api_key=""))
    asyncio.run(need_token_2LO_async(access_token=""))
    asyncio.run(need_token_3LO_async(access_token=""))

    # AWS IAM JWT flows (require IAM permissions)
    print("\n=== Testing AWS IAM JWT Flow (ES384) ===")
    asyncio.run(need_aws_jwt_token_async(access_token=""))

    print("\n=== Testing AWS IAM JWT Flow (Sync) ===")
    need_aws_jwt_token_sync(access_token="")

    print("\n=== Testing AWS IAM JWT Flow (RS256) ===")
    asyncio.run(need_aws_jwt_token_rs256(access_token=""))

    print("\n=== Testing AWS IAM JWT Flow (With Tags) ===")
    asyncio.run(need_aws_jwt_token_with_tags(access_token=""))

    print("\n=== Testing AWS IAM JWT Flow (Custom Param) ===")
    asyncio.run(need_aws_jwt_custom_param(jwt_token=""))
