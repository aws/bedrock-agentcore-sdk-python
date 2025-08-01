import asyncio

from bedrock_agentcore.identity.auth import requires_access_token, requires_api_key


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
    scopes=[],
    auth_flow="M2M",
)
async def need_token_2LO_async(*, access_token: str):
    print(f"received 2LO token for async func: {access_token}")


@requires_api_key(
    provider_name="test-api-key-provider"  # replace with your own credential provider name
)
async def need_api_key(*, api_key: str):
    print(f"received api key for async func: {api_key}")


if __name__ == "__main__":
    asyncio.run(need_api_key(api_key=""))
    asyncio.run(need_token_2LO_async(access_token=""))
    asyncio.run(need_token_3LO_async(access_token=""))
