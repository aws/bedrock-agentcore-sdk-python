import asyncio

from bedrock_agentcore.identity.auth import requires_access_token, requires_api_key


@requires_access_token(
    provider_name="auth0_3lo",  # replace with your Auth0  credential provider name
    scopes=["list"],
    auth_flow="USER_FEDERATION",
    on_auth_url=lambda x: print(x),
    custom_parameters={
        "audience": "Auth0Gateway"
    },  # replace with the audience associated to your API
    force_authentication=True,
)
async def need_token_3LO_async(*, access_token: str):
    print(access_token)


@requires_access_token(
    provider_name="auth0_2lo",  # replace with your Auth0 credential provider name
    scopes=[],
    custom_parameters={
        "audience": "Auth0Gateway"
    },  # replace with the audience associated to your API
    auth_flow="M2M",
)
async def need_token_2LO_async(*, access_token: str):
    print(f"received 2LO token for async func: {access_token}")


if __name__ == "__main__":
    asyncio.run(need_token_2LO_async(access_token=""))
    asyncio.run(need_token_3LO_async(access_token=""))
