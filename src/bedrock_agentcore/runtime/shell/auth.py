"""Authentication modes for open_shell."""

from dataclasses import dataclass
from typing import Literal, Union


@dataclass
class OAuthAuth:
    """OAuth bearer token authentication for ``open_shell``.

    Use this when connecting from a browser relay or any context where an OAuth
    token (not AWS IAM credentials) is the auth mechanism.  The token is
    embedded in the ``Sec-WebSocket-Protocol`` header — the only auth mechanism
    browsers can provide on a WebSocket upgrade (RFC 6455 §4.1).

    Attributes:
        bearer_token: OAuth bearer token obtained from your identity provider.

    Example:
        async with client.open_shell(
            runtime_arn,
            auth=OAuthAuth(bearer_token=await get_oauth_token()),
        ) as shell:
            ...
    """

    bearer_token: str


@dataclass
class PresignedAuth:
    """Presigned URL authentication for ``open_shell``.

    Use this when you want auth embedded in the URL query string — useful when
    handing off to another process or service without sharing AWS credentials,
    or when the WebSocket client cannot set custom headers.

    Attributes:
        expires: Seconds until the presigned URL expires (max 300).

    Example:
        async with client.open_shell(
            runtime_arn,
            auth=PresignedAuth(expires=120),
        ) as shell:
            ...
    """

    expires: int = 300


# SigV4 is expressed as a string literal for brevity; the two dataclasses carry
# the extra fields their auth paths need.
AuthMode = Union[Literal["sigv4"], PresignedAuth, OAuthAuth]
