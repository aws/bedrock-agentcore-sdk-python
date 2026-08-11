"""Utilities for propagating the Identity WAT header on outbound boto3 calls.

The X-Amz-Bedrock-AgentCore-Identity-WAT header carries the Workload Access Token
across service hops, enabling the workload identity chain to be extended automatically.
When present in the request context, it must be attached to outbound calls to
Runtime and Gateway services so downstream services can verify and extend the chain.

WAT is NOT propagated to Identity calls (Identity is the WAT issuer, not a consumer).

Note on Gateway MCP calls:
    Gateway invocations use the MCP protocol over raw HTTP (no boto3 API exists for
    invoke_gateway). For these calls, the developer must manually attach the WAT header
    using ``BedrockAgentCoreContext.get_workload_access_token()``::

        wat = BedrockAgentCoreContext.get_workload_access_token()
        if wat:
            headers["X-Amz-Bedrock-AgentCore-Identity-WAT"] = wat
"""

import logging

from botocore.client import BaseClient

from ..runtime.context import BedrockAgentCoreContext
from ..runtime.models import IDENTITY_WAT_HEADER

logger = logging.getLogger(__name__)

# Operations that should receive the WAT header (Runtime and Gateway invocations)
_WAT_PROPAGATION_OPERATIONS = (
    "InvokeAgentRuntime",
    "InvokeAgentRuntimeCommand",
    "InvokeHarness",
    "InvokeGateway",
)


def _inject_identity_wat_header(request, **kwargs) -> None:
    """Inject the Identity WAT header into an outgoing boto3 request.

    Reads the WAT from ``BedrockAgentCoreContext.get_workload_access_token()``
    and attaches it as the ``X-Amz-Bedrock-AgentCore-Identity-WAT`` header.
    No-ops if no WAT is present in the current context.

    Args:
        request: The ``AWSPreparedRequest`` about to be sent.
        **kwargs: Additional event keyword arguments (ignored).
    """
    wat = BedrockAgentCoreContext.get_workload_access_token()
    if wat:
        request.headers[IDENTITY_WAT_HEADER] = wat


def register_identity_wat_propagation(client: BaseClient) -> None:
    """Register the Identity WAT propagation handler on a boto3 client.

    Registers a ``before-sign`` event handler for each allowlisted operation
    so that Runtime and Gateway invocation requests automatically include the
    ``X-Amz-Bedrock-AgentCore-Identity-WAT`` header when a WAT is present
    in the current request context.

    Identity operations (token minting, credential retrieval) are excluded.

    Args:
        client: A boto3/botocore client instance (bedrock-agentcore data plane).
    """
    for op in _WAT_PROPAGATION_OPERATIONS:
        client.meta.events.register(
            f"before-sign.bedrock-agentcore.{op}",
            _inject_identity_wat_header,
            unique_id=f"identity-wat-{op}",
        )
