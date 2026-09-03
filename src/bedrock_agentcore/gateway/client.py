"""AgentCore Gateway SDK - Client for MCP gateway and target operations."""

import logging
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config

from .._utils.config import WaitConfig
from .._utils.polling import wait_until, wait_until_deleted
from .._utils.snake_case import accept_snake_case_kwargs, convert_kwargs
from .._utils.user_agent import build_user_agent_suffix

logger = logging.getLogger(__name__)

_GATEWAY_FAILED_STATUSES = {"FAILED", "UPDATE_UNSUCCESSFUL"}
_TARGET_FAILED_STATUSES = {"FAILED", "UPDATE_UNSUCCESSFUL", "SYNCHRONIZE_UNSUCCESSFUL"}

#: Default name for a web search target. Gateway prefixes every tool with the name of
#: the target it came from, so this is what makes the tool read as
#: "amazon-web-search___WebSearch" to the agent.
DEFAULT_WEB_SEARCH_TARGET_NAME = "amazon-web-search"

#: First connector version that accepts a target-level include list. The connector's
#: default is older, so an include list has to pin this.
_INCLUDE_DOMAINS_MIN_CONNECTOR_VERSION = "1.2.0"

#: Documented maximum length of either domain list on a web search target.
_MAX_DOMAIN_FILTER_ENTRIES = 100


class GatewayClient:
    """Client for Bedrock AgentCore Gateway operations.

    Provides access to gateway and gateway target CRUD operations.
    Allowlisted boto3 methods can be called directly on this client.
    Parameters accept both camelCase and snake_case (auto-converted).

    Example::

        client = GatewayClient(region_name="us-west-2")

        # Pass-through to boto3 control plane client
        gateway = client.create_gateway(
            name="my-gateway",
            roleArn="arn:aws:iam::123456789:role/gateway-role",
            protocolType="MCP",
        )
    """

    _ALLOWED_CP_METHODS = {
        # Gateway CRUD
        "create_gateway",
        "get_gateway",
        "list_gateways",
        "update_gateway",
        "delete_gateway",
        # Gateway target CRUD
        "create_gateway_target",
        "get_gateway_target",
        "list_gateway_targets",
        "update_gateway_target",
        "delete_gateway_target",
    }

    def __init__(
        self,
        region_name: Optional[str] = None,
        integration_source: Optional[str] = None,
        boto3_session: Optional[boto3.Session] = None,
    ):
        """Initialize the Gateway client.

        Args:
            region_name: AWS region name. If not provided, uses the session's region or "us-west-2".
            integration_source: Optional integration source for user-agent telemetry.
            boto3_session: Optional boto3 Session to use. If not provided, a default session
                          is created. Useful for named profiles or custom credentials.
        """
        session = boto3_session if boto3_session else boto3.Session()
        self.region_name = region_name or session.region_name or "us-west-2"
        self.integration_source = integration_source

        user_agent_extra = build_user_agent_suffix(integration_source)
        client_config = Config(user_agent_extra=user_agent_extra)

        self.cp_client = session.client("bedrock-agentcore-control", region_name=self.region_name, config=client_config)

        logger.info("Initialized GatewayClient for region: %s", self.cp_client.meta.region_name)

    # Pass-through
    # -------------------------------------------------------------------------
    def __getattr__(self, name: str):
        """Dynamically forward allowlisted method calls to the control plane boto3 client."""
        if name in self._ALLOWED_CP_METHODS and hasattr(self.cp_client, name):
            method = getattr(self.cp_client, name)
            logger.debug("Forwarding method '%s' to cp_client", name)
            return accept_snake_case_kwargs(method)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
            f"Method not found on cp_client. "
            f"Available methods can be found in the boto3 documentation for "
            f"'bedrock-agentcore-control' service."
        )

    # *_and_wait methods
    # -------------------------------------------------------------------------
    def create_gateway_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Create a gateway and wait for it to reach READY status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the create_gateway API.

        Returns:
            Gateway details when READY.

        Raises:
            RuntimeError: If the gateway reaches a failed state.
            TimeoutError: If the gateway doesn't become READY within max_wait.
        """
        response = self.cp_client.create_gateway(**convert_kwargs(kwargs))
        gw_id = response["gatewayId"]
        return wait_until(
            lambda: self.cp_client.get_gateway(gatewayIdentifier=gw_id),
            "READY",
            _GATEWAY_FAILED_STATUSES,
            wait_config,
        )

    def update_gateway_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Update a gateway and wait for it to reach READY status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the update_gateway API.

        Returns:
            Gateway details when READY.

        Raises:
            RuntimeError: If the gateway reaches a failed state.
            TimeoutError: If the gateway doesn't become READY within max_wait.
        """
        response = self.cp_client.update_gateway(**convert_kwargs(kwargs))
        gw_id = response["gatewayId"]
        return wait_until(
            lambda: self.cp_client.get_gateway(gatewayIdentifier=gw_id),
            "READY",
            _GATEWAY_FAILED_STATUSES,
            wait_config,
        )

    def create_gateway_target_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Create a gateway target and wait for it to reach READY status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the create_gateway_target API.
                Must include gatewayIdentifier.

        Returns:
            Gateway target details when READY.

        Raises:
            RuntimeError: If the target reaches a failed state.
            TimeoutError: If the target doesn't become READY within max_wait.
        """
        response = self.cp_client.create_gateway_target(**convert_kwargs(kwargs))
        gw_id = response["gatewayArn"].rsplit("/", 1)[-1]
        target_id = response["targetId"]
        return wait_until(
            lambda: self.cp_client.get_gateway_target(
                gatewayIdentifier=gw_id,
                targetId=target_id,
            ),
            "READY",
            _TARGET_FAILED_STATUSES,
            wait_config,
        )

    def update_gateway_target_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Update a gateway target and wait for it to reach READY status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the update_gateway_target API.
                Must include gatewayIdentifier and targetId.

        Returns:
            Gateway target details when READY.

        Raises:
            RuntimeError: If the target reaches a failed state.
            TimeoutError: If the target doesn't become READY within max_wait.
        """
        response = self.cp_client.update_gateway_target(**convert_kwargs(kwargs))
        gw_id = response["gatewayArn"].rsplit("/", 1)[-1]
        target_id = response["targetId"]
        return wait_until(
            lambda: self.cp_client.get_gateway_target(
                gatewayIdentifier=gw_id,
                targetId=target_id,
            ),
            "READY",
            _TARGET_FAILED_STATUSES,
            wait_config,
        )

    def delete_gateway_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> None:
        """Delete a gateway and wait for deletion to complete.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the delete_gateway API.

        Raises:
            TimeoutError: If the gateway isn't deleted within max_wait.
        """
        response = self.cp_client.delete_gateway(**convert_kwargs(kwargs))
        gw_id = response["gatewayId"]
        wait_until_deleted(
            lambda: self.cp_client.get_gateway(gatewayIdentifier=gw_id),
            wait_config=wait_config,
        )

    def delete_gateway_target_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> None:
        """Delete a gateway target and wait for deletion to complete.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the delete_gateway_target API.

        Raises:
            TimeoutError: If the target isn't deleted within max_wait.
        """
        response = self.cp_client.delete_gateway_target(**convert_kwargs(kwargs))
        gw_id = response["gatewayArn"].rsplit("/", 1)[-1]
        target_id = response["targetId"]
        wait_until_deleted(
            lambda: self.cp_client.get_gateway_target(
                gatewayIdentifier=gw_id,
                targetId=target_id,
            ),
            wait_config=wait_config,
        )

    # Knowledge Base target helpers
    # -------------------------------------------------------------------------
    def create_knowledge_base_target(
        self,
        gateway_identifier: str,
        knowledge_base_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        retrieval_configuration: Optional[Dict[str, Any]] = None,
        parameter_overrides: Optional[List[Dict[str, Any]]] = None,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a gateway target that exposes a Knowledge Base as an MCP Retrieve tool.

        Args:
            gateway_identifier: Gateway ID or ARN.
            knowledge_base_id: The Knowledge Base to expose.
            name: Target name. Defaults to "kb-{knowledge_base_id}".
            description: Agent-facing description of the Retrieve tool.
            retrieval_configuration: Optional retrieval config (vectorSearchConfiguration, etc.).
            parameter_overrides: Optional per-parameter visibility/description overrides.
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Additional arguments forwarded to create_gateway_target
                (e.g., credentialProviderConfigurations, roleArn). Overrides built values on conflict.

        Returns:
            Gateway target details when READY.
        """
        parameter_values: Dict[str, Any] = {"knowledgeBaseId": knowledge_base_id}
        if retrieval_configuration:
            parameter_values["retrievalConfiguration"] = retrieval_configuration

        tool_config: Dict[str, Any] = {
            "name": "Retrieve",
            "parameterValues": parameter_values,
        }
        if description:
            tool_config["description"] = description
        if parameter_overrides:
            tool_config["parameterOverrides"] = parameter_overrides

        target_kwargs = {
            "gatewayIdentifier": gateway_identifier,
            "name": name or f"kb-{knowledge_base_id}",
            "targetConfiguration": {
                "mcp": {
                    "connector": {
                        "source": {"connectorId": "bedrock-knowledge-bases"},
                        "enabled": ["Retrieve"],
                        "configurations": [tool_config],
                    },
                },
            },
            "credentialProviderConfigurations": [
                {"credentialProviderType": "GATEWAY_IAM_ROLE"},
            ],
        }
        target_kwargs.update(kwargs)

        return self.create_gateway_target_and_wait(
            wait_config=wait_config,
            **target_kwargs,
        )

    def create_agentic_retrieve_target(
        self,
        gateway_identifier: str,
        retrievers: List[Dict[str, Any]],
        model_arn: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_agent_iteration: Optional[int] = None,
        parameter_overrides: Optional[List[Dict[str, Any]]] = None,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a gateway target that exposes Knowledge Bases as an MCP AgenticRetrieveStream tool.

        Args:
            gateway_identifier: Gateway ID or ARN.
            retrievers: List of retriever configurations, each with knowledge_base_id and optional
                retrieval_overrides. Example: [{"knowledgeBaseId": "KB1", "description": "..."}]
            model_arn: Foundation model ARN for orchestration.
            name: Target name. Defaults to "agentic-retrieve-{timestamp}".
            description: Agent-facing description of the AgenticRetrieveStream tool.
            max_agent_iteration: Max iterations for the agentic loop (default: service default).
            parameter_overrides: Optional per-parameter visibility/description overrides.
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Additional arguments forwarded to create_gateway_target. Overrides built values on conflict.

        Returns:
            Gateway target details when READY.
        """
        import time as _time

        agentic_config: Dict[str, Any] = {
            "foundationModelConfiguration": {"bedrock": {"modelArn": model_arn}},
        }
        if max_agent_iteration:
            agentic_config["maxAgentIteration"] = max_agent_iteration

        parameter_values: Dict[str, Any] = {
            "retrievers": retrievers,
            "agenticRetrieveConfiguration": agentic_config,
        }

        tool_config: Dict[str, Any] = {
            "name": "AgenticRetrieveStream",
            "parameterValues": parameter_values,
        }
        if description:
            tool_config["description"] = description
        if parameter_overrides:
            tool_config["parameterOverrides"] = parameter_overrides

        target_kwargs = {
            "gatewayIdentifier": gateway_identifier,
            "name": name or f"agentic-retrieve-{int(_time.time())}",
            "targetConfiguration": {
                "mcp": {
                    "connector": {
                        "source": {"connectorId": "bedrock-agentic-retrieve"},
                        "enabled": ["AgenticRetrieveStream"],
                        "configurations": [tool_config],
                    },
                },
            },
            "credentialProviderConfigurations": [
                {"credentialProviderType": "GATEWAY_IAM_ROLE"},
            ],
        }
        target_kwargs.update(kwargs)

        return self.create_gateway_target_and_wait(
            wait_config=wait_config,
            **target_kwargs,
        )

    # Web Search target helpers
    # -------------------------------------------------------------------------
    def create_web_search_target(
        self,
        gateway_identifier: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        exclude_domains: Optional[List[str]] = None,
        include_domains: Optional[List[str]] = None,
        connector_version: Optional[str] = None,
        parameter_overrides: Optional[List[Dict[str, Any]]] = None,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a gateway target that exposes Amazon Web Search as an MCP WebSearch tool.

        The tool the agent discovers is named "<target name>___WebSearch", because Gateway
        prefixes every tool with its target name. The default target name is therefore chosen
        so the agent-facing tool reads as "amazon-web-search___WebSearch".

        The gateway's service role needs bedrock-agentcore:InvokeWebSearch on the connector,
        and whoever calls the resulting tool needs bedrock-agentcore:InvokeGateway on the
        gateway ARN. Web search takes no API key of its own.

        Args:
            gateway_identifier: Gateway ID or ARN.
            name: Target name, and the prefix of the agent-facing tool name.
                Defaults to "amazon-web-search".
            description: Agent-facing description of the WebSearch tool.
            exclude_domains: Optional list of domains to drop from results, up to 100.
                Enforced server-side and hidden from the calling agent. A result is
                dropped if its domain is on this list or on the caller's own exclude
                list, so the agent can narrow this but never relax it.
            include_domains: Optional list of domains to restrict results to, up to 100.
                Needs connector version 1.2.0 or later, which this method pins for you
                when you pass one and do not pin a version yourself. A result is
                returned only if its domain appears on every include list that is set,
                so a caller passing its own include list narrows to the intersection
                with this one, and disjoint lists return no results at all. A root
                domain matches its subdomains.
            connector_version: Optional connector version to pin, e.g. "1.2.0". Defaults
                to the connector's current default version, except that an include list
                pins 1.2.0 as described above.
            parameter_overrides: Optional per-parameter visibility/description overrides,
                keyed by JSONPath, e.g. {"path": "$.maxResults", "visible": True}.
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Additional arguments forwarded to create_gateway_target
                (e.g., credentialProviderConfigurations, roleArn). Overrides built values on conflict.

        Returns:
            Gateway target details when READY.

        Raises:
            ValueError: If either domain list is longer than 100, or if include_domains
                is combined with a pinned connector version that predates it.
        """
        # parameterValues is always sent, even when empty. The service drops every
        # configuration whose parameterValues is absent before it validates them, so a
        # configuration carrying nothing but a name leaves nothing to validate and the
        # request is rejected with "Connector configurations must not be empty".
        # An empty object is accepted.
        tool_config: Dict[str, Any] = {"name": "WebSearch", "parameterValues": {}}
        domain_filter: Dict[str, List[str]] = {}
        if include_domains:
            domain_filter["include"] = _checked_domains("include_domains", include_domains)
        if exclude_domains:
            domain_filter["exclude"] = _checked_domains("exclude_domains", exclude_domains)
        if domain_filter:
            tool_config["parameterValues"]["domainFilter"] = domain_filter
        if description:
            tool_config["description"] = description
        if parameter_overrides:
            tool_config["parameterOverrides"] = parameter_overrides

        source: Dict[str, Any] = {"connectorId": "web-search"}
        if include_domains:
            # A target-level include list only exists from 1.2.0 on, and the connector
            # default is older, so sending one unpinned is rejected server-side. Pin the
            # first version that accepts it rather than build a request that cannot
            # validate.
            connector_version = _connector_version_for_include_domains(connector_version)
        if connector_version:
            source["version"] = connector_version

        target_kwargs = {
            "gatewayIdentifier": gateway_identifier,
            "name": name or DEFAULT_WEB_SEARCH_TARGET_NAME,
            "targetConfiguration": {
                "mcp": {
                    "connector": {
                        "source": source,
                        "enabled": ["WebSearch"],
                        "configurations": [tool_config],
                    },
                },
            },
            "credentialProviderConfigurations": [
                {"credentialProviderType": "GATEWAY_IAM_ROLE"},
            ],
        }
        target_kwargs.update(kwargs)

        return self.create_gateway_target_and_wait(
            wait_config=wait_config,
            **target_kwargs,
        )

    # Name-based lookup
    # -------------------------------------------------------------------------
    def get_gateway_by_name(self, name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Look up a gateway by name.

        Paginates through gateways and returns the full resource details
        for the first match. Short-circuits on first match without fetching
        remaining pages. Returns None if no gateway with that name exists.

        Args:
            name: The gateway name to search for.
            **kwargs: Additional arguments forwarded to the list_gateways API.

        Returns:
            Gateway details from get_gateway, or None if not found.
        """
        params = convert_kwargs(kwargs)
        params.pop("nextToken", None)
        while True:
            response = self.cp_client.list_gateways(**params)
            for gw in response.get("items", []):
                if gw.get("name") == name:
                    return self.cp_client.get_gateway(
                        gatewayIdentifier=gw["gatewayId"],
                    )
            if not response.get("nextToken"):
                return None
            params["nextToken"] = response["nextToken"]

    def get_gateway_target_by_name(self, gateway_identifier: str, name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Look up a gateway target by name.

        Paginates through targets for the given gateway and returns the
        full resource details for the first match. Short-circuits on first
        match without fetching remaining pages. Returns None if not found.

        Args:
            gateway_identifier: Gateway ID or ARN.
            name: The target name to search for.
            **kwargs: Additional arguments forwarded to the list_gateway_targets API.

        Returns:
            Gateway target details from get_gateway_target, or None if not found.
        """
        params = convert_kwargs(kwargs)
        params.pop("nextToken", None)
        params["gatewayIdentifier"] = gateway_identifier
        while True:
            response = self.cp_client.list_gateway_targets(**params)
            for target in response.get("items", []):
                if target.get("name") == name:
                    return self.cp_client.get_gateway_target(
                        gatewayIdentifier=gateway_identifier,
                        targetId=target["targetId"],
                    )
            if not response.get("nextToken"):
                return None
            params["nextToken"] = response["nextToken"]


def _checked_domains(argument: str, domains: List[str]) -> List[str]:
    """Return the domain list, rejecting one longer than the documented maximum."""
    values = list(domains)
    if len(values) > _MAX_DOMAIN_FILTER_ENTRIES:
        raise ValueError(f"{argument} accepts at most {_MAX_DOMAIN_FILTER_ENTRIES} domains, got {len(values)}")
    return values


def _connector_version_for_include_domains(connector_version: Optional[str]) -> str:
    """Return the connector version to pin when a target-level include list is set.

    Raises:
        ValueError: If the caller pinned a version that predates the include list.
    """
    if connector_version is None:
        return _INCLUDE_DOMAINS_MIN_CONNECTOR_VERSION
    if _version_tuple(connector_version) < _version_tuple(_INCLUDE_DOMAINS_MIN_CONNECTOR_VERSION):
        raise ValueError(
            f"include_domains requires connector version {_INCLUDE_DOMAINS_MIN_CONNECTOR_VERSION} or later, "
            f"got {connector_version}"
        )
    return connector_version


def _version_tuple(version: str) -> tuple:
    """Read a dotted version into comparable integers, ignoring anything unparseable.

    Missing components count as zero, so "1.2" is not read as older than "1.2.0". An
    unrecognized version sorts high, so a version this SDK does not understand is passed
    through to the service to accept or reject rather than rejected locally.
    """
    parts = [0, 0, 0]
    for index, part in enumerate(version.split(".")):
        if not part.isdigit() or index >= len(parts):
            return (float("inf"),)
        parts[index] = int(part)
    return tuple(parts)
