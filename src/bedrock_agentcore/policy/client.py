"""AgentCore Policy Engine SDK - Client for Cedar policy engine operations."""

import logging
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .._utils.config import ListConfig, WaitConfig
from .._utils.pagination import list_all
from .._utils.polling import wait_until
from .._utils.snake_case import accept_snake_case_kwargs
from .._utils.user_agent import build_user_agent_suffix

logger = logging.getLogger(__name__)

_FAILED_STATUSES = {"CREATE_FAILED", "UPDATE_FAILED", "DELETE_FAILED"}


class PolicyEngineClient:
    """Client for Bedrock AgentCore Policy Engine operations.

    Provides access to policy engine and Cedar policy CRUD operations.
    Allowlisted boto3 methods can be called directly on this client.
    Parameters accept both camelCase and snake_case (auto-converted).

    Example::

        client = PolicyEngineClient(region_name="us-west-2")

        # These are forwarded to the underlying boto3 control plane client
        engine = client.create_policy_engine(name="my-engine")
        client.create_policy(
            policy_engine_id=engine["policyEngineId"],
            name="my-policy",
            definition={"cedar": {"statement": "permit(principal, action, resource);"}},
        )
    """

    _ALLOWED_CP_METHODS = {
        # Policy engine CRUD
        "create_policy_engine",
        "get_policy_engine",
        "list_policy_engines",
        "update_policy_engine",
        "delete_policy_engine",
        # Policy CRUD
        "create_policy",
        "get_policy",
        "list_policies",
        "update_policy",
        "delete_policy",
        # Policy generation
        "start_policy_generation",
        "get_policy_generation",
        "list_policy_generations",
        "list_policy_generation_assets",
    }

    def __init__(
        self,
        region_name: Optional[str] = None,
        integration_source: Optional[str] = None,
        boto3_session: Optional[boto3.Session] = None,
    ):
        """Initialize the Policy Engine client.

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

        logger.info("Initialized PolicyEngineClient for region: %s", self.cp_client.meta.region_name)

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

    # list_all_* methods
    # -------------------------------------------------------------------------
    def list_all_policy_engines(self, list_config: Optional[ListConfig] = None, **kwargs) -> List[Dict[str, Any]]:
        """List all policy engines with automatic pagination.

        Args:
            list_config: Optional ListConfig to control max items returned (default: 100).
            **kwargs: Additional arguments forwarded to the list_policy_engines API.

        Returns:
            List of policy engine summaries.
        """
        return list_all(self.cp_client, "list_policy_engines", "policyEngines", list_config, **kwargs)

    def list_all_policies(self, list_config: Optional[ListConfig] = None, **kwargs) -> List[Dict[str, Any]]:
        """List all policies with automatic pagination.

        Args:
            list_config: Optional ListConfig to control max items returned (default: 100).
            **kwargs: Additional arguments forwarded to the list_policies API.
                Must include policyEngineId.

        Returns:
            List of policy summaries.
        """
        return list_all(self.cp_client, "list_policies", "policies", list_config, **kwargs)

    def list_all_policy_generations(self, list_config: Optional[ListConfig] = None, **kwargs) -> List[Dict[str, Any]]:
        """List all policy generations with automatic pagination.

        Args:
            list_config: Optional ListConfig to control max items returned (default: 100).
            **kwargs: Additional arguments forwarded to the list_policy_generations API.
                Must include policyEngineId.

        Returns:
            List of policy generation summaries.
        """
        return list_all(self.cp_client, "list_policy_generations", "policyGenerations", list_config, **kwargs)

    def list_all_policy_generation_assets(
        self, list_config: Optional[ListConfig] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """List all policy generation assets with automatic pagination.

        Args:
            list_config: Optional ListConfig to control max items returned (default: 100).
            **kwargs: Additional arguments forwarded to the list_policy_generation_assets API.
                Must include policyEngineId and policyGenerationId.

        Returns:
            List of policy generation asset summaries.
        """
        return list_all(
            self.cp_client, "list_policy_generation_assets", "policyGenerationAssets", list_config, **kwargs
        )

    # *_and_wait methods
    # -------------------------------------------------------------------------
    def create_policy_engine_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Create a policy engine and wait for it to reach ACTIVE status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the create_policy_engine API.

        Returns:
            Policy engine details when ACTIVE.

        Raises:
            RuntimeError: If the engine reaches a failed state.
            TimeoutError: If the engine doesn't become ACTIVE within max_wait.
        """
        response = self.cp_client.create_policy_engine(**kwargs)
        engine_id = response["policyEngineId"]
        return wait_until(
            lambda: self.cp_client.get_policy_engine(policyEngineId=engine_id),
            "ACTIVE",
            _FAILED_STATUSES,
            wait_config,
        )

    def update_policy_engine_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Update a policy engine and wait for it to reach ACTIVE status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the update_policy_engine API.

        Returns:
            Policy engine details when ACTIVE.

        Raises:
            RuntimeError: If the engine reaches a failed state.
            TimeoutError: If the engine doesn't become ACTIVE within max_wait.
        """
        response = self.cp_client.update_policy_engine(**kwargs)
        engine_id = response["policyEngineId"]
        return wait_until(
            lambda: self.cp_client.get_policy_engine(policyEngineId=engine_id),
            "ACTIVE",
            _FAILED_STATUSES,
            wait_config,
        )

    def create_policy_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Create a policy and wait for it to reach ACTIVE status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the create_policy API.
                Must include policyEngineId.

        Returns:
            Policy details when ACTIVE.

        Raises:
            RuntimeError: If the policy reaches a failed state.
            TimeoutError: If the policy doesn't become ACTIVE within max_wait.
        """
        response = self.cp_client.create_policy(**kwargs)
        engine_id = response["policyEngineId"]
        policy_id = response["policyId"]
        return wait_until(
            lambda: self.cp_client.get_policy(
                policyEngineId=engine_id,
                policyId=policy_id,
            ),
            "ACTIVE",
            _FAILED_STATUSES,
            wait_config,
        )

    # Higher-level orchestration methods
    # -------------------------------------------------------------------------
    def generate_policy(
        self,
        policy_engine_id: str,
        name: str,
        resource: Dict[str, Any],
        content: Dict[str, Any],
        client_token: Optional[str] = None,
        wait_config: Optional[WaitConfig] = None,
        fetch_assets: bool = False,
    ) -> Dict[str, Any]:
        """Generate Cedar policies from natural language and wait for completion.

        Starts policy generation, polls until complete, and optionally fetches
        the generated policy assets.

        Args:
            policy_engine_id: ID of the policy engine.
            name: Name for the generation.
            resource: Resource for which policies will be generated (e.g., {"arn": "..."}).
            content: Natural language input (e.g., {"rawText": "allow refunds..."}).
            client_token: Optional idempotency token.
            wait_config: Optional WaitConfig for polling behavior.
            fetch_assets: If True, fetch generated policies and include in response.

        Returns:
            Generation details. If fetch_assets=True, includes 'generatedPolicies' field.

        Raises:
            RuntimeError: If generation fails.
            TimeoutError: If generation doesn't complete within max_wait.
        """
        request: Dict[str, Any] = {
            "policyEngineId": policy_engine_id,
            "name": name,
            "resource": resource,
            "content": content,
        }
        if client_token:
            request["clientToken"] = client_token

        generation = self.cp_client.start_policy_generation(**request)
        generation_id = generation["policyGenerationId"]
        logger.info("Started policy generation %s, waiting for completion...", generation_id)

        _generation_failed = {"GENERATE_FAILED", "DELETE_FAILED"}
        result = wait_until(
            lambda: self.cp_client.get_policy_generation(
                policyEngineId=policy_engine_id,
                policyGenerationId=generation_id,
            ),
            "GENERATED",
            _generation_failed,
            wait_config,
        )

        if fetch_assets:
            assets = self.cp_client.list_policy_generation_assets(
                policyEngineId=policy_engine_id,
                policyGenerationId=generation_id,
            )
            result["generatedPolicies"] = assets.get("policyGenerationAssets", [])

        return result

    def create_policy_from_generation_asset(
        self,
        policy_engine_id: str,
        name: str,
        policy_generation_id: str,
        policy_generation_asset_id: str,
        description: Optional[str] = None,
        validation_mode: Optional[str] = None,
        client_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a policy from a generation asset.

        Args:
            policy_engine_id: ID of the policy engine.
            name: Name of the policy.
            policy_generation_id: ID of the policy generation.
            policy_generation_asset_id: ID of the generation asset.
            description: Optional description.
            validation_mode: Optional validation mode (FAIL_ON_ANY_FINDINGS, IGNORE_ALL_FINDINGS).
            client_token: Optional idempotency token.

        Returns:
            Policy details including policyId, ARN, and status.
        """
        definition = {
            "policyGeneration": {
                "policyGenerationId": policy_generation_id,
                "policyGenerationAssetId": policy_generation_asset_id,
            }
        }

        request: Dict[str, Any] = {
            "policyEngineId": policy_engine_id,
            "name": name,
            "definition": definition,
        }
        if description:
            request["description"] = description
        if validation_mode:
            request["validationMode"] = validation_mode
        if client_token:
            request["clientToken"] = client_token

        return self.cp_client.create_policy(**request)

    # Idempotent creates
    # -------------------------------------------------------------------------
    def create_or_get_policy_engine(
        self,
        name: str,
        description: Optional[str] = None,
        encryption_key_arn: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        client_token: Optional[str] = None,
        wait_config: Optional[WaitConfig] = None,
    ) -> Dict[str, Any]:
        """Create a policy engine or return existing one with the same name.

        Idempotent — if a ConflictException occurs, finds the existing engine
        by name. Waits for ACTIVE status before returning.

        Args:
            name: Name of the policy engine.
            description: Optional description.
            encryption_key_arn: Optional KMS key ARN.
            tags: Optional tags.
            client_token: Optional idempotency token.
            wait_config: Optional WaitConfig for polling behavior.

        Returns:
            Policy engine details in ACTIVE status.
        """
        try:
            request: Dict[str, Any] = {"name": name}
            if description:
                request["description"] = description
            if encryption_key_arn:
                request["encryptionKeyArn"] = encryption_key_arn
            if tags:
                request["tags"] = tags
            if client_token:
                request["clientToken"] = client_token

            resp = self.cp_client.create_policy_engine(**request)
            engine_id = resp["policyEngineId"]
            logger.info("Created policy engine %s, waiting for ACTIVE...", engine_id)
        except ClientError as e:
            if "ConflictException" not in e.response["Error"]["Code"]:
                raise
            logger.info("Policy engine '%s' already exists, looking up...", name)
            engine_id = self._find_policy_engine_by_name(name)
            if not engine_id:
                raise

        return wait_until(
            lambda: self.cp_client.get_policy_engine(policyEngineId=engine_id),
            "ACTIVE",
            _FAILED_STATUSES,
            wait_config,
        )

    def create_or_get_policy(
        self,
        policy_engine_id: str,
        name: str,
        definition: Dict[str, Any],
        description: Optional[str] = None,
        validation_mode: Optional[str] = None,
        client_token: Optional[str] = None,
        wait_config: Optional[WaitConfig] = None,
    ) -> Dict[str, Any]:
        """Create a policy or return existing one with the same name.

        Idempotent — if a ConflictException occurs, finds the existing policy
        by name. Waits for ACTIVE status before returning.

        Args:
            policy_engine_id: ID of the policy engine.
            name: Name of the policy.
            definition: Policy definition.
            description: Optional description.
            validation_mode: Optional validation mode.
            client_token: Optional idempotency token.
            wait_config: Optional WaitConfig for polling behavior.

        Returns:
            Policy details in ACTIVE status.
        """
        try:
            request: Dict[str, Any] = {
                "policyEngineId": policy_engine_id,
                "name": name,
                "definition": definition,
            }
            if description:
                request["description"] = description
            if validation_mode:
                request["validationMode"] = validation_mode
            if client_token:
                request["clientToken"] = client_token

            resp = self.cp_client.create_policy(**request)
            policy_id = resp["policyId"]
            logger.info("Created policy %s, waiting for ACTIVE...", policy_id)
        except ClientError as e:
            if "ConflictException" not in e.response["Error"]["Code"]:
                raise
            logger.info("Policy '%s' already exists, looking up...", name)
            policy_id = self._find_policy_by_name(policy_engine_id, name)
            if not policy_id:
                raise

        return wait_until(
            lambda: self.cp_client.get_policy(
                policyEngineId=policy_engine_id,
                policyId=policy_id,
            ),
            "ACTIVE",
            _FAILED_STATUSES,
            wait_config,
        )

    # Helper methods
    # -------------------------------------------------------------------------
    def _find_policy_engine_by_name(self, name: str) -> Optional[str]:
        """Find a policy engine ID by name. Returns None if not found."""
        next_token = None
        while True:
            params: Dict[str, Any] = {"maxResults": 100}
            if next_token:
                params["nextToken"] = next_token
            resp = self.cp_client.list_policy_engines(**params)
            for engine in resp.get("policyEngines", []):
                if engine.get("name") == name:
                    return engine["policyEngineId"]
            next_token = resp.get("nextToken")
            if not next_token:
                return None

    def _find_policy_by_name(self, policy_engine_id: str, name: str) -> Optional[str]:
        """Find a policy ID by name within an engine. Returns None if not found."""
        next_token = None
        while True:
            params: Dict[str, Any] = {"policyEngineId": policy_engine_id, "maxResults": 100}
            if next_token:
                params["nextToken"] = next_token
            resp = self.cp_client.list_policies(**params)
            for policy in resp.get("policies", []):
                if policy.get("name") == name:
                    return policy["policyId"]
            next_token = resp.get("nextToken")
            if not next_token:
                return None
