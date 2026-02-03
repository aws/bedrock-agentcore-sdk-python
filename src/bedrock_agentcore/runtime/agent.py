"""Agent class for managing Bedrock AgentCore Runtimes.

This module provides a high-level Agent class that wraps runtime operations
with Build strategy support for container and code deployment.
"""

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from bedrock_agentcore._utils.user_agent import build_user_agent_suffix

from .config import (
    BuildConfigModel,
    BuildStrategyType,
    NetworkConfigurationModel,
    NetworkMode,
    RuntimeArtifactModel,
    RuntimeConfigModel,
    VpcConfigModel,
)

if TYPE_CHECKING:
    from .build import Build

logger = logging.getLogger(__name__)


class Agent:
    """Represents a Bedrock AgentCore Runtime with Build strategy support.

    Each Agent instance manages a single runtime. Use Project.from_json()
    to load agents from configuration files.

    Example:
        from bedrock_agentcore.runtime import Agent
        from bedrock_agentcore.runtime.build import ECR, DirectCodeDeploy

        # Pre-built ECR image
        agent = Agent(
            name="my-agent",
            build=ECR(image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-agent:latest"),
        )
        agent.launch()

        # Build from source with CodeBuild + ECR
        agent = Agent(
            name="my-agent",
            build=ECR(source_path="./agent-src", entrypoint="main.py:app"),
        )
        agent.launch()  # Builds and launches

        # Direct code deploy (zip to S3)
        agent = Agent(
            name="my-agent",
            build=DirectCodeDeploy(source_path="./agent-src", entrypoint="main.py:app"),
        )
        agent.launch()

    Attributes:
        name: Agent name
        config: Runtime configuration model
        runtime_arn: ARN of deployed runtime (if deployed)
        runtime_id: ID of deployed runtime (if deployed)
        is_deployed: Whether the agent is deployed
    """

    def __init__(
        self,
        name: str,
        build: "Build",
        description: Optional[str] = None,
        network_mode: str = "PUBLIC",
        security_groups: Optional[List[str]] = None,
        subnets: Optional[List[str]] = None,
        environment_variables: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        region: Optional[str] = None,
    ):
        """Create an Agent instance with a build strategy.

        Args:
            name: Unique agent name (used for runtime name)
            build: Build strategy (PrebuiltImage, CodeBuild, LocalBuild, or DirectCodeDeploy)
            description: Optional description of the agent
            network_mode: "PUBLIC" or "VPC"
            security_groups: Security group IDs (required if network_mode="VPC")
            subnets: Subnet IDs (required if network_mode="VPC")
            environment_variables: Environment variables for the container
            tags: Resource tags
            region: AWS region (defaults to boto3 default or us-west-2)
        """
        self._name = name
        self._region = region or boto3.Session().region_name or "us-west-2"
        self._runtime_id: Optional[str] = None
        self._runtime_arn: Optional[str] = None
        self._build_strategy: "Build" = build

        # Build config model
        vpc_config = None
        if network_mode == "VPC" and security_groups and subnets:
            vpc_config = VpcConfigModel(securityGroups=security_groups, subnets=subnets)

        network_config = NetworkConfigurationModel(
            networkMode=NetworkMode(network_mode),
            vpcConfig=vpc_config,
        )

        # Build artifact config from build strategy if image_uri available
        artifact = None
        if build.image_uri:
            artifact = RuntimeArtifactModel(imageUri=build.image_uri)

        # Build the build config for serialization
        build_config = self._create_build_config(build)

        self._config = RuntimeConfigModel(
            name=name,
            description=description,
            artifact=artifact,
            build=build_config,
            networkConfiguration=network_config,
            environmentVariables=environment_variables,
            tags=tags,
        )

        # Initialize boto3 clients
        user_agent_extra = build_user_agent_suffix()
        client_config = Config(user_agent_extra=user_agent_extra)

        self._control_plane = boto3.client(
            "bedrock-agentcore-control",
            region_name=self._region,
            config=client_config,
        )
        self._data_plane = boto3.client(
            "bedrock-agentcore",
            region_name=self._region,
            config=client_config,
        )

        logger.info("Initialized Agent '%s' in region %s", name, self._region)

    # ==================== PROPERTIES ====================

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    @property
    def config(self) -> RuntimeConfigModel:
        """Current configuration."""
        return self._config

    @property
    def runtime_arn(self) -> Optional[str]:
        """Runtime ARN if deployed."""
        return self._runtime_arn

    @property
    def runtime_id(self) -> Optional[str]:
        """Runtime ID if deployed."""
        return self._runtime_id

    @property
    def is_deployed(self) -> bool:
        """Whether agent is deployed (has runtime ARN)."""
        return self._runtime_arn is not None

    @property
    def image_uri(self) -> Optional[str]:
        """Current image URI from the build strategy."""
        return self._build_strategy.image_uri

    @property
    def build_strategy(self) -> "Build":
        """Build strategy for this agent."""
        return self._build_strategy

    # ==================== OPERATIONS ====================

    def build_and_launch(
        self,
        tag: str = "latest",
        max_wait_build: int = 600,
        max_wait_launch: int = 600,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Build, push, and launch the agent in one step.

        This is the primary method for deploying an agent. It handles:
        1. Building and pushing the artifact (via build strategy's launch())
        2. Creating or updating the runtime in AWS (via launch())

        For ECR strategy: builds container and pushes to ECR, then launches runtime
        For DirectCodeDeploy: packages code and uploads to S3, then launches runtime
        For pre-built images: skips build, just launches runtime

        This method is idempotent - it will create the runtime if it doesn't exist,
        or update it if it does.

        Args:
            tag: Image tag for build (default: "latest")
            max_wait_build: Maximum seconds to wait for build
            max_wait_launch: Maximum seconds to wait for launch
            poll_interval: Seconds between status checks

        Returns:
            Launch result including runtime ARN and status

        Raises:
            RuntimeError: If build or launch fails
        """
        # Launch artifact (build + push) if image not yet available
        if not self._build_strategy.image_uri:
            logger.info("Launching build artifact...")
            self._build_strategy.validate_prerequisites()

            result = self._build_strategy.launch(
                agent_name=self._name,
                region_name=self._region,
                tag=tag,
                max_wait=max_wait_build,
            )

            # Update the config artifact with the built image
            if result.get("imageUri"):
                self._config.artifact = RuntimeArtifactModel(imageUri=result["imageUri"])
                logger.info("Artifact ready. Image URI: %s", result["imageUri"])

        # Launch the agent runtime (create or update)
        return self.launch(max_wait=max_wait_launch, poll_interval=poll_interval)

    def launch(
        self,
        max_wait: int = 600,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Deploy the agent to AWS (create or update).

        This method is idempotent - it will create the runtime if it doesn't exist,
        or update it if it already exists.

        Waits for the runtime to become ACTIVE before returning.

        Args:
            max_wait: Max seconds to wait
            poll_interval: Seconds between status checks

        Returns:
            Runtime details dict

        Raises:
            ClientError: If AWS API call fails
            TimeoutError: If wait times out
        """
        # Get image URI (either provided or built)
        current_image_uri = self.image_uri
        if not current_image_uri:
            raise ValueError(
                "Cannot launch agent without image_uri. "
                "Either provide image_uri or call build_and_launch() for source-based agents."
            )

        # Check if runtime already exists
        self._refresh_runtime_state()

        if self._runtime_id:
            # Runtime exists - update it
            return self._update_runtime(current_image_uri, max_wait, poll_interval)
        else:
            # Runtime doesn't exist - create it
            return self._create_runtime(current_image_uri, max_wait, poll_interval)

    def _create_runtime(
        self,
        image_uri: str,
        max_wait: int,
        poll_interval: int,
    ) -> Dict[str, Any]:
        """Create a new agent runtime."""
        params: Dict[str, Any] = {
            "agentRuntimeName": self._name,
            "agentRuntimeArtifact": {
                "containerConfiguration": {
                    "containerUri": image_uri,
                },
            },
        }

        if self._config.description:
            params["description"] = self._config.description

        if self._config.network_configuration:
            network_config: Dict[str, Any] = {
                "networkMode": self._config.network_configuration.network_mode.value,
            }
            if self._config.network_configuration.vpc_config:
                network_config["vpcConfiguration"] = {
                    "securityGroupIds": self._config.network_configuration.vpc_config.security_groups,
                    "subnetIds": self._config.network_configuration.vpc_config.subnets,
                }
            params["networkConfiguration"] = network_config

        if self._config.environment_variables:
            params["environmentVariables"] = self._config.environment_variables

        logger.info("Creating agent runtime '%s'...", self._name)

        try:
            response = self._control_plane.create_agent_runtime(**params)
            self._runtime_arn = response.get("agentRuntimeArn")
            self._runtime_id = response.get("agentRuntimeId")

            logger.info("Created runtime with ARN: %s", self._runtime_arn)

            return self._wait_for_active(max_wait, poll_interval)

        except ClientError as e:
            logger.error("Failed to create agent runtime: %s", e)
            raise

    def _update_runtime(
        self,
        image_uri: str,
        max_wait: int,
        poll_interval: int,
    ) -> Dict[str, Any]:
        """Update an existing agent runtime."""
        params: Dict[str, Any] = {
            "agentRuntimeId": self._runtime_id,
            "agentRuntimeArtifact": {
                "containerConfiguration": {
                    "containerUri": image_uri,
                },
            },
        }

        if self._config.description:
            params["description"] = self._config.description

        if self._config.network_configuration:
            network_config: Dict[str, Any] = {
                "networkMode": self._config.network_configuration.network_mode.value,
            }
            if self._config.network_configuration.vpc_config:
                network_config["vpcConfiguration"] = {
                    "securityGroupIds": self._config.network_configuration.vpc_config.security_groups,
                    "subnetIds": self._config.network_configuration.vpc_config.subnets,
                }
            params["networkConfiguration"] = network_config

        if self._config.environment_variables:
            params["environmentVariables"] = self._config.environment_variables

        logger.info("Updating agent runtime '%s'...", self._name)

        try:
            response = self._control_plane.update_agent_runtime(**params)
            self._runtime_arn = response.get("agentRuntimeArn")

            logger.info("Updated runtime with ARN: %s", self._runtime_arn)

            return self._wait_for_active(max_wait, poll_interval)

        except ClientError as e:
            logger.error("Failed to update agent runtime: %s", e)
            raise

    def invoke(
        self,
        payload: Union[Dict[str, Any], str, bytes],
        session_id: Optional[str] = None,
        endpoint_name: str = "DEFAULT",
    ) -> Dict[str, Any]:
        """Invoke the agent with a payload.

        Args:
            payload: Request payload (dict will be JSON-encoded)
            session_id: Session ID for stateful interactions
            endpoint_name: Endpoint qualifier

        Returns:
            Response dict with payload and metadata

        Raises:
            ValueError: If agent is not deployed
            ClientError: If AWS API call fails
        """
        if not self._runtime_arn:
            raise ValueError("Agent is not deployed. Call launch() first.")

        # Encode payload
        if isinstance(payload, dict):
            payload_bytes = json.dumps(payload).encode("utf-8")
        elif isinstance(payload, str):
            payload_bytes = payload.encode("utf-8")
        else:
            payload_bytes = payload

        params: Dict[str, Any] = {
            "agentRuntimeArn": self._runtime_arn,
            "payload": payload_bytes,
            "qualifier": endpoint_name,
        }

        if session_id:
            params["sessionId"] = session_id

        logger.debug("Invoking agent with payload...")

        response = self._data_plane.invoke_agent_runtime(**params)

        # Parse response payload
        response_payload = response.get("payload", b"")
        if isinstance(response_payload, bytes):
            try:
                response_payload = json.loads(response_payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        return {
            "payload": response_payload,
            "sessionId": response.get("sessionId"),
            "contentType": response.get("contentType"),
        }

    def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Stop a specific runtime session.

        Args:
            session_id: Session to stop

        Returns:
            Stop operation result

        Raises:
            ValueError: If agent is not deployed
            ClientError: If AWS API call fails
        """
        if not self._runtime_arn:
            raise ValueError("Agent is not deployed. Call launch() first.")

        logger.info("Stopping session '%s'...", session_id)

        response = self._data_plane.stop_agent_runtime_session(
            agentRuntimeArn=self._runtime_arn,
            sessionId=session_id,
        )

        return dict(response)

    def destroy(
        self,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Delete the runtime from AWS.

        Waits for deletion to complete before returning.

        Args:
            max_wait: Max seconds to wait
            poll_interval: Seconds between status checks

        Returns:
            Deletion result

        Raises:
            ValueError: If agent is not deployed
            ClientError: If AWS API call fails
        """
        if not self._runtime_id:
            logger.warning("Agent '%s' is not deployed, nothing to destroy", self._name)
            return {"status": "NOT_DEPLOYED"}

        logger.info("Destroying agent '%s'...", self._name)

        try:
            response = self._control_plane.delete_agent_runtime(
                agentRuntimeId=self._runtime_id,
            )

            self._wait_for_deleted(max_wait, poll_interval)

            # Clear state
            self._runtime_arn = None
            self._runtime_id = None

            logger.info("Agent '%s' destroyed", self._name)
            return dict(response)

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.warning("Agent '%s' not found, may already be deleted", self._name)
                self._runtime_arn = None
                self._runtime_id = None
                return {"status": "NOT_FOUND"}
            raise

    # ==================== HELPERS ====================

    def _create_build_config(self, build: "Build") -> BuildConfigModel:
        """Create a BuildConfigModel from a Build strategy for serialization.

        Args:
            build: Build strategy instance

        Returns:
            BuildConfigModel for YAML serialization
        """
        from .build import DirectCodeDeploy, ECR

        if isinstance(build, ECR):
            return BuildConfigModel(
                strategy=BuildStrategyType.ECR,
                imageUri=build.image_uri,
                sourcePath=build.source_path,
                entrypoint=build.entrypoint,
            )
        elif isinstance(build, DirectCodeDeploy):
            return BuildConfigModel(
                strategy=BuildStrategyType.DIRECT_CODE_DEPLOY,
                sourcePath=build.source_path,
                entrypoint=build.entrypoint,
                s3Bucket=build._s3_bucket,
            )
        else:
            # Unknown strategy - try to serialize with minimal info
            return BuildConfigModel(
                strategy=BuildStrategyType.ECR,
                imageUri=build.image_uri,
            )

    def _refresh_runtime_state(self) -> None:
        """Fetch current runtime state from AWS by name."""
        try:
            paginator = self._control_plane.get_paginator("list_agent_runtimes")

            for page in paginator.paginate():
                for runtime in page.get("agentRuntimeSummaries", []):
                    if runtime.get("agentRuntimeName") == self._name:
                        self._runtime_id = runtime.get("agentRuntimeId")
                        self._runtime_arn = runtime.get("agentRuntimeArn")
                        logger.debug(
                            "Found existing runtime: %s (ARN: %s)",
                            self._runtime_id,
                            self._runtime_arn,
                        )
                        return

            logger.debug("No existing runtime found for agent '%s'", self._name)

        except ClientError as e:
            logger.warning("Failed to refresh runtime state: %s", e)

    def _wait_for_active(self, max_wait: int, poll_interval: int) -> Dict[str, Any]:
        """Poll until runtime is ACTIVE.

        Args:
            max_wait: Maximum seconds to wait
            poll_interval: Seconds between polls

        Returns:
            Final runtime details

        Raises:
            TimeoutError: If max_wait exceeded
            RuntimeError: If runtime enters FAILED state
        """
        if not self._runtime_id:
            raise ValueError("No runtime ID to wait for")

        start_time = time.time()
        logger.info("Waiting for runtime to become ACTIVE...")

        while time.time() - start_time < max_wait:
            try:
                response = self._control_plane.get_agent_runtime(
                    agentRuntimeId=self._runtime_id,
                )

                status = response.get("status")
                logger.debug("Runtime status: %s", status)

                if status == "ACTIVE":
                    logger.info("Runtime is ACTIVE")
                    return dict(response)

                if status == "FAILED":
                    raise RuntimeError(
                        f"Runtime failed to launch: {response.get('failureReason', 'Unknown')}"
                    )

                time.sleep(poll_interval)

            except ClientError as e:
                logger.warning("Error checking runtime status: %s", e)
                time.sleep(poll_interval)

        raise TimeoutError(f"Timeout waiting for runtime to become ACTIVE after {max_wait}s")

    def _wait_for_deleted(self, max_wait: int, poll_interval: int) -> None:
        """Poll until runtime is deleted.

        Args:
            max_wait: Maximum seconds to wait
            poll_interval: Seconds between polls

        Raises:
            TimeoutError: If max_wait exceeded
        """
        if not self._runtime_id:
            return

        start_time = time.time()
        logger.info("Waiting for runtime deletion...")

        while time.time() - start_time < max_wait:
            try:
                response = self._control_plane.get_agent_runtime(
                    agentRuntimeId=self._runtime_id,
                )

                status = response.get("status")
                logger.debug("Runtime status: %s", status)

                if status == "DELETING":
                    time.sleep(poll_interval)
                    continue

            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    logger.info("Runtime deleted")
                    return
                raise

            time.sleep(poll_interval)

        raise TimeoutError(f"Timeout waiting for runtime deletion after {max_wait}s")
