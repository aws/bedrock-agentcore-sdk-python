"""Agent class for managing Bedrock AgentCore Runtimes.

This module provides a high-level Agent class that wraps runtime operations
with YAML-based configuration persistence and container build support.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError

from bedrock_agentcore._utils.user_agent import build_user_agent_suffix

from .config import (
    BuildConfigModel,
    NetworkConfigurationModel,
    NetworkMode,
    RuntimeArtifactModel,
    RuntimeConfigModel,
    RuntimeStatus,
    VpcConfigModel,
)

logger = logging.getLogger(__name__)


class Agent:
    """Represents a Bedrock AgentCore Runtime with YAML-based configuration.

    Each Agent instance manages a single runtime. Configuration is provided
    at construction time and can be saved to/loaded from YAML files.

    Supports two deployment modes:
    1. Pre-built image: Provide image_uri directly
    2. Build from source: Provide source_path and entrypoint

    Example:
        # Mode 1: Pre-built image
        agent = Agent(name="my-agent", image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-agent:latest")
        agent.launch()

        # Mode 2: Build from source
        agent = Agent(
            name="my-agent",
            source_path="./my-agent-code",
            entrypoint="agent.py:app",
            use_codebuild=True,
        )
        agent.deploy()  # Builds image, pushes to ECR, and launches

        # Or load from file
        agent = Agent.from_yaml("my-agent.agentcore.yaml")
        agent.invoke({"message": "Hello"})

    Attributes:
        name: Agent name
        config: Runtime configuration model
        runtime_arn: ARN of deployed runtime (if deployed)
        runtime_id: ID of deployed runtime (if deployed)
        is_deployed: Whether the agent is deployed
        is_built: Whether the agent image has been built
    """

    def __init__(
        self,
        name: str,
        image_uri: Optional[str] = None,
        source_path: Optional[str] = None,
        entrypoint: Optional[str] = None,
        description: Optional[str] = None,
        network_mode: str = "PUBLIC",
        security_groups: Optional[List[str]] = None,
        subnets: Optional[List[str]] = None,
        environment_variables: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        region: Optional[str] = None,
    ):
        """Create an Agent instance with full configuration.

        Supports two modes:
        1. Pre-built image: Provide image_uri
        2. Build from source: Provide source_path and entrypoint

        Args:
            name: Unique agent name (used for runtime name)
            image_uri: ECR image URI for pre-built container (Mode 1)
            source_path: Path to agent source code (Mode 2)
            entrypoint: Entry point e.g. "agent.py:app" (Mode 2)
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
        self._built_image_uri: Optional[str] = None

        # Validate: must provide either image_uri OR source_path
        if not image_uri and not source_path:
            raise ValueError("Must provide either image_uri or source_path")

        # Build config model
        vpc_config = None
        if network_mode == "VPC" and security_groups and subnets:
            vpc_config = VpcConfigModel(securityGroups=security_groups, subnets=subnets)

        network_config = NetworkConfigurationModel(
            networkMode=NetworkMode(network_mode),
            vpcConfig=vpc_config,
        )

        # Build artifact config (only if image_uri provided)
        artifact = None
        if image_uri:
            artifact = RuntimeArtifactModel(imageUri=image_uri)

        # Build config (only if source_path provided)
        build_config = None
        if source_path:
            build_config = BuildConfigModel(
                sourcePath=source_path,
                entrypoint=entrypoint,
            )

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

    @classmethod
    def from_yaml(cls, file_path: str, region: Optional[str] = None) -> "Agent":
        """Load an agent from a YAML configuration file.

        Args:
            file_path: Path to the YAML config file
            region: AWS region (overrides any region in config)

        Returns:
            Agent instance with loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = RuntimeConfigModel.model_validate(data)

        # Extract network config
        network_mode = "PUBLIC"
        security_groups = None
        subnets = None

        if config.network_configuration:
            network_mode = config.network_configuration.network_mode.value
            if config.network_configuration.vpc_config:
                security_groups = config.network_configuration.vpc_config.security_groups
                subnets = config.network_configuration.vpc_config.subnets

        # Extract build config
        source_path = None
        entrypoint = None

        if config.build:
            source_path = config.build.source_path
            entrypoint = config.build.entrypoint

        # Create agent instance
        agent = cls(
            name=config.name,
            image_uri=config.artifact.image_uri if config.artifact else None,
            source_path=source_path,
            entrypoint=entrypoint,
            description=config.description,
            network_mode=network_mode,
            security_groups=security_groups,
            subnets=subnets,
            environment_variables=config.environment_variables,
            tags=config.tags,
            region=region,
        )

        # Try to find existing runtime
        agent._refresh_runtime_state()

        logger.info("Loaded Agent '%s' from %s", config.name, file_path)
        return agent

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
    def is_built(self) -> bool:
        """Whether agent image has been built (for source-based agents)."""
        # If image_uri was provided directly, consider it "built"
        if self._config.artifact and self._config.artifact.image_uri:
            return True
        # Otherwise check if we've built an image
        return self._built_image_uri is not None

    @property
    def image_uri(self) -> Optional[str]:
        """Current image URI (either provided or built)."""
        if self._built_image_uri:
            return self._built_image_uri
        if self._config.artifact:
            return self._config.artifact.image_uri
        return None

    # ==================== OPERATIONS ====================

    def save(self, file_path: str) -> str:
        """Save the agent configuration to a YAML file.

        Args:
            file_path: Path to save the YAML config file

        Returns:
            The file path where config was saved
        """
        path = Path(file_path)
        data = self._config.model_dump(mode="json", by_alias=True, exclude_none=True)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Saved Agent config to %s", file_path)
        return str(path)

    def build(
        self,
        tag: str = "latest",
        wait: bool = True,
        max_wait: int = 600,
    ) -> Dict[str, Any]:
        """Build the agent container image and push to ECR.

        This method is only applicable for source-based agents (those created
        with source_path). It will:
        1. Generate a Dockerfile automatically
        2. Create ECR repository automatically
        3. Create IAM execution role automatically
        4. Build the Docker image via CodeBuild (ARM64)
        5. Push the image to ECR

        Args:
            tag: Image tag (default: "latest")
            wait: Wait for build to complete
            max_wait: Maximum seconds to wait for build

        Returns:
            Build result including imageUri

        Raises:
            ValueError: If agent was created with image_uri (not source_path)
            FileNotFoundError: If source_path doesn't exist
            RuntimeError: If build fails
        """
        if not self._config.build:
            raise ValueError(
                "Cannot build agent created with image_uri. "
                "Use source_path and entrypoint for source-based builds."
            )

        if not self._config.build.source_path:
            raise ValueError("source_path is required for building")

        if not self._config.build.entrypoint:
            raise ValueError("entrypoint is required for building")

        # Import builder module (lazy import to avoid circular dependencies)
        from .builder import build_and_push

        logger.info("Building agent '%s'...", self._name)

        result = build_and_push(
            source_path=self._config.build.source_path,
            agent_name=self._name,
            entrypoint=self._config.build.entrypoint,
            region_name=self._region,
            tag=tag,
            wait=wait,
            max_wait=max_wait,
        )

        # Store the built image URI
        self._built_image_uri = result.get("imageUri")

        # Update the config artifact with the built image
        if self._built_image_uri:
            self._config.artifact = RuntimeArtifactModel(imageUri=self._built_image_uri)

        logger.info("Build complete. Image URI: %s", self._built_image_uri)
        return result

    def deploy(
        self,
        tag: str = "latest",
        wait: bool = True,
        max_wait_build: int = 600,
        max_wait_launch: int = 600,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Build and launch the agent in one step.

        This is a convenience method that combines build() and launch().
        For source-based agents, it will build the image first.
        For image_uri-based agents, it will just launch.

        Args:
            tag: Image tag for build (default: "latest")
            wait: Wait for operations to complete
            max_wait_build: Maximum seconds to wait for build
            max_wait_launch: Maximum seconds to wait for launch
            poll_interval: Seconds between status checks

        Returns:
            Launch result including runtime ARN and status

        Raises:
            RuntimeError: If build or launch fails
        """
        # Build if this is a source-based agent and not already built
        if self._config.build and not self.is_built:
            logger.info("Building agent before deploy...")
            self.build(tag=tag, wait=wait, max_wait=max_wait_build)

        # Launch the agent
        return self.launch(wait=wait, max_wait=max_wait_launch, poll_interval=poll_interval)

    def launch(
        self,
        wait: bool = True,
        max_wait: int = 600,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Deploy the agent to AWS.

        Calls create_agent_runtime API using the saved configuration.

        Args:
            wait: Wait for ACTIVE status
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
                "Either provide image_uri or call build() first for source-based agents."
            )

        # Build request params
        params: Dict[str, Any] = {
            "agentRuntimeName": self._name,
            "agentRuntimeArtifact": {
                "containerConfiguration": {
                    "containerUri": current_image_uri,
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

        logger.info("Launching agent '%s'...", self._name)

        try:
            response = self._control_plane.create_agent_runtime(**params)
            self._runtime_arn = response.get("agentRuntimeArn")
            self._runtime_id = response.get("agentRuntimeId")

            logger.info("Created runtime with ARN: %s", self._runtime_arn)

            if wait:
                return self._wait_for_active(max_wait, poll_interval)

            return dict(response)

        except ClientError as e:
            logger.error("Failed to launch agent: %s", e)
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
        wait: bool = True,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Delete the runtime from AWS.

        Args:
            wait: Wait for deletion to complete
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

            if wait:
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
