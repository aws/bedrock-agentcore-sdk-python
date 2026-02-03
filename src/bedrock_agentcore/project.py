"""Project class for managing Bedrock AgentCore resources.

This module provides a Project class that loads/saves agentcore.json
configuration and manages collections of Agent and Memory objects.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3

from .memory import Memory
from .project_config import (
    AgentConfig,
    AgentDeployedState,
    AWSTarget,
    DeployedState,
    MemoryProviderConfig,
    MemoryRelation,
    MemoryStrategyConfig,
    MemoryStrategyType,
    NetworkMode,
    ProjectConfig,
    PythonVersion,
    RuntimeConfig,
    TargetDeployedState,
    TargetResources,
)
from .runtime import Agent
from .runtime.build import DirectCodeDeploy

logger = logging.getLogger(__name__)


class Project:
    """Manages Bedrock AgentCore resources with agentcore.json configuration.

    Project provides a container for Agent and Memory resources with
    JSON-based persistence matching the starter-toolkit format.

    Example:
        # Load from agentcore.json
        project = Project.from_json("agentcore.json")

        # Access resources
        agents = project.agents
        memories = project.memories

        # Launch all resources
        project.launch_all()

        # Save back to JSON
        project.save("agentcore.json")

    Attributes:
        name: Project name
        agents: List of Agent objects
        memories: List of Memory objects
    """

    def __init__(
        self,
        name: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """Create a Project instance.

        Args:
            name: Project name (max 23 chars)
            version: Project version
            description: Project description
            region: AWS region (applied to all resources)
        """
        self._name = name
        self._version = version
        self._description = description
        self._region = region or boto3.Session().region_name or "us-west-2"
        self._agents: Dict[str, Agent] = {}
        self._memories: Dict[str, Memory] = {}

        logger.info("Initialized Project '%s' in region %s", name, self._region)

    @classmethod
    def from_json(cls, file_path: str, region: Optional[str] = None) -> "Project":
        """Load a project from an agentcore.json configuration file.

        Args:
            file_path: Path to the agentcore.json file
            region: AWS region (overrides config)

        Returns:
            Project instance with all resources loaded

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(path, "r") as f:
            data = json.load(f)

        config = ProjectConfig.model_validate(data)
        project = cls(
            name=config.name,
            version=config.version,
            description=config.description,
            region=region,
        )

        # Process agents and their embedded memories
        if config.agents:
            for agent_config in config.agents:
                # Create Memory objects from memoryProviders
                if agent_config.memory_providers:
                    for mem_provider in agent_config.memory_providers:
                        if mem_provider.type == "AgentCoreMemory" and mem_provider.relation == MemoryRelation.OWN:
                            # Only create memories that this agent owns
                            strategies = None
                            if mem_provider.memory_strategies:
                                strategies = [
                                    {
                                        "type": _map_memory_strategy_type(s.type),
                                        "namespace": f"{s.type.value.lower()}/{{sessionId}}/",
                                    }
                                    for s in mem_provider.memory_strategies
                                ]

                            memory = Memory(
                                name=mem_provider.name,
                                strategies=strategies,
                                region=region or project._region,
                            )
                            project.add_memory(memory)

                # Create Agent object
                runtime = agent_config.runtime
                network_mode = runtime.network_mode.value if runtime.network_mode else "PUBLIC"

                build = DirectCodeDeploy(
                    source_path=runtime.code_location,
                    entrypoint=runtime.entrypoint,
                )

                agent = Agent(
                    name=agent_config.name,
                    build=build,
                    network_mode=network_mode,
                    region=region or project._region,
                )
                project.add_agent(agent)

        logger.info("Loaded Project '%s' from %s", config.name, file_path)
        return project

    # ==================== PROPERTIES ====================

    @property
    def name(self) -> str:
        """Project name."""
        return self._name

    @property
    def version(self) -> Optional[str]:
        """Project version."""
        return self._version

    @property
    def description(self) -> Optional[str]:
        """Project description."""
        return self._description

    @property
    def region(self) -> str:
        """AWS region."""
        return self._region

    @property
    def agents(self) -> List[Agent]:
        """List of all agents in the project."""
        return list(self._agents.values())

    @property
    def memories(self) -> List[Memory]:
        """List of all memories in the project."""
        return list(self._memories.values())

    # ==================== RESOURCE MANAGEMENT ====================

    def add_agent(self, agent: Agent) -> "Project":
        """Add an agent to the project.

        Args:
            agent: Agent object to add

        Returns:
            Self for method chaining
        """
        self._agents[agent.name] = agent
        return self

    def add_memory(self, memory: Memory) -> "Project":
        """Add a memory to the project.

        Args:
            memory: Memory object to add

        Returns:
            Self for method chaining
        """
        self._memories[memory.name] = memory
        return self

    def get_agent(self, name: str) -> Agent:
        """Get an agent by name.

        Args:
            name: Agent name

        Returns:
            Agent object

        Raises:
            KeyError: If agent not found
        """
        if name not in self._agents:
            raise KeyError(f"Agent not found: {name}")
        return self._agents[name]

    def get_memory(self, name: str) -> Memory:
        """Get a memory by name.

        Args:
            name: Memory name

        Returns:
            Memory object

        Raises:
            KeyError: If memory not found
        """
        if name not in self._memories:
            raise KeyError(f"Memory not found: {name}")
        return self._memories[name]

    def remove_agent(self, name: str) -> "Project":
        """Remove an agent from the project.

        Args:
            name: Agent name to remove

        Returns:
            Self for method chaining
        """
        del self._agents[name]
        return self

    def remove_memory(self, name: str) -> "Project":
        """Remove a memory from the project.

        Args:
            name: Memory name to remove

        Returns:
            Self for method chaining
        """
        del self._memories[name]
        return self

    # ==================== PERSISTENCE ====================

    def save(self, file_path: str) -> str:
        """Save the project configuration to agentcore.json format.

        Args:
            file_path: Path to save the JSON config file

        Returns:
            The file path where config was saved
        """
        # Build agent configs
        agent_configs = []
        for agent in self._agents.values():
            # Find associated memories for this agent
            memory_providers = []
            for memory in self._memories.values():
                mem_strategies = []
                if memory.config.strategies:
                    for s in memory.config.strategies:
                        mem_strategies.append(
                            MemoryStrategyConfig(type=_reverse_map_memory_strategy_type(s.strategy_type.value))
                        )

                memory_providers.append(
                    MemoryProviderConfig(
                        type="AgentCoreMemory",
                        relation=MemoryRelation.OWN,
                        name=memory.name,
                        memory_strategies=mem_strategies if mem_strategies else None,
                    )
                )

            # Extract runtime info from agent's build strategy
            build = agent.build_strategy
            code_location = getattr(build, "source_path", "./src")
            entrypoint = getattr(build, "entrypoint", "main.py:handler")

            network_mode = NetworkMode.PUBLIC
            if agent.config.network_configuration:
                network_mode = NetworkMode(agent.config.network_configuration.network_mode.value)

            runtime = RuntimeConfig(
                artifact="CodeZip",
                entrypoint=entrypoint,
                code_location=code_location,
                network_mode=network_mode,
            )

            agent_config = AgentConfig(
                name=agent.name,
                id=agent.runtime_id,
                runtime=runtime,
                memory_providers=memory_providers if memory_providers else None,
            )
            agent_configs.append(agent_config)

        config = ProjectConfig(
            name=self._name,
            version=self._version,
            description=self._description,
            agents=agent_configs if agent_configs else None,
        )

        path = Path(file_path)
        with open(path, "w") as f:
            json.dump(config.model_dump(mode="json", by_alias=True, exclude_none=True), f, indent=2)

        logger.info("Saved Project config to %s", file_path)
        return str(path)

    def save_deployed_state(self, file_path: str, target_name: Optional[str] = None) -> str:
        """Save the deployed state to deployed-state.json format.

        Args:
            file_path: Path to save the deployed state file
            target_name: Target name (defaults to region)

        Returns:
            The file path where state was saved
        """
        target = target_name or self._region

        # Build agent deployed states
        agent_states: Dict[str, AgentDeployedState] = {}
        for agent in self._agents.values():
            memory_ids = [m.memory_id for m in self._memories.values() if m.memory_id]

            agent_states[agent.name] = AgentDeployedState(
                runtime_id=agent.runtime_id,
                runtime_arn=agent.runtime_arn,
                memory_ids=memory_ids if memory_ids else None,
            )

        state = DeployedState(
            targets={
                target: TargetDeployedState(
                    resources=TargetResources(agents=agent_states if agent_states else None)
                )
            }
        )

        path = Path(file_path)
        with open(path, "w") as f:
            json.dump(state.model_dump(mode="json", by_alias=True, exclude_none=True), f, indent=2)

        logger.info("Saved deployed state to %s", file_path)
        return str(path)

    def save_aws_targets(self, file_path: str, account: Optional[str] = None) -> str:
        """Save AWS targets to aws-targets.json format.

        Args:
            file_path: Path to save the targets file
            account: AWS account ID (auto-detected if not provided)

        Returns:
            The file path where targets were saved
        """
        if not account:
            sts = boto3.client("sts")
            account = sts.get_caller_identity()["Account"]

        targets = [
            AWSTarget(
                name=self._region.replace("-", ""),
                account=account,
                region=self._region,
            )
        ]

        path = Path(file_path)
        with open(path, "w") as f:
            json.dump([t.model_dump(mode="json", by_alias=True) for t in targets], f, indent=2)

        logger.info("Saved AWS targets to %s", file_path)
        return str(path)

    # ==================== BULK OPERATIONS ====================

    def launch_all(self, max_wait: int = 600, poll_interval: int = 10) -> Dict[str, Any]:
        """Launch all memories and agents.

        Memories are created first, then agents are launched.

        Args:
            max_wait: Max seconds to wait for each resource
            poll_interval: Seconds between status checks

        Returns:
            Dictionary with launch results for each resource
        """
        results: Dict[str, Any] = {"memories": {}, "agents": {}}

        # Launch memories first
        for memory in self._memories.values():
            logger.info("Launching memory '%s'...", memory.name)
            results["memories"][memory.name] = memory.launch(
                max_wait=max_wait,
                poll_interval=poll_interval,
            )

        # Then launch agents
        for agent in self._agents.values():
            logger.info("Launching agent '%s'...", agent.name)
            results["agents"][agent.name] = agent.launch(
                max_wait=max_wait,
                poll_interval=poll_interval,
            )

        return results

    def destroy_all(self, max_wait: int = 300, poll_interval: int = 10) -> Dict[str, Any]:
        """Destroy all agents and memories.

        Agents are destroyed first, then memories.

        Args:
            max_wait: Max seconds to wait for each resource
            poll_interval: Seconds between status checks

        Returns:
            Dictionary with destroy results for each resource
        """
        results: Dict[str, Any] = {"agents": {}, "memories": {}}

        # Destroy agents first
        for agent in self._agents.values():
            logger.info("Destroying agent '%s'...", agent.name)
            results["agents"][agent.name] = agent.destroy(
                max_wait=max_wait,
                poll_interval=poll_interval,
            )

        # Then destroy memories
        for memory in self._memories.values():
            logger.info("Destroying memory '%s'...", memory.name)
            results["memories"][memory.name] = memory.delete(
                max_wait=max_wait,
                poll_interval=poll_interval,
            )

        return results

    def status(self) -> Dict[str, Any]:
        """Get status of all resources.

        Returns:
            Dictionary with status for each agent and memory
        """
        return {
            "agents": {a.name: {"deployed": a.is_deployed, "runtime_id": a.runtime_id} for a in self._agents.values()},
            "memories": {m.name: {"active": m.is_active, "memory_id": m.memory_id} for m in self._memories.values()},
        }


def _map_memory_strategy_type(strategy_type: MemoryStrategyType) -> str:
    """Map project config strategy type to Memory strategy type."""
    mapping = {
        MemoryStrategyType.SEMANTIC: "SEMANTIC",
        MemoryStrategyType.SUMMARIZATION: "SUMMARY",
        MemoryStrategyType.USER_PREFERENCE: "USER_PREFERENCE",
        MemoryStrategyType.CUSTOM: "CUSTOM_SEMANTIC",
    }
    return mapping.get(strategy_type, strategy_type.value)


def _reverse_map_memory_strategy_type(strategy_type: str) -> MemoryStrategyType:
    """Map Memory strategy type back to project config strategy type."""
    mapping = {
        "SEMANTIC": MemoryStrategyType.SEMANTIC,
        "SUMMARY": MemoryStrategyType.SUMMARIZATION,
        "USER_PREFERENCE": MemoryStrategyType.USER_PREFERENCE,
        "CUSTOM_SEMANTIC": MemoryStrategyType.CUSTOM,
    }
    return mapping.get(strategy_type, MemoryStrategyType.SEMANTIC)
