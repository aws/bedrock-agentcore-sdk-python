"""Project class for managing multiple Bedrock AgentCore resources.

This module provides a Project class that acts as a resource registry
for managing multiple Agents and Memories with unified YAML persistence.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import yaml

from .memory.config import MemoryConfigModel
from .memory.memory import Memory
from .project_config import ProjectConfigModel
from .runtime.agent import Agent
from .runtime.config import RuntimeConfigModel

logger = logging.getLogger(__name__)


class Project:
    """Resource registry for managing multiple Agents and Memories.

    Project provides a container for Agent and Memory resources with
    bulk operations and unified YAML persistence.

    Example:
        project = Project(name="my-project")
        project.add_agent(Agent(...))
        project.add_memory(Memory(...))
        project.save("my-project.agentcore.yaml")

        # Later
        project = Project.from_yaml("my-project.agentcore.yaml")
        project.launch_all()

    Attributes:
        name: Project name
        agents: List of Agent instances
        memories: List of Memory instances
    """

    def __init__(self, name: str, region: Optional[str] = None):
        """Create a Project instance.

        Args:
            name: Project name
            region: AWS region (applied to all resources)
        """
        self._name = name
        self._region = region or boto3.Session().region_name or "us-west-2"
        self._agents: Dict[str, Agent] = {}
        self._memories: Dict[str, Memory] = {}

        logger.info("Initialized Project '%s' in region %s", name, self._region)

    @classmethod
    def from_yaml(cls, file_path: str, region: Optional[str] = None) -> "Project":
        """Load a project from a YAML configuration file.

        Args:
            file_path: Path to the YAML config file
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
            data = yaml.safe_load(f)

        config = ProjectConfigModel.model_validate(data)
        project = cls(name=config.name, region=region)

        # Reconstruct Agent objects from config
        if config.agents:
            for agent_config in config.agents:
                agent = _create_agent_from_config(agent_config, region or project._region)
                project.add_agent(agent)

        # Reconstruct Memory objects from config
        if config.memories:
            for mem_config in config.memories:
                memory = _create_memory_from_config(mem_config, region or project._region)
                project.add_memory(memory)

        logger.info(
            "Loaded Project '%s' from %s (%d agents, %d memories)",
            config.name,
            file_path,
            len(project._agents),
            len(project._memories),
        )
        return project

    # ==================== PROPERTIES ====================

    @property
    def name(self) -> str:
        """Project name."""
        return self._name

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
            agent: Agent instance to add

        Returns:
            self (for method chaining)
        """
        self._agents[agent.name] = agent
        logger.debug("Added agent '%s' to project", agent.name)
        return self

    def add_memory(self, memory: Memory) -> "Project":
        """Add a memory to the project.

        Args:
            memory: Memory instance to add

        Returns:
            self (for method chaining)
        """
        self._memories[memory.name] = memory
        logger.debug("Added memory '%s' to project", memory.name)
        return self

    def get_agent(self, name: str) -> Agent:
        """Get an agent by name.

        Args:
            name: Agent name

        Returns:
            Agent instance

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
            Memory instance

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
            self (for method chaining)

        Raises:
            KeyError: If agent not found
        """
        del self._agents[name]
        logger.debug("Removed agent '%s' from project", name)
        return self

    def remove_memory(self, name: str) -> "Project":
        """Remove a memory from the project.

        Args:
            name: Memory name to remove

        Returns:
            self (for method chaining)

        Raises:
            KeyError: If memory not found
        """
        del self._memories[name]
        logger.debug("Removed memory '%s' from project", name)
        return self

    # ==================== PERSISTENCE ====================

    def save(self, file_path: str) -> str:
        """Save the project configuration to a YAML file.

        Args:
            file_path: Path to save the YAML config file

        Returns:
            The file path where config was saved
        """
        path = Path(file_path)

        data: Dict[str, Any] = {
            "name": self._name,
        }

        if self._agents:
            data["agents"] = [
                a.config.model_dump(mode="json", by_alias=True, exclude_none=True)
                for a in self._agents.values()
            ]

        if self._memories:
            data["memories"] = [
                m.config.model_dump(mode="json", by_alias=True, exclude_none=True)
                for m in self._memories.values()
            ]

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Saved Project config to %s", file_path)
        return str(path)

    # ==================== BULK OPERATIONS ====================

    def create_all(self, wait: bool = True) -> Dict[str, Any]:
        """Create all memory resources in AWS.

        Args:
            wait: Wait for each memory to become ACTIVE

        Returns:
            Dict mapping memory names to creation results
        """
        results: Dict[str, Any] = {}

        for memory in self._memories.values():
            logger.info("Creating memory '%s'...", memory.name)
            try:
                results[memory.name] = memory.create(wait=wait)
            except Exception as e:
                logger.error("Failed to create memory '%s': %s", memory.name, e)
                results[memory.name] = {"error": str(e)}

        return results

    def launch_all(self, wait: bool = True) -> Dict[str, Any]:
        """Launch all agents in AWS.

        Args:
            wait: Wait for each agent to become ACTIVE

        Returns:
            Dict mapping agent names to launch results
        """
        results: Dict[str, Any] = {}

        for agent in self._agents.values():
            logger.info("Launching agent '%s'...", agent.name)
            try:
                results[agent.name] = agent.launch(wait=wait)
            except Exception as e:
                logger.error("Failed to launch agent '%s': %s", agent.name, e)
                results[agent.name] = {"error": str(e)}

        return results

    def status(self) -> Dict[str, Any]:
        """Get status of all resources.

        Returns:
            Dict with 'agents' and 'memories' status
        """
        return {
            "agents": {a.name: a.status() for a in self._agents.values()},
            "memories": {m.name: m.status() for m in self._memories.values()},
        }

    def destroy_all(self, wait: bool = True) -> Dict[str, Any]:
        """Destroy all resources in AWS.

        Args:
            wait: Wait for each resource to be deleted

        Returns:
            Dict with 'agents' and 'memories' deletion results
        """
        results: Dict[str, Any] = {"agents": {}, "memories": {}}

        # Destroy agents first
        for agent in self._agents.values():
            logger.info("Destroying agent '%s'...", agent.name)
            try:
                results["agents"][agent.name] = agent.destroy(wait=wait)
            except Exception as e:
                logger.error("Failed to destroy agent '%s': %s", agent.name, e)
                results["agents"][agent.name] = {"error": str(e)}

        # Then destroy memories
        for memory in self._memories.values():
            logger.info("Deleting memory '%s'...", memory.name)
            try:
                results["memories"][memory.name] = memory.delete(wait=wait)
            except Exception as e:
                logger.error("Failed to delete memory '%s': %s", memory.name, e)
                results["memories"][memory.name] = {"error": str(e)}

        return results


def _create_agent_from_config(config: RuntimeConfigModel, region: str) -> Agent:
    """Create an Agent instance from a config model.

    Args:
        config: Runtime configuration model
        region: AWS region

    Returns:
        Agent instance
    """
    # Extract network config
    network_mode = "PUBLIC"
    security_groups = None
    subnets = None

    if config.network_configuration:
        network_mode = config.network_configuration.network_mode.value
        if config.network_configuration.vpc_config:
            security_groups = config.network_configuration.vpc_config.security_groups
            subnets = config.network_configuration.vpc_config.subnets

    return Agent(
        name=config.name,
        image_uri=config.artifact.image_uri if config.artifact else "",
        description=config.description,
        network_mode=network_mode,
        security_groups=security_groups,
        subnets=subnets,
        environment_variables=config.environment_variables,
        tags=config.tags,
        region=region,
    )


def _create_memory_from_config(config: MemoryConfigModel, region: str) -> Memory:
    """Create a Memory instance from a config model.

    Args:
        config: Memory configuration model
        region: AWS region

    Returns:
        Memory instance
    """
    strategies = None
    if config.strategies:
        strategies = [
            {
                "type": s.strategy_type.value,
                "namespace": s.namespace,
                "customPrompt": s.custom_prompt,
            }
            for s in config.strategies
        ]

    return Memory(
        name=config.name,
        description=config.description,
        strategies=strategies,
        encryption_key_arn=config.encryption_key_arn,
        tags=config.tags,
        region=region,
    )
