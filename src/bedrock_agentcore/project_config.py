"""Configuration models for Bedrock AgentCore Project.

This module provides Pydantic models for Project configuration
with YAML serialization support.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from .memory.config import MemoryConfigModel
from .runtime.config import RuntimeConfigModel


class ProjectConfigModel(BaseModel):
    """Complete project configuration model.

    This model represents the configuration for a Bedrock AgentCore project,
    containing multiple agents and memories.

    Attributes:
        name: Project name
        agents: List of agent configurations
        memories: List of memory configurations
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    agents: Optional[List[RuntimeConfigModel]] = None
    memories: Optional[List[MemoryConfigModel]] = None
