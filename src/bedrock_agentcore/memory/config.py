"""Configuration models for Bedrock AgentCore Memory.

This module provides Pydantic models for Memory configuration
with YAML serialization support.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class StrategyType(str, Enum):
    """Memory strategy types."""

    SEMANTIC = "SEMANTIC"
    SUMMARY = "SUMMARY"
    USER_PREFERENCE = "USER_PREFERENCE"
    CUSTOM_SEMANTIC = "CUSTOM_SEMANTIC"


class StrategyConfigModel(BaseModel):
    """Memory strategy configuration."""

    model_config = ConfigDict(populate_by_name=True)

    strategy_type: StrategyType = Field(alias="type")
    namespace: str
    custom_prompt: Optional[str] = Field(default=None, alias="customPrompt")


class MemoryConfigModel(BaseModel):
    """Complete memory configuration model.

    This model represents the configuration for a Bedrock AgentCore memory,
    suitable for YAML serialization and deserialization.

    Attributes:
        name: Unique memory name
        description: Optional description
        strategies: List of memory extraction strategies
        encryption_key_arn: Optional KMS key ARN for encryption
        tags: Resource tags
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: Optional[str] = None
    strategies: Optional[List[StrategyConfigModel]] = None
    encryption_key_arn: Optional[str] = Field(default=None, alias="encryptionKeyArn")
    tags: Optional[Dict[str, str]] = None
