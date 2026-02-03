"""Tests for memory configuration models."""

import pytest

from bedrock_agentcore.memory.config import (
    MemoryConfigModel,
    StrategyConfigModel,
    StrategyType,
)


class TestStrategyType:
    """Tests for StrategyType enum."""

    def test_strategy_type_values(self) -> None:
        """Test that all expected strategy type values exist."""
        assert StrategyType.SEMANTIC == "SEMANTIC"
        assert StrategyType.SUMMARY == "SUMMARY"
        assert StrategyType.USER_PREFERENCE == "USER_PREFERENCE"
        assert StrategyType.CUSTOM_SEMANTIC == "CUSTOM_SEMANTIC"


class TestStrategyConfigModel:
    """Tests for StrategyConfigModel."""

    def test_create_with_alias(self) -> None:
        """Test creating StrategyConfigModel with alias names."""
        config = StrategyConfigModel(
            type=StrategyType.SEMANTIC,
            namespace="facts/{sessionId}/",
        )
        assert config.strategy_type == StrategyType.SEMANTIC
        assert config.namespace == "facts/{sessionId}/"
        assert config.custom_prompt is None

    def test_create_with_custom_prompt(self) -> None:
        """Test creating StrategyConfigModel with custom prompt."""
        config = StrategyConfigModel(
            type=StrategyType.CUSTOM_SEMANTIC,
            namespace="custom/{sessionId}/",
            customPrompt="Extract important facts from the conversation.",
        )
        assert config.strategy_type == StrategyType.CUSTOM_SEMANTIC
        assert config.namespace == "custom/{sessionId}/"
        assert config.custom_prompt == "Extract important facts from the conversation."

    def test_dump_by_alias(self) -> None:
        """Test dumping config with alias names."""
        config = StrategyConfigModel(
            type=StrategyType.SUMMARY,
            namespace="summaries/",
        )
        data = config.model_dump(by_alias=True)
        assert data["type"] == "SUMMARY"
        assert data["namespace"] == "summaries/"


class TestMemoryConfigModel:
    """Tests for MemoryConfigModel."""

    def test_minimal_config(self) -> None:
        """Test minimal config with just name."""
        config = MemoryConfigModel(name="test-memory")
        assert config.name == "test-memory"
        assert config.description is None
        assert config.strategies is None
        assert config.encryption_key_arn is None
        assert config.tags is None

    def test_full_config(self) -> None:
        """Test full config with all fields."""
        strategies = [
            StrategyConfigModel(type=StrategyType.SEMANTIC, namespace="facts/"),
            StrategyConfigModel(type=StrategyType.SUMMARY, namespace="summaries/"),
        ]
        config = MemoryConfigModel(
            name="test-memory",
            description="Test memory description",
            strategies=strategies,
            encryptionKeyArn="arn:aws:kms:us-west-2:123456789012:key/abc123",
            tags={"Environment": "test"},
        )
        assert config.name == "test-memory"
        assert config.description == "Test memory description"
        assert config.strategies is not None
        assert len(config.strategies) == 2
        assert config.encryption_key_arn == "arn:aws:kms:us-west-2:123456789012:key/abc123"
        assert config.tags == {"Environment": "test"}

    def test_dump_excludes_none(self) -> None:
        """Test that dump excludes None values."""
        config = MemoryConfigModel(name="test-memory")
        data = config.model_dump(by_alias=True, exclude_none=True)
        assert "name" in data
        assert "description" not in data
        assert "strategies" not in data
        assert "encryptionKeyArn" not in data

    def test_validate_from_dict(self) -> None:
        """Test validating from dict (YAML-like structure)."""
        data = {
            "name": "test-memory",
            "strategies": [
                {"type": "SEMANTIC", "namespace": "facts/"},
                {"type": "SUMMARY", "namespace": "summaries/"},
            ],
        }
        config = MemoryConfigModel.model_validate(data)
        assert config.name == "test-memory"
        assert config.strategies is not None
        assert len(config.strategies) == 2
        assert config.strategies[0].strategy_type == StrategyType.SEMANTIC
        assert config.strategies[1].strategy_type == StrategyType.SUMMARY
