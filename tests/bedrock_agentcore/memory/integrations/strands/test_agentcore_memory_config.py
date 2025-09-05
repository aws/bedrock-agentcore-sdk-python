"""Tests for AgentCore Memory configuration models."""

import pytest
from pydantic import ValidationError

from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig


class TestRetrievalConfig:
    """Test RetrievalConfig validation."""

    def test_valid_config(self):
        """Test valid RetrievalConfig creation."""
        config = RetrievalConfig(top_k=5, relevance_score=0.5, strategy_id="test")
        assert config.top_k == 5
        assert config.relevance_score == 0.5
        assert config.strategy_id == "test"

    def test_defaults(self):
        """Test default values."""
        config = RetrievalConfig()
        assert config.top_k == 10
        assert config.relevance_score == 0.2
        assert config.strategy_id is None
        assert config.initialization_query is None

    def test_optional_fields(self):
        """Test optional fields with custom values."""
        config = RetrievalConfig(
            initialization_query="custom query for memories",
        )

        assert config.initialization_query == "custom query for memories"

    def test_all_fields(self):
        """Test all fields together."""
        config = RetrievalConfig(
            top_k=15,
            relevance_score=0.7,
            strategy_id="test_strategy",
            initialization_query="test query",
        )

        assert config.top_k == 15
        assert config.relevance_score == 0.7
        assert config.strategy_id == "test_strategy"
        assert config.initialization_query == "test query"

    def test_top_k_validation(self):
        """Test top_k validation."""
        with pytest.raises(ValidationError):
            RetrievalConfig(top_k=0)
        with pytest.raises(ValidationError):
            RetrievalConfig(top_k=1001)

    def test_relevance_score_validation(self):
        """Test relevance_score validation."""
        with pytest.raises(ValidationError):
            RetrievalConfig(relevance_score=-0.1)
        with pytest.raises(ValidationError):
            RetrievalConfig(relevance_score=1.1)


class TestAgentCoreMemoryConfig:
    """Test AgentCoreMemoryConfig validation."""

    def test_valid_config(self):
        """Test valid config creation."""
        config = AgentCoreMemoryConfig(memory_id="mem-123", session_id="sess-456", actor_id="actor-789")
        assert config.memory_id == "mem-123"
        assert config.session_id == "sess-456"
        assert config.actor_id == "actor-789"

    def test_empty_string_validation(self):
        """Test empty string validation."""
        with pytest.raises(ValidationError):
            AgentCoreMemoryConfig(memory_id="", session_id="sess", actor_id="actor")
        with pytest.raises(ValidationError):
            AgentCoreMemoryConfig(memory_id="mem", session_id="", actor_id="actor")
        with pytest.raises(ValidationError):
            AgentCoreMemoryConfig(memory_id="mem", session_id="sess", actor_id="")

    def test_with_retrieval_config(self):
        """Test config with retrieval configuration."""
        retrieval = RetrievalConfig(top_k=5)
        config = AgentCoreMemoryConfig(
            memory_id="mem-123", session_id="sess-456", actor_id="actor-789", retrieval_config={"namespace1": retrieval}
        )
        assert config.retrieval_config["namespace1"].top_k == 5
