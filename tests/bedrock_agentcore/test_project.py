"""Tests for Project class."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from bedrock_agentcore.project import Project


class TestProjectInit:
    """Tests for Project initialization."""

    @patch("bedrock_agentcore.project.boto3")
    def test_minimal_init(self, mock_boto3: MagicMock) -> None:
        """Test minimal project initialization."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")

        assert project.name == "test-project"
        assert project.agents == []
        assert project.memories == []

    @patch("bedrock_agentcore.project.boto3")
    def test_init_with_region(self, mock_boto3: MagicMock) -> None:
        """Test project initialization with explicit region."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project", region="us-east-1")

        assert project.name == "test-project"
        assert project._region == "us-east-1"


class TestProjectAddRemove:
    """Tests for Project add/remove operations."""

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_add_agent(self, mock_agent_boto3: MagicMock, mock_project_boto3: MagicMock) -> None:
        """Test adding agent to project."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"
        mock_agent_boto3.Session.return_value.region_name = "us-west-2"

        from bedrock_agentcore.runtime import Agent

        project = Project(name="test-project")
        agent = Agent(
            name="test-agent",
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
        )

        result = project.add_agent(agent)

        assert result is project  # Returns self for chaining
        assert len(project.agents) == 1
        assert project.agents[0].name == "test-agent"

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_add_memory(self, mock_memory_client: MagicMock, mock_project_boto3: MagicMock) -> None:
        """Test adding memory to project."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"
        mock_memory_client.return_value.region_name = "us-west-2"

        from bedrock_agentcore.memory import Memory

        project = Project(name="test-project")
        memory = Memory(
            name="test-memory",
            strategies=[{"type": "SEMANTIC", "namespace": "facts/"}],
        )

        result = project.add_memory(memory)

        assert result is project  # Returns self for chaining
        assert len(project.memories) == 1
        assert project.memories[0].name == "test-memory"

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_get_agent(self, mock_agent_boto3: MagicMock, mock_project_boto3: MagicMock) -> None:
        """Test getting agent by name."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"
        mock_agent_boto3.Session.return_value.region_name = "us-west-2"

        from bedrock_agentcore.runtime import Agent

        project = Project(name="test-project")
        agent = Agent(
            name="test-agent",
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
        )
        project.add_agent(agent)

        result = project.get_agent("test-agent")

        assert result is agent

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_get_agent_not_found_raises(self, mock_agent_boto3: MagicMock, mock_project_boto3: MagicMock) -> None:
        """Test getting non-existent agent raises KeyError."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")

        with pytest.raises(KeyError, match="Agent not found"):
            project.get_agent("nonexistent-agent")

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_get_memory(self, mock_memory_client: MagicMock, mock_project_boto3: MagicMock) -> None:
        """Test getting memory by name."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"
        mock_memory_client.return_value.region_name = "us-west-2"

        from bedrock_agentcore.memory import Memory

        project = Project(name="test-project")
        memory = Memory(name="test-memory")
        project.add_memory(memory)

        result = project.get_memory("test-memory")

        assert result is memory

    @patch("bedrock_agentcore.project.boto3")
    def test_get_memory_not_found_raises(self, mock_project_boto3: MagicMock) -> None:
        """Test getting non-existent memory raises KeyError."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")

        with pytest.raises(KeyError, match="Memory not found"):
            project.get_memory("nonexistent-memory")

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_remove_agent(self, mock_agent_boto3: MagicMock, mock_project_boto3: MagicMock) -> None:
        """Test removing agent from project."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"
        mock_agent_boto3.Session.return_value.region_name = "us-west-2"

        from bedrock_agentcore.runtime import Agent

        project = Project(name="test-project")
        agent = Agent(
            name="test-agent",
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
        )
        project.add_agent(agent)

        result = project.remove_agent("test-agent")

        assert result is project
        assert len(project.agents) == 0

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_remove_memory(self, mock_memory_client: MagicMock, mock_project_boto3: MagicMock) -> None:
        """Test removing memory from project."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"
        mock_memory_client.return_value.region_name = "us-west-2"

        from bedrock_agentcore.memory import Memory

        project = Project(name="test-project")
        memory = Memory(name="test-memory")
        project.add_memory(memory)

        result = project.remove_memory("test-memory")

        assert result is project
        assert len(project.memories) == 0


class TestProjectSaveLoad:
    """Tests for Project save/load operations."""

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.runtime.agent.boto3")
    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_save_to_yaml(
        self, mock_memory_client: MagicMock, mock_agent_boto3: MagicMock, mock_project_boto3: MagicMock
    ) -> None:
        """Test saving project to YAML file."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"
        mock_agent_boto3.Session.return_value.region_name = "us-west-2"
        mock_memory_client.return_value.region_name = "us-west-2"

        from bedrock_agentcore.memory import Memory
        from bedrock_agentcore.runtime import Agent

        project = Project(name="test-project")
        project.add_agent(
            Agent(
                name="test-agent",
                image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
                description="Test agent",
            )
        )
        project.add_memory(
            Memory(
                name="test-memory",
                description="Test memory",
                strategies=[{"type": "SEMANTIC", "namespace": "facts/"}],
            )
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            file_path = f.name

        try:
            result = project.save(file_path)
            assert result == file_path

            # Verify file contents
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)

            assert data["name"] == "test-project"
            assert len(data["agents"]) == 1
            assert data["agents"][0]["name"] == "test-agent"
            assert len(data["memories"]) == 1
            assert data["memories"][0]["name"] == "test-memory"
        finally:
            Path(file_path).unlink(missing_ok=True)

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.runtime.agent.boto3")
    @patch("bedrock_agentcore.memory.memory.MemoryClient")
    def test_from_yaml(
        self, mock_memory_client: MagicMock, mock_agent_boto3: MagicMock, mock_project_boto3: MagicMock
    ) -> None:
        """Test loading project from YAML file."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"
        mock_agent_boto3.Session.return_value.region_name = "us-west-2"
        mock_memory_client.return_value.region_name = "us-west-2"
        mock_memory_client.return_value.list_memories.return_value = []

        # Mock paginator for Agent
        mock_control_plane = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"agentRuntimeSummaries": []}]
        mock_control_plane.get_paginator.return_value = mock_paginator
        mock_agent_boto3.client.return_value = mock_control_plane

        yaml_content = """
name: test-project
agents:
  - name: agent-1
    description: First agent
    artifact:
      imageUri: 123456789012.dkr.ecr.us-west-2.amazonaws.com/agent-1:latest
    networkConfiguration:
      networkMode: PUBLIC
  - name: agent-2
    description: Second agent
    artifact:
      imageUri: 123456789012.dkr.ecr.us-west-2.amazonaws.com/agent-2:latest
memories:
  - name: memory-1
    description: First memory
    strategies:
      - type: SEMANTIC
        namespace: facts/
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            file_path = f.name

        try:
            project = Project.from_yaml(file_path)

            assert project.name == "test-project"
            assert len(project.agents) == 2
            assert project.agents[0].name == "agent-1"
            assert project.agents[1].name == "agent-2"
            assert len(project.memories) == 1
            assert project.memories[0].name == "memory-1"
        finally:
            Path(file_path).unlink(missing_ok=True)

    @patch("bedrock_agentcore.project.boto3")
    def test_from_yaml_file_not_found(self, mock_project_boto3: MagicMock) -> None:
        """Test that from_yaml raises FileNotFoundError for missing file."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Project.from_yaml("/nonexistent/path/config.yaml")


class TestProjectBulkOperations:
    """Tests for Project bulk operations."""

    @patch("bedrock_agentcore.project.boto3")
    def test_status_empty_project(self, mock_project_boto3: MagicMock) -> None:
        """Test status on empty project."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")

        status = project.status()

        assert status["agents"] == {}
        assert status["memories"] == {}

    @patch("bedrock_agentcore.project.boto3")
    def test_destroy_all_empty_project(self, mock_project_boto3: MagicMock) -> None:
        """Test destroy_all on empty project."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")

        result = project.destroy_all()

        assert result["agents"] == {}
        assert result["memories"] == {}

    @patch("bedrock_agentcore.project.boto3")
    def test_create_all_empty_project(self, mock_project_boto3: MagicMock) -> None:
        """Test create_all on empty project."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")

        result = project.create_all()

        assert result == {}

    @patch("bedrock_agentcore.project.boto3")
    def test_launch_all_empty_project(self, mock_project_boto3: MagicMock) -> None:
        """Test launch_all on empty project."""
        mock_project_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")

        result = project.launch_all()

        assert result == {}
