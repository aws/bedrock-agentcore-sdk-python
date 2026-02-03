"""Tests for Project class."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore import Project
from bedrock_agentcore.memory import Memory
from bedrock_agentcore.runtime import Agent
from bedrock_agentcore.runtime.build import DirectCodeDeploy, ECR


class TestProjectInit:
    """Tests for Project initialization."""

    @patch("bedrock_agentcore.project.boto3")
    def test_basic_init(self, mock_boto3: MagicMock) -> None:
        """Test basic project initialization."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")

        assert project.name == "test-project"
        assert project.region == "us-west-2"
        assert project.agents == []
        assert project.memories == []

    @patch("bedrock_agentcore.project.boto3")
    def test_init_with_all_params(self, mock_boto3: MagicMock) -> None:
        """Test project initialization with all parameters."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(
            name="test-project",
            version="1.0.0",
            description="Test description",
            region="us-east-1",
        )

        assert project.name == "test-project"
        assert project.version == "1.0.0"
        assert project.description == "Test description"
        assert project.region == "us-east-1"


class TestProjectFromJson:
    """Tests for Project.from_json()."""

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.runtime.agent.boto3")
    @patch("bedrock_agentcore.memory.client.boto3")
    def test_from_json_minimal(
        self, mock_mem_boto3: MagicMock, mock_agent_boto3: MagicMock, mock_proj_boto3: MagicMock
    ) -> None:
        """Test loading minimal project from JSON."""
        mock_proj_boto3.Session.return_value.region_name = "us-west-2"
        mock_agent_boto3.Session.return_value.region_name = "us-west-2"
        mock_mem_boto3.Session.return_value.region_name = "us-west-2"

        config = {
            "name": "test-project",
            "agents": [
                {
                    "name": "test-agent",
                    "runtime": {
                        "artifact": "CodeZip",
                        "entrypoint": "main.py:handler",
                        "codeLocation": "./src",
                    },
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()

            project = Project.from_json(f.name)

            assert project.name == "test-project"
            assert len(project.agents) == 1
            assert project.agents[0].name == "test-agent"

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.runtime.agent.boto3")
    @patch("bedrock_agentcore.memory.client.boto3")
    def test_from_json_with_memory(
        self, mock_mem_boto3: MagicMock, mock_agent_boto3: MagicMock, mock_proj_boto3: MagicMock
    ) -> None:
        """Test loading project with memory providers from JSON."""
        mock_proj_boto3.Session.return_value.region_name = "us-west-2"
        mock_agent_boto3.Session.return_value.region_name = "us-west-2"
        mock_mem_boto3.Session.return_value.region_name = "us-west-2"

        config = {
            "name": "test-project",
            "agents": [
                {
                    "name": "test-agent",
                    "runtime": {
                        "entrypoint": "main.py:handler",
                        "codeLocation": "./src",
                    },
                    "memoryProviders": [
                        {
                            "type": "AgentCoreMemory",
                            "relation": "own",
                            "name": "test-memory",
                            "memoryStrategies": [{"type": "SEMANTIC"}],
                        }
                    ],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()

            project = Project.from_json(f.name)

            assert len(project.memories) == 1
            assert project.memories[0].name == "test-memory"

    def test_from_json_file_not_found(self) -> None:
        """Test from_json raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            Project.from_json("/nonexistent/file.json")


class TestProjectResourceManagement:
    """Tests for resource management methods."""

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_add_agent(self, mock_agent_boto3: MagicMock, mock_proj_boto3: MagicMock) -> None:
        """Test adding an agent to project."""
        mock_proj_boto3.Session.return_value.region_name = "us-west-2"
        mock_agent_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")
        build = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(name="test-agent", build=build)

        result = project.add_agent(agent)

        assert result is project  # Method chaining
        assert len(project.agents) == 1
        assert project.agents[0].name == "test-agent"

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.memory.client.boto3")
    def test_add_memory(self, mock_mem_boto3: MagicMock, mock_proj_boto3: MagicMock) -> None:
        """Test adding a memory to project."""
        mock_proj_boto3.Session.return_value.region_name = "us-west-2"
        mock_mem_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")
        memory = Memory(name="test-memory")

        result = project.add_memory(memory)

        assert result is project  # Method chaining
        assert len(project.memories) == 1
        assert project.memories[0].name == "test-memory"

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_get_agent(self, mock_agent_boto3: MagicMock, mock_proj_boto3: MagicMock) -> None:
        """Test getting an agent by name."""
        mock_proj_boto3.Session.return_value.region_name = "us-west-2"
        mock_agent_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")
        build = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(name="test-agent", build=build)
        project.add_agent(agent)

        result = project.get_agent("test-agent")
        assert result is agent

    @patch("bedrock_agentcore.project.boto3")
    def test_get_agent_not_found(self, mock_boto3: MagicMock) -> None:
        """Test get_agent raises KeyError for missing agent."""
        mock_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")

        with pytest.raises(KeyError, match="Agent not found"):
            project.get_agent("nonexistent")

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.runtime.agent.boto3")
    def test_remove_agent(self, mock_agent_boto3: MagicMock, mock_proj_boto3: MagicMock) -> None:
        """Test removing an agent from project."""
        mock_proj_boto3.Session.return_value.region_name = "us-west-2"
        mock_agent_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")
        build = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(name="test-agent", build=build)
        project.add_agent(agent)

        result = project.remove_agent("test-agent")

        assert result is project  # Method chaining
        assert len(project.agents) == 0


class TestProjectSave:
    """Tests for Project.save()."""

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.runtime.agent.boto3")
    @patch("bedrock_agentcore.memory.client.boto3")
    def test_save_to_json(
        self, mock_mem_boto3: MagicMock, mock_agent_boto3: MagicMock, mock_proj_boto3: MagicMock
    ) -> None:
        """Test saving project to JSON file."""
        mock_proj_boto3.Session.return_value.region_name = "us-west-2"
        mock_agent_boto3.Session.return_value.region_name = "us-west-2"
        mock_mem_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project", version="1.0.0")
        build = DirectCodeDeploy(source_path="./src", entrypoint="main.py:handler")
        agent = Agent(name="test-agent", build=build)
        memory = Memory(
            name="test-memory", strategies=[{"type": "SEMANTIC", "namespace": "facts/{sessionId}/"}]
        )
        project.add_agent(agent)
        project.add_memory(memory)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "agentcore.json"
            result = project.save(str(output_path))

            assert Path(result).exists()

            with open(result) as f:
                saved_config = json.load(f)

            assert saved_config["name"] == "test-project"
            assert saved_config["version"] == "1.0.0"
            assert len(saved_config["agents"]) == 1
            assert saved_config["agents"][0]["name"] == "test-agent"


class TestProjectStatus:
    """Tests for Project.status()."""

    @patch("bedrock_agentcore.project.boto3")
    @patch("bedrock_agentcore.runtime.agent.boto3")
    @patch("bedrock_agentcore.memory.client.boto3")
    def test_status(
        self, mock_mem_boto3: MagicMock, mock_agent_boto3: MagicMock, mock_proj_boto3: MagicMock
    ) -> None:
        """Test getting status of all resources."""
        mock_proj_boto3.Session.return_value.region_name = "us-west-2"
        mock_agent_boto3.Session.return_value.region_name = "us-west-2"
        mock_mem_boto3.Session.return_value.region_name = "us-west-2"

        project = Project(name="test-project")
        build = ECR(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest")
        agent = Agent(name="test-agent", build=build)
        memory = Memory(name="test-memory")
        project.add_agent(agent)
        project.add_memory(memory)

        status = project.status()

        assert "agents" in status
        assert "memories" in status
        assert "test-agent" in status["agents"]
        assert "test-memory" in status["memories"]
        assert status["agents"]["test-agent"]["deployed"] is False
        assert status["memories"]["test-memory"]["active"] is False
