"""Tests for A2A models."""

import pytest

from bedrock_agentcore.runtime.a2a_models import (
    A2A_DEFAULT_PORT,
    A2A_METHOD_MESSAGE_SEND,
    A2AArtifact,
    A2AMessage,
    A2AMessagePart,
    AgentCard,
    AgentSkill,
    JsonRpcErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
    build_runtime_url,
)


class TestAgentSkill:
    def test_basic_creation(self):
        """Test creating a basic AgentSkill."""
        skill = AgentSkill(
            id="calc",
            name="Calculator",
            description="Perform arithmetic calculations",
        )
        assert skill.id == "calc"
        assert skill.name == "Calculator"
        assert skill.description == "Perform arithmetic calculations"
        assert skill.tags == []

    def test_creation_with_tags(self):
        """Test creating AgentSkill with tags."""
        skill = AgentSkill(
            id="search",
            name="Web Search",
            description="Search the web",
            tags=["search", "web", "information"],
        )
        assert skill.tags == ["search", "web", "information"]

    def test_to_dict(self):
        """Test AgentSkill serialization to dict."""
        skill = AgentSkill(
            id="calc",
            name="Calculator",
            description="Math operations",
            tags=["math"],
        )
        result = skill.to_dict()
        assert result == {
            "id": "calc",
            "name": "Calculator",
            "description": "Math operations",
            "tags": ["math"],
        }


class TestAgentCard:
    def test_basic_creation(self):
        """Test creating a basic AgentCard."""
        card = AgentCard(
            name="Test Agent",
            description="A test agent",
        )
        assert card.name == "Test Agent"
        assert card.description == "A test agent"
        assert card.version == "1.0.0"
        assert card.protocol_version == "0.3.0"
        assert card.preferred_transport == "JSONRPC"
        assert card.capabilities == {"streaming": True}
        assert card.default_input_modes == ["text"]
        assert card.default_output_modes == ["text"]
        assert card.skills == []

    def test_creation_with_skills(self):
        """Test AgentCard with skills."""
        skills = [
            AgentSkill(id="s1", name="Skill 1", description="First skill"),
            AgentSkill(id="s2", name="Skill 2", description="Second skill"),
        ]
        card = AgentCard(
            name="Multi-Skill Agent",
            description="An agent with multiple skills",
            skills=skills,
        )
        assert len(card.skills) == 2
        assert card.skills[0].id == "s1"
        assert card.skills[1].id == "s2"

    def test_to_dict_without_url(self):
        """Test AgentCard serialization without URL."""
        card = AgentCard(
            name="Test Agent",
            description="A test agent",
        )
        result = card.to_dict()
        assert result["name"] == "Test Agent"
        assert result["description"] == "A test agent"
        assert result["protocolVersion"] == "0.3.0"
        assert result["preferredTransport"] == "JSONRPC"
        assert "url" not in result

    def test_to_dict_with_url(self):
        """Test AgentCard serialization with URL."""
        card = AgentCard(
            name="Test Agent",
            description="A test agent",
        )
        result = card.to_dict(url="https://example.com/agent")
        assert result["url"] == "https://example.com/agent"

    def test_to_dict_with_skills(self):
        """Test AgentCard serialization with skills."""
        skills = [AgentSkill(id="s1", name="Skill 1", description="First skill")]
        card = AgentCard(
            name="Test Agent",
            description="A test agent",
            skills=skills,
        )
        result = card.to_dict()
        assert len(result["skills"]) == 1
        assert result["skills"][0]["id"] == "s1"


class TestJsonRpcRequest:
    def test_from_dict(self):
        """Test creating JsonRpcRequest from dict."""
        data = {
            "jsonrpc": "2.0",
            "id": "req-001",
            "method": "message/send",
            "params": {"message": {"text": "Hello"}},
        }
        request = JsonRpcRequest.from_dict(data)
        assert request.jsonrpc == "2.0"
        assert request.id == "req-001"
        assert request.method == "message/send"
        assert request.params == {"message": {"text": "Hello"}}

    def test_from_dict_minimal(self):
        """Test creating JsonRpcRequest with minimal data."""
        data = {"method": "test"}
        request = JsonRpcRequest.from_dict(data)
        assert request.jsonrpc == "2.0"
        assert request.id is None
        assert request.method == "test"
        assert request.params is None


class TestJsonRpcResponse:
    def test_success_response(self):
        """Test creating a success response."""
        response = JsonRpcResponse.success("req-001", {"result": "success"})
        assert response.id == "req-001"
        assert response.result == {"result": "success"}
        assert response.error is None

    def test_error_response(self):
        """Test creating an error response."""
        response = JsonRpcResponse.error_response(
            "req-001",
            JsonRpcErrorCode.INTERNAL_ERROR,
            "Something went wrong",
        )
        assert response.id == "req-001"
        assert response.result is None
        assert response.error is not None
        assert response.error.code == JsonRpcErrorCode.INTERNAL_ERROR
        assert response.error.message == "Something went wrong"

    def test_success_to_dict(self):
        """Test success response serialization."""
        response = JsonRpcResponse.success("req-001", {"data": "test"})
        result = response.to_dict()
        assert result == {
            "jsonrpc": "2.0",
            "id": "req-001",
            "result": {"data": "test"},
        }

    def test_error_to_dict(self):
        """Test error response serialization."""
        response = JsonRpcResponse.error_response(
            "req-001",
            -32600,
            "Invalid request",
        )
        result = response.to_dict()
        assert result == {
            "jsonrpc": "2.0",
            "id": "req-001",
            "error": {
                "code": -32600,
                "message": "Invalid request",
            },
        }


class TestA2AMessagePart:
    def test_text_part(self):
        """Test creating a text message part."""
        part = A2AMessagePart(kind="text", text="Hello, world!")
        assert part.kind == "text"
        assert part.text == "Hello, world!"
        assert part.file is None
        assert part.data is None

    def test_to_dict(self):
        """Test message part serialization."""
        part = A2AMessagePart(kind="text", text="Test message")
        result = part.to_dict()
        assert result == {"kind": "text", "text": "Test message"}

    def test_from_dict(self):
        """Test creating message part from dict."""
        data = {"kind": "text", "text": "Hello"}
        part = A2AMessagePart.from_dict(data)
        assert part.kind == "text"
        assert part.text == "Hello"


class TestA2AMessage:
    def test_basic_creation(self):
        """Test creating a basic A2A message."""
        parts = [A2AMessagePart(kind="text", text="Hello")]
        message = A2AMessage(role="user", parts=parts)
        assert message.role == "user"
        assert len(message.parts) == 1
        assert message.message_id is None

    def test_with_message_id(self):
        """Test message with ID."""
        parts = [A2AMessagePart(kind="text", text="Hello")]
        message = A2AMessage(role="user", parts=parts, message_id="msg-001")
        assert message.message_id == "msg-001"

    def test_get_text(self):
        """Test extracting text from message."""
        parts = [
            A2AMessagePart(kind="text", text="Line 1"),
            A2AMessagePart(kind="text", text="Line 2"),
        ]
        message = A2AMessage(role="user", parts=parts)
        assert message.get_text() == "Line 1\nLine 2"

    def test_to_dict(self):
        """Test message serialization."""
        parts = [A2AMessagePart(kind="text", text="Hello")]
        message = A2AMessage(role="user", parts=parts, message_id="msg-001")
        result = message.to_dict()
        assert result == {
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello"}],
            "messageId": "msg-001",
        }

    def test_from_dict(self):
        """Test creating message from dict."""
        data = {
            "role": "agent",
            "parts": [{"kind": "text", "text": "Response"}],
            "messageId": "msg-002",
        }
        message = A2AMessage.from_dict(data)
        assert message.role == "agent"
        assert len(message.parts) == 1
        assert message.parts[0].text == "Response"
        assert message.message_id == "msg-002"


class TestA2AArtifact:
    def test_basic_creation(self):
        """Test creating a basic artifact."""
        parts = [A2AMessagePart(kind="text", text="Result")]
        artifact = A2AArtifact(
            artifact_id="art-001",
            name="response",
            parts=parts,
        )
        assert artifact.artifact_id == "art-001"
        assert artifact.name == "response"
        assert len(artifact.parts) == 1

    def test_from_text(self):
        """Test creating text artifact."""
        artifact = A2AArtifact.from_text("art-001", "response", "Hello")
        assert artifact.artifact_id == "art-001"
        assert artifact.name == "response"
        assert len(artifact.parts) == 1
        assert artifact.parts[0].kind == "text"
        assert artifact.parts[0].text == "Hello"

    def test_to_dict(self):
        """Test artifact serialization."""
        artifact = A2AArtifact.from_text("art-001", "response", "Result")
        result = artifact.to_dict()
        assert result == {
            "artifactId": "art-001",
            "name": "response",
            "parts": [{"kind": "text", "text": "Result"}],
        }


class TestBuildRuntimeUrl:
    def test_basic_url(self):
        """Test building runtime URL."""
        arn = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-agent"
        url = build_runtime_url(arn)
        # ARN should be URL-encoded
        assert "us-west-2" in url
        assert "arn%3Aaws%3Abedrock-agentcore" in url
        assert url.startswith("https://bedrock-agentcore.us-west-2.amazonaws.com/runtimes/")
        assert url.endswith("/invocations/")

    def test_url_with_region(self):
        """Test building URL with custom region."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-agent"
        url = build_runtime_url(arn, region="us-east-1")
        assert "us-east-1" in url
        assert "bedrock-agentcore.us-east-1.amazonaws.com" in url

    def test_special_characters_encoded(self):
        """Test that special characters in ARN are properly encoded."""
        arn = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/agent-with-special"
        url = build_runtime_url(arn)
        # Colon and slash should be encoded
        assert "%3A" in url  # Encoded colon
        assert "%2F" in url  # Encoded slash


class TestConstants:
    def test_default_port(self):
        """Test A2A default port."""
        assert A2A_DEFAULT_PORT == 9000

    def test_method_constants(self):
        """Test A2A method constants."""
        assert A2A_METHOD_MESSAGE_SEND == "message/send"
