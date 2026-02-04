"""Models for Bedrock AgentCore A2A runtime.

Contains data models for A2A protocol including Agent Card, JSON-RPC 2.0 messages,
and related types.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote


class JsonRpcErrorCode(int, Enum):
    """Standard JSON-RPC 2.0 error codes and A2A-specific error codes."""

    # Standard JSON-RPC 2.0 errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # A2A-specific error codes (AgentCore Runtime)
    RESOURCE_NOT_FOUND = -32501
    VALIDATION_ERROR = -32502
    THROTTLING = -32503
    RESOURCE_CONFLICT = -32504
    RUNTIME_CLIENT_ERROR = -32505


@dataclass
class AgentSkill:
    """A2A Agent Skill definition.

    Skills describe specific capabilities that the agent can perform.
    """

    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
        }


@dataclass
class AgentCard:
    """A2A Agent Card metadata.

    Agent Cards describe an agent's identity, capabilities, and how to communicate with it.
    This metadata is served at /.well-known/agent-card.json endpoint.
    """

    name: str
    description: str
    version: str = "1.0.0"
    protocol_version: str = "0.3.0"
    preferred_transport: str = "JSONRPC"
    capabilities: Dict[str, Any] = field(default_factory=lambda: {"streaming": True})
    default_input_modes: List[str] = field(default_factory=lambda: ["text"])
    default_output_modes: List[str] = field(default_factory=lambda: ["text"])
    skills: List[AgentSkill] = field(default_factory=list)

    def to_dict(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Args:
            url: The URL where this agent is accessible. If not provided,
                 the 'url' field will be omitted from the output.

        Returns:
            Dictionary representation of the Agent Card.
        """
        result = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "protocolVersion": self.protocol_version,
            "preferredTransport": self.preferred_transport,
            "capabilities": self.capabilities,
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "skills": [skill.to_dict() for skill in self.skills],
        }
        if url:
            result["url"] = url
        return result


@dataclass
class JsonRpcRequest:
    """JSON-RPC 2.0 Request object."""

    method: str
    id: Optional[Union[str, int]] = None
    params: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JsonRpcRequest":
        """Create from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method", ""),
            params=data.get("params"),
        )


@dataclass
class JsonRpcError:
    """JSON-RPC 2.0 Error object."""

    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 Response object."""

    id: Optional[Union[str, int]]
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        response = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            response["error"] = self.error.to_dict()
        else:
            response["result"] = self.result
        return response

    @classmethod
    def success(cls, id: Optional[Union[str, int]], result: Any) -> "JsonRpcResponse":
        """Create a success response."""
        return cls(id=id, result=result)

    @classmethod
    def error_response(
        cls,
        id: Optional[Union[str, int]],
        code: int,
        message: str,
        data: Optional[Any] = None,
    ) -> "JsonRpcResponse":
        """Create an error response."""
        return cls(id=id, error=JsonRpcError(code=code, message=message, data=data))


@dataclass
class A2AMessagePart:
    """A2A message part (text, file, data, etc.)."""

    kind: str  # "text", "file", "data", etc.
    text: Optional[str] = None
    file: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: Dict[str, Any] = {"kind": self.kind}
        if self.text is not None:
            result["text"] = self.text
        if self.file is not None:
            result["file"] = self.file
        if self.data is not None:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessagePart":
        """Create from dictionary."""
        return cls(
            kind=data.get("kind", "text"),
            text=data.get("text"),
            file=data.get("file"),
            data=data.get("data"),
        )


@dataclass
class A2AMessage:
    """A2A protocol message."""

    role: str  # "user", "agent"
    parts: List[A2AMessagePart]
    message_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: Dict[str, Any] = {
            "role": self.role,
            "parts": [part.to_dict() for part in self.parts],
        }
        if self.message_id:
            result["messageId"] = self.message_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        """Create from dictionary."""
        parts = [A2AMessagePart.from_dict(p) for p in data.get("parts", [])]
        return cls(
            role=data.get("role", "user"),
            parts=parts,
            message_id=data.get("messageId"),
        )

    def get_text(self) -> str:
        """Extract text content from message parts."""
        texts = []
        for part in self.parts:
            if part.kind == "text" and part.text:
                texts.append(part.text)
        return "\n".join(texts)


@dataclass
class A2AArtifact:
    """A2A protocol artifact (response content)."""

    artifact_id: str
    name: str
    parts: List[A2AMessagePart]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "artifactId": self.artifact_id,
            "name": self.name,
            "parts": [part.to_dict() for part in self.parts],
        }

    @classmethod
    def from_text(cls, artifact_id: str, name: str, text: str) -> "A2AArtifact":
        """Create a text artifact."""
        return cls(
            artifact_id=artifact_id,
            name=name,
            parts=[A2AMessagePart(kind="text", text=text)],
        )


def build_runtime_url(agent_arn: str, region: str = "us-west-2") -> str:
    """Build the AgentCore Runtime URL from an agent ARN.

    Args:
        agent_arn: The ARN of the agent runtime
        region: AWS region (default: us-west-2)

    Returns:
        The full runtime URL with properly encoded ARN
    """
    # URL encode the ARN (safe='' means encode all special characters)
    escaped_arn = quote(agent_arn, safe="")
    return f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{escaped_arn}/invocations/"


# A2A Protocol Methods
A2A_METHOD_MESSAGE_SEND = "message/send"
A2A_METHOD_MESSAGE_STREAM = "message/stream"
A2A_METHOD_TASKS_GET = "tasks/get"
A2A_METHOD_TASKS_CANCEL = "tasks/cancel"

# Default A2A port for AgentCore Runtime
A2A_DEFAULT_PORT = 9000
