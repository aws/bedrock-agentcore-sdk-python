"""Session manager tests with OpenAI/auto converter."""

import json
from unittest.mock import Mock, patch

from strands.types.session import Session, SessionMessage, SessionType

from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.converters import AutoConverseConverter, OpenAIConverseConverter
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager


def test_create_message_uses_tool_role_with_openai_converter():
    """When configured with OpenAI converter, create_event should receive TOOL role for tool outputs."""
    config = AgentCoreMemoryConfig(
        memory_id="test-memory-123",
        session_id="test-session-456",
        actor_id="test-actor-789",
    )

    mock_memory_client = Mock()
    mock_memory_client.create_event.return_value = {"eventId": "event_123"}
    mock_memory_client.list_events.return_value = []
    mock_memory_client.gmcp_client = Mock()
    mock_memory_client.gmdp_client = Mock()

    with (
        patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ),
        patch("boto3.Session") as mock_boto_session,
        patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
    ):
        mock_session = Mock()
        mock_session.region_name = "us-west-2"
        mock_session.client.return_value = Mock()
        mock_boto_session.return_value = mock_session

        manager = AgentCoreMemorySessionManager(config, converter=OpenAIConverseConverter)
        manager.session_id = config.session_id
        manager.session = Session(session_id=config.session_id, session_type=SessionType.AGENT)

        message = SessionMessage(
            message={
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_777",
                            "content": [{"text": "ok"}],
                            "status": "success",
                        }
                    }
                ],
            },
            message_id=1,
            created_at="2024-01-01T12:00:00Z",
        )

        manager.create_message(config.session_id, "agent-1", message)

        kwargs = mock_memory_client.create_event.call_args.kwargs
        assert kwargs["messages"][0][1] == "tool"


def test_list_messages_filters_restored_tool_context():
    """Restored history should exclude toolUse/toolResult blocks."""
    config = AgentCoreMemoryConfig(
        memory_id="test-memory-123",
        session_id="test-session-456",
        actor_id="test-actor-789",
        filter_restored_tool_context=True,
    )

    mock_memory_client = Mock()
    mock_memory_client.list_events.return_value = [{"payload": []}]
    mock_memory_client.gmcp_client = Mock()
    mock_memory_client.gmdp_client = Mock()

    with (
        patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ),
        patch("boto3.Session") as mock_boto_session,
        patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
    ):
        mock_session = Mock()
        mock_session.region_name = "us-west-2"
        mock_session.client.return_value = Mock()
        mock_boto_session.return_value = mock_session

        manager = AgentCoreMemorySessionManager(config, converter=OpenAIConverseConverter)
        manager.session_id = config.session_id
        manager.session = Session(session_id=config.session_id, session_type=SessionType.AGENT)

        manager.converter = Mock()
        manager.converter.events_to_messages.return_value = [
            SessionMessage(message={"role": "user", "content": [{"text": "hello"}]}, message_id=0),
            SessionMessage(
                message={
                    "role": "assistant",
                    "content": [
                        {"text": "calling tool"},
                        {"toolUse": {"toolUseId": "t1", "name": "foo", "input": {}}},
                    ],
                },
                message_id=1,
            ),
            SessionMessage(
                message={
                    "role": "user",
                    "content": [{"toolResult": {"toolUseId": "t1", "status": "success", "content": [{"text": "ok"}]}}],
                },
                message_id=2,
            ),
            SessionMessage(message={"role": "assistant", "content": [{"text": "done"}]}, message_id=3),
        ]

        messages = manager.list_messages(config.session_id, "agent-1")

        assert [m.message for m in messages] == [
            {"role": "user", "content": [{"text": "hello"}]},
            {"role": "assistant", "content": [{"text": "calling tool"}]},
            {"role": "assistant", "content": [{"text": "done"}]},
        ]


def test_auto_converter_mode_selects_openai_for_openai_model():
    config = AgentCoreMemoryConfig(
        memory_id="test-memory-123",
        session_id="test-session-456",
        actor_id="test-actor-789",
    )

    mock_memory_client = Mock()
    mock_memory_client.list_events.return_value = []
    mock_memory_client.gmcp_client = Mock()
    mock_memory_client.gmdp_client = Mock()

    with (
        patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ),
        patch("boto3.Session") as mock_boto_session,
        patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
        patch("strands.session.repository_session_manager.RepositorySessionManager.initialize", return_value=None),
    ):
        mock_session = Mock()
        mock_session.region_name = "us-west-2"
        mock_session.client.return_value = Mock()
        mock_boto_session.return_value = mock_session

        manager = AgentCoreMemorySessionManager(config, converter="auto")
        assert manager.converter is AutoConverseConverter

        mock_agent = Mock()
        mock_agent.model = Mock()
        mock_agent.model.__class__.__name__ = "OpenAIModel"
        mock_agent.model.__class__.__module__ = "strands.models.openai"

        manager.initialize(mock_agent)
        assert AutoConverseConverter._write_converter is OpenAIConverseConverter


def test_list_messages_with_auto_converter_restores_mixed_provider_history():
    config = AgentCoreMemoryConfig(
        memory_id="test-memory-123",
        session_id="test-session-456",
        actor_id="test-actor-789",
    )

    openai_payload = json.dumps({"role": "assistant", "content": "hello from openai"})
    anthropic_payload = json.dumps(
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "trace", "signature": "sig"}]}
    )
    gemini_payload = json.dumps({"role": "model", "parts": [{"text": "hello from gemini"}]})

    mock_memory_client = Mock()
    mock_memory_client.list_events.return_value = [
        {"payload": [{"conversational": {"content": {"text": openai_payload}}}]},
        {"payload": [{"conversational": {"content": {"text": anthropic_payload}}}]},
        {"payload": [{"conversational": {"content": {"text": gemini_payload}}}]},
    ]
    mock_memory_client.gmcp_client = Mock()
    mock_memory_client.gmdp_client = Mock()

    with (
        patch(
            "bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient",
            return_value=mock_memory_client,
        ),
        patch("boto3.Session") as mock_boto_session,
        patch("strands.session.repository_session_manager.RepositorySessionManager.__init__", return_value=None),
    ):
        mock_session = Mock()
        mock_session.region_name = "us-west-2"
        mock_session.client.return_value = Mock()
        mock_boto_session.return_value = mock_session

        manager = AgentCoreMemorySessionManager(config, converter="auto")
        manager.session_id = config.session_id
        manager.session = Session(session_id=config.session_id, session_type=SessionType.AGENT)

        messages = manager.list_messages(config.session_id, "agent-1")
        assert len(messages) == 3
        assert any(
            block.get("text") == "hello from openai" for msg in messages for block in msg.message.get("content", [])
        )
        assert any(
            "reasoningContent" in block for msg in messages for block in msg.message.get("content", [])
        )
        assert any(
            block.get("text") == "hello from gemini" for msg in messages for block in msg.message.get("content", [])
        )
