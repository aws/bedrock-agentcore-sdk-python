"""Session manager tests with OpenAI converter."""

from unittest.mock import Mock, patch

from strands.types.session import Session, SessionMessage, SessionType

from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.converters import OpenAIConverseConverter
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager


def test_create_message_uses_tool_role_with_openai_converter():
    """When configured with OpenAI converter, create_event should receive TOOL role for tool outputs."""
    config = AgentCoreMemoryConfig(memory_id="test-memory-123", session_id="test-session-456", actor_id="test-actor-789")

    mock_memory_client = Mock()
    mock_memory_client.create_event.return_value = {"eventId": "event_123"}
    mock_memory_client.list_events.return_value = []
    mock_memory_client.gmcp_client = Mock()
    mock_memory_client.gmdp_client = Mock()

    with (
        patch("bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient", return_value=mock_memory_client),
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
    config = AgentCoreMemoryConfig(memory_id="test-memory-123", session_id="test-session-456", actor_id="test-actor-789")

    mock_memory_client = Mock()
    mock_memory_client.list_events.return_value = [{"payload": []}]
    mock_memory_client.gmcp_client = Mock()
    mock_memory_client.gmdp_client = Mock()

    with (
        patch("bedrock_agentcore.memory.integrations.strands.session_manager.MemoryClient", return_value=mock_memory_client),
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
