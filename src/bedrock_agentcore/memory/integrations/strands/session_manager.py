"""AgentCore Memory-based session manager for Bedrock AgentCore Memory integration."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

import boto3
from botocore.config import Config as BotocoreConfig
from strands.hooks import MessageAddedEvent
from strands.hooks.registry import HookRegistry
from strands.session.repository_session_manager import RepositorySessionManager
from strands.session.session_repository import SessionRepository
from strands.types.content import Message
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage
from typing_extensions import override

from bedrock_agentcore.memory.client import MemoryClient

from .bedrock_converter import AgentCoreMemoryConverter
from .config import AgentCoreMemoryConfig

if TYPE_CHECKING:
    from strands.agent.agent import Agent

logger = logging.getLogger(__name__)

SESSION_PREFIX = "session_"
AGENT_PREFIX = "agent_"
MESSAGE_PREFIX = "message_"


class AgentCoreMemorySessionManager(RepositorySessionManager, SessionRepository):
    """AgentCore Memory-based session manager for Bedrock AgentCore Memory integration.

    This session manager integrates Strands agents with Amazon Bedrock AgentCore Memory,
    providing seamless synchronization between Strands' session management and Bedrock's
    short-term and long-term memory capabilities.

    Key Features:
    - Automatic synchronization of conversation messages to Bedrock AgentCore Memory events
    - Loading of conversation history from short-term memory during agent initialization
    - Integration with long-term memory for context injection into agent state
    - Support for custom retrieval configurations per namespace
    - Consistent with existing Strands Session managers (such as: FileSessionManager, S3SessionManager)
    """

    # Class-level timestamp tracking for monotonic ordering
    _timestamp_lock = threading.Lock()
    _last_timestamp: Optional[datetime] = None

    @classmethod
    def _get_monotonic_timestamp(cls, desired_timestamp: Optional[datetime] = None) -> datetime:
        """Get a monotonically increasing timestamp.

        Args:
            desired_timestamp (Optional[datetime]): The desired timestamp. If None, uses current time.

        Returns:
            datetime: A timestamp guaranteed to be greater than any previously returned timestamp.
        """
        if desired_timestamp is None:
            desired_timestamp = datetime.now(timezone.utc)

        with cls._timestamp_lock:
            if cls._last_timestamp is None:
                cls._last_timestamp = desired_timestamp
                return desired_timestamp

            # Why the 1 second check? Because Boto3 does NOT support sub 1 second resolution.
            if desired_timestamp <= cls._last_timestamp + timedelta(seconds=1):
                # Increment by 1 second to ensure ordering
                new_timestamp = cls._last_timestamp + timedelta(seconds=1)
            else:
                new_timestamp = desired_timestamp

            cls._last_timestamp = new_timestamp
            return new_timestamp

    def __init__(
        self,
        agentcore_memory_config: AgentCoreMemoryConfig,
        region_name: Optional[str] = None,
        boto_session: Optional[boto3.Session] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
        **kwargs: Any,
    ):
        """Initialize AgentCoreMemorySessionManager with Bedrock AgentCore Memory.

        Args:
            agentcore_memory_config (AgentCoreMemoryConfig): Configuration for AgentCore Memory integration.
            region_name (Optional[str], optional): AWS region for Bedrock AgentCore Memory. Defaults to None.
            boto_session (Optional[boto3.Session], optional): Optional boto3 session. Defaults to None.
            boto_client_config (Optional[BotocoreConfig], optional): Optional boto3 client configuration.
               Defaults to None.
            **kwargs (Any): Additional keyword arguments.
        """
        self.config = agentcore_memory_config
        self.memory_client = MemoryClient(region_name=region_name)
        session = boto_session or boto3.Session(region_name=region_name)
        self.has_existing_agent = False

        # Override the clients if custom boto session or config is provided
        # Add strands-agents to the request user agent
        if boto_client_config:
            existing_user_agent = getattr(boto_client_config, "user_agent_extra", None)
            if existing_user_agent:
                new_user_agent = f"{existing_user_agent} strands-agents"
            else:
                new_user_agent = "strands-agents"
            client_config = boto_client_config.merge(BotocoreConfig(user_agent_extra=new_user_agent))
        else:
            client_config = BotocoreConfig(user_agent_extra="strands-agents")

        # Override the memory client's boto3 clients
        self.memory_client.gmcp_client = session.client(
            "bedrock-agentcore-control", region_name=region_name or session.region_name, config=client_config
        )
        self.memory_client.gmdp_client = session.client(
            "bedrock-agentcore", region_name=region_name or session.region_name, config=client_config
        )
        super().__init__(session_id=self.config.session_id, session_repository=self)

    def _get_full_session_id(self, session_id: str) -> str:
        """Get the full session ID with the configured prefix.

        Args:
            session_id (str): The session ID.

        Returns:
            str: The full session ID with the prefix.
        """
        full_session_id = f"{SESSION_PREFIX}{session_id}"
        if full_session_id == self.config.actor_id:
            raise SessionException(
                f"Cannot have session [ {full_session_id} ] with the same ID as the actor ID: {self.config.actor_id}"
            )
        return full_session_id

    def _get_full_agent_id(self, agent_id: str) -> str:
        """Get the full agent ID with the configured prefix.

        Args:
            agent_id (str): The agent ID.

        Returns:
            str: The full agent ID with the prefix.
        """
        full_agent_id = f"{AGENT_PREFIX}{agent_id}"
        if full_agent_id == self.config.actor_id:
            raise SessionException(
                f"Cannot create agent [ {full_agent_id} ] with the same ID as the actor ID: {self.config.actor_id}"
            )
        return full_agent_id

    # region SessionRepository interface implementation
    def create_session(self, session: Session, **kwargs: Any) -> Session:
        """Create a new session in AgentCore Memory.

        Note: AgentCore Memory doesn't have explicit session creation,
        so we just validate the session and return it.

        Args:
            session (Session): The session to create.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Session: The created session.

        Raises:
            SessionException: If session ID doesn't match configuration.
        """
        if session.session_id != self.config.session_id:
            raise SessionException(f"Session ID mismatch: expected {self.config.session_id}, got {session.session_id}")

        event = self.memory_client.gmdp_client.create_event(
            memoryId=self.config.memory_id,
            actorId=self._get_full_session_id(session.session_id),
            sessionId=self.session_id,
            payload=[
                {"blob": json.dumps(session.to_dict())},
            ],
            eventTimestamp=self._get_monotonic_timestamp(),
        )
        logger.info("Created session: %s with event: %s", session.session_id, event.get("event", {}).get("eventId"))
        return session

    def read_session(self, session_id: str, **kwargs: Any) -> Optional[Session]:
        """Read session data.

        AgentCore Memory does not have a `get_session` method.
        Which is fine as AgentCore Memory is a managed service we therefore do not need to read/update
        the session data. We just return the session object.

        Args:
            session_id (str): The session ID to read.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Optional[Session]: The session if found, None otherwise.
        """
        if session_id != self.config.session_id:
            return None

        events = self.memory_client.list_events(
            memory_id=self.config.memory_id,
            actor_id=self._get_full_session_id(session_id),
            session_id=session_id,
            max_results=1,
        )
        if not events:
            return None

        session_data = json.loads(events[0].get("payload", {})[0].get("blob"))
        return Session.from_dict(session_data)

    def delete_session(self, session_id: str, **kwargs: Any) -> None:
        """Delete session and all associated data.

        Note: AgentCore Memory doesn't support deletion of events,
        so this is a no-op operation.

        Args:
            session_id (str): The session ID to delete.
            **kwargs (Any): Additional keyword arguments.
        """
        logger.warning("Session deletion not supported in AgentCore Memory: %s", session_id)

    def create_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Create a new agent in the session.

        For AgentCore Memory, we don't need to explicitly create agents; we have Implicit Agent Existence
        The agent's existence is inferred from the presence of events/messages in the memory system,
        but we validate the session_id matches our config.

        Args:
            session_id (str): The session ID to create the agent in.
            session_agent (SessionAgent): The agent to create.
            **kwargs (Any): Additional keyword arguments.

        Raises:
            SessionException: If session ID doesn't match configuration.
        """
        if session_id != self.config.session_id:
            raise SessionException(f"Session ID mismatch: expected {self.config.session_id}, got {session_id}")

        event = self.memory_client.gmdp_client.create_event(
            memoryId=self.config.memory_id,
            actorId=self._get_full_agent_id(session_agent.agent_id),
            sessionId=self.session_id,
            payload=[
                {"blob": json.dumps(session_agent.to_dict())},
            ],
            eventTimestamp=self._get_monotonic_timestamp(),
        )
        logger.info(
            "Created agent: %s in session: %s with event %s",
            session_agent.agent_id,
            session_id,
            event.get("event", {}).get("eventId"),
        )

    def read_agent(self, session_id: str, agent_id: str, **kwargs: Any) -> Optional[SessionAgent]:
        """Read agent data from AgentCore Memory events.

        We reconstruct the agent state from the conversation history.

        Args:
            session_id (str): The session ID to read from.
            agent_id (str): The agent ID to read.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Optional[SessionAgent]: The agent if found, None otherwise.
        """
        if session_id != self.config.session_id:
            return None
        try:
            events = self.memory_client.list_events(
                memory_id=self.config.memory_id,
                actor_id=self._get_full_agent_id(agent_id),
                session_id=session_id,
                max_results=1,
            )

            if not events:
                return None

            agent_data = json.loads(events[0].get("payload", {})[0].get("blob"))
            return SessionAgent.from_dict(agent_data)
        except Exception as e:
            logger.error("Failed to read agent %s", e)
            return None

    def update_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Update agent data.

        Args:
            session_id (str): The session ID containing the agent.
            session_agent (SessionAgent): The agent to update.
            **kwargs (Any): Additional keyword arguments.

        Raises:
            SessionException: If session ID doesn't match configuration.
        """
        agent_id = session_agent.agent_id
        previous_agent = self.read_agent(session_id=session_id, agent_id=agent_id)
        if previous_agent is None:
            raise SessionException(f"Agent {agent_id} in session {session_id} does not exist")

        session_agent.created_at = previous_agent.created_at
        # Create a new agent as AgentCore Memory is immutable. We always get the latest one in `read_agent`
        self.create_agent(session_id, session_agent)

    def create_message(
        self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any
    ) -> Optional[dict[str, Any]]:
        """Create a new message in AgentCore Memory.

        Args:
            session_id (str): The session ID to create the message in.
            agent_id (str): The agent ID associated with the message (only here for the interface.
               We use the actorId for AgentCore).
            session_message (SessionMessage): The message to create.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Optional[dict[str, Any]]: The created event data from AgentCore Memory.

        Raises:
            SessionException: If session ID doesn't match configuration or message creation fails.

        Note:
            The returned created message `event` looks like:
            ```python
                {
                    "memoryId": "my-mem-id",
                    "actorId": "user_1",
                    "sessionId": "test_session_id",
                    "eventId": "0000001752235548000#97f30a6b",
                    "eventTimestamp": datetime.datetime(2025, 8, 18, 12, 45, 48, tzinfo=tzlocal()),
                    "branch": {"name": "main"},
                }
            ```
        """
        if session_id != self.config.session_id:
            raise SessionException(f"Session ID mismatch: expected {self.config.session_id}, got {session_id}")

        try:
            messages = AgentCoreMemoryConverter.message_to_payload(session_message)
            if not messages:
                return

            # Parse the original timestamp and use it as desired timestamp
            original_timestamp = datetime.fromisoformat(session_message.created_at.replace("Z", "+00:00"))
            monotonic_timestamp = self._get_monotonic_timestamp(original_timestamp)

            if not AgentCoreMemoryConverter.exceeds_conversational_limit(messages[0]):
                event = self.memory_client.create_event(
                    memory_id=self.config.memory_id,
                    actor_id=self.config.actor_id,
                    session_id=session_id,
                    messages=messages,
                    event_timestamp=monotonic_timestamp,
                )
            else:
                event = self.memory_client.gmdp_client.create_event(
                    memoryId=self.config.memory_id,
                    actorId=self.config.actor_id,
                    sessionId=session_id,
                    payload=[
                        {"blob": json.dumps(messages[0])},
                    ],
                    eventTimestamp=monotonic_timestamp,
                )
            logger.debug("Created event: %s for message: %s", event.get("eventId"), session_message.message_id)
            return event
        except Exception as e:
            logger.error("Failed to create message in AgentCore Memory: %s", e)
            raise SessionException(f"Failed to create message: {e}") from e

    def read_message(self, session_id: str, agent_id: str, message_id: int, **kwargs: Any) -> Optional[SessionMessage]:
        """Read a specific message by ID from AgentCore Memory.

        Args:
            session_id (str): The session ID to read from.
            agent_id (str): The agent ID associated with the message.
            message_id (int): The message ID to read.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Optional[SessionMessage]: The message if found, None otherwise.

        Note:
            This should not be called as (as of now) only the `update_message` method calls this method and
            updating messages is not supported in AgentCore Memory.
        """
        result = self.memory_client.gmdp_client.get_event(
            memoryId=self.config.memory_id, actorId=self.config.actor_id, sessionId=session_id, eventId=message_id
        )
        return SessionMessage.from_dict(result) if result else None

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Update message data.

        Note: AgentCore Memory doesn't support updating events,
        so this is primarily for validation and logging.

        Args:
            session_id (str): The session ID containing the message.
            agent_id (str): The agent ID associated with the message.
            session_message (SessionMessage): The message to update.
            **kwargs (Any): Additional keyword arguments.

        Raises:
            SessionException: If session ID doesn't match configuration.
        """
        if session_id != self.config.session_id:
            raise SessionException(f"Session ID mismatch: expected {self.config.session_id}, got {session_id}")

        logger.debug(
            "Message update requested for message: %s (AgentCore Memory doesn't support updates)",
            {session_message.message_id},
        )

    def list_messages(
        self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0, **kwargs: Any
    ) -> list[SessionMessage]:
        """List messages for an agent from AgentCore Memory with pagination.

        Args:
            session_id (str): The session ID to list messages from.
            agent_id (str): The agent ID to list messages for.
            limit (Optional[int], optional): Maximum number of messages to return. Defaults to None.
            offset (int, optional): Number of messages to skip. Defaults to 0.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            list[SessionMessage]: list of messages for the agent.

        Raises:
            SessionException: If session ID doesn't match configuration.
        """
        if session_id != self.config.session_id:
            raise SessionException(f"Session ID mismatch: expected {self.config.session_id}, got {session_id}")

        try:
            max_results = (limit + offset) if limit else 100
            events = self.memory_client.list_events(
                memory_id=self.config.memory_id,
                actor_id=self.config.actor_id,
                session_id=session_id,
                max_results=max_results,
            )
            messages = AgentCoreMemoryConverter.events_to_messages(events)
            if limit is not None:
                return messages[offset : offset + limit]
            else:
                return messages[offset:]

        except Exception as e:
            logger.error("Failed to list messages from AgentCore Memory: %s", e)
            return []

    # endregion SessionRepository interface implementation

    # region RepositorySessionManager overrides
    @override
    def append_message(self, message: Message, agent: "Agent", **kwargs: Any) -> None:
        """Append a message to the agent's session using AgentCore's eventId as message_id.

        Args:
            message: Message to add to the agent in the session
            agent: Agent to append the message to
            **kwargs: Additional keyword arguments for future extensibility.
        """
        created_message = self.create_message(self.session_id, agent.agent_id, SessionMessage.from_message(message, 0))
        session_message = SessionMessage.from_message(message, created_message.get("eventId"))
        self._latest_agent_message[agent.agent_id] = session_message

    def retrieve_customer_context(self, event: MessageAddedEvent) -> None:
        """Retrieve customer LTM context before processing support query.

        Args:
            event (MessageAddedEvent): The message added event containing the agent and message data.
        """
        messages = event.agent.messages
        if not messages or messages[-1].get("role") != "user" or "toolResult" in messages[-1].get("content")[0]:
            return None
        if not self.config.retrieval_config:
            # Only retrieve LTM
            return None

        user_query = messages[-1]["content"][0]["text"]

        def retrieve_for_namespace(namespace: str, retrieval_config: AgentCoreMemoryConfig):
            """Helper function to retrieve memories for a single namespace."""
            resolved_namespace = namespace.format(
                actorId=self.config.actor_id,
                sessionId=self.config.session_id,
                memoryStrategyId=retrieval_config.strategy_id or "",
            )

            memories = self.memory_client.retrieve_memories(
                memory_id=self.config.memory_id,
                namespace=resolved_namespace,
                query=user_query,
                top_k=retrieval_config.top_k,
            )
            context_items = []
            for memory in memories:
                if isinstance(memory, dict):
                    content = memory.get("content", {})
                    if isinstance(content, dict):
                        text = content.get("text", "").strip()
                        if text:
                            context_items.append(text)
            return context_items

        try:
            # Retrieve customer context from all namespaces in parallel
            all_context = []

            with ThreadPoolExecutor() as executor:
                future_to_namespace = {
                    executor.submit(retrieve_for_namespace, namespace, retrieval_config): namespace
                    for namespace, retrieval_config in self.config.retrieval_config.items()
                }
                for future in as_completed(future_to_namespace):
                    try:
                        context_items = future.result()
                        all_context.extend(context_items)
                    except Exception as e:
                        # Continue processing other futures event if one fails rather than failing the entire operation
                        namespace = future_to_namespace[future]
                        logger.error("Failed to retrieve memories for namespace %s: %s", namespace, e)

            # Inject customer context into the query
            if all_context:
                context_text = "\n".join(all_context)
                ltm_msg: Message = {
                    "role": "assistant",
                    "content": [{"text": f"<user_context>{context_text}</user_context>"}],
                }
                event.agent.messages.append(ltm_msg)
                logger.info("Retrieved %s customer context items", len(all_context))

        except Exception as e:
            logger.error("Failed to retrieve customer context: %s", e)

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        """Register additional hooks.

        Args:
            registry (HookRegistry): The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments.
        """
        RepositorySessionManager.register_hooks(self, registry, **kwargs)
        registry.add_callback(MessageAddedEvent, lambda event: self.retrieve_customer_context(event))

    @override
    def initialize(self, agent: "Agent", **kwargs: Any) -> None:
        if self.has_existing_agent:
            logger.warning(
                "An Agent already exists in session %s. We currently support one agent per session.", self.session_id
            )
        else:
            self.has_existing_agent = True
        RepositorySessionManager.initialize(self, agent, **kwargs)

    # endregion RepositorySessionManager overrides
