"""AgentCore Memory-based session manager for Bedrock AgentCore Memory integration."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from enum import Enum
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
from bedrock_agentcore.memory.models.filters import EventMetadataFilter, LeftExpression, OperatorType, RightExpression

from .bedrock_converter import AgentCoreMemoryConverter
from .config import AgentCoreMemoryConfig, RetrievalConfig

if TYPE_CHECKING:
    from strands.agent.agent import Agent

logger = logging.getLogger(__name__)

MAX_FETCH_ALL_RESULTS = 10000

# Legacy prefixes for backwards compatibility with old events
LEGACY_SESSION_PREFIX = "session_"
LEGACY_AGENT_PREFIX = "agent_"

# Metadata keys for event identification
STATE_TYPE_KEY = "stateType"
AGENT_ID_KEY = "agentId"


class StateType(Enum):
    """State type for distinguishing session and agent metadata in events."""

    SESSION = "SESSION"
    AGENT = "AGENT"


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

        # Batching support - stores pre-processed messages: (session_id, messages, is_blob, timestamp)
        self._message_buffer: list[tuple[str, list[tuple[str, str]], bool, datetime]] = []
        self._buffer_lock = threading.Lock()

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
            actorId=self.config.actor_id,
            sessionId=self.session_id,
            payload=[
                {"blob": json.dumps(session.to_dict())},
            ],
            eventTimestamp=self._get_monotonic_timestamp(),
            metadata={STATE_TYPE_KEY: {"stringValue": StateType.SESSION.value}},
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

        # 1. Try new approach (metadata filter)
        event_metadata = [
            EventMetadataFilter.build_expression(
                left_operand=LeftExpression.build(STATE_TYPE_KEY),
                operator=OperatorType.EQUALS_TO,
                right_operand=RightExpression.build(StateType.SESSION.value),
            )
        ]

        events = self.memory_client.list_events(
            memory_id=self.config.memory_id,
            actor_id=self.config.actor_id,
            session_id=session_id,
            event_metadata=event_metadata,
            max_results=1,
        )
        if events:
            session_data = json.loads(events[0].get("payload", {})[0].get("blob"))
            return Session.from_dict(session_data)

        # 2. Fallback: check for legacy event and migrate
        legacy_actor_id = f"{LEGACY_SESSION_PREFIX}{session_id}"
        events = self.memory_client.list_events(
            memory_id=self.config.memory_id,
            actor_id=legacy_actor_id,
            session_id=session_id,
            max_results=1,
        )
        if events:
            old_event = events[0]
            session_data = json.loads(old_event.get("payload", {})[0].get("blob"))
            session = Session.from_dict(session_data)
            # Migrate: create new event with metadata, delete old
            self.create_session(session)
            self.memory_client.gmdp_client.delete_event(
                memoryId=self.config.memory_id,
                actorId=legacy_actor_id,
                sessionId=session_id,
                eventId=old_event.get("eventId"),
            )
            logger.info("Migrated legacy session event for session: %s", session_id)
            return session

        return None

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
            actorId=self.config.actor_id,
            sessionId=self.session_id,
            payload=[
                {"blob": json.dumps(session_agent.to_dict())},
            ],
            eventTimestamp=self._get_monotonic_timestamp(),
            metadata={
                STATE_TYPE_KEY: {"stringValue": StateType.AGENT.value},
                AGENT_ID_KEY: {"stringValue": session_agent.agent_id},
            },
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
            # 1. Try new approach (metadata filter)
            event_metadata = [
                EventMetadataFilter.build_expression(
                    left_operand=LeftExpression.build(STATE_TYPE_KEY),
                    operator=OperatorType.EQUALS_TO,
                    right_operand=RightExpression.build(StateType.AGENT.value),
                ),
                EventMetadataFilter.build_expression(
                    left_operand=LeftExpression.build(AGENT_ID_KEY),
                    operator=OperatorType.EQUALS_TO,
                    right_operand=RightExpression.build(agent_id),
                ),
            ]

            events = self.memory_client.list_events(
                memory_id=self.config.memory_id,
                actor_id=self.config.actor_id,
                session_id=session_id,
                event_metadata=event_metadata,
                max_results=1,
            )

            if events:
                agent_data = json.loads(events[0].get("payload", {})[0].get("blob"))
                return SessionAgent.from_dict(agent_data)

            # 2. Fallback: check for legacy event and migrate
            legacy_actor_id = f"{LEGACY_AGENT_PREFIX}{agent_id}"
            events = self.memory_client.list_events(
                memory_id=self.config.memory_id,
                actor_id=legacy_actor_id,
                session_id=session_id,
                max_results=1,
            )
            if events:
                old_event = events[0]
                agent_data = json.loads(old_event.get("payload", {})[0].get("blob"))
                agent = SessionAgent.from_dict(agent_data)
                # Migrate: create new event with metadata, delete old
                self.create_agent(session_id, agent)
                self.memory_client.gmdp_client.delete_event(
                    memoryId=self.config.memory_id,
                    actorId=legacy_actor_id,
                    sessionId=session_id,
                    eventId=old_event.get("eventId"),
                )
                logger.info("Migrated legacy agent event for agent: %s", agent_id)
                return agent

            return None
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
        else:
            session_agent.created_at = previous_agent.created_at

        # Create a new agent as AgentCore Memory is immutable. We always get the latest one in `read_agent`
        self.create_agent(session_id, session_agent)

    def create_message(
        self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any
    ) -> Optional[dict[str, Any]]:
        """Create a new message in AgentCore Memory.

        If batch_size > 1, the message is buffered and sent when the buffer reaches batch_size.
        Use _flush_messages() or close() to send any remaining buffered messages.

        Args:
            session_id (str): The session ID to create the message in.
            agent_id (str): The agent ID associated with the message (only here for the interface.
               We use the actorId for AgentCore).
            session_message (SessionMessage): The message to create.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Optional[dict[str, Any]]: The created event data from AgentCore Memory.
                Returns empty dict if message is buffered (batch_size > 1).

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

        # Convert and check size ONCE (not again at flush)
        messages = AgentCoreMemoryConverter.message_to_payload(session_message)
        if not messages:
            return None

        is_blob = AgentCoreMemoryConverter.exceeds_conversational_limit(messages[0])

        # Parse the original timestamp and use it as desired timestamp
        original_timestamp = datetime.fromisoformat(session_message.created_at.replace("Z", "+00:00"))
        monotonic_timestamp = self._get_monotonic_timestamp(original_timestamp)

        if self.config.batch_size > 1:
            # Buffer the pre-processed message
            should_flush = False
            with self._buffer_lock:
                self._message_buffer.append((session_id, messages, is_blob, monotonic_timestamp))
                should_flush = len(self._message_buffer) >= self.config.batch_size

            # Flush outside the lock to prevent deadlock
            if should_flush:
                self._flush_messages()

            return {}  # No eventId yet

        # Immediate send (batch_size == 1)
        try:
            if not is_blob:
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
        self,
        session_id: str,
        agent_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        **kwargs: Any,
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
            max_results = (limit + offset) if limit else MAX_FETCH_ALL_RESULTS

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

        def retrieve_for_namespace(namespace: str, retrieval_config: RetrievalConfig):
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
            if retrieval_config.relevance_score:
                memories = [
                    m
                    for m in memories
                    if m.get("relevanceScore", retrieval_config.relevance_score) >= retrieval_config.relevance_score
                ]
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

    # region Batching support

    def _flush_messages(self) -> list[dict[str, Any]]:
        """Flush all buffered messages to AgentCore Memory.

        Call this method to send any remaining buffered messages when batch_size > 1.
        This is automatically called when the buffer reaches batch_size, but should
        also be called explicitly when the session is complete (via close() or context manager).

        Messages are batched by session_id - all conversational messages for the same
        session are combined into a single create_event() call to reduce API calls.
        Blob messages (>9KB) are sent individually as they require a different API path.

        Returns:
            list[dict[str, Any]]: List of created event responses from AgentCore Memory.

        Raises:
            SessionException: If any message creation fails. On failure, all messages
                remain in the buffer to prevent data loss.
        """
        with self._buffer_lock:
            messages_to_send = list(self._message_buffer)

        if not messages_to_send:
            return []

        # Group conversational messages by session_id, preserve order
        # Structure: {session_id: {"messages": [...], "timestamp": latest_timestamp}}
        session_groups: dict[str, dict[str, Any]] = {}
        blob_messages: list[tuple[str, list[tuple[str, str]], datetime]] = []

        for session_id, messages, is_blob, monotonic_timestamp in messages_to_send:
            if is_blob:
                # Blobs cannot be combined - collect them separately
                blob_messages.append((session_id, messages, monotonic_timestamp))
            else:
                # Group conversational messages by session_id
                if session_id not in session_groups:
                    session_groups[session_id] = {"messages": [], "timestamp": monotonic_timestamp}
                # Extend messages list to preserve order (earlier messages first)
                session_groups[session_id]["messages"].extend(messages)
                # Use the latest timestamp for the combined event
                if monotonic_timestamp > session_groups[session_id]["timestamp"]:
                    session_groups[session_id]["timestamp"] = monotonic_timestamp

        results = []
        try:
            # Send one create_event per session_id with combined messages
            for session_id, group in session_groups.items():
                event = self.memory_client.create_event(
                    memory_id=self.config.memory_id,
                    actor_id=self.config.actor_id,
                    session_id=session_id,
                    messages=group["messages"],
                    event_timestamp=group["timestamp"],
                )
                results.append(event)
                logger.debug("Flushed batched event for session %s: %s", session_id, event.get("eventId"))

            # Send blob messages individually (they use a different API path)
            for session_id, messages, monotonic_timestamp in blob_messages:
                event = self.memory_client.gmdp_client.create_event(
                    memoryId=self.config.memory_id,
                    actorId=self.config.actor_id,
                    sessionId=session_id,
                    payload=[
                        {"blob": json.dumps(messages[0])},
                    ],
                    eventTimestamp=monotonic_timestamp,
                )
                results.append(event)
                logger.debug("Flushed blob event for session %s: %s", session_id, event.get("eventId"))

            # Clear buffer only after ALL messages succeed
            with self._buffer_lock:
                self._message_buffer.clear()

        except Exception as e:
            logger.error("Failed to flush messages to AgentCore Memory for session: %s", e)
            raise SessionException(f"Failed to flush messages: {e}") from e

        logger.info("Flushed %d events to AgentCore Memory", len(results))
        return results

    def pending_message_count(self) -> int:
        """Return the number of messages pending in the buffer.

        Returns:
            int: Number of buffered messages waiting to be sent.
        """
        with self._buffer_lock:
            return len(self._message_buffer)

    def close(self) -> None:
        """Explicitly flush pending messages and close the session manager.

        Call this method when the session is complete to ensure all buffered
        messages are sent to AgentCore Memory. Alternatively, use the context
        manager protocol (with statement) for automatic cleanup.
        """
        self._flush_messages()

    def __enter__(self) -> "AgentCoreMemorySessionManager":
        """Enter the context manager.

        Returns:
            AgentCoreMemorySessionManager: This session manager instance.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager and flush any pending messages.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        try:
            self._flush_messages()
        except Exception as e:
            if exc_type is not None:
                logger.error("Failed to flush messages during exception handling: %s", e)
            else:
                raise

    # endregion Batching support
