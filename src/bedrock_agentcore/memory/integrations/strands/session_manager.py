"""AgentCore Memory-based session manager for Bedrock AgentCore Memory integration."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional

import boto3
from botocore.config import Config as BotocoreConfig
from strands.hooks import AfterInvocationEvent, MessageAddedEvent
from strands.hooks.registry import HookRegistry
from strands.session.repository_session_manager import RepositorySessionManager
from strands.session.session_repository import SessionRepository
from strands.types.content import Message
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage
from typing_extensions import override

from bedrock_agentcore.memory.client import MemoryClient
from bedrock_agentcore.memory.models.filters import (
    EventMetadataFilter,
    LeftExpression,
    MetadataValue,
    OperatorType,
    RightExpression,
)

from .bedrock_converter import AgentCoreMemoryConverter
from .config import AgentCoreMemoryConfig, PersistenceMode, RetrievalConfig, normalize_metadata
from .converters import MemoryConverter

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

# Maximum metadata key-value pairs per event (API limit)
MAX_METADATA_KEYS = 15

# Reserved internal metadata keys that users cannot override
RESERVED_METADATA_KEYS = frozenset({STATE_TYPE_KEY, AGENT_ID_KEY})


class BufferedMessage(NamedTuple):
    """A pre-processed message waiting to be flushed to AgentCore Memory."""

    session_id: str
    messages: list[tuple[str, str]]
    is_blob: bool
    timestamp: datetime
    metadata: Optional[Dict[str, MetadataValue]] = None


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
        *,
        converter: Optional[type[MemoryConverter]] = None,
        **kwargs: Any,
    ):
        """Initialize AgentCoreMemorySessionManager with Bedrock AgentCore Memory.

        Args:
            agentcore_memory_config (AgentCoreMemoryConfig): Configuration for AgentCore Memory integration.
            region_name (Optional[str], optional): AWS region for Bedrock AgentCore Memory. Defaults to None.
            boto_session (Optional[boto3.Session], optional): Optional boto3 session. Defaults to None.
            boto_client_config (Optional[BotocoreConfig], optional): Optional boto3 client configuration.
               Defaults to None.
            converter (Optional[type[MemoryConverter]], optional): Optional custom converter.
                If None, native Bedrock/Strands converter is used.
            **kwargs (Any): Additional keyword arguments.
        """
        self.converter = converter or AgentCoreMemoryConverter
        self.config = agentcore_memory_config
        self.read_only = agentcore_memory_config.persistence_mode is PersistenceMode.NONE
        self.memory_client = MemoryClient(region_name=region_name)
        session = boto_session or boto3.Session(region_name=region_name)
        self.has_existing_agent = False

        # Batching support - stores pre-processed messages
        self._message_buffer: list[BufferedMessage] = []
        self._message_lock = threading.Lock()

        # Agent state buffering - stores all agent state updates: (session_id, agent)
        self._agent_state_buffer: list[tuple[str, SessionAgent]] = []
        self._agent_state_lock = threading.Lock()

        # Cache for agent created_at timestamps to avoid fetching on every update
        self._agent_created_at_cache: dict[str, datetime] = {}

        # Interval-based flushing support
        self._flush_timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()
        self._shutdown = False

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

        # Start interval-based flush timer if configured
        if self.config.flush_interval_seconds:
            self._start_flush_timer()

    def _build_metadata(
        self,
        internal_metadata: Optional[Dict[str, MetadataValue]] = None,
        per_call_metadata: Optional[Dict[str, MetadataValue]] = None,
    ) -> Optional[Dict[str, MetadataValue]]:
        """Build merged metadata from config defaults, provider, per-call overrides, and internal keys.

        Merge precedence (highest wins):
            1. internal_metadata (stateType, agentId) — always wins
            2. per_call_metadata (passed via **kwargs)
            3. metadata_provider() (called at event creation time for dynamic values)
            4. self.config.default_metadata (set at config construction time)

        Args:
            internal_metadata: System-reserved metadata (e.g. stateType, agentId).
            per_call_metadata: Caller-supplied metadata for a single operation.

        Returns:
            Merged metadata dict, or None if empty.

        Raises:
            ValueError: If user metadata contains reserved keys or total keys exceed MAX_METADATA_KEYS.
        """
        merged: Dict[str, MetadataValue] = {}

        if self.config.default_metadata:
            merged.update(self.config.default_metadata)

        if self.config.metadata_provider:
            merged.update(normalize_metadata(self.config.metadata_provider()))

        if per_call_metadata:
            merged.update(per_call_metadata)

        # Validate user-supplied keys before merging internal keys
        user_reserved = RESERVED_METADATA_KEYS & merged.keys()
        if user_reserved:
            raise ValueError(
                f"Metadata keys {user_reserved} are reserved for internal use. Reserved keys: {RESERVED_METADATA_KEYS}"
            )

        if internal_metadata:
            merged.update(internal_metadata)

        if len(merged) > MAX_METADATA_KEYS:
            raise ValueError(f"Combined metadata has {len(merged)} keys, exceeding the maximum of {MAX_METADATA_KEYS}.")

        return merged or None

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

        if not self.read_only:
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
            logger.info(
                "Created session: %s with event: %s", session.session_id, event.get("event", {}).get("eventId")
            )

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
            if not self.read_only:
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

        # Cache the created_at timestamp to avoid re-fetching on updates
        if session_agent.created_at:
            self._agent_created_at_cache[session_agent.agent_id] = session_agent.created_at

        if self.read_only:
            return

        if self.config.batch_size > 1:
            # Buffer the agent state events
            should_flush = False
            with self._agent_state_lock:
                self._agent_state_buffer.append((session_id, session_agent))
                should_flush = len(self._agent_state_buffer) >= self.config.batch_size

            # Flush only agent states outside the lock to prevent deadlock
            if should_flush:
                self._flush_agent_states_only()

            logger.info(
                "Buffered agent creation: %s in session: %s",
                session_agent.agent_id,
                session_id,
            )
        else:
            # Immediate send when batching is disabled
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
                agent = SessionAgent.from_dict(agent_data)
                # Cache the created_at timestamp to avoid re-fetching on updates
                if agent.created_at:
                    self._agent_created_at_cache[agent_id] = agent.created_at
                return agent

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
                if not self.read_only:
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

        # Verify agent exists and get created_at timestamp if not cached
        if agent_id not in self._agent_created_at_cache:
            previous_agent = self.read_agent(session_id=session_id, agent_id=agent_id)
            if previous_agent is None:
                raise SessionException(f"Agent {agent_id} in session {session_id} does not exist")

        # Set created_at from cache before creating the update event
        session_agent.created_at = self._agent_created_at_cache[agent_id]

        # Create a new agent event (AgentCore Memory is immutable)
        # create_agent will handle batching and caching appropriately
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
        messages = self.converter.message_to_payload(session_message)
        if not messages:
            return None

        if self.read_only:
            return {}

        is_blob = self.converter.exceeds_conversational_limit(messages[0])

        # Build merged metadata from config defaults + per-call overrides
        merged_metadata = self._build_metadata(per_call_metadata=kwargs.get("metadata"))

        # Parse the original timestamp and use it as desired timestamp
        original_timestamp = datetime.fromisoformat(session_message.created_at.replace("Z", "+00:00"))
        monotonic_timestamp = self._get_monotonic_timestamp(original_timestamp)

        if self.config.batch_size > 1:
            # Buffer the pre-processed message
            should_flush = False
            with self._message_lock:
                self._message_buffer.append(
                    BufferedMessage(
                        session_id=session_id,
                        messages=messages,
                        is_blob=is_blob,
                        timestamp=monotonic_timestamp,
                        metadata=merged_metadata,
                    )
                )
                should_flush = len(self._message_buffer) >= self.config.batch_size

            # Flush only messages outside the lock to prevent deadlock
            if should_flush:
                self._flush_messages_only()

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
                    metadata=merged_metadata,
                )
            else:
                create_event_kwargs: dict[str, Any] = {
                    "memoryId": self.config.memory_id,
                    "actorId": self.config.actor_id,
                    "sessionId": session_id,
                    "payload": [{"blob": json.dumps(messages[0])}],
                    "eventTimestamp": monotonic_timestamp,
                }
                if merged_metadata:
                    create_event_kwargs["metadata"] = merged_metadata
                event = self.memory_client.gmdp_client.create_event(**create_event_kwargs)
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
            messages = self.converter.events_to_messages(events)
            if self.config.filter_restored_tool_context:
                messages = self._filter_restored_tool_context(messages)
            if limit is not None:
                return messages[offset : offset + limit]
            else:
                return messages[offset:]

        except Exception as e:
            logger.error("Failed to list messages from AgentCore Memory: %s", e)
            return []

    def _filter_restored_tool_context(self, messages: list[SessionMessage]) -> list[SessionMessage]:
        """Strip historical toolUse/toolResult context from restored messages."""
        filtered_messages: list[SessionMessage] = []
        for session_message in messages:
            message = session_message.to_message()
            filtered_content = [
                content
                for content in message.get("content", [])
                if "toolUse" not in content and "toolResult" not in content
            ]

            if not filtered_content:
                continue

            filtered_message: Message = {"role": message["role"], "content": filtered_content}
            filtered_messages.append(
                SessionMessage(
                    message=filtered_message,
                    message_id=session_message.message_id,
                    redact_message=session_message.redact_message,
                    created_at=session_message.created_at,
                    updated_at=session_message.updated_at,
                )
            )

        return filtered_messages

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
        if created_message is None:
            return
        session_message = SessionMessage.from_message(message, created_message.get("eventId"))
        self._latest_agent_message[agent.agent_id] = session_message

    def retrieve_customer_context(self, event: MessageAddedEvent) -> None:
        """Retrieve customer LTM context before processing support query.

        Args:
            event (MessageAddedEvent): The message added event containing the agent and message data.
        """
        messages = event.agent.messages
        if not messages or messages[-1].get("role") != "user" or "text" not in messages[-1].get("content")[0]:
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

            # Inject retrieved memory as a content block in the last user message.
            # Prepended so the user's query text remains last (avoids assistant-prefill
            # errors on Claude 4.6+ and keeps the user request in the position models
            # attend to most).
            if all_context:
                context_text = "\n".join(all_context)
                event.agent.messages[-1]["content"].insert(
                    0, {"text": f"<{self.config.context_tag}>{context_text}</{self.config.context_tag}>"}
                )
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

        # Only register AfterInvocationEvent hook when batching is enabled
        if self.config.batch_size > 1:
            registry.add_callback(AfterInvocationEvent, lambda event: self._flush_messages())

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

    def _flush_messages_only(self) -> list[dict[str, Any]]:
        """Flush only buffered messages to AgentCore Memory.

        Call this method to send any remaining buffered messages when batch_size > 1.
        This is called when the message buffer reaches batch_size.
        Messages are batched by session_id - all conversational messages for the same
        session are combined into a single create_event() call to reduce API calls.
        Messages that exceed the conversational payload limit are sent as blob events individually
        as they require a different API path.

        Returns:
            list[dict[str, Any]]: List of created event responses from AgentCore Memory.

        Raises:
            SessionException: If message creation fails. On failure, messages remain in the buffer.
        """
        if self.read_only:
            return []

        with self._message_lock:
            messages_to_send = list(self._message_buffer)

        if not messages_to_send:
            return []

        # Group all messages by session_id, combining conversational and blob messages
        # Structure: {session_id: {"payload": [...], "timestamp": latest_timestamp, "metadata": {...}}}
        session_groups: dict[str, dict[str, Any]] = {}

        for buffered_msg in messages_to_send:
            sid = buffered_msg.session_id
            if sid not in session_groups:
                session_groups[sid] = {"payload": [], "timestamp": buffered_msg.timestamp, "metadata": {}}

            if buffered_msg.is_blob:
                for msg in buffered_msg.messages:
                    session_groups[sid]["payload"].append({"blob": json.dumps(msg)})
            else:
                for text, role in buffered_msg.messages:
                    session_groups[sid]["payload"].append(
                        {"conversational": {"content": {"text": text}, "role": role.upper()}}
                    )

            # Use the latest timestamp for the combined event
            if buffered_msg.timestamp > session_groups[sid]["timestamp"]:
                session_groups[sid]["timestamp"] = buffered_msg.timestamp

            # Merge metadata (later entries override earlier for same key)
            if buffered_msg.metadata:
                session_groups[sid]["metadata"].update(buffered_msg.metadata)

        results = []
        try:
            # Send one create_event per session_id with all messages (conversational + blob)
            for session_id, group in session_groups.items():
                create_event_kwargs: dict[str, Any] = {
                    "memoryId": self.config.memory_id,
                    "actorId": self.config.actor_id,
                    "sessionId": session_id,
                    "payload": group["payload"],
                    "eventTimestamp": group["timestamp"],
                }
                if group["metadata"]:
                    create_event_kwargs["metadata"] = group["metadata"]
                event = self.memory_client.gmdp_client.create_event(**create_event_kwargs)
                results.append(event)
                logger.debug(
                    "Flushed batched event for session %s with %d messages: %s",
                    session_id,
                    len(group["payload"]),
                    event.get("eventId"),
                )

            # Clear message buffer only after ALL messages succeed
            with self._message_lock:
                self._message_buffer.clear()

        except Exception as e:
            logger.error("Failed to flush messages to AgentCore Memory: %s", e)
            raise SessionException(f"Failed to flush messages: {e}") from e

        logger.info("Flushed %d message events to AgentCore Memory", len(results))
        return results

    def _flush_agent_states_only(self) -> list[dict[str, Any]]:
        """Flush only buffered agent states to AgentCore Memory.

        Call this method to send any remaining agent state when batch_size > 1.
        Agent states are grouped by agent_id and sent as separate events so that
        each event carries the correct agentId metadata for read_agent() lookups.

        Returns:
            list[dict[str, Any]]: List of created event responses from AgentCore Memory.

        Raises:
            SessionException: If agent state creation fails. On failure, agent states remain in the buffer.
        """
        if self.read_only:
            return []

        with self._agent_state_lock:
            agent_states_to_send = list(self._agent_state_buffer)

        if not agent_states_to_send:
            return []

        results = []
        try:
            # Group agent states by agent_id
            agent_groups: dict[str, list[dict]] = {}
            for _session_id, session_agent in agent_states_to_send:
                agent_id = session_agent.agent_id
                if agent_id not in agent_groups:
                    agent_groups[agent_id] = []
                agent_groups[agent_id].append({"blob": json.dumps(session_agent.to_dict())})

            # Send one event per agent_id with correct metadata
            for agent_id, payloads in agent_groups.items():
                event = self.memory_client.gmdp_client.create_event(
                    memoryId=self.config.memory_id,
                    actorId=self.config.actor_id,
                    sessionId=self.config.session_id,
                    payload=payloads,
                    eventTimestamp=self._get_monotonic_timestamp(),
                    metadata={
                        STATE_TYPE_KEY: {"stringValue": StateType.AGENT.value},
                        AGENT_ID_KEY: {"stringValue": agent_id},
                    },
                )
                results.append(event)
                logger.debug("Flushed %d agent states for agent %s: %s", len(payloads), agent_id, event.get("eventId"))

            # Clear agent state buffer only after ALL events succeed
            with self._agent_state_lock:
                self._agent_state_buffer.clear()

        except Exception as e:
            logger.error("Failed to flush agent states to AgentCore Memory: %s", e)
            raise SessionException(f"Failed to flush agent states: {e}") from e

        logger.info("Flushed %d agent state events to AgentCore Memory", len(results))
        return results

    def _flush_messages(self) -> list[dict[str, Any]]:
        """Flush all buffered messages and agent state to AgentCore Memory.

        Call this method to send any remaining buffered messages and agent state messages.
        This is automatically called when the session is complete (via close() or context manager).

        Returns:
            list[dict[str, Any]]: List of created event responses from AgentCore Memory.

        Raises:
            SessionException: If any message or agent state creation fails.
        """
        results = []
        results.extend(self._flush_messages_only())
        results.extend(self._flush_agent_states_only())
        return results

    def pending_message_count(self) -> int:
        """Return the number of messages pending in the buffer.

        Returns:
            int: Number of buffered messages waiting to be sent.
        """
        with self._message_lock:
            return len(self._message_buffer)

    def pending_agent_state_count(self) -> int:
        """Return the number of agent states pending in the buffer.

        Returns:
            int: Number of buffered agent states waiting to be sent.
        """
        with self._agent_state_lock:
            return len(self._agent_state_buffer)

    def close(self) -> None:
        """Explicitly flush pending messages and close the session manager.

        Call this method when the session is complete to ensure all buffered
        messages are sent to AgentCore Memory. Alternatively, use the context
        manager protocol (with statement) for automatic cleanup.
        """
        self._stop_flush_timer()
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
            self._stop_flush_timer()
            self._flush_messages()
        except Exception as e:
            if exc_type is not None:
                logger.error("Failed to flush messages during exception handling: %s", e)
            else:
                raise

    # endregion Batching support

    # region Interval-based flushing support

    def _start_flush_timer(self) -> None:
        """Start the interval-based flush timer.

        This method schedules a recurring timer that flushes the message buffer
        at regular intervals if flush_interval_seconds is configured.
        """
        with self._timer_lock:
            if self._shutdown:
                return

            # Cancel existing timer if any
            if self._flush_timer is not None:
                self._flush_timer.cancel()

            # Schedule next flush
            self._flush_timer = threading.Timer(
                self.config.flush_interval_seconds,
                self._interval_flush_callback,
            )
            self._flush_timer.daemon = True
            self._flush_timer.start()
            logger.debug(
                "Scheduled interval flush in %.1f seconds",
                self.config.flush_interval_seconds,
            )

    def _interval_flush_callback(self) -> None:
        """Callback executed by the flush timer.

        Flushes the buffer if it contains messages or agent states, then reschedules the timer.
        """
        try:
            # Only flush if there are messages or agent states in the buffer
            pending_messages = self.pending_message_count()
            pending_agent_states = self.pending_agent_state_count()
            if pending_messages > 0 or pending_agent_states > 0:
                logger.debug(
                    "Interval flush triggered: %d message(s) and %d agent state(s) pending",
                    pending_messages,
                    pending_agent_states,
                )
                self._flush_messages()
            else:
                logger.debug("Interval flush skipped: buffers are empty")

            # Reschedule the timer (unless shutdown)
            if not self._shutdown and self.config.flush_interval_seconds:
                self._start_flush_timer()

        except Exception as e:
            logger.error("Error during interval flush: %s", e)
            # Attempt to reschedule even after error
            if not self._shutdown and self.config.flush_interval_seconds:
                self._start_flush_timer()

    def _stop_flush_timer(self) -> None:
        """Stop the interval-based flush timer.

        This method cancels the timer and prevents it from rescheduling.
        Should be called during cleanup (close() or __exit__).
        """
        with self._timer_lock:
            self._shutdown = True
            if self._flush_timer is not None:
                self._flush_timer.cancel()
                self._flush_timer = None
                logger.debug("Stopped interval flush timer")

    # endregion Interval-based flushing support
