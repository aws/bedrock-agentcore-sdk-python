"""AgentCore Memory-based session manager for Bedrock AgentCore Memory integration."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

import boto3
from botocore.config import Config as BotocoreConfig
from strands.experimental.hooks.multiagent import (
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    MultiAgentInitializedEvent,
)
from strands.hooks import AfterInvocationEvent, AgentInitializedEvent, MessageAddedEvent
from strands.hooks.registry import HookRegistry
from strands.session.repository_session_manager import RepositorySessionManager
from strands.session.session_repository import SessionRepository
from strands.types.content import Message
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage
from typing_extensions import override

from bedrock_agentcore.memory.client import MemoryClient

from .bedrock_converter import AgentCoreMemoryConverter
from .config import AgentCoreMemoryConfig, RetrievalConfig

if TYPE_CHECKING:
    from strands.agent.agent import Agent

logger = logging.getLogger(__name__)

SESSION_PREFIX = "session_"
AGENT_PREFIX = "agent_"
MESSAGE_PREFIX = "message_"
MAX_FETCH_ALL_RESULTS = 10000

# Payload type markers for batched storage
PAYLOAD_TYPE_MESSAGE = "message"
PAYLOAD_TYPE_AGENT_STATE = "agent_state"


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
        self._last_synced_state_hash: Optional[int] = None

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

    # region Internal Storage Methods

    def _build_agent_state_payload(self, agent: "Agent") -> dict:
        """Create agent state payload for unified storage.

        Creates a SessionAgent-compatible payload that can be reconstructed
        via SessionAgent.from_dict().

        Args:
            agent (Agent): The agent whose state to capture.

        Returns:
            dict: Agent state payload with type markers.
        """
        session_agent = SessionAgent.from_agent(agent)
        return {
            "_type": PAYLOAD_TYPE_AGENT_STATE,
            "_agent_id": agent.agent_id,
            **session_agent.to_dict(),
        }

    def _compute_state_hash(self, agent: "Agent") -> int:
        """Compute hash of agent state for change detection.

        Excludes timestamps (created_at, updated_at) as they change on every call.

        Args:
            agent (Agent): The agent whose state to hash.

        Returns:
            int: Hash of the agent state.
        """
        session_agent = SessionAgent.from_agent(agent)
        state_dict = session_agent.to_dict()
        # Exclude timestamps that change on every call
        state_dict.pop("created_at", None)
        state_dict.pop("updated_at", None)
        return hash(json.dumps(state_dict, sort_keys=True))

    def _save_message_with_state(self, message: Message, agent: "Agent") -> None:
        """Save message and agent state in a single batched API call.

        Combines message and agent state into one API call instead of two separate calls.

        Args:
            message (Message): The message to save.
            agent (Agent): The agent whose state to sync.
        """
        session_message = SessionMessage.from_message(message, 0)
        messages = AgentCoreMemoryConverter.message_to_payload(session_message)
        if not messages:
            return

        message_tuple = messages[0]
        message_payload = {"_type": PAYLOAD_TYPE_MESSAGE, "data": list(message_tuple)}
        agent_state_payload = self._build_agent_state_payload(agent)

        original_timestamp = datetime.fromisoformat(session_message.created_at.replace("Z", "+00:00"))
        monotonic_timestamp = self._get_monotonic_timestamp(original_timestamp)

        try:
            event = self.memory_client.gmdp_client.create_event(
                memoryId=self.config.memory_id,
                actorId=self.config.actor_id,
                sessionId=self.session_id,
                payload=[
                    {"blob": json.dumps(message_payload)},
                    {"blob": json.dumps(agent_state_payload)},
                ],
                eventTimestamp=monotonic_timestamp,
            )
            logger.debug(
                "Saved message and agent state in single call: event=%s, agent=%s",
                event.get("event", {}).get("eventId"),
                agent.agent_id,
            )

            session_message = SessionMessage.from_message(message, event.get("event", {}).get("eventId"))
            self._latest_agent_message[agent.agent_id] = session_message
            self._last_synced_state_hash = self._compute_state_hash(agent)

        except Exception as e:
            logger.error("Failed to save message with state: %s", e)
            raise SessionException(f"Failed to save message with state: {e}") from e

    def _sync_agent_state_if_changed(self, agent: "Agent") -> None:
        """Sync agent state to AgentCore Memory only if state changed since last sync.

        Args:
            agent (Agent): The agent to sync.
        """
        current_hash = self._compute_state_hash(agent)
        if current_hash == self._last_synced_state_hash:
            logger.debug("Agent state unchanged, skipping sync for agent=%s", agent.agent_id)
            return

        agent_state_payload = self._build_agent_state_payload(agent)

        try:
            event = self.memory_client.gmdp_client.create_event(
                memoryId=self.config.memory_id,
                actorId=self.config.actor_id,
                sessionId=self.session_id,
                payload=[{"blob": json.dumps(agent_state_payload)}],
                eventTimestamp=self._get_monotonic_timestamp(),
            )
            self._last_synced_state_hash = current_hash
            logger.debug(
                "Synced agent state: event=%s, agent=%s", event.get("event", {}).get("eventId"), agent.agent_id
            )
        except Exception as e:
            logger.error("Failed to sync agent state: %s", e)
            raise SessionException(f"Failed to sync agent state: {e}") from e

    # endregion Internal Storage Methods

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

        Uses unified actorId with type markers for optimized storage.

        Args:
            session_id (str): The session ID to create the agent in.
            session_agent (SessionAgent): The agent to create.
            **kwargs (Any): Additional keyword arguments.

        Raises:
            SessionException: If session ID doesn't match configuration.
        """
        if session_id != self.config.session_id:
            raise SessionException(f"Session ID mismatch: expected {self.config.session_id}, got {session_id}")

        agent_state_payload = {
            "_type": PAYLOAD_TYPE_AGENT_STATE,
            "_agent_id": session_agent.agent_id,
            **session_agent.to_dict(),
        }
        event = self.memory_client.gmdp_client.create_event(
            memoryId=self.config.memory_id,
            actorId=self.config.actor_id,
            sessionId=self.session_id,
            payload=[{"blob": json.dumps(agent_state_payload)}],
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

        Uses dual-read approach: tries new format first (unified actorId with type markers),
        falls back to legacy format (separate actorId) for backward compatibility.

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
                actor_id=self.config.actor_id,
                session_id=session_id,
                max_results=MAX_FETCH_ALL_RESULTS,
            )

            # Events are returned oldest-first, so reverse to get latest state
            for event in reversed(events):
                for payload_item in event.get("payload", []):
                    blob = payload_item.get("blob")
                    if blob:
                        try:
                            data = json.loads(blob)
                            if data.get("_type") == PAYLOAD_TYPE_AGENT_STATE and data.get("_agent_id") == agent_id:
                                agent_data = {k: v for k, v in data.items() if k not in ("_type", "_agent_id")}
                                return SessionAgent.from_dict(agent_data)
                        except json.JSONDecodeError:
                            continue

            # Fallback to legacy format for backward compatibility
            legacy_events = self.memory_client.list_events(
                memory_id=self.config.memory_id,
                actor_id=self._get_full_agent_id(agent_id),
                session_id=session_id,
                max_results=1,
            )
            if legacy_events:
                agent_data = json.loads(legacy_events[0].get("payload", {})[0].get("blob"))
                return SessionAgent.from_dict(agent_data)

        except Exception as e:
            logger.error("Failed to read agent: %s", e)

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

    def create_message(  # type: ignore[override]
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
                return None

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
        """Append a message to the agent's session with batched agent state sync.

        Saves message and agent state in a single API call for optimization.

        Args:
            message: Message to add to the agent in the session
            agent: Agent to append the message to
            **kwargs: Additional keyword arguments for future extensibility.
        """
        self._save_message_with_state(message, agent)

    @override
    def sync_agent(self, agent: "Agent", **kwargs: Any) -> None:
        """Sync agent state only if it changed since last sync.

        Skips sync if agent state is unchanged, avoiding redundant API calls.

        Args:
            agent: Agent to sync to the session.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        self._sync_agent_state_if_changed(agent)

    def retrieve_customer_context(self, event: MessageAddedEvent) -> None:
        """Retrieve customer LTM context before processing support query.

        Args:
            event (MessageAddedEvent): The message added event containing the agent and message data.
        """
        messages = event.agent.messages
        content = messages[-1].get("content") if messages else None
        if not messages or messages[-1].get("role") != "user" or not content or "toolResult" in content[0]:
            return None
        if not self.config.retrieval_config:
            # Only retrieve LTM
            return None

        user_query = messages[-1]["content"][0]["text"]

        def retrieve_for_namespace(namespace: str, retrieval_config: RetrievalConfig) -> list[str]:
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
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks for session management with optimized storage."""
        registry.add_callback(AgentInitializedEvent, lambda event: self.initialize(event.agent))
        registry.add_callback(MessageAddedEvent, lambda e: self._save_message_with_state(e.message, e.agent))
        registry.add_callback(AfterInvocationEvent, lambda event: self._sync_agent_state_if_changed(event.agent))
        registry.add_callback(MessageAddedEvent, lambda event: self.retrieve_customer_context(event))

        # Multi-agent hooks
        registry.add_callback(MultiAgentInitializedEvent, lambda event: self.initialize_multi_agent(event.source))
        registry.add_callback(AfterNodeCallEvent, lambda event: self.sync_multi_agent(event.source))
        registry.add_callback(AfterMultiAgentInvocationEvent, lambda event: self.sync_multi_agent(event.source))

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
