"""AgentCore Memory-based session manager for Bedrock AgentCore Memory integration."""

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

import boto3
from botocore.config import Config as BotocoreConfig
from strands.agent.state import AgentState
from strands.hooks import AfterInvocationEvent, AgentInitializedEvent, MessageAddedEvent
from strands.hooks.registry import HookRegistry
from strands.session.repository_session_manager import RepositorySessionManager
from strands.session.session_repository import SessionRepository
from strands.types.content import Message
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage
from typing_extensions import override

from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole
from bedrock_agentcore.memory.models import (
    EventMetadataFilter,
    LeftExpression,
    OperatorType,
    RightExpression,
    StringValue,
)
from bedrock_agentcore.memory.session import MemorySessionManager

from .bedrock_converter import AgentCoreMemoryConverter
from .config import AgentCoreMemoryConfig

if TYPE_CHECKING:
    from strands.agent.agent import Agent

logger = logging.getLogger(__name__)


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

    def _validate_session_id(self, session_id: str) -> None:
        """Validate session ID matches configuration."""
        if session_id != self.config.session_id:
            raise SessionException(
                f"Session ID mismatch: expected {self.config.session_id}, got {session_id}"
            )

    def _prepare_event_params(
        self,
        payload: Any,
        metadata: Optional[dict] = None,
        branch=True,
    ) -> dict:
        """Prepare common event parameters."""
        event_params = {
            "memoryId": self.config.memory_id,
            "actorId": self.config.actor_id,
            "sessionId": self.config.session_id,
            "payload": payload,
            "eventTimestamp": self._get_monotonic_timestamp(),
        }

        if metadata:
            event_params["metadata"] = metadata

        if (
            branch
            and self.config.default_branch
            and self.config.default_branch.name != "main"
        ):
            event_params["branch"] = self.config.default_branch.to_agentcore_format()

        return event_params

    def _prepare_list_params(self, branch=True, **kwargs) -> dict:
        """Prepare common list_events parameters."""
        params = {
            "actor_id": self.config.actor_id,
            "session_id": self.config.session_id,
            **kwargs,
        }

        if (
            self.config.default_branch
            and branch
            and self.config.short_term_retrieval_config.branch_filter
            and self.config.default_branch.name != "main"
        ):
            params["branch_name"] = self.config.default_branch.name

        return params

    @classmethod
    def _get_monotonic_timestamp(
        cls, desired_timestamp: Optional[datetime] = None
    ) -> datetime:
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
        self._session_cache: Optional[SessionAgent] = None
        self._branch_root_events: dict[str, str] = {}
        # Validate session_id length
        if len(self.config.session_id) < 33:
            raise SessionException(
                f"Session ID must be at least 33 characters long to ensure uniqueness: {self.config.session_id}"
            )

        # Initialize the new MemorySessionManager
        self.memory_session_manager = MemorySessionManager(
            memory_id=self.config.memory_id,
            region_name=region_name,
            boto3_session=boto_session,
            boto_client_config=boto_client_config,
        )
        self.agent_id = None

    def _get_or_create_branch_root(
        self, branch_name: str, session_agent: SessionAgent
    ) -> str:
        """Get branch root event ID and cache it."""
        if self._branch_root_events.get(branch_name):
            return self._branch_root_events[branch_name]
        if branch_name == "main":
            return None
        # Check if branch exists
        list_params = self._prepare_list_params(max_results=1)
        branch_events = self.memory_session_manager.list_events(**list_params)

        if branch_events:
            root_event_id = branch_events[0].get("eventId")
            self._branch_root_events[branch_name] = root_event_id
            return root_event_id

        # Branch doesn't exist - get main branch root
        if branch_name != "main":
            main_params = self._prepare_list_params(max_results=1, branch=False)

            main_events = self.memory_session_manager.list_events(**main_params)

            if not main_events:
                # Create event in main first
                metadata = {
                    "agent_id": StringValue.build(session_agent.agent_id),
                    "event_type": StringValue.build("session_state"),
                }
                if self.config.metadata:
                    metadata.update(self.config.metadata)

                event_params = self._prepare_event_params(
                    payload=[
                        {"blob": json.dumps({"session_state": session_agent.to_dict()})}
                    ],
                    metadata=metadata,
                    branch=False,
                )

                main_event = self.memory_session_manager.create_event(**event_params)
                root_event_id = main_event.get("eventId")
            else:
                root_event_id = main_events[0].get("eventId")

            self._branch_root_events[branch_name] = root_event_id
            return root_event_id

        return None

    def create_or_fetch_agent_branch(
        self, session_id: str, session_agent: SessionAgent
    ) -> Any:
        """Create event and update session cache."""
        self._validate_session_id(session_id)

        branch_name = self.config.default_branch.name

        try:
            # Get or create branch root
            root_event_id = self._get_or_create_branch_root(branch_name, session_agent)

            # Create new session_state event
            metadata = {
                "agent_id": StringValue.build(session_agent.agent_id),
                "event_type": StringValue.build("session_state"),
            }
            if self.config.metadata:
                metadata.update(self.config.metadata)

            if branch_name == "main":
                event_params = self._prepare_event_params(
                    payload=[
                        {"blob": json.dumps({"session_state": session_agent.to_dict()})}
                    ],
                    metadata=metadata,
                )
                created_event = self.memory_session_manager.create_event(**event_params)
            else:
                created_event = self.memory_session_manager.fork_conversation(
                    actor_id=self.config.actor_id,
                    session_id=self.config.session_id,
                    root_event_id=root_event_id,
                    branch_name=branch_name,
                    messages=[
                        {"blob": json.dumps({"session_state": session_agent.to_dict()})}
                    ],
                    metadata=metadata,
                    event_timestamp=self._get_monotonic_timestamp(),
                )

            self._session_cache = session_agent
            return created_event

        except Exception as e:
            logger.error("Failed to create or fetch agent branch: %s", e)
            raise SessionException(
                f"Failed to create or fetch agent branch: {e}"
            ) from e

    def create_session(self, session: Session, **kwargs: Any) -> Session:
        """Create a new session."""
        self._validate_session_id(session.session_id)
        return session

    def update_agent(
        self, session_id: str, session_agent: SessionAgent, **kwargs: Any
    ) -> None:
        """Update an existing agent."""
        self._validate_session_id(session_id)
        self._session_cache = session_agent

    def create_message(
        self,
        session_id: str,
        agent_id: str,
        session_message: SessionMessage,
        **kwargs: Any,
    ) -> None:
        """Create a new message."""
        self._validate_session_id(session_id)

    def update_message(
        self,
        session_id: str,
        agent_id: str,
        session_message: SessionMessage,
        **kwargs: Any,
    ) -> None:
        """Update an existing message."""
        self._validate_session_id(session_id)

    # region SessionRepository interface implementation
    def create_agent(
        self, session_id: str, session_agent: SessionAgent, **kwargs: Any
    ) -> SessionAgent:
        """Create a new session or get the existing session in AgentCore Memory."""
        logger.debug(
            f"Creating agent: {session_agent.agent_id} in session: {session_id}"
        )
        self._validate_session_id(session_id)
        self.agent_id = session_agent.agent_id
        try:
            if self.config.default_branch.name == "main":
                metadata = {
                    "agent_id": StringValue.build(session_agent.agent_id),
                    "event_type": StringValue.build("session_state"),
                }
                if self.config.metadata:
                    metadata.update(self.config.metadata)

                # Prepare event parameters
                event_params = self._prepare_event_params(
                    payload=[
                        {"blob": json.dumps({"session": session_agent.to_dict()})}
                    ],
                    metadata=metadata,
                )
                event = self.memory_session_manager.create_event(**event_params)
                self._session_cache = session_agent
            else:
                event = self.create_or_fetch_agent_branch(session_id, session_agent)

            self._session_cache = session_agent
            logger.info(
                "Created session: %s with event: %s",
                session_id,
                event.get("eventId"),
            )
            logger.debug(
                f"Successfully created session with event: {event.get('eventId')}"
            )
            return session_agent
        except Exception as e:
            logger.error("Failed to create session: %s", e)
            raise SessionException(f"Failed to create session: {e}") from e

    def read_session(self, session_id: str, **kwargs: Any) -> Optional[Session]:
        """Read session data - always from the main branch."""
        logger.debug(f"Reading session: {session_id}")

        # Return cached session if available
        if (
            self._session_cache
            and hasattr(self._session_cache, "session_id")
            and self._session_cache.session_id == session_id
        ):
            logger.debug(f"Returning cached session: {session_id}")
            return self._session_cache

        session = self.read_agent(
            session_id, agent_id=self.agent_id if self.agent_id else "default"
        )
        return session

    def read_agent(
        self, session_id: str, agent_id: str, **kwargs: Any
    ) -> Optional[SessionAgent]:
        """Read agent data from AgentCore Memory events (uses branch filtering)."""
        logger.debug(f"Reading agent: {agent_id} from session: {session_id}")
        self._validate_session_id(session_id)
        if self._session_cache:
            logger.debug(f"Returning cached session: {session_id}")
            return self._session_cache

        try:
            logger.debug(f"Building metadata filters for agent {agent_id}")
            # Agent operations use branch filtering
            filters = [
                EventMetadataFilter.build_expression(
                    left_operand=LeftExpression.build(key="event_type"),
                    operator=OperatorType.EQUALS_TO,
                    right_operand=RightExpression.build("session_state"),
                )
            ]
            if (
                agent_id
                and hasattr(self.config, "short_term_retrieval_config")
                and hasattr(self.config.short_term_retrieval_config, "metadata")
                and self.config.short_term_retrieval_config.metadata
                and "agent_id"
                in self.config.short_term_retrieval_config.metadata.keys()
            ):
                logger.debug(
                    f"Using metadata filter: {self.config.short_term_retrieval_config.metadata['agent_id']}"
                )
                filters.append(
                    EventMetadataFilter.build_expression(
                        left_operand=LeftExpression.build(key="agent_id"),
                        operator=OperatorType.EQUALS_TO,
                        right_operand=RightExpression.build(
                            self.config.short_term_retrieval_config.metadata["agent_id"]
                        ),
                    )
                )
            list_params = self._prepare_list_params(
                eventMetadata=filters, max_results=1
            )

            events = self.memory_session_manager.list_events(**list_params)
            logger.debug(f"Found {len(events)} agent events for {agent_id}")

            if not events:
                logger.debug(f"No events found for agent {agent_id}")
                return None

            payload = events[-1].get("payload", [])
            if payload and "blob" in payload[0]:
                blob_data = json.loads(payload[0]["blob"])
                # Extract SessionAgent from blob.
                session_data = blob_data.get("session_state") or blob_data.get(
                    "session"
                )
                agent_data = SessionAgent.from_dict(session_data)
                logger.debug(f"Successfully read agent {agent_id}")
                self._session_cache = agent_data
                return agent_data

            logger.debug(f"Agent {agent_id} not found - no valid payload")
            return None
        except Exception as e:
            logger.error("Failed to read agent %s: %s", agent_id, e)
            return None

    def read_message(
        self, session_id: str, agent_id: str, message_id: int, **kwargs: Any
    ) -> Optional[SessionMessage]:
        """Read a specific message by ID from AgentCore Memory."""
        try:
            result = self.memory_session_manager.get_event(
                actor_id=self.config.actor_id,
                session_id=session_id,
                event_id=str(message_id),
            )
            return SessionMessage.from_dict(result) if result else None
        except Exception as e:
            logger.error("Failed to read message: %s", e)
            return None

    def save_turn_messages(
        self, event: AfterInvocationEvent, **kwargs: Any
    ) -> Optional[dict[str, Any]]:
        """Save turn messages to MemorySessionManager with both SessionAgent and Session in blob."""
        try:
            logger.debug(f"Saving turn messages for agent: {event.agent.agent_id}")
            # Filter messages based on configured message types
            filtered_messages = []
            role_map = {
                "user": MessageRole.USER,
                "assistant": MessageRole.ASSISTANT,
                "tool": MessageRole.TOOL,
                "other": MessageRole.OTHER,
            }

            for message in reversed(event.agent.messages):
                role = message.get("role")
                if role in self.config.message_types:
                    content = message.get("content", [{}])[0].get("text", "")
                    mapped_role = role_map.get(role, MessageRole.ASSISTANT)
                    if role == "user":
                        content = self.remove_user_context(content)
                    filtered_messages.append(
                        ConversationalMessage(content, mapped_role)
                    )
                if role == "user":
                    break

            logger.debug(f"Filtered {len(filtered_messages)} messages to save..")
            if not filtered_messages:
                return None

            # Prepare enhanced metadata using agent properties
            event_metadata = {
                "agent_id": StringValue.build(event.agent.agent_id),
            }

            # Merge with config metadata
            if self.config.metadata:
                event_metadata.update(self.config.metadata)

            # Prepare event parameters
            event_params = self._prepare_event_params(
                payload=filtered_messages, metadata=event_metadata
            )
            # Add branch configuration if not "main"
            event_params = {
                "actor_id": self.config.actor_id,
                "session_id": self.config.session_id,
                "messages": filtered_messages,
                "metadata": event_metadata,
                "event_timestamp": self._get_monotonic_timestamp(),
            }
            if self.config.default_branch and self.config.default_branch.name != "main":
                event_params["branch"] = (
                    self.config.default_branch.to_agentcore_format()
                )

            return self.memory_session_manager.add_turns(**event_params)

        except Exception as e:
            logger.error("Failed to save turn messages: %s", e)
            raise SessionException(f"Failed to save turn messages: {e}") from e

    def list_messages(
        self,
        session_id: str,
        agent_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        **kwargs: Any,
    ) -> list[SessionMessage]:
        """List messages for an agent from AgentCore Memory with pagination."""
        self._validate_session_id(session_id)
        logger.debug(f"Listing messages for agent: {agent_id}, limit: {limit}")

        try:
            max_results = (limit + offset) if limit else 100

            # Filter for non-session-state events (conversation messages)
            message_filter = EventMetadataFilter.build_expression(
                left_operand=LeftExpression.build(key="event_type"),
                operator=OperatorType.NOT_EXISTS,
            )

            filters = [message_filter]

            # Create metadata filter for agent_id if configured
            if (
                hasattr(self.config, "retrieval_config")
                and hasattr(self.config.short_term_retrieval_config, "metadata")
                and self.config.short_term_retrieval_config.metadata
                and "agent_id"
                in self.config.short_term_retrieval_config.metadata.keys()
            ):
                agent_filter = EventMetadataFilter.build_expression(
                    left_operand=LeftExpression.build(key="agent_id"),
                    operator=OperatorType.EQUALS_TO,
                    right_operand=RightExpression.build(agent_id),
                )
                filters.append(agent_filter)

            list_params = self._prepare_list_params(
                max_results=max_results,
                eventMetadata=filters,
            )

            events = self.memory_session_manager.list_events(**list_params)
            logger.debug(f"Found {len(events)} events")
            messages = AgentCoreMemoryConverter.events_to_messages(events)
            return messages

        except Exception as e:
            logger.error("Failed to list messages from AgentCore Memory: %s", e)
            return []

    # endregion SessionRepository interface implementation

    # region RepositorySessionManager overrides
    @staticmethod
    def remove_user_context(text: str) -> str:
        """Remove user context from text."""
        return re.sub(
            r"<previous_context>.*?</previous_context>", "", text, flags=re.DOTALL
        ).strip()

    @override
    def initialize(self, agent: "Agent", **kwargs: Any) -> None:
        """Agent hydration with branch support for multiple actors."""
        try:
            self.agent_id = agent.agent_id
            # Use create_or_fetch_agent_branch for both main and custom branches
            if not self.read_agent(self.config.session_id, agent.agent_id):
                self.create_agent(
                    self.config.session_id, SessionAgent.from_agent(agent)
                )
            # Set agent state from cached session
            agent.state = AgentState(
                self._session_cache.state if self._session_cache else {}
            )

            # Load previous messages
            prev_messages = self.list_messages(
                self.config.session_id, agent.agent_id, limit=10
            )
            agent.messages = prev_messages if prev_messages else []

        except Exception as e:
            logger.error("Failed to initialize agent %s: %s", agent.agent_id, e)
            raise

    def retrieve_customer_context(self, event: MessageAddedEvent) -> None:
        """Retrieve customer context from both short-term and long-term memory."""
        messages = event.agent.messages
        if not messages or messages[-1].get("role") != "user":
            return

        logger.debug(f"Retrieving customer context for agent:")
        # Skip if message contains tool results
        last_content = messages[-1].get("content", [])
        if last_content and any(
            "toolResult" in str(content) for content in last_content
        ):
            return

        user_query = messages[-1]["content"][0]["text"]

        def retrieve_for_namespace(
            namespace: str, retrieval_config: AgentCoreMemoryConfig
        ):
            """Helper function to retrieve memories for a single namespace."""
            resolved_namespace = namespace.format(
                actorId=self.config.actor_id,
                sessionId=self.config.session_id,
                memoryStrategyId=retrieval_config.strategy_id or "",
            )
            memories = self.memory_session_manager.search_long_term_memories(
                query=user_query,
                namespace_prefix=resolved_namespace,
                top_k=retrieval_config.top_k,
            )

            context_items = []
            for memory in memories:
                if hasattr(memory, "content") and hasattr(memory.content, "text"):
                    text = memory.content.text.strip()
                    if text:
                        all_context.append(text)
                elif isinstance(memory, dict):
                    content = memory.get("content", {})
                    if isinstance(content, dict):
                        text = content.get("text", "").strip()
                        if text:
                            context_items.append(text)
            return context_items

        try:
            all_context = []
            # Retrieve from long-term memory in parallel
            with ThreadPoolExecutor() as executor:
                future_to_namespace = {
                    executor.submit(
                        retrieve_for_namespace, namespace, retrieval_config
                    ): namespace
                    for namespace, retrieval_config in self.config.retrieval_config.items()
                }
                for future in as_completed(future_to_namespace):
                    try:
                        context_items = future.result()
                        all_context.extend(context_items)
                    except Exception as e:
                        # Continue processing other futures event if one fails rather than failing the entire operation
                        namespace = future_to_namespace[future]
                        logger.error(
                            "Failed to retrieve memories for namespace %s: %s",
                            namespace,
                            e,
                        )

            if all_context:
                original_text = messages[-1]["content"][0]["text"]
                context_text = "\n".join(all_context)
                messages[-1]["content"][0][
                    "text"
                ] = f"<previous_context>\n{context_text}\n</previous_context>\n\n{original_text}\n"
                event.agent.messages[-1]["content"][0]["text"] = context_text
                logger.info("Retrieved %d customer context items", len(all_context))

        except Exception as e:
            logger.error("Failed to retrieve customer context: %s", e)

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        """Register hooks.

        Args:
            registry (HookRegistry): The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments.
        """
        # After the normal Agent initialization behavior, call the session initialize function to restore the agent
        registry.add_callback(
            AgentInitializedEvent, lambda event: self.initialize(event.agent)
        )
        # For each message appended to the Agents messages, store that message in the session
        # After an agent was invoked, sync it with the session to capture any conversation manager state updates
        registry.add_callback(
            AfterInvocationEvent, lambda event: self.save_turn_messages(event)
        )
        registry.add_callback(
            MessageAddedEvent, lambda event: self.retrieve_customer_context(event)
        )
