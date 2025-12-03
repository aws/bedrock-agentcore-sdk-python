"""Module containing session management classes for AgentCore Memory interactions."""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

from .constants import BlobMessage, ConversationalMessage, MessageRole, RetrievalConfig
from .models import (
    ActorSummary,
    Branch,
    DictWrapper,
    Event,
    EventMessage,
    EventMetadataFilter,
    MemoryRecord,
    MetadataValue,
    SessionSummary,
)

logger = logging.getLogger(__name__)


class MemorySessionManager:
    """Manages conversational sessions and memory operations for AWS Bedrock AgentCore.

    The MemorySessionManager provides a high-level interface for managing conversational AI sessions,
    handling both short-term (conversational events) and long-term (semantic memory) storage.
    It serves as the primary entry point for data plane operations with AWS Bedrock AgentCore
    Memory services.

    Key Capabilities:
        - **Conversation Management**: Store, retrieve, and organize conversational turns
        - **Memory Operations**: Search and manage long-term semantic memory records
        - **Branch Support**: Create and manage conversation branches for alternative flows
        - **LLM Integration**: Built-in callback pattern for LLM processing with memory context
        - **Actor & Session Tracking**: Multi-user, multi-session conversation management

    Usage Patterns:
        1. **Simple Conversation**: Store user/assistant message pairs
        2. **Memory-Enhanced Chat**: Retrieve relevant context before LLM processing
        3. **Branched Conversations**: Fork conversations for alternative responses
        4. **Multi-Modal**: Handle both text and binary data (images, files, etc.)

    Example:
        ```python
        # Initialize manager
        manager = MemorySessionManager(memory_id="my-memory-123", region_name="us-east-1")

        # Store a conversation turn
        manager.add_turns(
            actor_id="user-456",
            session_id="session-789",
            messages=[
                ConversationalMessage("Hello!", MessageRole.USER),
                ConversationalMessage("Hi there!", MessageRole.ASSISTANT)
            ]
        )

        # Search long-term memory and process with LLM
        def my_llm(user_input: str, memories: List[Dict]) -> str:
            # Your LLM processing logic here
            return "Response based on context"

        memories, response, event = manager.process_turn_with_llm(
            actor_id="user-456",
            session_id="session-789",
            user_input="What did we discuss?",
            llm_callback=my_llm,
            retrieval_namespace="support/facts/{sessionId}"
        )
        ```

    Thread Safety:
        This class is not thread-safe. Create separate instances for concurrent operations.

    AWS Permissions Required:
        - bedrock-agentcore:CreateEvent
        - bedrock-agentcore:GetEvent
        - bedrock-agentcore:ListEvents
        - bedrock-agentcore:DeleteEvent
        - bedrock-agentcore:RetrieveMemoryRecords
        - bedrock-agentcore:ListMemoryRecords
        - bedrock-agentcore:GetMemoryRecord
        - bedrock-agentcore:DeleteMemoryRecord
        - bedrock-agentcore:ListActors
        - bedrock-agentcore:ListSessions
        - bedrock-agentcore:BatchCreateMemoryRecords
        - bedrock-agentcore:BatchDeleteMemoryRecords
        - bedrock-agentcore:BatchUpdateMemoryRecords
    """

    def __init__(
        self,
        memory_id: str,
        region_name: Optional[str] = None,
        boto3_session: Optional[boto3.Session] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
    ):
        """Initialize a MemorySessionManager instance.

        Args:
            memory_id: The memory identifier for this session manager.
            region_name: AWS region for the bedrock-agentcore client. If not provided,
                   will use the region from boto3_session or default session.
            boto3_session: Optional boto3 Session to use. If provided and region_name
                          parameter is also specified, validation will ensure they match.
            boto_client_config: Optional boto3 client configuration. If provided, will be
                              merged with default configuration including user agent.

        Raises:
            ValueError: If region_name parameter conflicts with boto3_session region.
        """
        # Initialize core attributes
        self._memory_id = memory_id

        # Setup session and validate region consistency
        self.region_name = self._validate_and_resolve_region(region_name, boto3_session)
        session = boto3_session if boto3_session else boto3.Session()

        # Configure and create boto3 client
        client_config = self._build_client_config(boto_client_config)
        self._data_plane_client = session.client(
            "bedrock-agentcore", region_name=self.region_name, config=client_config
        )

        # Configure timestamp serialization to use float representation
        self._configure_timestamp_serialization()

        # Define allowed data plane methods
        self._ALLOWED_DATA_PLANE_METHODS = {
            "retrieve_memory_records",
            "get_memory_record",
            "delete_memory_record",
            "list_memory_records",
            "create_event",
            "get_event",
            "delete_event",
            "list_events",
            "batch_create_memory_records",
            "batch_delete_memory_records",
            "batch_update_memory_records",
        }

    def _validate_and_resolve_region(self, region_name: Optional[str], session: Optional[boto3.Session]) -> str:
        """Validate region consistency and resolve the final region to use.

        Args:
            region_name: Explicitly provided region name
            session: Optional Boto3 session instance

        Returns:
            The resolved region name to use

        Raises:
            ValueError: If region_name conflicts with session region
        """
        session_region = session.region_name if session else None

        # Validate region consistency if both are provided
        if region_name and session and session_region and (region_name != session_region):
            raise ValueError(
                f"Region mismatch: provided region_name '{region_name}' does not match "
                f"boto3_session region '{session_region}'. Please ensure both "
                f"parameters specify the same region or omit the region_name parameter "
                f"to use the session's region."
            )

        return (
            region_name or session_region or os.environ.get("AWS_REGION") or boto3.Session().region_name or "us-west-2"
        )

    def _build_client_config(self, boto_client_config: Optional[BotocoreConfig]) -> BotocoreConfig:
        """Build the final boto3 client configuration with SDK user agent.

        Args:
            boto_client_config: Optional user-provided client configuration

        Returns:
            Final client configuration with SDK user agent
        """
        sdk_user_agent = "bedrock-agentcore-sdk"

        if boto_client_config:
            existing_user_agent = getattr(boto_client_config, "user_agent_extra", None)
            if existing_user_agent:
                new_user_agent = f"{existing_user_agent} {sdk_user_agent}"
            else:
                new_user_agent = sdk_user_agent
            return boto_client_config.merge(BotocoreConfig(user_agent_extra=new_user_agent))
        else:
            return BotocoreConfig(user_agent_extra=sdk_user_agent)

    def _configure_timestamp_serialization(self) -> None:
        """Configure the boto3 client to serialize timestamps as float values.

        This method overrides the default timestamp serialization to convert datetime objects
        to float timestamps (seconds since Unix epoch) which preserves millisecond precision
        when sending datetime objects to the AgentCore Memory service.
        """
        original_serialize_timestamp = self._data_plane_client._serializer._serializer._serialize_type_timestamp

        def serialize_timestamp_as_float(serialized, value, shape, name):
            if isinstance(value, datetime):
                serialized[name] = value.timestamp()  # Convert to float (seconds since epoch with fractional seconds)
            else:
                original_serialize_timestamp(serialized, value, shape, name)

        self._data_plane_client._serializer._serializer._serialize_type_timestamp = serialize_timestamp_as_float

    def __getattr__(self, name: str):
        """Dynamically forward method calls to the appropriate boto3 client.

        This method enables access to all data_plane boto3 client methods without explicitly
        defining them. Methods are looked up in the following order:
        _data_plane_client (bedrock-agentcore) - for data plane operations

        Args:
            name: The method name being accessed

        Returns:
            A callable method from the boto3 client

        Raises:
            AttributeError: If the method doesn't exist on _data_plane_client

        Example:
            # Access any boto3 method directly
            manager = MemorySessionManager(region_name="us-east-1")

            # These calls are forwarded to the appropriate boto3 functions
            memory_records = manager.retrieve_memory_records()
            events = manager.list_events(...)
        """
        if name in self._ALLOWED_DATA_PLANE_METHODS and hasattr(self._data_plane_client, name):
            method = getattr(self._data_plane_client, name)
            logger.debug("Forwarding method '%s' to _data_plane_client", name)
            return method

        # Method not found on client
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
            f"Method not found on _data_plane_client. "
            f"Available methods can be found in the boto3 documentation for "
            f"'bedrock-agentcore' services."
        )

    def process_turn_with_llm(
        self,
        actor_id: str,
        session_id: str,
        user_input: str,
        llm_callback: Callable[[str, List[Dict[str, Any]]], str],
        retrieval_config: Optional[Dict[str, RetrievalConfig]],
        metadata: Optional[Dict[str, MetadataValue]] = None,
        event_timestamp: Optional[datetime] = None,
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        r"""Complete conversation turn with LLM callback integration.

        This method combines memory retrieval, LLM invocation, and response storage
        in a single call using a callback pattern.

        Args:
            actor_id: Actor identifier (e.g., "user-123")
            session_id: Session identifier
            user_input: The user's message
            llm_callback: Function that takes (user_input, memories) and returns agent_response
                         The callback receives the user input and retrieved memories,
                         and should return the agent's response string
            retrieval_config: Optional dictionary mapping namespaces to RetrievalConfig objects.
                            Each namespace can contain template variables like {actorId}, {sessionId},
                            {memoryStrategyId} that will be resolved at runtime.
            metadata: Optional custom key-value metadata to attach to an event.
            event_timestamp: Optional timestamp for the event

        Returns:
            Tuple of (retrieved_memories, agent_response, created_event)

        Example:
            from bedrock_agentcore.memory.constants import RetrievalConfig

            def my_llm(user_input: str, memories: List[Dict]) -> str:
                # Format context from memories
                context = "\\n".join([m.get('content', {}).get('text', '') for m in memories])

                # Call your LLM (Bedrock, OpenAI, etc.)
                response = bedrock.invoke_model(
                    messages=[
                        {"role": "system", "content": f"Context: {context}"},
                        {"role": "user", "content": user_input}
                    ]
                )
                return response['content']

            retrieval_config = {
                "support/facts/{sessionId}": RetrievalConfig(top_k=5, relevance_score=0.3),
                "user/preferences/{actorId}": RetrievalConfig(top_k=3, relevance_score=0.5)
            }

            memories, response, event = manager.process_turn_with_llm(
                actor_id="user-123",
                session_id="session-456",
                user_input="What did we discuss yesterday?",
                llm_callback=my_llm,
                retrieval_config=retrieval_config
            )
        """
        # Step 1: Retrieve relevant memories
        retrieved_memories = self._retrieve_memories_for_llm(actor_id, session_id, user_input, retrieval_config)

        # Step 2: Invoke LLM callback
        try:
            agent_response = llm_callback(user_input, retrieved_memories)
            if not isinstance(agent_response, str):
                raise ValueError("LLM callback must return a string response")
            logger.info("LLM callback generated response")
        except Exception as e:
            logger.error("LLM callback failed: %s", e)
            raise

        # Step 3: Save the conversation turn
        event = self._save_conversation_turn(
            actor_id, session_id, user_input, agent_response, metadata, event_timestamp
        )
        return retrieved_memories, agent_response, event

    async def process_turn_with_llm_async(
        self,
        actor_id: str,
        session_id: str,
        user_input: str,
        llm_callback: Callable[[str, List[Dict[str, Any]]], Awaitable[str]],
        retrieval_config: Optional[Dict[str, RetrievalConfig]],
        metadata: Optional[Dict[str, MetadataValue]] = None,
        event_timestamp: Optional[datetime] = None,
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        r"""Complete conversation turn with async LLM callback integration.

        This method combines memory retrieval, LLM invocation, and response storage
        in a single call using an async callback pattern.

        Args:
            actor_id: Actor identifier (e.g., "user-123")
            session_id: Session identifier
            user_input: The user's message
            llm_callback: Async function that takes (user_input, memories) and returns agent_response.
                         The callback receives the user input and retrieved memories,
                         and should return the agent's response string
            retrieval_config: Optional dictionary mapping namespaces to RetrievalConfig objects.
                            Each namespace can contain template variables like {actorId}, {sessionId},
                            {memoryStrategyId} that will be resolved at runtime.
            metadata: Optional custom key-value metadata to attach to an event.
            event_timestamp: Optional timestamp for the event

        Returns:
            Tuple of (retrieved_memories, agent_response, created_event)
        """
        # Step 1: Retrieve relevant memories
        retrieved_memories = self._retrieve_memories_for_llm(actor_id, session_id, user_input, retrieval_config)

        # Step 2: Invoke async LLM callback
        try:
            agent_response = await llm_callback(user_input, retrieved_memories)
            if not isinstance(agent_response, str):
                raise ValueError("LLM callback must return a string response")
            logger.info("LLM callback generated response")
        except Exception as e:
            logger.error("LLM callback failed: %s", e)
            raise

        # Step 3: Save the conversation turn
        event = self._save_conversation_turn(
            actor_id, session_id, user_input, agent_response, metadata, event_timestamp
        )
        return retrieved_memories, agent_response, event

    def _retrieve_memories_for_llm(
        self,
        actor_id: str,
        session_id: str,
        user_input: str,
        retrieval_config: Optional[Dict[str, RetrievalConfig]],
    ) -> List[Dict[str, Any]]:
        """Helper method to retrieve memories for LLM context."""
        retrieved_memories = []
        if retrieval_config:
            for namespace, config in retrieval_config.items():
                resolved_namespace = namespace.format(
                    actorId=actor_id,
                    sessionId=session_id,
                    strategyId=config.strategy_id or "",
                )
                search_query = f"{config.retrieval_query} {user_input}" if config.retrieval_query else user_input
                memory_records = self.search_long_term_memories(
                    query=search_query, namespace_prefix=resolved_namespace, top_k=config.top_k
                )
                # Filter memory records with a relevance score which is lower than config.relevance_score
                if config.relevance_score:
                    memory_records = [
                        record
                        for record in memory_records
                        if record.get("relevanceScore", config.relevance_score) >= config.relevance_score
                    ]
                retrieved_memories.extend(memory_records)

        logger.info("Retrieved %d memories for LLM context", len(retrieved_memories))
        return retrieved_memories

    def _save_conversation_turn(
        self,
        actor_id: str,
        session_id: str,
        user_input: str,
        agent_response: str,
        metadata: Optional[Dict[str, MetadataValue]],
        event_timestamp: Optional[datetime],
    ) -> Dict[str, Any]:
        """Helper method to save conversation turn."""
        event = self.add_turns(
            actor_id=actor_id,
            session_id=session_id,
            messages=[
                ConversationalMessage(user_input, MessageRole.USER),
                ConversationalMessage(agent_response, MessageRole.ASSISTANT),
            ],
            metadata=metadata,
            event_timestamp=event_timestamp,
        )
        logger.info("Completed full conversation turn with LLM")
        return event

    def add_turns(
        self,
        actor_id: str,
        session_id: str,
        messages: List[Union[ConversationalMessage, BlobMessage]],
        branch: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, MetadataValue]] = None,
        event_timestamp: Optional[datetime] = None,
    ) -> Event:
        """Adds conversational turns or blob objects to short-term memory.

        Maps to: bedrock-agentcore.create_event

        Args:
            actor_id: Actor identifier
            session_id: Session identifier
            messages: List of either:
                - ConversationalMessage objects for conversational messages
                - BlobMessage objects for blob data
            branch: Optional branch info
            metadata: Optional custom key-value metadata to attach to an event.
            event_timestamp: Optional timestamp for the event

        Returns:
            Created event

        Example:
        ```
            manager.add_turns(
                actor_id="user-123",
                session_id="session-456",
                messages=[
                    ConversationalMessage("Hello", USER),
                    BlobMessage({"file_data": "base64_content"}),
                    ConversationalMessage("How can I help?", ASSISTANT)
                ],
                metadata=[
                    {
                        'location': {
                            'stringValue': 'NYC'
                        }
                    }
                ]
            )
        ```
        """
        logger.info("  -> Storing %d messages in short-term memory...", len(messages))

        if not messages:
            raise ValueError("At least one message is required")

        payload = []
        for message in messages:
            if isinstance(message, ConversationalMessage):
                # Handle ConversationalMessage data class
                payload.append({"conversational": {"content": {"text": message.text}, "role": message.role.value}})

            elif isinstance(message, BlobMessage):
                # Handle BlobMessage data class
                payload.append({"blob": message.data})
            else:
                raise ValueError("Invalid message format. Must be ConversationalMessage or BlobMessage")

        # Use provided timestamp or current time
        if event_timestamp is None:
            event_timestamp = datetime.now(timezone.utc)

        params = {
            "memoryId": self._memory_id,
            "actorId": actor_id,
            "sessionId": session_id,
            "eventTimestamp": event_timestamp,
            "payload": payload,
        }

        if branch:
            params["branch"] = branch

        if metadata:
            params["metadata"] = metadata

        try:
            response = self._data_plane_client.create_event(**params)
            logger.info("     ✅ Turn stored successfully with Event ID: %s", response.get("eventId"))
            return Event(response["event"])
        except ClientError as e:
            logger.error("     ❌ Error storing turn: %s", e)
            raise

    def fork_conversation(
        self,
        actor_id: str,
        session_id: str,
        root_event_id: str,
        branch_name: str,
        messages: List[Union[ConversationalMessage, BlobMessage]],
        metadata: Optional[Dict[str, MetadataValue]] = None,
        event_timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Fork a conversation from a specific event to create a new branch."""
        try:
            branch = {"rootEventId": root_event_id, "name": branch_name}

            event = self.add_turns(
                actor_id=actor_id,
                session_id=session_id,
                messages=messages,
                event_timestamp=event_timestamp,
                branch=branch,
                metadata=metadata,
            )

            logger.info("Created branch '%s' from event %s", branch_name, root_event_id)
            return event

        except ClientError as e:
            logger.error("Failed to fork conversation: %s", e)
            raise

    def list_events(
        self,
        actor_id: str,
        session_id: str,
        branch_name: Optional[str] = None,
        include_parent_branches: bool = False,
        eventMetadata: Optional[List[EventMetadataFilter]] = None,
        max_results: int = 100,
        include_payload: bool = True,
    ) -> List[Event]:
        """List all events in a session with pagination support.

        This method provides direct access to the raw events API, allowing developers
        to retrieve all events without the turn grouping logic of get_last_k_turns.

        Args:
            actor_id: Actor identifier
            session_id: Session identifier
            branch_name: Optional branch name to filter events (None for all branches)
            include_parent_branches: Whether to include parent branch events (only applies with branch_name)
            eventMetadata: Optional list of event metadata filters to apply
            max_results: Maximum number of events to return
            include_payload: Whether to include event payloads in response

        Returns:
            List of event dictionaries in chronological order

        Example:
            # Get all events
            events = client.list_events(actor_id, session_id)

            # Get only main branch events
            main_events = client.list_events(actor_id, session_id, branch_name="main")

            # Get events from a specific branch
            branch_events = client.list_events(actor_id, session_id, branch_name="test-branch")

            #### Get events with event metadata filter
            ```
            filtered_events_with_metadata = client.list_events(
                actor_id=actor_id,
                session_id=session_id,
                eventMetadata=[
                    {
                        'left': {
                            'metadataKey': 'location'
                        },
                        'operator': 'EQUALS_TO',
                        'right': {
                            'metadataValue': {
                                'stringValue': 'NYC'
                            }
                        }
                    }
                ]
            )
            ```

            #### Get events with event metadata filter + specific branch filter
            ```
            branch_with_metadata_filtered_events = client.list_events(
                actor_id=actor_id,
                session_id=session_id,
                branch_name="test-branch",
                eventMetadata=[
                    {
                        'left': {
                            'metadataKey': 'location'
                        },
                        'operator': 'EQUALS_TO',
                        'right': {
                            'metadataValue': {
                                'stringValue': 'NYC'
                            }
                        }
                    }
                ]
            )
            ```
        """
        try:
            all_events: List[Event] = []
            next_token = None
            max_iterations = 1000  # Safety limit to prevent infinite loops

            iteration_count = 0
            while len(all_events) < max_results and iteration_count < max_iterations:
                iteration_count += 1

                params = {
                    "memoryId": self._memory_id,
                    "actorId": actor_id,
                    "sessionId": session_id,
                    "maxResults": min(100, max_results - len(all_events)),
                    "includePayloads": include_payload,
                }

                if next_token:
                    params["nextToken"] = next_token

                # Initialize the filterMap
                filterMap = {}

                # Add branch filter if specified (but not for "main")
                if branch_name and branch_name != "main":
                    filterMap["branch"] = {"name": branch_name, "includeParentBranches": include_parent_branches}

                # Add eventMetadata filter if specified
                if eventMetadata:
                    filterMap["eventMetadata"] = eventMetadata

                if filterMap:
                    params["filter"] = filterMap

                response = self._data_plane_client.list_events(**params)

                events = response.get("events", [])

                # If no events returned, break to prevent infinite loop
                if not events:
                    logger.debug("No more events returned, ending pagination")
                    break

                all_events.extend([Event(event) for event in events])

                next_token = response.get("nextToken")
                if not next_token or len(all_events) >= max_results:
                    break

            if iteration_count >= max_iterations:
                logger.warning("Reached maximum iteration limit (%d) in list_events pagination", max_iterations)

            logger.info("Retrieved total of %d events", len(all_events))
            return all_events[:max_results]

        except ClientError as e:
            logger.error("Failed to list events: %s", e)
            raise

    def list_branches(self, actor_id: str, session_id: str) -> List[Branch]:
        """List all branches in a session.

        This method handles pagination automatically and provides a structured view
        of all conversation branches, which would require complex pagination and
        grouping logic if done with raw boto3 calls.

        Returns:
            List of branch information including name and root event
        """
        try:
            # Get all events - need to handle pagination for complete list
            all_events = []
            next_token = None
            max_iterations = 1000  # Safety limit to prevent infinite loops

            iteration_count = 0
            while iteration_count < max_iterations:
                iteration_count += 1

                params = {"memoryId": self._memory_id, "actorId": actor_id, "sessionId": session_id, "maxResults": 100}

                if next_token:
                    params["nextToken"] = next_token

                response = self._data_plane_client.list_events(**params)
                events = response.get("events", [])

                # If no events returned, break to prevent infinite loop
                if not events:
                    logger.debug("No more events returned, ending pagination in list_branches")
                    break

                all_events.extend(events)

                next_token = response.get("nextToken")
                if not next_token:
                    break

            if iteration_count >= max_iterations:
                logger.warning("Reached maximum iteration limit (%d) in list_branches pagination", max_iterations)

            branches = {}
            main_branch_events = []

            for event in all_events:
                branch_info = event.get("branch")
                if branch_info:
                    branch_name = branch_info["name"]
                    if branch_name not in branches:
                        branches[branch_name] = {
                            "name": branch_name,
                            "rootEventId": branch_info.get("rootEventId"),
                            "firstEventId": event["eventId"],
                            "eventCount": 1,
                            "created": event["eventTimestamp"],
                        }
                    else:
                        branches[branch_name]["eventCount"] += 1
                else:
                    main_branch_events.append(event)

            # Build result list
            result: List[Branch] = []

            # Only add main branch if there are actual events
            if main_branch_events:
                result.append(
                    {
                        "name": "main",
                        "rootEventId": None,
                        "firstEventId": main_branch_events[0]["eventId"],
                        "eventCount": len(main_branch_events),
                        "created": main_branch_events[0]["eventTimestamp"],
                    }
                )

            # Add other branches
            result.extend(list(branches.values()))

            logger.info("Found %d branches in session %s", len(result), session_id)
            return [Branch(branch) for branch in result]

        except ClientError as e:
            logger.error("Failed to list branches: %s", e)
            raise

    def get_last_k_turns(
        self,
        actor_id: str,
        session_id: str,
        k: int = 5,
        branch_name: Optional[str] = None,
        include_parent_branches: bool = False,
        max_results: int = 100,
    ) -> List[List[EventMessage]]:
        """Get the last K conversation turns.

        A "turn" typically consists of a user message followed by assistant response(s).
        This method groups messages into logical turns for easier processing.

        Returns:
            List of turns, where each turn is a list of message dictionaries
        """
        try:
            events = self.list_events(
                actor_id=actor_id,
                session_id=session_id,
                branch_name=branch_name,
                include_parent_branches=include_parent_branches,
                max_results=max_results,
            )

            if not events:
                return []

            # Process events to group into turns
            turns = []
            current_turn = []

            for event in events:
                if len(turns) >= k:
                    break  # Only need last K turns
                for payload_item in event.get("payload", []):
                    if "conversational" in payload_item:
                        role = payload_item["conversational"].get("role")

                        # Start new turn on USER message
                        if role == MessageRole.USER.value and current_turn:
                            turns.append(current_turn)
                            current_turn = []

                        current_turn.append(EventMessage(payload_item["conversational"]))

            # Don't forget the last turn
            if current_turn:
                turns.append(current_turn)

            # Return the last k turns
            return turns[:k] if len(turns) > k else turns

        except ClientError as e:
            logger.error("Failed to get last K turns: %s", e)
            raise

    def get_event(self, actor_id: str, session_id: str, event_id: str) -> Event:
        """Retrieves a specific event from short-term memory by its ID.

        Maps to: bedrock-agentcore.get_event.
        """
        logger.info("  -> Retrieving event by ID: %s...", event_id)
        try:
            response = self._data_plane_client.get_event(
                memoryId=self._memory_id, actorId=actor_id, sessionId=session_id, eventId=event_id
            )
            logger.info("     ✅ Event retrieved.")
            return Event(response.get("event", {}))
        except ClientError as e:
            logger.error("     ❌ Error retrieving event: %s", e)
            raise

    def delete_event(self, actor_id: str, session_id: str, event_id: str):
        """Deletes a specific event from short-term memory by its ID.

        Maps to: bedrock-agentcore.delete_event.
        """
        logger.info("  -> Deleting event by ID: %s...", event_id)
        try:
            self._data_plane_client.delete_event(
                memoryId=self._memory_id, actorId=actor_id, sessionId=session_id, eventId=event_id
            )
            logger.info("     ✅ Event deleted successfully.")
        except ClientError as e:
            logger.error("     ❌ Error deleting event: %s", e)
            raise

    def search_long_term_memories(
        self,
        query: str,
        namespace_prefix: str,
        top_k: int = 3,
        strategy_id: str = None,
        max_results: int = 20,
    ) -> List[MemoryRecord]:
        """Performs a semantic search against the long-term memory for this actor.

        Maps to: bedrock-agentcore.retrieve_memory_records.
        """
        logger.info("  -> Querying long-term memory in namespace '%s' with query: '%s'...", namespace_prefix, query)
        search_criteria = {"searchQuery": query, "topK": top_k}
        if strategy_id:
            search_criteria["strategyId"] = strategy_id

        namespace = namespace_prefix
        params = {
            "memoryId": self._memory_id,
            "searchCriteria": search_criteria,
            "namespace": namespace,
            "maxResults": max_results,
        }

        try:
            response = self._data_plane_client.retrieve_memory_records(**params)
            records = response.get("memoryRecordSummaries", [])
            logger.info("     ✅ Found %d relevant long-term records.", len(records))
            return [MemoryRecord(record) for record in records]
        except ClientError as e:
            logger.info("     ❌ Error querying long-term memory: %s", e)
            raise

    def list_long_term_memory_records(
        self, namespace_prefix: str, strategy_id: Optional[str] = None, max_results: int = 10
    ) -> List[MemoryRecord]:
        """Lists all long-term memory records for this actor without a semantic query.

        Maps to: bedrock-agentcore.list_memory_records.
        """
        logger.info("  -> Listing all long-term records in namespace '%s'...", namespace_prefix)

        try:
            paginator = self._data_plane_client.get_paginator("list_memory_records")

            params = {
                "memoryId": self._memory_id,
                "namespace": namespace_prefix,
            }

            if strategy_id:
                params["memoryStrategyId"] = strategy_id

            pages = paginator.paginate(**params)
            all_records: List[MemoryRecord] = []

            for page in pages:
                memory_records = page.get("memoryRecords", [])
                # Also check for memoryRecordSummaries (which is what the API actually returns)
                if not memory_records:
                    memory_records = page.get("memoryRecordSummaries", [])

                all_records.extend([MemoryRecord(record) for record in memory_records])

                # Stop if we've reached max_results
                if len(all_records) >= max_results:
                    break

            logger.info("     ✅ Found a total of %d long-term records.", len(all_records))
            return all_records[:max_results]

        except ClientError as e:
            logger.error("     ❌ Error listing long-term records: %s", e)
            raise

    def list_actors(self) -> List[ActorSummary]:
        """Lists all actors who have events in a specific memory.

        Maps to: bedrock-agentcore.list_actors.
        """
        logger.info("👥 Listing all actors for memory %s...", self._memory_id)
        try:
            paginator = self._data_plane_client.get_paginator("list_actors")
            pages = paginator.paginate(memoryId=self._memory_id)
            all_actors = []
            for page in pages:
                actor_summaries = page.get("actorSummaries", [])
                all_actors.extend([ActorSummary(actor) for actor in actor_summaries])
            logger.info("  ✅ Found %d actors.", len(all_actors))
            return all_actors
        except ClientError as e:
            logger.error("  ❌ Error listing actors: %s", e)
            raise

    def get_memory_record(self, record_id: str) -> MemoryRecord:
        """Retrieves a specific long-term memory record by its ID.

        Maps to: bedrock-agentcore.get_memory_record.
        """
        logger.info("📄 Retrieving long-term record by ID: %s from memory %s...", record_id, self._memory_id)
        try:
            response = self._data_plane_client.get_memory_record(memoryId=self._memory_id, memoryRecordId=record_id)
            logger.info("  ✅ Record retrieved.")
            memory_record = response.get("memoryRecord", {})
            return MemoryRecord(memory_record)
        except ClientError as e:
            logger.error("  ❌ Error retrieving record: %s", e)
            raise

    def delete_memory_record(self, record_id: str):
        """Deletes a specific long-term memory record by its ID.

        Maps to: bedrock-agentcore.delete_memory_record.
        """
        logger.info("🗑️ Deleting long-term record by ID: %s from memory %s...", record_id, self._memory_id)
        try:
            self._data_plane_client.delete_memory_record(memoryId=self._memory_id, memoryRecordId=record_id)
            logger.info("  ✅ Record deleted successfully.")
        except ClientError as e:
            logger.error("  ❌ Error deleting record: %s", e)
            raise

    def list_actor_sessions(self, actor_id: str) -> List[SessionSummary]:
        """Lists all sessions for a specific actor in a specific memory.

        Maps to: bedrock-agentcore.list_sessions.
        """
        logger.info("🗂️ Listing all sessions for actor '%s' in memory %s...", actor_id, self._memory_id)
        try:
            paginator = self._data_plane_client.get_paginator("list_sessions")
            pages = paginator.paginate(memoryId=self._memory_id, actorId=actor_id)
            all_sessions: List[SessionSummary] = []
            for page in pages:
                response = page.get("sessionSummaries", [])
                all_sessions.extend([SessionSummary(session) for session in response])
            logger.info("  ✅ Found %d sessions.", len(all_sessions))
            return all_sessions
        except ClientError as e:
            logger.error("  ❌ Error listing sessions: %s", e)
            raise

    def delete_all_long_term_memories_in_namespace(self, namespace: str) -> Dict[str, Any]:
        """Delete all long-term memory records within a specific namespace.

        This method retrieves all memory records in the specified namespace and performs
        batch deletion operations using the AWS Bedrock AgentCore API, processing in chunks of 100.

        Args:
            namespace: The namespace prefix to delete memories from

        Returns:
            Dictionary containing batch deletion results with successfulRecords and failedRecords
        """
        logger.info("🗑️ Deleting all long-term memories in namespace '%s'...", namespace)

        # Retrieve all memory records in the specified namespace
        memory_records = self.list_long_term_memory_records(namespace_prefix=namespace)
        logger.info("  -> Found %d memory records to delete", len(memory_records))

        if not memory_records:
            logger.info("  ✅ No records found to delete")
            return {"successfulRecords": [], "failedRecords": []}

        # Format record IDs for batch deletion API
        memory_record_ids = [{"memoryRecordId": record["memoryRecordId"]} for record in memory_records]

        all_successful = []
        all_failed = []

        # Process in chunks of 100
        for i in range(0, len(memory_record_ids), 100):
            chunk = memory_record_ids[i : i + 100]
            try:
                result = self._data_plane_client.batch_delete_memory_records(memoryId=self._memory_id, records=chunk)
                all_successful.extend(result.get("successfulRecords", []))
                all_failed.extend(result.get("failedRecords", []))
            except ClientError as e:
                logger.error("  ❌ Error deleting chunk: %s", e)
                raise

        logger.info("  ✅ Successfully deleted %d records", len(all_successful))
        if all_failed:
            logger.warning("  ⚠️ Failed to delete %d records", len(all_failed))

        return {"successfulRecords": all_successful, "failedRecords": all_failed}

    def create_memory_session(self, actor_id: str, session_id: str = None) -> "MemorySession":
        """Creates a new MemorySession instance."""
        session_id = session_id or str(uuid.uuid4())
        logger.info("💬 Creating new conversation for actor '%s' in session '%s'...", actor_id, session_id)
        return MemorySession(memory_id=self._memory_id, actor_id=actor_id, session_id=session_id, manager=self)


class MemorySession(DictWrapper):
    """Represents a single, AgentCore MemorySession resource.

    This class provides convenient delegation to MemorySessionManager operations.
    """

    def __init__(self, memory_id: str, actor_id: str, session_id: str, manager: MemorySessionManager):
        """Initialize a MemorySession instance.

        Args:
            memory_id: The memory identifier for this session.
            actor_id: The actor identifier for this session.
            session_id: The session identifier.
            manager: The MemorySessionManager instance to delegate operations to.
        """
        self._memory_id = memory_id
        self._actor_id = actor_id
        self._session_id = session_id
        self._manager = manager
        super().__init__(self._construct_session_dict())

    def _construct_session_dict(self) -> Dict[str, Any]:
        """Constructs a dictionary representing the session."""
        return {"memoryId": self._memory_id, "actorId": self._actor_id, "sessionId": self._session_id}

    def add_turns(
        self,
        messages: List[Union[ConversationalMessage, BlobMessage]],
        branch: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, MetadataValue]] = None,
        event_timestamp: Optional[datetime] = None,
    ) -> Event:
        """Delegates to manager.add_turns."""
        return self._manager.add_turns(self._actor_id, self._session_id, messages, branch, metadata, event_timestamp)

    def fork_conversation(
        self,
        messages: List[Union[ConversationalMessage, BlobMessage]],
        root_event_id: str,
        branch_name: str,
        metadata: Optional[Dict[str, MetadataValue]] = None,
        event_timestamp: Optional[datetime] = None,
    ) -> Event:
        """Delegates to manager.fork_conversation."""
        return self._manager.fork_conversation(
            self._actor_id, self._session_id, root_event_id, branch_name, messages, metadata, event_timestamp
        )

    def process_turn_with_llm(
        self,
        user_input: str,
        llm_callback: Callable[[str, List[Dict[str, Any]]], str],
        retrieval_config: Optional[Dict[str, RetrievalConfig]],
        metadata: Optional[Dict[str, MetadataValue]] = None,
        event_timestamp: Optional[datetime] = None,
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """Delegates to manager.process_turn_with_llm."""
        return self._manager.process_turn_with_llm(
            self._actor_id,
            self._session_id,
            user_input,
            llm_callback,
            retrieval_config,
            metadata,
            event_timestamp,
        )

    async def process_turn_with_llm_async(
        self,
        user_input: str,
        llm_callback: Callable[[str, List[Dict[str, Any]]], Awaitable[str]],
        retrieval_config: Optional[Dict[str, RetrievalConfig]],
        metadata: Optional[Dict[str, MetadataValue]] = None,
        event_timestamp: Optional[datetime] = None,
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """Delegates to manager.process_turn_with_llm_async."""
        return await self._manager.process_turn_with_llm_async(
            self._actor_id,
            self._session_id,
            user_input,
            llm_callback,
            retrieval_config,
            metadata,
            event_timestamp,
        )

    def get_last_k_turns(
        self,
        k: int = 5,
        branch_name: Optional[str] = None,
        include_parent_branches: Optional[bool] = None,
        max_results: int = 100,
    ) -> List[List[EventMessage]]:
        """Delegates to manager.get_last_k_turns."""
        return self._manager.get_last_k_turns(
            self._actor_id, self._session_id, k, branch_name, include_parent_branches, max_results
        )

    def get_event(self, event_id: str) -> Event:
        """Delegates to manager.get_event."""
        return self._manager.get_event(self._actor_id, self._session_id, event_id)

    def delete_event(self, event_id: str):
        """Delegates to manager.delete_event."""
        return self._manager.delete_event(self._actor_id, self._session_id, event_id)

    def get_memory_record(self, record_id: str) -> MemoryRecord:
        """Delegates to manager.get_memory_record."""
        return self._manager.get_memory_record(record_id)

    def delete_memory_record(self, record_id: str):
        """Delegates to manager.delete_memory_record."""
        return self._manager.delete_memory_record(record_id)

    def search_long_term_memories(
        self,
        query: str,
        namespace_prefix: str,
        top_k: int = 3,
        strategy_id: Optional[str] = None,
        max_results: int = 20,
    ) -> List[MemoryRecord]:
        """Delegates to manager.search_long_term_memories."""
        return self._manager.search_long_term_memories(query, namespace_prefix, top_k, strategy_id, max_results)

    def list_long_term_memory_records(
        self, namespace_prefix: str, strategy_id: Optional[str] = None, max_results: int = 10
    ) -> List[MemoryRecord]:
        """Delegates to manager.list_long_term_memory_records."""
        return self._manager.list_long_term_memory_records(namespace_prefix, strategy_id, max_results)

    def list_actors(self) -> List[ActorSummary]:
        """Delegates to manager.list_actors."""
        return self._manager.list_actors()

    def list_events(
        self,
        branch_name: Optional[str] = None,
        include_parent_branches: bool = False,
        eventMetadata: Optional[List[EventMetadataFilter]] = None,
        max_results: int = 100,
        include_payload: bool = True,
    ) -> List[Event]:
        """Delegates to manager.list_events."""
        return self._manager.list_events(
            actor_id=self._actor_id,
            session_id=self._session_id,
            branch_name=branch_name,
            include_parent_branches=include_parent_branches,
            eventMetadata=eventMetadata,
            include_payload=include_payload,
            max_results=max_results,
        )

    def list_branches(self) -> List[Branch]:
        """Delegates to manager.list_branches."""
        return self._manager.list_branches(self._actor_id, self._session_id)

    def get_actor(self) -> "Actor":
        """Returns an Actor instance for this conversation's actor."""
        return Actor(self._actor_id, self._manager)


class Actor(DictWrapper):
    """Represents an actor within a session."""

    def __init__(self, actor_id: str, session_manager: MemorySessionManager):
        """Represents an actor within a session.

        :param actor_id: id of the actor
        :param session_manager: Behaviour manager for the operations
        """
        self._id = actor_id
        self._session_manager = session_manager
        super().__init__(self._construct_session_dict())

    def _construct_session_dict(self) -> Dict[str, Any]:
        """Constructs a dictionary representing the actor."""
        return {
            "actorId": self._id,
        }

    def list_sessions(self) -> List[SessionSummary]:
        """Delegates to _session_manager.list_actor_sessions."""
        return self._session_manager.list_actor_sessions(self._id)
