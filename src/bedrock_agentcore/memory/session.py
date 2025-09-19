"""Module containing session management classes for AgentCore Memory interactions."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import boto3
from botocore.exceptions import ClientError

from .constants import BlobMessage, ConversationalMessage, MessageRole
from .models import (
    ActorSummary,
    Branch,
    BranchContext,
    ConversationTree,
    DictWrapper,
    Event,
    EventMessage,
    MemoryRecord,
    SessionSummary,
)

logger = logging.getLogger(__name__)


class SessionManager:
    """Represents a stateful conversation with a specific user (actor) in a session.

    This class encapsulates all context and handles DATA PLANE interactions.
    """

    def __init__(self, memory_id: str, region: str):
        """Initialize a SessionManager instance.

        Args:
            memory_id: The memory identifier for this session manager.
            region: AWS region for the bedrock-agentcore client.
        """
        self._memory_id = memory_id
        self._data_plane_client = boto3.client("bedrock-agentcore", region_name=region)

        # AgentCore Memory data plane methods
        self._ALLOWED_DATA_PLANE_METHODS = {
            "retrieve_memory_records",
            "get_memory_record",
            "delete_memory_record",
            "list_memory_records",
            "create_event",
            "get_event",
            "delete_event",
            "list_events",
        }

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
            manager = SessionManager(region_name="us-east-1")

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

    def _event_sort_key(self, event: Dict[str, Any]) -> Tuple[int, str]:
        """Generate sort key for chronological event ordering with stable tie-breaker.

        Uses the millisecond prefix of eventId to break same-second ties, ensuring
        deterministic ordering even when multiple events share the same eventTimestamp.

        Args:
            event: Event dictionary containing eventId and eventTimestamp

        Returns:
            Tuple of (milliseconds_since_epoch, suffix) for stable sorting
        """
        event_id = event.get("eventId", "")

        # eventId format: "<epoch_ms>#<suffix>"
        head, _, tail = event_id.partition("#")

        try:
            # Extract milliseconds from eventId prefix
            ms = int(head)
        except (ValueError, TypeError):
            # Fallback to timestamp if eventId format is unexpected
            event_timestamp = event.get("eventTimestamp")
            if event_timestamp:
                if isinstance(event_timestamp, datetime):
                    ms = int(event_timestamp.timestamp() * 1000)
                else:
                    # Handle string timestamps (ISO format)
                    try:
                        dt = datetime.fromisoformat(event_timestamp.replace("Z", "+00:00"))
                        ms = int(dt.timestamp() * 1000)
                    except (ValueError, AttributeError):
                        ms = 0
            else:
                ms = 0

        return (ms, tail)

    def process_turn_with_llm(
        self,
        actor_id: str,
        session_id: str,
        user_input: str,
        llm_callback: Callable[[str, List[Dict[str, Any]]], str],
        retrieval_namespace: Optional[str] = None,
        retrieval_query: Optional[str] = None,
        top_k: int = 3,
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
            retrieval_namespace: Namespace to search for memories (optional)
            retrieval_query: Custom search query (defaults to user_input)
            top_k: Number of memories to retrieve
            event_timestamp: Optional timestamp for the event

        Returns:
            Tuple of (retrieved_memories, agent_response, created_event)

        Example:
            def my_llm(user_input: str, memories: List[Dict]) -> str:
                # Format context from memories
                context = "\\n".join([m['content']['text'] for m in memories])

                # Call your LLM (Bedrock, OpenAI, etc.)
                response = bedrock.invoke_model(
                    messages=[
                        {"role": "system", "content": f"Context: {context}"},
                        {"role": "user", "content": user_input}
                    ]
                )
                return response['content']

            memories, response, event = client.process_turn_with_llm(
                actor_id="user-123",
                session_id="session-456",
                user_input="What did we discuss yesterday?",
                llm_callback=my_llm,
                retrieval_namespace="support/facts/{sessionId}"
            )
        """
        # Step 1: Retrieve relevant memories
        retrieved_memories = []
        if retrieval_namespace:
            search_query = retrieval_query or user_input
            retrieved_memories = self.search_long_term_memory(
                query=search_query, namespace_prefix=retrieval_namespace, top_k=top_k
            )
            logger.info("Retrieved %d memories for LLM context", len(retrieved_memories))

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
        event = self.add_turns(
            actor_id=actor_id,
            session_id=session_id,
            messages=[
                ConversationalMessage(user_input, MessageRole.USER),
                ConversationalMessage(agent_response, MessageRole.ASSISTANT),
            ],
            event_timestamp=event_timestamp,
        )

        logger.info("Completed full conversation turn with LLM")
        return retrieved_memories, agent_response, event

    def add_turns(
        self,
        actor_id: str,
        session_id: str,
        messages: List[Union[ConversationalMessage, BlobMessage]],
        branch: Optional[Dict[str, str]] = None,
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
            event_timestamp: Optional timestamp for the event

        Returns:
            Created event

        Example:
            manager.add_turns(
                actor_id="user-123",
                session_id="session-456",
                messages=[
                    ConversationalMessage("Hello", USER),
                    BlobMessage({"file_data": "base64_content"}),
                    ConversationalMessage("How can I help?", ASSISTANT)
                ]
            )
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
        include_parent_events: bool = False,
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
            include_parent_events: Whether to include parent branch events (only applies with branch_name)
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
        """
        try:
            all_events: List[Event] = []
            next_token = None

            while len(all_events) < max_results:
                params = {
                    "memoryId": self._memory_id,
                    "actorId": actor_id,
                    "sessionId": session_id,
                    "maxResults": min(100, max_results - len(all_events)),
                    "includePayloads": include_payload,
                }

                if next_token:
                    params["nextToken"] = next_token

                # Add branch filter if specified (but not for "main")
                if branch_name and branch_name != "main":
                    params["filter"] = {"branch": {"name": branch_name, "includeParentBranches": include_parent_events}}

                response = self._data_plane_client.list_events(**params)

                events = response.get("events", [])
                all_events.extend([Event(event) for event in events])

                next_token = response.get("nextToken")
                if not next_token or len(all_events) >= max_results:
                    break

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

            while True:
                params = {"memoryId": self._memory_id, "actorId": actor_id, "sessionId": session_id, "maxResults": 100}

                if next_token:
                    params["nextToken"] = next_token

                response = self._data_plane_client.list_events(**params)
                all_events.extend(response.get("events", []))

                next_token = response.get("nextToken")
                if not next_token:
                    break

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

    def get_conversation_tree(self, actor_id: str, session_id: str) -> ConversationTree:
        """Get a tree structure of the conversation with all branches.

        This method transforms a flat list of events into a hierarchical tree structure,
        providing visualization-ready data that would be complex to build from raw events.
        It handles:
        - Full pagination to get all events
        - Grouping by branches
        - Message summarization
        - Tree structure building

        Returns:
            Dictionary representing the conversation tree structure
        """
        try:
            # Get all events - need to handle pagination for complete list
            all_events = []
            next_token = None

            while True:
                params = {"memoryId": self._memory_id, "actorId": actor_id, "sessionId": session_id, "maxResults": 100}

                if next_token:
                    params["nextToken"] = next_token

                response = self._data_plane_client.list_events(**params)
                all_events.extend(response.get("events", []))

                next_token = response.get("nextToken")
                if not next_token:
                    break

            # Build tree structure
            tree = {"session_id": session_id, "actor_id": actor_id, "main_branch": {"events": [], "branches": {}}}

            # Group events by branch
            for event in all_events:
                event_summary = {"eventId": event["eventId"], "timestamp": event["eventTimestamp"], "messages": []}

                # Extract message summaries
                if "payload" in event:
                    for payload_item in event.get("payload", []):
                        if "conversational" in payload_item:
                            conv = payload_item["conversational"]
                            event_summary["messages"].append(
                                {"role": conv.get("role"), "text": conv.get("content", {}).get("text", "")[:50] + "..."}
                            )

                branch_info = event.get("branch")
                if branch_info:
                    branch_name = branch_info["name"]
                    root_event = branch_info.get("rootEventId")  # Use .get() to handle missing field

                    if branch_name not in tree["main_branch"]["branches"]:
                        tree["main_branch"]["branches"][branch_name] = {"root_event_id": root_event, "events": []}

                    tree["main_branch"]["branches"][branch_name]["events"].append(event_summary)
                else:
                    tree["main_branch"]["events"].append(event_summary)

            logger.info("Built conversation tree with %d branches", len(tree["main_branch"]["branches"]))
            return ConversationTree(tree)

        except ClientError as e:
            logger.error("Failed to build conversation tree: %s", e)
            raise

    def merge_branch_context(
        self, actor_id: str, session_id: str, branch_name: str, include_parent: bool = True
    ) -> List[BranchContext]:
        """Get all messages from a branch for context building.

        Args:
            actor_id: Actor identifier
            session_id: Session identifier
            branch_name: Branch to get context from
            include_parent: Whether to include parent branch events

        Returns:
            List of all messages in chronological order
        """
        events = self.list_events(
            actor_id=actor_id,
            session_id=session_id,
            branch_name=branch_name,
            include_parent_events=include_parent,
            max_results=100,
        )

        messages = []
        for event in events:
            if "payload" in event:
                for payload_item in event.get("payload", []):
                    if "conversational" in payload_item:
                        conv = payload_item["conversational"]
                        messages.append(
                            {
                                "timestamp": event["eventTimestamp"],
                                "eventId": event["eventId"],
                                "branch": event.get("branch", {}).get("name", "main"),
                                "role": conv.get("role"),
                                "content": conv.get("content", {}).get("text", ""),
                            }
                        )

        # Sort by timestamp
        messages.sort(key=lambda x: x["timestamp"])

        logger.info("Retrieved %d messages from branch '%s'", len(messages), branch_name)
        return [BranchContext(message) for message in messages]

    def get_last_k_turns(
        self,
        actor_id: str,
        sesssion_id: str,
        k: int = 5,
        branch_name: Optional[str] = None,
        include_branches: bool = False,
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
                session_id=sesssion_id,
                branch_name=branch_name,
                include_parent_events=include_branches,
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

    def search_long_term_memory(
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
            logger.info("     ❌ Error querying long-term memory", e)
            raise

    def list_long_term_memory_records(
        self, namespace_prefix: str, strategy_id: str = None, max_results: int = 10
    ) -> List[MemoryRecord]:
        """Lists all long-term memory records for this actor without a semantic query.

        Maps to: bedrock-agentcore.list_memory_records.
        """
        logger.info("  -> Listing all long-term records in namespace '%s'...", namespace_prefix)
        params = {
            "memoryId": self._memory_id,
            "namespace": namespace_prefix,
            "maxResults": max_results,
        }

        if strategy_id:
            params["strategyId"] = strategy_id

        try:
            paginator = self._data_plane_client.get_paginator("list_memory_records")
            pages = paginator.paginate(**params)
            all_records: List[MemoryRecord] = []
            for page in pages:
                memory_records = page.get("memoryRecords", [])
                all_records.extend([MemoryRecord(record) for record in memory_records])
            logger.info("     ✅ Found a total of %d long-term records.", len(all_records))
            return all_records
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

    def create_session(self, actor_id: str, session_id: str = None) -> "Session":
        """Creates a new Session instance."""
        session_id = session_id or str(uuid.uuid4())
        logger.info("💬 Creating new conversation for actor '%s' in session '%s'...", actor_id, session_id)
        return Session(memory_id=self._memory_id, actor_id=actor_id, session_id=session_id, manager=self)


class Session(DictWrapper):
    """Represents a single, AgentCore Session resource.

    This class provides convenient delegation to SessionManager operations.
    """

    def __init__(self, memory_id: str, actor_id: str, session_id: str, manager: SessionManager):
        """Initialize a Session instance.

        Args:
            memory_id: The memory identifier for this session.
            actor_id: The actor identifier for this session.
            session_id: The session identifier.
            manager: The SessionManager instance to delegate operations to.
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
        branch_name: Optional[Dict[str, str]] = None,
        event_timestamp: Optional[datetime] = None,
    ) -> Event:
        """Delegates to manager.add_turns."""
        return self._manager.add_turns(self._actor_id, self._session_id, messages, event_timestamp, branch_name)

    def fork_conversation(
        self,
        messages: List[Union[ConversationalMessage, BlobMessage]],
        root_event_id: str,
        branch_name: str,
        event_timestamp: Optional[datetime] = None,
    ) -> Event:
        """Delegates to manager.fork_conversation."""
        return self._manager.fork_conversation(
            self._actor_id, self._session_id, root_event_id, branch_name, messages, event_timestamp
        )

    def process_turn_with_llm(
        self,
        user_input: str,
        llm_callback: Callable[[str, List[Dict[str, Any]]], str],
        retrieval_namespace: Optional[str] = None,
        retrieval_query: Optional[str] = None,
        top_k: int = 3,
        event_timestamp: Optional[datetime] = None,
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """Delegates to manager.process_turn_with_llm."""
        return self._manager.process_turn_with_llm(
            self._actor_id,
            self._session_id,
            user_input,
            llm_callback,
            retrieval_namespace,
            retrieval_query,
            top_k,
            event_timestamp,
        )

    def get_last_k_turns(
        self, k: int = 5, branch_name: Optional[str] = None, max_results: int = 100
    ) -> List[List[EventMessage]]:
        """Delegates to manager.get_last_k_turns."""
        return self._manager.get_last_k_turns(self._actor_id, self._session_id, k, branch_name, max_results=max_results)

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

    def search_long_term_memory(
        self, query: str, namespace_prefix: str, top_k: int = 3, strategy_id: str = None, max_results: int = 20
    ) -> List[MemoryRecord]:
        """Delegates to manager.search_long_term_memory."""
        return self._manager.search_long_term_memory(query, namespace_prefix, top_k, strategy_id, max_results)

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
        include_parent_events: bool = False,
        max_results: int = 100,
        include_payload: bool = True,
    ) -> List[Event]:
        """Delegates to manager.list_events."""
        return self._manager.list_events(
            actor_id=self._actor_id,
            session_id=self._session_id,
            branch_name=branch_name,
            include_parent_events=include_parent_events,
            include_payload=include_payload,
            max_results=max_results,
        )

    def list_branches(self) -> List[Branch]:
        """Delegates to manager.list_branches."""
        return self._manager.list_branches(self._actor_id, self._session_id)

    def get_conversation_tree(self) -> ConversationTree:
        """Delegates to manager.get_conversation_tree."""
        return self._manager.get_conversation_tree(self._actor_id, self._session_id)

    def merge_branch_context(self, branch_name: str, include_parent: bool = True) -> List[BranchContext]:
        """Delegates to manager.merge_branch_context."""
        return self._manager.merge_branch_context(self._actor_id, self._session_id, branch_name, include_parent)

    def get_actor(self) -> "Actor":
        """Returns an Actor instance for this conversation's actor."""
        return Actor(self._actor_id, self._manager)


class Actor(DictWrapper):
    """Represents an actor within a session."""

    def __init__(self, actor_id: str, session_manager: SessionManager):
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

    def list_sessions(self) -> List[Session]:
        """Delegates to _session_manager.list_actor_sessions."""
        return self._session_manager.list_actor_sessions(self._id)
