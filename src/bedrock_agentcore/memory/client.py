"""AgentCore Memory SDK - High-level client for memory operations.

This SDK handles the asymmetric API where:
- Input parameters use old field names (memoryStrategies, memoryStrategyId, etc.)
- Output responses use new field names (strategies, strategyId, etc.)

The SDK automatically normalizes responses to provide both field names for
backward compatibility.
"""

import copy
import logging
import time
import uuid
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

from .constants import (
    CUSTOM_CONSOLIDATION_WRAPPER_KEYS,
    CUSTOM_EXTRACTION_WRAPPER_KEYS,
    CUSTOM_REFLECTION_WRAPPER_KEYS,
    DEFAULT_NAMESPACES,
    EXTRACTION_WRAPPER_KEYS,
    MemoryStatus,
    MemoryStrategyTypeEnum,
    MessageRole,
    OverrideType,
    Role,
    StrategyType,
)

logger = logging.getLogger(__name__)


class MemoryClient:
    """High-level Bedrock AgentCore Memory client with essential operations."""

    # AgentCore Memory data plane methods
    _ALLOWED_GMDP_METHODS = {
        "retrieve_memory_records",
        "get_memory_record",
        "delete_memory_record",
        "list_memory_records",
        "create_event",
        "get_event",
        "delete_event",
        "list_events",
    }

    # AgentCore Memory control plane methods
    _ALLOWED_GMCP_METHODS = {
        "create_memory",
        "get_memory",
        "list_memories",
        "update_memory",
        "delete_memory",
        "list_memory_strategies",
    }

    def __init__(self, region_name: Optional[str] = None):
        """Initialize the Memory client."""
        self.region_name = region_name or boto3.Session().region_name or "us-west-2"

        self.gmcp_client = boto3.client("bedrock-agentcore-control", region_name=self.region_name)
        self.gmdp_client = boto3.client("bedrock-agentcore", region_name=self.region_name)

        logger.info(
            "Initialized MemoryClient for control plane: %s, data plane: %s",
            self.gmcp_client.meta.region_name,
            self.gmdp_client.meta.region_name,
        )

    def __getattr__(self, name: str):
        """Dynamically forward method calls to the appropriate boto3 client.

        This method enables access to all boto3 client methods without explicitly
        defining them. Methods are looked up in the following order:
        1. gmdp_client (bedrock-agentcore) - for data plane operations
        2. gmcp_client (bedrock-agentcore-control) - for control plane operations

        Args:
            name: The method name being accessed

        Returns:
            A callable method from the appropriate boto3 client

        Raises:
            AttributeError: If the method doesn't exist on either client

        Example:
            # Access any boto3 method directly
            client = MemoryClient()

            # These calls are forwarded to the appropriate boto3 client
            response = client.list_memory_records(memoryId="mem-123", namespace="test")
            metadata = client.get_memory_metadata(memoryId="mem-123")
        """
        if name in self._ALLOWED_GMDP_METHODS and hasattr(self.gmdp_client, name):
            method = getattr(self.gmdp_client, name)
            logger.debug("Forwarding method '%s' to gmdp_client", name)
            return method

        if name in self._ALLOWED_GMCP_METHODS and hasattr(self.gmcp_client, name):
            method = getattr(self.gmcp_client, name)
            logger.debug("Forwarding method '%s' to gmcp_client", name)
            return method

        # Method not found on either client
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
            f"Method not found on gmdp_client or gmcp_client. "
            f"Available methods can be found in the boto3 documentation for "
            f"'bedrock-agentcore' and 'bedrock-agentcore-control' services."
        )

    def create_memory(
        self,
        name: str,
        strategies: Optional[List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        event_expiry_days: int = 90,
        memory_execution_role_arn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a memory with simplified configuration."""
        if strategies is None:
            strategies = []

        try:
            processed_strategies = self._add_default_namespaces(strategies)

            params = {
                "name": name,
                "eventExpiryDuration": event_expiry_days,
                "memoryStrategies": processed_strategies,  # Using old field name for input
                "clientToken": str(uuid.uuid4()),
            }

            if description is not None:
                params["description"] = description

            if memory_execution_role_arn is not None:
                params["memoryExecutionRoleArn"] = memory_execution_role_arn

            response = self.gmcp_client.create_memory(**params)

            memory = response["memory"]
            # Normalize response to handle new field names
            memory = self._normalize_memory_response(memory)

            logger.info("Created memory: %s", memory["memoryId"])
            return memory

        except ClientError as e:
            logger.error("Failed to create memory: %s", e)
            raise

    def create_or_get_memory(
        self,
        name: str,
        strategies: Optional[List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        event_expiry_days: int = 90,
        memory_execution_role_arn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a memory resource or fetch the existing memory details if it already exists.

        Returns:
            Memory object, either newly created or existing
        """
        try:
            memory = self.create_memory_and_wait(
                name=name,
                strategies=strategies,
                description=description,
                event_expiry_days=event_expiry_days,
                memory_execution_role_arn=memory_execution_role_arn,
            )
            return memory
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException" and "already exists" in str(e):
                memories = self.list_memories()
                memory = next((m for m in memories if m["id"].startswith(name)), None)
                logger.info("Memory already exists. Using existing memory ID: %s", memory["id"])
                return memory
            else:
                logger.error("ClientError: Failed to create or get memory: %s", e)
                raise
        except Exception:
            raise

    def create_memory_and_wait(
        self,
        name: str,
        strategies: List[Dict[str, Any]],
        description: Optional[str] = None,
        event_expiry_days: int = 90,
        memory_execution_role_arn: Optional[str] = None,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Create a memory and wait for it to become ACTIVE.

        This method creates a memory and polls until it reaches ACTIVE status,
        providing a convenient way to ensure the memory is ready for use.

        Args:
            name: Name for the memory resource
            strategies: List of strategy configurations
            description: Optional description
            event_expiry_days: How long to retain events (default: 90 days)
            memory_execution_role_arn: IAM role ARN for memory execution
            max_wait: Maximum seconds to wait (default: 300)
            poll_interval: Seconds between status checks (default: 10)

        Returns:
            Created memory object in ACTIVE status

        Raises:
            TimeoutError: If memory doesn't become ACTIVE within max_wait
            RuntimeError: If memory creation fails
        """
        # Create the memory
        memory = self.create_memory(
            name=name,
            strategies=strategies,
            description=description,
            event_expiry_days=event_expiry_days,
            memory_execution_role_arn=memory_execution_role_arn,
        )

        memory_id = memory.get("memoryId", memory.get("id"))  # Handle both field names
        if memory_id is None:
            memory_id = ""
        logger.info("Created memory %s, waiting for ACTIVE status...", memory_id)

        start_time = time.time()
        while time.time() - start_time < max_wait:
            elapsed = int(time.time() - start_time)

            try:
                status = self.get_memory_status(memory_id)

                if status == MemoryStatus.ACTIVE.value:
                    logger.info("Memory %s is now ACTIVE (took %d seconds)", memory_id, elapsed)
                    # Get fresh memory details
                    response = self.gmcp_client.get_memory(memoryId=memory_id)  # Input uses old field name
                    memory = self._normalize_memory_response(response["memory"])
                    return memory
                elif status == MemoryStatus.FAILED.value:
                    # Get failure reason if available
                    response = self.gmcp_client.get_memory(memoryId=memory_id)  # Input uses old field name
                    failure_reason = response["memory"].get("failureReason", "Unknown")
                    raise RuntimeError("Memory creation failed: %s" % failure_reason)
                else:
                    logger.debug("Memory status: %s (%d seconds elapsed)", status, elapsed)

            except ClientError as e:
                logger.error("Error checking memory status: %s", e)
                raise

            time.sleep(poll_interval)

        raise TimeoutError("Memory %s did not become ACTIVE within %d seconds" % (memory_id, max_wait))

    def retrieve_memories(
        self, memory_id: str, namespace: str, query: str, actor_id: Optional[str] = None, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from a namespace.

        Note: Wildcards (*) are NOT supported in namespaces. You must provide the
        exact namespace path with all variables resolved.

        Args:
            memory_id: Memory resource ID
            namespace: Exact namespace path (no wildcards)
            query: Search query
            actor_id: Optional actor ID (deprecated, use namespace)
            top_k: Number of results to return

        Returns:
            List of memory records

        Example:
            # Correct - exact namespace
            memories = client.retrieve_memories(
                memory_id="mem-123",
                namespace="support/facts/session-456",
                query="customer preferences"
            )

            # Incorrect - wildcards not supported
            # memories = client.retrieve_memories(..., namespace="support/facts/*", ...)
        """
        if "*" in namespace:
            logger.error("Wildcards are not supported in namespaces. Please provide exact namespace.")
            return []

        try:
            # Let service handle all namespace validation
            response = self.gmdp_client.retrieve_memory_records(
                memoryId=memory_id, namespace=namespace, searchCriteria={"searchQuery": query, "topK": top_k}
            )

            memories = response.get("memoryRecordSummaries", [])
            logger.info("Retrieved %d memories from namespace: %s", len(memories), namespace)
            return memories

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_msg = e.response["Error"]["Message"]

            if error_code == "ResourceNotFoundException":
                logger.warning(
                    "Memory or namespace not found. Ensure memory %s exists and namespace '%s' is configured",
                    memory_id,
                    namespace,
                )
            elif error_code == "ValidationException":
                logger.warning("Invalid search parameters: %s", error_msg)
            elif error_code == "ServiceException":
                logger.warning("Service error: %s. This may be temporary - try again later", error_msg)
            else:
                logger.warning("Memory retrieval failed (%s): %s", error_code, error_msg)

            return []

    def create_event(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        messages: List[Tuple[str, str]],
        event_timestamp: Optional[datetime] = None,
        branch: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Save an event of an agent interaction or conversation with a user.

        This is the basis of short-term memory. If you configured your Memory resource
        to have MemoryStrategies, then events that are saved in short-term memory via
        create_event will be used to extract long-term memory records.

        Args:
            memory_id: Memory resource ID
            actor_id: Actor identifier (could be id of your user or an agent)
            session_id: Session identifier (meant to logically group a series of events)
            messages: List of (text, role) tuples. Role can be USER, ASSISTANT, TOOL, etc.
            event_timestamp: timestamp for the entire event (not per message)
            branch: Optional branch info. For new branches: {"rootEventId": "...", "name": "..."}
                   For continuing existing branch: {"name": "..."} or {"name": "...", "rootEventId": "..."}
                   A branch is used when you want to have a different history of events.

        Returns:
            Created event

        Example:
            event = client.create_event(
                memory_id=memory.get("id"),
                actor_id="weatherWorrier",
                session_id="WeatherSession",
                messages=[
                    ("What's the weather?", "USER"),
                    ("Today is sunny", "ASSISTANT")
                ]
            )
            root_event_id = event.get("eventId")
            print(event)

            # Continue the conversation
            event = client.create_event(
                memory_id=memory.get("id"),
                actor_id="weatherWorrier",
                session_id="WeatherSession",
                messages=[
                    ("How about the weather tomorrow", "USER"),
                    ("Tomorrow is cold!", "ASSISTANT")
                ]
            )
            print(event)

            # branch the conversation so that the previous message is not part of the history
            # (suppose you did not mean to ask about the weather tomorrow and want to undo
            # that, and replace with a new message)
            event = client.create_event(
                memory_id=memory.get("id"),
                actor_id="weatherWorrier",
                session_id="WeatherSession",
                branch={"name": "differentWeatherQuestion", "rootEventId": root_event_id},
                messages=[
                    ("How about the weather a year from now", "USER"),
                    ("I can't predict that far into the future!", "ASSISTANT")
                ]
            )
            print(event)
        """
        try:
            if not messages:
                raise ValueError("At least one message is required")

            payload = []
            for msg in messages:
                if len(msg) != 2:
                    raise ValueError("Each message must be (text, role)")

                text, role = msg

                try:
                    role_enum = MessageRole(role.upper())
                except ValueError as err:
                    raise ValueError(
                        "Invalid role '%s'. Must be one of: %s" % (role, ", ".join([r.value for r in MessageRole]))
                    ) from err

                payload.append({"conversational": {"content": {"text": text}, "role": role_enum.value}})

            # Use provided timestamp or current time
            if event_timestamp is None:
                event_timestamp = datetime.utcnow()

            params = {
                "memoryId": memory_id,
                "actorId": actor_id,
                "sessionId": session_id,
                "eventTimestamp": event_timestamp,
                "payload": payload,
            }

            if branch:
                params["branch"] = branch

            response = self.gmdp_client.create_event(**params)

            event = response["event"]
            logger.info("Created event: %s", event["eventId"])

            return event

        except ClientError as e:
            logger.error("Failed to create event: %s", e)
            raise

    def create_blob_event(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        blob_data: Any,
        event_timestamp: Optional[datetime] = None,
        branch: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Save a blob event to AgentCore Memory.

        Args:
            memory_id: Memory resource ID
            actor_id: Actor identifier
            session_id: Session identifier
            blob_data: Binary or structured data to store
            event_timestamp: Optional timestamp for the event
            branch: Optional branch info

        Returns:
            Created event

        Example:
            # Store binary data
            event = client.create_blob_event(
                memory_id="mem-xyz",
                actor_id="user-123",
                session_id="session-456",
                blob_data={"file_content": "base64_encoded_data", "metadata": {"type": "image"}}
            )
        """
        try:
            payload = [{"blob": blob_data}]

            if event_timestamp is None:
                event_timestamp = datetime.utcnow()

            params = {
                "memoryId": memory_id,
                "actorId": actor_id,
                "sessionId": session_id,
                "eventTimestamp": event_timestamp,
                "payload": payload,
            }

            if branch:
                params["branch"] = branch

            response = self.gmdp_client.create_event(**params)

            event = response["event"]
            logger.info("Created blob event: %s", event["eventId"])

            return event

        except ClientError as e:
            logger.error("Failed to create blob event: %s", e)
            raise

    def save_conversation(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        messages: List[Tuple[str, str]],
        event_timestamp: Optional[datetime] = None,
        branch: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """DEPRECATED: Use create_event() instead.

        Args:
            memory_id: Memory resource ID
            actor_id: Actor identifier
            session_id: Session identifier
            messages: List of (text, role) tuples. Role can be USER, ASSISTANT, TOOL, etc.
            event_timestamp: Optional timestamp for the entire event (not per message)
            branch: Optional branch info. For new branches: {"rootEventId": "...", "name": "..."}
                   For continuing existing branch: {"name": "..."} or {"name": "...", "rootEventId": "..."}

        Returns:
            Created event

        Example:
            # Save multi-turn conversation
            event = client.save_conversation(
                memory_id="mem-xyz",
                actor_id="user-123",
                session_id="session-456",
                messages=[
                    ("What's the weather?", "USER"),
                    ("And tomorrow?", "USER"),
                    ("Checking weather...", "TOOL"),
                    ("Today sunny, tomorrow rain", "ASSISTANT")
                ]
            )

            # Continue existing branch (only name required)
            event = client.save_conversation(
                memory_id="mem-xyz",
                actor_id="user-123",
                session_id="session-456",
                messages=[("Continue conversation", "USER")],
                branch={"name": "existing-branch"}
            )
        """
        try:
            if not messages:
                raise ValueError("At least one message is required")

            # Build payload
            payload = []

            for msg in messages:
                if len(msg) != 2:
                    raise ValueError("Each message must be (text, role)")

                text, role = msg

                # Validate role
                try:
                    role_enum = MessageRole(role.upper())
                except ValueError as err:
                    raise ValueError(
                        "Invalid role '%s'. Must be one of: %s" % (role, ", ".join([r.value for r in MessageRole]))
                    ) from err

                payload.append({"conversational": {"content": {"text": text}, "role": role_enum.value}})

            # Use provided timestamp or current time
            if event_timestamp is None:
                event_timestamp = datetime.utcnow()

            params = {
                "memoryId": memory_id,
                "actorId": actor_id,
                "sessionId": session_id,
                "eventTimestamp": event_timestamp,
                "payload": payload,
                "clientToken": str(uuid.uuid4()),
            }

            if branch:
                params["branch"] = branch

            response = self.gmdp_client.create_event(**params)

            event = response["event"]
            logger.info("Created event: %s", event["eventId"])

            return event

        except ClientError as e:
            logger.error("Failed to create event: %s", e)
            raise

    def save_turn(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        user_input: str,
        agent_response: str,
        event_timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """DEPRECATED: Use save_conversation() for more flexibility.

        This method will be removed in v1.0.0.
        """
        warnings.warn(
            "save_turn() is deprecated and will be removed in v1.0.0. "
            "Use save_conversation() for flexible message handling.",
            DeprecationWarning,
            stacklevel=2,
        )

        messages = [(user_input, "USER"), (agent_response, "ASSISTANT")]

        return self.create_event(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=messages,
            event_timestamp=event_timestamp,
        )

    def process_turn(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        user_input: str,
        agent_response: str,
        event_timestamp: Optional[datetime] = None,
        retrieval_namespace: Optional[str] = None,
        retrieval_query: Optional[str] = None,
        top_k: int = 3,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """DEPRECATED: Use retrieve_memories() and save_conversation() separately.

        This method will be removed in v1.0.0.
        """
        warnings.warn(
            "process_turn() is deprecated and will be removed in v1.0.0. "
            "Use retrieve_memories() and save_conversation() separately, or use process_turn_with_llm().",
            DeprecationWarning,
            stacklevel=2,
        )

        retrieved_memories = []

        if retrieval_namespace:
            search_query = retrieval_query or user_input
            retrieved_memories = self.retrieve_memories(
                memory_id=memory_id, namespace=retrieval_namespace, query=search_query, top_k=top_k
            )

        event = self.save_turn(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            user_input=user_input,
            agent_response=agent_response,
            event_timestamp=event_timestamp,
        )

        return retrieved_memories, event

    def process_turn_with_llm(
        self,
        memory_id: str,
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
            memory_id: Memory resource ID
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
                memory_id="mem-xyz",
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
            retrieved_memories = self.retrieve_memories(
                memory_id=memory_id, namespace=retrieval_namespace, query=search_query, top_k=top_k
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
        event = self.create_event(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=[(user_input, "USER"), (agent_response, "ASSISTANT")],
            event_timestamp=event_timestamp,
        )

        logger.info("Completed full conversation turn with LLM")
        return retrieved_memories, agent_response, event

    def list_events(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        branch_name: Optional[str] = None,
        include_parent_branches: bool = False,
        max_results: int = 100,
        include_payload: bool = True,
    ) -> List[Dict[str, Any]]:
        """List all events in a session with pagination support.

        This method provides direct access to the raw events API, allowing developers
        to retrieve all events without the turn grouping logic of get_last_k_turns.

        Args:
            memory_id: Memory resource ID
            actor_id: Actor identifier
            session_id: Session identifier
            branch_name: Optional branch name to filter events (None for all branches)
            include_parent_branches: Whether to include parent branch events (only applies with branch_name)
            max_results: Maximum number of events to return
            include_payload: Whether to include event payloads in response

        Returns:
            List of event dictionaries in chronological order

        Example:
            # Get all events
            events = client.list_events(memory_id, actor_id, session_id)

            # Get only main branch events
            main_events = client.list_events(memory_id, actor_id, session_id, branch_name="main")

            # Get events from a specific branch
            branch_events = client.list_events(memory_id, actor_id, session_id, branch_name="test-branch")
        """
        try:
            all_events = []
            next_token = None

            while len(all_events) < max_results:
                params = {
                    "memoryId": memory_id,
                    "actorId": actor_id,
                    "sessionId": session_id,
                    "maxResults": min(100, max_results - len(all_events)),
                    "includePayloads": include_payload,
                }

                if next_token:
                    params["nextToken"] = next_token

                # Add branch filter if specified (but not for "main")
                if branch_name and branch_name != "main":
                    params["filter"] = {
                        "branch": {"name": branch_name, "includeParentBranches": include_parent_branches}
                    }

                response = self.gmdp_client.list_events(**params)

                events = response.get("events", [])
                all_events.extend(events)

                next_token = response.get("nextToken")
                if not next_token or len(all_events) >= max_results:
                    break

            logger.info("Retrieved total of %d events", len(all_events))
            return all_events[:max_results]

        except ClientError as e:
            logger.error("Failed to list events: %s", e)
            raise

    def list_branches(self, memory_id: str, actor_id: str, session_id: str) -> List[Dict[str, Any]]:
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
                params = {"memoryId": memory_id, "actorId": actor_id, "sessionId": session_id, "maxResults": 100}

                if next_token:
                    params["nextToken"] = next_token

                response = self.gmdp_client.list_events(**params)
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
            result = []

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
            return result

        except ClientError as e:
            logger.error("Failed to list branches: %s", e)
            raise

    def list_branch_events(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        branch_name: Optional[str] = None,
        include_parent_branches: bool = False,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """List events in a specific branch.

        This method provides complex filtering and pagination that would require
        significant boilerplate code with raw boto3. It handles:
        - Automatic pagination across multiple API calls
        - Branch filtering with parent event inclusion logic
        - Main branch isolation (events without branch info)

        Args:
            memory_id: Memory resource ID
            actor_id: Actor identifier
            session_id: Session identifier
            branch_name: Branch name (None for main branch)
            include_parent_branches: Whether to include events from parent branches
            max_results: Maximum events to return

        Returns:
            List of events in the branch
        """
        try:
            params = {
                "memoryId": memory_id,
                "actorId": actor_id,
                "sessionId": session_id,
                "maxResults": min(100, max_results),
            }

            # Only add filter when we have a specific branch name
            if branch_name:
                params["filter"] = {"branch": {"name": branch_name, "includeParentBranches": include_parent_branches}}

            response = self.gmdp_client.list_events(**params)
            events = response.get("events", [])

            # Handle pagination
            next_token = response.get("nextToken")
            while next_token and len(events) < max_results:
                params["nextToken"] = next_token
                params["maxResults"] = min(100, max_results - len(events))
                response = self.gmdp_client.list_events(**params)
                events.extend(response.get("events", []))
                next_token = response.get("nextToken")

            # Filter for main branch if no branch specified
            if not branch_name:
                events = [e for e in events if not e.get("branch")]

            logger.info("Retrieved %d events from branch '%s'", len(events), branch_name or "main")
            return events

        except ClientError as e:
            logger.error("Failed to list branch events: %s", e)
            raise

    def get_conversation_tree(self, memory_id: str, actor_id: str, session_id: str) -> Dict[str, Any]:
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
                params = {"memoryId": memory_id, "actorId": actor_id, "sessionId": session_id, "maxResults": 100}

                if next_token:
                    params["nextToken"] = next_token

                response = self.gmdp_client.list_events(**params)
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
            return tree

        except ClientError as e:
            logger.error("Failed to build conversation tree: %s", e)
            raise

    def merge_branch_context(
        self, memory_id: str, actor_id: str, session_id: str, branch_name: str, include_parent: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all messages from a branch for context building.

        Args:
            memory_id: Memory resource ID
            actor_id: Actor identifier
            session_id: Session identifier
            branch_name: Branch to get context from
            include_parent: Whether to include parent branch events

        Returns:
            List of all messages in chronological order
        """
        events = self.list_branch_events(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            branch_name=branch_name,
            include_parent_branches=include_parent,
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
        return messages

    def get_last_k_turns(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        k: int = 5,
        branch_name: Optional[str] = None,
        include_branches: bool = False,
        max_results: int = 100,
    ) -> List[List[Dict[str, Any]]]:
        """Get the last K conversation turns.

        A "turn" typically consists of a user message followed by assistant response(s).
        This method groups messages into logical turns for easier processing.

        Returns:
            List of turns, where each turn is a list of message dictionaries
        """
        try:
            # Use the new list_events method
            events = self.list_events(
                memory_id=memory_id,
                actor_id=actor_id,
                session_id=session_id,
                branch_name=branch_name,
                include_parent_branches=False,
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
                        if role == Role.USER.value and current_turn:
                            turns.append(current_turn)
                            current_turn = []

                        current_turn.append(payload_item["conversational"])

            # Don't forget the last turn
            if current_turn:
                turns.append(current_turn)

            # Return the last k turns
            return turns[:k] if len(turns) > k else turns

        except ClientError as e:
            logger.error("Failed to get last K turns: %s", e)
            raise

    def fork_conversation(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        root_event_id: str,
        branch_name: str,
        new_messages: List[Tuple[str, str]],
        event_timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Fork a conversation from a specific event to create a new branch."""
        try:
            branch = {"rootEventId": root_event_id, "name": branch_name}

            event = self.create_event(
                memory_id=memory_id,
                actor_id=actor_id,
                session_id=session_id,
                messages=new_messages,
                branch=branch,
                event_timestamp=event_timestamp,
            )

            logger.info("Created branch '%s' from event %s", branch_name, root_event_id)
            return event

        except ClientError as e:
            logger.error("Failed to fork conversation: %s", e)
            raise

    def get_memory_strategies(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get all strategies for a memory."""
        try:
            response = self.gmcp_client.get_memory(memoryId=memory_id)  # Input uses old field name
            memory = response["memory"]

            # Handle both old and new field names in response
            strategies = memory.get("strategies", memory.get("memoryStrategies", []))

            # Normalize strategy fields
            normalized_strategies = []
            for strategy in strategies:
                # Create normalized version with both old and new field names
                normalized = strategy.copy()

                # Ensure both field name versions exist
                if "strategyId" in strategy and "memoryStrategyId" not in normalized:
                    normalized["memoryStrategyId"] = strategy["strategyId"]
                elif "memoryStrategyId" in strategy and "strategyId" not in normalized:
                    normalized["strategyId"] = strategy["memoryStrategyId"]

                if "type" in strategy and "memoryStrategyType" not in normalized:
                    normalized["memoryStrategyType"] = strategy["type"]
                elif "memoryStrategyType" in strategy and "type" not in normalized:
                    normalized["type"] = strategy["memoryStrategyType"]

                normalized_strategies.append(normalized)

            return normalized_strategies
        except ClientError as e:
            logger.error("Failed to get memory strategies: %s", e)
            raise

    def get_memory_status(self, memory_id: str) -> str:
        """Get current memory status."""
        try:
            response = self.gmcp_client.get_memory(memoryId=memory_id)  # Input uses old field name
            return response["memory"]["status"]
        except ClientError as e:
            logger.error("Failed to get memory status: %s", e)
            raise

    def list_memories(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """List all memories for the account."""
        try:
            # Ensure max_results doesn't exceed API limit per request
            results_per_request = min(max_results, 100)

            response = self.gmcp_client.list_memories(maxResults=results_per_request)
            memories = response.get("memories", [])

            next_token = response.get("nextToken")
            while next_token and len(memories) < max_results:
                remaining = max_results - len(memories)
                results_per_request = min(remaining, 100)

                response = self.gmcp_client.list_memories(maxResults=results_per_request, nextToken=next_token)
                memories.extend(response.get("memories", []))
                next_token = response.get("nextToken")

            # Normalize memory summaries if they contain new field names
            normalized_memories = []
            for memory in memories[:max_results]:
                normalized = memory.copy()
                # Ensure both field name versions exist
                if "id" in memory and "memoryId" not in normalized:
                    normalized["memoryId"] = memory["id"]
                elif "memoryId" in memory and "id" not in normalized:
                    normalized["id"] = memory["memoryId"]
                normalized_memories.append(normalized)

            return normalized_memories

        except ClientError as e:
            logger.error("Failed to list memories: %s", e)
            raise

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory resource."""
        try:
            response = self.gmcp_client.delete_memory(
                memoryId=memory_id, clientToken=str(uuid.uuid4())
            )  # Input uses old field name
            logger.info("Deleted memory: %s", memory_id)
            return response
        except ClientError as e:
            logger.error("Failed to delete memory: %s", e)
            raise

    def delete_memory_and_wait(self, memory_id: str, max_wait: int = 300, poll_interval: int = 10) -> Dict[str, Any]:
        """Delete a memory and wait for deletion to complete.

        This method deletes a memory and polls until it's fully deleted,
        ensuring clean resource cleanup.

        Args:
            memory_id: Memory resource ID to delete
            max_wait: Maximum seconds to wait (default: 300)
            poll_interval: Seconds between checks (default: 10)

        Returns:
            Final deletion response

        Raises:
            TimeoutError: If deletion doesn't complete within max_wait
        """
        # Initiate deletion
        response = self.delete_memory(memory_id)
        logger.info("Initiated deletion of memory %s", memory_id)

        start_time = time.time()
        while time.time() - start_time < max_wait:
            elapsed = int(time.time() - start_time)

            try:
                # Try to get the memory - if it doesn't exist, deletion is complete
                self.gmcp_client.get_memory(memoryId=memory_id)  # Input uses old field name
                logger.debug("Memory still exists, waiting... (%d seconds elapsed)", elapsed)

            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    logger.info("Memory %s successfully deleted (took %d seconds)", memory_id, elapsed)
                    return response
                else:
                    logger.error("Error checking memory status: %s", e)
                    raise

            time.sleep(poll_interval)

        raise TimeoutError("Memory %s was not deleted within %d seconds" % (memory_id, max_wait))

    def add_semantic_strategy(
        self,
        memory_id: str,
        name: str,
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add a semantic memory strategy.

        Note: Configuration is no longer provided for built-in strategies as per API changes.
        """
        strategy: Dict = {
            StrategyType.SEMANTIC.value: {
                "name": name,
            }
        }

        if description:
            strategy[StrategyType.SEMANTIC.value]["description"] = description
        if namespaces:
            strategy[StrategyType.SEMANTIC.value]["namespaces"] = namespaces

        return self._add_strategy(memory_id, strategy)

    def add_semantic_strategy_and_wait(
        self,
        memory_id: str,
        name: str,
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Add a semantic strategy and wait for memory to return to ACTIVE state.

        This addresses the issue where adding a strategy puts the memory into
        CREATING state temporarily, preventing subsequent operations.
        """
        # Add the strategy
        self.add_semantic_strategy(memory_id, name, description, namespaces)

        # Wait for memory to return to ACTIVE
        return self._wait_for_memory_active(memory_id, max_wait, poll_interval)

    def add_summary_strategy(
        self,
        memory_id: str,
        name: str,
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add a summary memory strategy.

        Note: Configuration is no longer provided for built-in strategies as per API changes.
        """
        strategy: Dict = {
            StrategyType.SUMMARY.value: {
                "name": name,
            }
        }

        if description:
            strategy[StrategyType.SUMMARY.value]["description"] = description
        if namespaces:
            strategy[StrategyType.SUMMARY.value]["namespaces"] = namespaces

        return self._add_strategy(memory_id, strategy)

    def add_summary_strategy_and_wait(
        self,
        memory_id: str,
        name: str,
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Add a summary strategy and wait for memory to return to ACTIVE state."""
        self.add_summary_strategy(memory_id, name, description, namespaces)
        return self._wait_for_memory_active(memory_id, max_wait, poll_interval)

    def add_user_preference_strategy(
        self,
        memory_id: str,
        name: str,
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add a user preference memory strategy.

        Note: Configuration is no longer provided for built-in strategies as per API changes.
        """
        strategy: Dict = {
            StrategyType.USER_PREFERENCE.value: {
                "name": name,
            }
        }

        if description:
            strategy[StrategyType.USER_PREFERENCE.value]["description"] = description
        if namespaces:
            strategy[StrategyType.USER_PREFERENCE.value]["namespaces"] = namespaces

        return self._add_strategy(memory_id, strategy)

    def add_user_preference_strategy_and_wait(
        self,
        memory_id: str,
        name: str,
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Add a user preference strategy and wait for memory to return to ACTIVE state."""
        self.add_user_preference_strategy(memory_id, name, description, namespaces)
        return self._wait_for_memory_active(memory_id, max_wait, poll_interval)

    def add_episodic_strategy(
        self,
        memory_id: str,
        name: str,
        reflection_namespaces: List[str],
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add an episodic memory strategy.

        Args:
            memory_id: Memory resource ID
            name: Strategy name
            reflection_namespaces: Namespaces for reflections (can be less nested than episode namespaces)
            description: Optional description
            namespaces: Optional namespaces for episodes
        """
        strategy: Dict = {
            StrategyType.EPISODIC.value: {
                "name": name,
                "reflectionConfiguration": {"namespaces": reflection_namespaces},
            }
        }

        if description:
            strategy[StrategyType.EPISODIC.value]["description"] = description
        if namespaces:
            strategy[StrategyType.EPISODIC.value]["namespaces"] = namespaces

        return self._add_strategy(memory_id, strategy)

    def add_episodic_strategy_and_wait(
        self,
        memory_id: str,
        name: str,
        reflection_namespaces: List[str],
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Add an episodic strategy and wait for memory to return to ACTIVE state."""
        self.add_episodic_strategy(memory_id, name, reflection_namespaces, description, namespaces)
        return self._wait_for_memory_active(memory_id, max_wait, poll_interval)

    def add_custom_semantic_strategy(
        self,
        memory_id: str,
        name: str,
        extraction_config: Dict[str, Any],
        consolidation_config: Dict[str, Any],
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add a custom semantic strategy with prompts.

        Args:
            memory_id: Memory resource ID
            name: Strategy name
            extraction_config: Extraction configuration with prompt and model:
                {"prompt": "...", "modelId": "..."}
            consolidation_config: Consolidation configuration with prompt and model:
                {"prompt": "...", "modelId": "..."}
            description: Optional description
            namespaces: Optional namespaces list
        """
        strategy = {
            StrategyType.CUSTOM.value: {
                "name": name,
                "configuration": {
                    "semanticOverride": {
                        "extraction": {
                            "appendToPrompt": extraction_config["prompt"],
                            "modelId": extraction_config["modelId"],
                        },
                        "consolidation": {
                            "appendToPrompt": consolidation_config["prompt"],
                            "modelId": consolidation_config["modelId"],
                        },
                    }
                },
            }
        }

        if description:
            strategy[StrategyType.CUSTOM.value]["description"] = description
        if namespaces:
            strategy[StrategyType.CUSTOM.value]["namespaces"] = namespaces

        return self._add_strategy(memory_id, strategy)

    def add_custom_semantic_strategy_and_wait(
        self,
        memory_id: str,
        name: str,
        extraction_config: Dict[str, Any],
        consolidation_config: Dict[str, Any],
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Add a custom semantic strategy and wait for memory to return to ACTIVE state."""
        self.add_custom_semantic_strategy(
            memory_id, name, extraction_config, consolidation_config, description, namespaces
        )
        return self._wait_for_memory_active(memory_id, max_wait, poll_interval)

    def add_custom_episodic_strategy(
        self,
        memory_id: str,
        name: str,
        extraction_config: Dict[str, Any],
        consolidation_config: Dict[str, Any],
        reflection_config: Dict[str, Any],
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add a custom episodic strategy with prompts.

        Args:
            memory_id: Memory resource ID
            name: Strategy name
            extraction_config: {"prompt": "...", "modelId": "..."}
            consolidation_config: {"prompt": "...", "modelId": "..."}
            reflection_config: {"prompt": "...", "modelId": "...", "namespaces": [...]}
            description: Optional description
            namespaces: Optional namespaces list
        """
        for config, config_name in [
            (extraction_config, "extraction_config"),
            (consolidation_config, "consolidation_config"),
            (reflection_config, "reflection_config"),
        ]:
            for key in ("prompt", "modelId"):
                if key not in config:
                    raise ValueError(f"{config_name} missing required key: {key}")

        strategy = {
            StrategyType.CUSTOM.value: {
                "name": name,
                "configuration": {
                    "episodicOverride": {
                        "extraction": {
                            "appendToPrompt": extraction_config["prompt"],
                            "modelId": extraction_config["modelId"],
                        },
                        "consolidation": {
                            "appendToPrompt": consolidation_config["prompt"],
                            "modelId": consolidation_config["modelId"],
                        },
                        "reflection": {
                            "appendToPrompt": reflection_config["prompt"],
                            "modelId": reflection_config["modelId"],
                            **(
                                {"namespaces": reflection_config["namespaces"]}
                                if "namespaces" in reflection_config
                                else {}
                            ),
                        },
                    }
                },
            }
        }

        if description:
            strategy[StrategyType.CUSTOM.value]["description"] = description
        if namespaces:
            strategy[StrategyType.CUSTOM.value]["namespaces"] = namespaces

        return self._add_strategy(memory_id, strategy)

    def add_custom_episodic_strategy_and_wait(
        self,
        memory_id: str,
        name: str,
        extraction_config: Dict[str, Any],
        consolidation_config: Dict[str, Any],
        reflection_config: Dict[str, Any],
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Add a custom episodic strategy and wait for memory to return to ACTIVE state."""
        self.add_custom_episodic_strategy(
            memory_id, name, extraction_config, consolidation_config, reflection_config, description, namespaces
        )
        return self._wait_for_memory_active(memory_id, max_wait, poll_interval)

    def modify_strategy(
        self,
        memory_id: str,
        strategy_id: str,
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Modify a strategy with full control over configuration."""
        modify_config: Dict = {"memoryStrategyId": strategy_id}  # Using old field name for input

        if description is not None:
            modify_config["description"] = description
        if namespaces is not None:
            modify_config["namespaces"] = namespaces
        if configuration is not None:
            modify_config["configuration"] = configuration

        return self.update_memory_strategies(memory_id=memory_id, modify_strategies=[modify_config])

    def delete_strategy(self, memory_id: str, strategy_id: str) -> Dict[str, Any]:
        """Delete a strategy from a memory."""
        return self.update_memory_strategies(memory_id=memory_id, delete_strategy_ids=[strategy_id])

    def update_memory_strategies(
        self,
        memory_id: str,
        add_strategies: Optional[List[Dict[str, Any]]] = None,
        modify_strategies: Optional[List[Dict[str, Any]]] = None,
        delete_strategy_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Update memory strategies - add, modify, or delete."""
        try:
            memory_strategies = {}

            if add_strategies:
                processed_add = self._add_default_namespaces(add_strategies)
                memory_strategies["addMemoryStrategies"] = processed_add  # Using old field name for input

            if modify_strategies:
                current_strategies = self.get_memory_strategies(memory_id)
                strategy_map = {s["memoryStrategyId"]: s for s in current_strategies}  # Using normalized field

                modify_list = []
                for strategy in modify_strategies:
                    if "memoryStrategyId" not in strategy:  # Using old field name
                        raise ValueError("Each modify strategy must include memoryStrategyId")

                    strategy_id = strategy["memoryStrategyId"]  # Using old field name
                    strategy_info = strategy_map.get(strategy_id)

                    if not strategy_info:
                        raise ValueError("Strategy %s not found in memory %s" % (strategy_id, memory_id))

                    strategy_type = strategy_info["memoryStrategyType"]  # Using normalized field
                    override_type = strategy_info.get("configuration", {}).get("type")

                    strategy_copy = copy.deepcopy(strategy)

                    if "configuration" in strategy_copy:
                        wrapped_config = self._wrap_configuration(
                            strategy_copy["configuration"], strategy_type, override_type
                        )
                        strategy_copy["configuration"] = wrapped_config

                    modify_list.append(strategy_copy)

                memory_strategies["modifyMemoryStrategies"] = modify_list  # Using old field name for input

            if delete_strategy_ids:
                delete_list = [{"memoryStrategyId": sid} for sid in delete_strategy_ids]  # Using old field name
                memory_strategies["deleteMemoryStrategies"] = delete_list  # Using old field name for input

            if not memory_strategies:
                raise ValueError("No strategy operations provided")

            response = self.gmcp_client.update_memory(
                memoryId=memory_id,
                memoryStrategies=memory_strategies,
                clientToken=str(uuid.uuid4()),  # Using old field names for input
            )

            logger.info("Updated memory strategies for: %s", memory_id)
            memory = self._normalize_memory_response(response["memory"])
            return memory

        except ClientError as e:
            logger.error("Failed to update memory strategies: %s", e)
            raise

    def update_memory_strategies_and_wait(
        self,
        memory_id: str,
        add_strategies: Optional[List[Dict[str, Any]]] = None,
        modify_strategies: Optional[List[Dict[str, Any]]] = None,
        delete_strategy_ids: Optional[List[str]] = None,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Update memory strategies and wait for memory to return to ACTIVE state.

        This method handles the temporary CREATING state that occurs when
        updating strategies, preventing subsequent update errors.
        """
        # Update strategies
        self.update_memory_strategies(memory_id, add_strategies, modify_strategies, delete_strategy_ids)

        # Wait for memory to return to ACTIVE
        return self._wait_for_memory_active(memory_id, max_wait, poll_interval)

    def wait_for_memories(
        self, memory_id: str, namespace: str, test_query: str = "test", max_wait: int = 180, poll_interval: int = 15
    ) -> bool:
        """Wait for memory extraction to complete by polling.

        IMPORTANT LIMITATIONS:
        1. This method only works reliably on empty namespaces. If there are already
           existing memories in the namespace, this method may return True immediately
           even if new extractions haven't completed.
        2. Wildcards (*) are NOT supported in namespaces. You must provide the exact
           namespace path with all variables resolved (e.g., "support/facts/session-123"
           not "support/facts/*").

        For subsequent extractions in populated namespaces, use a fixed wait time:
            time.sleep(150)  # Wait 2.5 minutes for extraction

        Args:
            memory_id: Memory resource ID
            namespace: Exact namespace to check (no wildcards)
            test_query: Query to test with (default: "test")
            max_wait: Maximum seconds to wait (default: 180)
            poll_interval: Seconds between checks (default: 15)

        Returns:
            True if memories found, False if timeout

        Note:
            This method will be deprecated in future versions once the API
            provides extraction status or timestamps.
        """
        if "*" in namespace:
            logger.error("Wildcards are not supported in namespaces. Please provide exact namespace.")
            return False

        logger.warning(
            "wait_for_memories() only works reliably on empty namespaces. "
            "For populated namespaces, consider using a fixed wait time instead."
        )

        logger.info("Waiting for memory extraction in namespace: %s", namespace)
        start_time = time.time()
        service_errors = 0

        while time.time() - start_time < max_wait:
            elapsed = int(time.time() - start_time)

            try:
                memories = self.retrieve_memories(memory_id=memory_id, namespace=namespace, query=test_query, top_k=1)

                if memories:
                    logger.info("Memory extraction complete after %d seconds", elapsed)
                    return True

                # Reset service error count on successful call
                service_errors = 0

            except Exception as e:
                if "ServiceException" in str(e):
                    service_errors += 1
                    if service_errors >= 3:
                        logger.warning("Multiple service errors - the service may be experiencing issues")
                logger.debug("Retrieval attempt failed: %s", e)

            if time.time() - start_time < max_wait:
                time.sleep(poll_interval)

        logger.warning("No memories found after %d seconds", max_wait)
        if service_errors > 0:
            logger.info("Note: Encountered %d service errors during polling", service_errors)
        return False

    def add_strategy(self, memory_id: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Add a strategy to a memory (without waiting).

        WARNING: After adding a strategy, the memory enters CREATING state temporarily.
        Use add_*_strategy_and_wait() methods instead to avoid errors.

        Args:
            memory_id: Memory resource ID
            strategy: Strategy configuration dictionary

        Returns:
            Updated memory response
        """
        warnings.warn(
            "add_strategy() may leave memory in CREATING state. "
            "Use add_*_strategy_and_wait() methods to avoid subsequent errors.",
            UserWarning,
            stacklevel=2,
        )
        return self._add_strategy(memory_id, strategy)

    # Private methods

    def _normalize_memory_response(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize memory response to include both old and new field names.

        The API returns new field names but SDK users might expect old ones.
        This ensures compatibility by providing both.
        """
        # Ensure both versions of memory ID exist
        if "id" in memory and "memoryId" not in memory:
            memory["memoryId"] = memory["id"]
        elif "memoryId" in memory and "id" not in memory:
            memory["id"] = memory["memoryId"]

        # Ensure both versions of strategies exist
        if "strategies" in memory and "memoryStrategies" not in memory:
            memory["memoryStrategies"] = memory["strategies"]
        elif "memoryStrategies" in memory and "strategies" not in memory:
            memory["strategies"] = memory["memoryStrategies"]

        # Normalize strategies within memory
        if "strategies" in memory:
            normalized_strategies = []
            for strategy in memory["strategies"]:
                normalized = strategy.copy()

                # Ensure both field name versions exist for strategies
                if "strategyId" in strategy and "memoryStrategyId" not in normalized:
                    normalized["memoryStrategyId"] = strategy["strategyId"]
                elif "memoryStrategyId" in strategy and "strategyId" not in normalized:
                    normalized["strategyId"] = strategy["memoryStrategyId"]

                if "type" in strategy and "memoryStrategyType" not in normalized:
                    normalized["memoryStrategyType"] = strategy["type"]
                elif "memoryStrategyType" in strategy and "type" not in normalized:
                    normalized["type"] = strategy["memoryStrategyType"]

                normalized_strategies.append(normalized)

            memory["strategies"] = normalized_strategies
            memory["memoryStrategies"] = normalized_strategies

        return memory

    def _add_strategy(self, memory_id: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to add a single strategy."""
        return self.update_memory_strategies(memory_id=memory_id, add_strategies=[strategy])

    def _wait_for_memory_active(self, memory_id: str, max_wait: int, poll_interval: int) -> Dict[str, Any]:
        """Wait for memory to return to ACTIVE state after strategy update."""
        logger.info("Waiting for memory %s to return to ACTIVE state...", memory_id)

        start_time = time.time()
        while time.time() - start_time < max_wait:
            elapsed = int(time.time() - start_time)

            try:
                status = self.get_memory_status(memory_id)

                if status == MemoryStatus.ACTIVE.value:
                    logger.info("Memory %s is ACTIVE again (took %d seconds)", memory_id, elapsed)
                    response = self.gmcp_client.get_memory(memoryId=memory_id)  # Input uses old field name
                    memory = self._normalize_memory_response(response["memory"])
                    return memory
                elif status == MemoryStatus.FAILED.value:
                    response = self.gmcp_client.get_memory(memoryId=memory_id)  # Input uses old field name
                    failure_reason = response["memory"].get("failureReason", "Unknown")
                    raise RuntimeError("Memory update failed: %s" % failure_reason)
                else:
                    logger.debug("Memory status: %s (%d seconds elapsed)", status, elapsed)

            except ClientError as e:
                logger.error("Error checking memory status: %s", e)
                raise

            time.sleep(poll_interval)

        raise TimeoutError("Memory %s did not return to ACTIVE state within %d seconds" % (memory_id, max_wait))

    def _add_default_namespaces(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add default namespaces to strategies that don't have them."""
        processed = []

        for strategy in strategies:
            strategy_copy = copy.deepcopy(strategy)

            strategy_type_key = list(strategy.keys())[0]
            strategy_config = strategy_copy[strategy_type_key]

            if "namespaces" not in strategy_config:
                strategy_type = StrategyType(strategy_type_key)
                strategy_config["namespaces"] = DEFAULT_NAMESPACES.get(strategy_type, ["custom/{actorId}/{sessionId}"])

            self._validate_strategy_config(strategy_copy, strategy_type_key)

            processed.append(strategy_copy)

        return processed

    def _validate_namespace(self, namespace: str) -> bool:
        """Validate namespace format - basic check only."""
        # Only check for template variables in namespace definition
        # Note: Using memoryStrategyId (old name) as it's still used in input parameters
        if "{" in namespace and not (
            "{actorId}" in namespace or "{sessionId}" in namespace or "{memoryStrategyId}" in namespace
        ):
            logger.warning("Namespace with templates should contain valid variables: %s", namespace)

        return True

    def _validate_strategy_config(self, strategy: Dict[str, Any], strategy_type: str) -> None:
        """Validate strategy configuration parameters."""
        strategy_config = strategy[strategy_type]

        namespaces = strategy_config.get("namespaces", [])
        for namespace in namespaces:
            self._validate_namespace(namespace)

    def _wrap_configuration(
        self, config: Dict[str, Any], strategy_type: str, override_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Wrap configuration based on strategy type."""
        wrapped_config = {}

        if "extraction" in config:
            extraction = config["extraction"]

            builtin_config_keys = ["triggerEveryNMessages", "historicalContextWindowSize"]

            if strategy_type == "CUSTOM" and override_type:
                override_enum = OverrideType(override_type)
                if override_enum in CUSTOM_EXTRACTION_WRAPPER_KEYS:
                    wrapped_config["extraction"] = {
                        "customExtractionConfiguration": {CUSTOM_EXTRACTION_WRAPPER_KEYS[override_enum]: extraction}
                    }
                else:
                    wrapped_config["extraction"] = extraction
            elif any(key in extraction for key in builtin_config_keys):
                strategy_type_enum = MemoryStrategyTypeEnum(strategy_type)
                if strategy_type in ("SEMANTIC", "USER_PREFERENCE"):
                    wrapped_config["extraction"] = {EXTRACTION_WRAPPER_KEYS[strategy_type_enum]: extraction}
                else:
                    wrapped_config["extraction"] = extraction
            else:
                wrapped_config["extraction"] = extraction

        if "consolidation" in config:
            consolidation = config["consolidation"]

            raw_keys = ["triggerEveryNMessages", "appendToPrompt", "modelId"]
            if any(key in consolidation for key in raw_keys):
                if strategy_type == "SUMMARIZATION":
                    if "triggerEveryNMessages" in consolidation:
                        wrapped_config["consolidation"] = {
                            "summaryConsolidationConfiguration": {
                                "triggerEveryNMessages": consolidation["triggerEveryNMessages"]
                            }
                        }
                elif strategy_type == "CUSTOM" and override_type:
                    override_enum = OverrideType(override_type)
                    if override_enum in CUSTOM_CONSOLIDATION_WRAPPER_KEYS:
                        wrapped_config["consolidation"] = {
                            "customConsolidationConfiguration": {
                                CUSTOM_CONSOLIDATION_WRAPPER_KEYS[override_enum]: consolidation
                            }
                        }
            else:
                wrapped_config["consolidation"] = consolidation

        if "reflection" in config:
            reflection = config["reflection"]

            if strategy_type == "CUSTOM" and override_type:
                override_enum = OverrideType(override_type)
                if override_enum in CUSTOM_REFLECTION_WRAPPER_KEYS:
                    wrapped_config["reflection"] = {
                        "customReflectionConfiguration": {CUSTOM_REFLECTION_WRAPPER_KEYS[override_enum]: reflection}
                    }
            else:
                wrapped_config["reflection"] = reflection

        return wrapped_config
