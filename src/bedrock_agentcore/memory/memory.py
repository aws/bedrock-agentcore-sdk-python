"""Memory class for managing Bedrock AgentCore Memory resources.

This module provides a high-level Memory class that wraps memory operations
for Bedrock AgentCore Memory resources.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from botocore.exceptions import ClientError

from .client import MemoryClient
from .config import MemoryConfigModel, StrategyConfigModel, StrategyType

if TYPE_CHECKING:
    from .session import MemorySession

logger = logging.getLogger(__name__)


class Memory:
    """Represents a Bedrock AgentCore Memory resource.

    Each Memory instance manages a single memory resource. Use Project.from_json()
    to load memories from configuration files.

    Example:
        # Create with config
        memory = Memory(
            name="my-memory",
            strategies=[{"type": "SEMANTIC", "namespace": "facts/{sessionId}/"}]
        )
        memory.launch()

        # Get a session for conversational operations
        session = memory.get_session(actor_id="user-123", session_id="sess-456")

    Attributes:
        name: Memory name
        config: Memory configuration model
        memory_id: ID of created memory resource (if created)
        is_active: Whether the memory is active
    """

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        strategies: Optional[List[Dict[str, Any]]] = None,
        encryption_key_arn: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        region: Optional[str] = None,
    ):
        """Create a Memory instance with full configuration.

        Args:
            name: Unique memory name
            description: Optional description
            strategies: List of strategy configs [{"type": "SEMANTIC", "namespace": "..."}]
            encryption_key_arn: Optional KMS key ARN for encryption
            tags: Resource tags
            region: AWS region (defaults to boto3 default or us-west-2)
        """
        self._name = name
        self._region = region
        self._memory_id: Optional[str] = None

        # Build config model
        strategy_models = None
        if strategies:
            strategy_models = [
                StrategyConfigModel(
                    type=StrategyType(s["type"]),
                    namespace=s["namespace"],
                    customPrompt=s.get("customPrompt"),
                )
                for s in strategies
            ]

        self._config = MemoryConfigModel(
            name=name,
            description=description,
            strategies=strategy_models,
            encryptionKeyArn=encryption_key_arn,
            tags=tags,
        )

        # Initialize client
        self._client = MemoryClient(region_name=region)

        logger.info("Initialized Memory '%s' in region %s", name, self._client.region_name)

    # ==================== PROPERTIES ====================

    @property
    def name(self) -> str:
        """Memory name."""
        return self._name

    @property
    def config(self) -> MemoryConfigModel:
        """Current configuration."""
        return self._config

    @property
    def memory_id(self) -> Optional[str]:
        """Memory ID if created."""
        return self._memory_id

    @property
    def is_active(self) -> bool:
        """Whether memory is active."""
        if not self._memory_id:
            return False
        try:
            status = self._client.get_memory_status(self._memory_id)
            return status == "ACTIVE"
        except ClientError:
            return False

    # ==================== OPERATIONS ====================

    def launch(
        self,
        max_wait: int = 600,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Launch the memory resource in AWS (create if not exists).

        This method is idempotent - it will create the memory if it doesn't exist,
        or return the existing memory if it already exists.

        To update strategies on an existing memory, use add_strategy().

        Waits for the memory to become ACTIVE before returning.

        Args:
            max_wait: Max seconds to wait
            poll_interval: Seconds between status checks

        Returns:
            Memory details

        Raises:
            ClientError: If AWS API call fails
            TimeoutError: If wait times out
        """
        # Check if memory already exists
        self._refresh_memory_state()

        if self._memory_id:
            # Memory exists - return current state
            logger.info("Memory '%s' already exists with ID: %s", self._name, self._memory_id)
            return self._client.get_memory(self._memory_id)

        # Convert strategies to API format
        strategies = []
        if self._config.strategies:
            for s in self._config.strategies:
                strategy = {
                    "memoryStrategyType": s.strategy_type.value,
                    "namespace": s.namespace,
                }
                if s.custom_prompt:
                    strategy["customPrompt"] = s.custom_prompt
                strategies.append(strategy)

        # Memory doesn't exist - create it
        logger.info("Creating memory '%s'...", self._name)
        memory = self._client.create_memory_and_wait(
            name=self._name,
            strategies=strategies,
            description=self._config.description,
            max_wait=max_wait,
            poll_interval=poll_interval,
        )
        self._memory_id = memory.get("memoryId", memory.get("id"))
        logger.info("Created memory with ID: %s", self._memory_id)

        return memory

    def delete(self, max_wait: int = 300, poll_interval: int = 10) -> Dict[str, Any]:
        """Delete the memory resource from AWS.

        Waits for deletion to complete before returning.

        Args:
            max_wait: Max seconds to wait
            poll_interval: Seconds between status checks

        Returns:
            Deletion result

        Raises:
            ClientError: If AWS API call fails
        """
        if not self._memory_id:
            logger.warning("Memory '%s' is not created, nothing to delete", self._name)
            return {"status": "NOT_CREATED"}

        logger.info("Deleting memory '%s'...", self._name)

        try:
            response = self._client.delete_memory(memory_id=self._memory_id)

            self._wait_for_deleted(max_wait, poll_interval)

            # Clear state
            self._memory_id = None

            logger.info("Memory '%s' deleted", self._name)
            return response

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.warning("Memory '%s' not found, may already be deleted", self._name)
                self._memory_id = None
                return {"status": "NOT_FOUND"}
            raise

    def add_strategy(
        self,
        strategy_type: str,
        namespace: str,
        custom_prompt: Optional[str] = None,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Add a strategy to the memory.

        Waits for the update to complete before returning.

        Args:
            strategy_type: Strategy type (SEMANTIC, SUMMARY, USER_PREFERENCE, CUSTOM_SEMANTIC)
            namespace: Namespace for the strategy
            custom_prompt: Custom extraction prompt (for CUSTOM_SEMANTIC)
            max_wait: Max seconds to wait
            poll_interval: Seconds between status checks

        Returns:
            Updated memory details

        Raises:
            ValueError: If memory is not created
            ClientError: If AWS API call fails
        """
        if not self._memory_id:
            raise ValueError("Memory is not launched. Call launch() first.")

        strategy = {
            "memoryStrategyType": strategy_type,
            "namespace": namespace,
        }
        if custom_prompt:
            strategy["customPrompt"] = custom_prompt

        logger.info("Adding strategy '%s' to memory '%s'...", strategy_type, self._name)

        return self._client.add_strategy_and_wait(
            memory_id=self._memory_id,
            strategy=strategy,
            max_wait=max_wait,
            poll_interval=poll_interval,
        )

    def get_session(self, actor_id: str, session_id: str) -> "MemorySession":
        """Get a session for conversational operations.

        Args:
            actor_id: Actor identifier (e.g., user ID)
            session_id: Session identifier (e.g., conversation ID)

        Returns:
            MemorySession instance with methods:
                - add_turns(messages): Add conversation messages
                - get_last_k_turns(k): Get recent conversation history
                - process_turn_with_llm(user_input, llm_callback, retrieval_config): Process with LLM
                - fork_conversation(messages, root_event_id, branch_name): Create conversation branch
                - get_event(event_id): Get a specific event

        Raises:
            ValueError: If memory is not launched

        Example:
            session = memory.get_session(actor_id="user-123", session_id="conv-456")

            # Add conversation turns
            session.add_turns([
                ConversationalMessage("Hello!", MessageRole.USER),
                ConversationalMessage("Hi there!", MessageRole.ASSISTANT)
            ])

            # Get recent history
            turns = session.get_last_k_turns(k=5)

            # Process with LLM and memory context
            memories, response, event = session.process_turn_with_llm(
                user_input="What did we discuss?",
                llm_callback=my_llm,
                retrieval_config={"facts": RetrievalConfig(namespace="facts/{sessionId}/")}
            )
        """
        if not self._memory_id:
            raise ValueError("Memory is not launched. Call launch() first.")

        from .session import MemorySessionManager

        manager = MemorySessionManager(memory_id=self._memory_id, region_name=self._client.region_name)
        return manager.create_memory_session(actor_id=actor_id, session_id=session_id)

    def list_events(
        self,
        actor_id: str,
        session_id: str,
        branch_name: Optional[str] = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """List events in a session.

        Args:
            actor_id: Actor identifier
            session_id: Session identifier
            branch_name: Optional branch name to filter
            max_results: Maximum results to return

        Returns:
            List of events

        Raises:
            ValueError: If memory is not created
        """
        if not self._memory_id:
            raise ValueError("Memory is not launched. Call launch() first.")

        params: Dict[str, Any] = {
            "memoryId": self._memory_id,
            "actorId": actor_id,
            "sessionId": session_id,
            "maxResults": max_results,
        }

        if branch_name:
            params["branchName"] = branch_name

        response = self._client.gmdp_client.list_events(**params)
        events = response.get("events", [])
        return list(events) if events else []

    def search_records(
        self,
        query: str,
        namespace: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search memory records.

        Args:
            query: Search query
            namespace: Namespace to search in
            top_k: Maximum results to return

        Returns:
            List of matching memory records

        Raises:
            ValueError: If memory is not created
        """
        if not self._memory_id:
            raise ValueError("Memory is not launched. Call launch() first.")

        return self._client.retrieve_memories(
            memory_id=self._memory_id,
            namespace=namespace,
            query=query,
            top_k=top_k,
        )

    # ==================== HELPERS ====================

    def _refresh_memory_state(self) -> None:
        """Fetch current memory state from AWS by name."""
        try:
            memories = self._client.list_memories()

            for memory in memories:
                # Handle both old and new field names
                memory_name = memory.get("name") or memory.get("id", "").split("-")[0]
                if memory_name == self._name or memory.get("id", "").startswith(self._name):
                    self._memory_id = memory.get("memoryId", memory.get("id"))
                    logger.debug("Found existing memory: %s", self._memory_id)
                    return

            logger.debug("No existing memory found for '%s'", self._name)

        except ClientError as e:
            logger.warning("Failed to refresh memory state: %s", e)

    def _wait_for_deleted(self, max_wait: int, poll_interval: int) -> None:
        """Poll until memory is deleted.

        Args:
            max_wait: Maximum seconds to wait
            poll_interval: Seconds between polls

        Raises:
            TimeoutError: If max_wait exceeded
        """
        if not self._memory_id:
            return

        start_time = time.time()
        logger.info("Waiting for memory deletion...")

        while time.time() - start_time < max_wait:
            try:
                response = self._client.gmcp_client.get_memory(memoryId=self._memory_id)
                status = response.get("memory", {}).get("status")
                logger.debug("Memory status: %s", status)

                if status == "DELETING":
                    time.sleep(poll_interval)
                    continue

            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    logger.info("Memory deleted")
                    return
                raise

            time.sleep(poll_interval)

        raise TimeoutError(f"Timeout waiting for memory deletion after {max_wait}s")
