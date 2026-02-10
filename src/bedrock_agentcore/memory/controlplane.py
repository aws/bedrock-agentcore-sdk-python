"""AgentCore Memory SDK - Control Plane Client (DEPRECATED).

This module is deprecated. Use MemoryClient from bedrock_agentcore.memory.client instead,
which provides all control plane operations plus data plane features.
"""

import logging
import time
import uuid
import warnings
from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError

from .client import MemoryClient
from .constants import MemoryStatus

logger = logging.getLogger(__name__)


class MemoryControlPlaneClient:
    """Client for Bedrock AgentCore Memory control plane operations.

    .. deprecated::
        Use :class:`MemoryClient` instead, which provides all control plane operations
        plus data plane features.
    """

    def __init__(self, region_name: str = "us-west-2", environment: str = "prod"):
        """Initialize the Memory Control Plane client.

        Args:
            region_name: AWS region name
            environment: Environment name (unused, kept for backward compatibility)
        """
        warnings.warn(
            "MemoryControlPlaneClient is deprecated and will be removed in v1.4.0. "
            "Use MemoryClient instead, which provides all control plane operations "
            "plus data plane features. See: https://github.com/aws/bedrock-agentcore-sdk-python/issues/247",
            DeprecationWarning,
            stacklevel=2,
        )
        self.region_name = region_name
        self.environment = environment
        self._memory_client = MemoryClient(region_name=region_name)

        logger.info("Initialized MemoryControlPlaneClient for %s in %s", environment, region_name)

    @property
    def client(self):
        """The underlying boto3 control plane client."""
        return self._memory_client.gmcp_client

    @client.setter
    def client(self, value):
        """Allow overriding the underlying boto3 client (used by tests)."""
        self._memory_client.gmcp_client = value

    # ==================== MEMORY OPERATIONS ====================

    def create_memory(
        self,
        name: str,
        event_expiry_days: int = 90,
        description: Optional[str] = None,
        memory_execution_role_arn: Optional[str] = None,
        strategies: Optional[List[Dict[str, Any]]] = None,
        wait_for_active: bool = False,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Create a memory resource with optional strategies.

        Args:
            name: Name for the memory resource
            event_expiry_days: How long to retain events (default: 90 days)
            description: Optional description
            memory_execution_role_arn: IAM role ARN for memory execution
            strategies: Optional list of strategy configurations
            wait_for_active: Whether to wait for memory to become ACTIVE
            max_wait: Maximum seconds to wait if wait_for_active is True
            poll_interval: Seconds between status checks if wait_for_active is True

        Returns:
            Created memory object
        """
        if wait_for_active:
            return self._memory_client.create_memory_and_wait(
                name=name,
                strategies=strategies or [],
                description=description,
                event_expiry_days=event_expiry_days,
                memory_execution_role_arn=memory_execution_role_arn,
                max_wait=max_wait,
                poll_interval=poll_interval,
            )
        return self._memory_client.create_memory(
            name=name,
            strategies=strategies or [],
            description=description,
            event_expiry_days=event_expiry_days,
            memory_execution_role_arn=memory_execution_role_arn,
        )

    def get_memory(self, memory_id: str, include_strategies: bool = True) -> Dict[str, Any]:
        """Get a memory resource by ID.

        Args:
            memory_id: Memory resource ID
            include_strategies: Whether to include strategy details in response

        Returns:
            Memory resource details
        """
        try:
            response = self.client.get_memory(memoryId=memory_id)
            memory = response["memory"]

            # Add strategy count
            strategies = memory.get("strategies", [])
            memory["strategyCount"] = len(strategies)

            # Remove strategies if not requested
            if not include_strategies and "strategies" in memory:
                del memory["strategies"]

            return memory

        except ClientError as e:
            logger.error("Failed to get memory: %s", e)
            raise

    def list_memories(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """List all memories for the account with pagination support.

        Args:
            max_results: Maximum number of memories to return

        Returns:
            List of memory summaries
        """
        return self._memory_client.list_memories(max_results=max_results)

    def update_memory(
        self,
        memory_id: str,
        description: Optional[str] = None,
        event_expiry_days: Optional[int] = None,
        memory_execution_role_arn: Optional[str] = None,
        add_strategies: Optional[List[Dict[str, Any]]] = None,
        modify_strategies: Optional[List[Dict[str, Any]]] = None,
        delete_strategy_ids: Optional[List[str]] = None,
        wait_for_active: bool = False,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Update a memory resource properties and/or strategies.

        Args:
            memory_id: Memory resource ID
            description: Optional new description
            event_expiry_days: Optional new event expiry duration
            memory_execution_role_arn: Optional new execution role ARN
            add_strategies: Optional list of strategies to add
            modify_strategies: Optional list of strategies to modify
            delete_strategy_ids: Optional list of strategy IDs to delete
            wait_for_active: Whether to wait for memory to become ACTIVE
            max_wait: Maximum seconds to wait if wait_for_active is True
            poll_interval: Seconds between status checks if wait_for_active is True

        Returns:
            Updated memory object
        """
        params: Dict = {
            "memoryId": memory_id,
            "clientToken": str(uuid.uuid4()),
        }

        # Add memory properties if provided
        if description is not None:
            params["description"] = description

        if event_expiry_days is not None:
            params["eventExpiryDuration"] = event_expiry_days

        if memory_execution_role_arn is not None:
            params["memoryExecutionRoleArn"] = memory_execution_role_arn

        # Add strategy operations if provided
        memory_strategies = {}

        if add_strategies:
            memory_strategies["addMemoryStrategies"] = add_strategies

        if modify_strategies:
            memory_strategies["modifyMemoryStrategies"] = modify_strategies

        if delete_strategy_ids:
            memory_strategies["deleteMemoryStrategies"] = [
                {"memoryStrategyId": strategy_id} for strategy_id in delete_strategy_ids
            ]

        if memory_strategies:
            params["memoryStrategies"] = memory_strategies

        try:
            response = self.client.update_memory(**params)
            memory = response["memory"]
            logger.info("Updated memory: %s", memory_id)

            if wait_for_active:
                return self._wait_for_memory_active(memory_id, max_wait, poll_interval)

            return memory

        except ClientError as e:
            logger.error("Failed to update memory: %s", e)
            raise

    def delete_memory(
        self,
        memory_id: str,
        wait_for_deletion: bool = False,
        wait_for_strategies: bool = False,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Delete a memory resource.

        Args:
            memory_id: Memory resource ID to delete
            wait_for_deletion: Whether to wait for complete deletion
            wait_for_strategies: Whether to wait for strategies to become ACTIVE before deletion
            max_wait: Maximum seconds to wait if wait_for_deletion is True
            poll_interval: Seconds between checks if wait_for_deletion is True

        Returns:
            Deletion response
        """
        try:
            # If requested, wait for all strategies to become ACTIVE before deletion
            if wait_for_strategies:
                try:
                    memory = self.get_memory(memory_id)
                    strategies = memory.get("strategies", [])

                    # Check if any strategies are in a transitional state
                    transitional_strategies = [
                        s
                        for s in strategies
                        if s.get("status") not in [MemoryStatus.ACTIVE.value, MemoryStatus.FAILED.value]
                    ]

                    if transitional_strategies:
                        logger.info(
                            "Waiting for %d strategies to become ACTIVE before deletion", len(transitional_strategies)
                        )
                        start_time = time.time()
                        while time.time() - start_time < max_wait:
                            memory = self.get_memory(memory_id)
                            strategies = memory.get("strategies", [])
                            all_ready = all(
                                s.get("status") in [MemoryStatus.ACTIVE.value, MemoryStatus.FAILED.value]
                                for s in strategies
                            )
                            if all_ready:
                                break
                            time.sleep(poll_interval)
                except Exception as e:
                    logger.warning("Error waiting for strategies to become ACTIVE: %s", e)

            if wait_for_deletion:
                return self._memory_client.delete_memory_and_wait(
                    memory_id=memory_id,
                    max_wait=max_wait,
                    poll_interval=poll_interval,
                )

            return self._memory_client.delete_memory(memory_id=memory_id)

        except ClientError as e:
            logger.error("Failed to delete memory: %s", e)
            raise

    # ==================== STRATEGY OPERATIONS ====================

    def add_strategy(
        self,
        memory_id: str,
        strategy: Dict[str, Any],
        wait_for_active: bool = False,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Add a strategy to a memory resource.

        Args:
            memory_id: Memory resource ID
            strategy: Strategy configuration dictionary
            wait_for_active: Whether to wait for strategy to become ACTIVE
            max_wait: Maximum seconds to wait if wait_for_active is True
            poll_interval: Seconds between status checks if wait_for_active is True

        Returns:
            Updated memory object
        """
        if wait_for_active:
            return self._memory_client.update_memory_strategies_and_wait(
                memory_id=memory_id,
                add_strategies=[strategy],
                max_wait=max_wait,
                poll_interval=poll_interval,
            )
        return self._memory_client.update_memory_strategies(
            memory_id=memory_id,
            add_strategies=[strategy],
        )

    def get_strategy(self, memory_id: str, strategy_id: str) -> Dict[str, Any]:
        """Get a specific strategy from a memory resource.

        Args:
            memory_id: Memory resource ID
            strategy_id: Strategy ID

        Returns:
            Strategy details
        """
        try:
            strategies = self._memory_client.get_memory_strategies(memory_id)

            for strategy in strategies:
                if strategy.get("strategyId") == strategy_id:
                    return strategy

            raise ValueError(f"Strategy {strategy_id} not found in memory {memory_id}")

        except ClientError as e:
            logger.error("Failed to get strategy: %s", e)
            raise

    def update_strategy(
        self,
        memory_id: str,
        strategy_id: str,
        description: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
        configuration: Optional[Dict[str, Any]] = None,
        wait_for_active: bool = False,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Update a strategy in a memory resource.

        Args:
            memory_id: Memory resource ID
            strategy_id: Strategy ID to update
            description: Optional new description
            namespaces: Optional new namespaces list
            configuration: Optional new configuration
            wait_for_active: Whether to wait for strategy to become ACTIVE
            max_wait: Maximum seconds to wait if wait_for_active is True
            poll_interval: Seconds between status checks if wait_for_active is True

        Returns:
            Updated memory object
        """
        modify_config: Dict = {"memoryStrategyId": strategy_id}

        if description is not None:
            modify_config["description"] = description

        if namespaces is not None:
            modify_config["namespaces"] = namespaces

        if configuration is not None:
            modify_config["configuration"] = configuration

        if wait_for_active:
            return self._memory_client.update_memory_strategies_and_wait(
                memory_id=memory_id,
                modify_strategies=[modify_config],
                max_wait=max_wait,
                poll_interval=poll_interval,
            )
        return self._memory_client.update_memory_strategies(
            memory_id=memory_id,
            modify_strategies=[modify_config],
        )

    def remove_strategy(
        self,
        memory_id: str,
        strategy_id: str,
        wait_for_active: bool = False,
        max_wait: int = 300,
        poll_interval: int = 10,
    ) -> Dict[str, Any]:
        """Remove a strategy from a memory resource.

        Args:
            memory_id: Memory resource ID
            strategy_id: Strategy ID to remove
            wait_for_active: Whether to wait for memory to become ACTIVE
            max_wait: Maximum seconds to wait if wait_for_active is True
            poll_interval: Seconds between status checks if wait_for_active is True

        Returns:
            Updated memory object
        """
        if wait_for_active:
            return self._memory_client.update_memory_strategies_and_wait(
                memory_id=memory_id,
                delete_strategy_ids=[strategy_id],
                max_wait=max_wait,
                poll_interval=poll_interval,
            )
        return self._memory_client.update_memory_strategies(
            memory_id=memory_id,
            delete_strategy_ids=[strategy_id],
        )

    # ==================== HELPER METHODS ====================

    def _wait_for_memory_active(self, memory_id: str, max_wait: int, poll_interval: int) -> Dict[str, Any]:
        """Wait for memory and all strategies to reach ACTIVE state.

        Used by update_memory(wait_for_active=True).
        """
        logger.info("Waiting for memory %s to become ACTIVE...", memory_id)

        start_time = time.time()
        last_memory_status = None

        while time.time() - start_time < max_wait:
            try:
                memory = self.get_memory(memory_id)
                status = memory.get("status")

                if status != last_memory_status:
                    logger.info("Memory %s status: %s", memory_id, status)
                    last_memory_status = status

                if status == MemoryStatus.ACTIVE.value:
                    strategies = memory.get("strategies", [])
                    all_strategies_active = all(
                        s.get("status") in [MemoryStatus.ACTIVE.value, MemoryStatus.FAILED.value] for s in strategies
                    )

                    if not all_strategies_active:
                        logger.info(
                            "Memory %s is ACTIVE but %d strategies are still processing",
                            memory_id,
                            len([s for s in strategies if s.get("status") != MemoryStatus.ACTIVE.value]),
                        )
                        time.sleep(poll_interval)
                        continue

                    elapsed = time.time() - start_time
                    logger.info("Memory %s and all strategies are now ACTIVE (took %.1f seconds)", memory_id, elapsed)
                    return memory
                elif status == MemoryStatus.FAILED.value:
                    failure_reason = memory.get("failureReason", "Unknown")
                    raise RuntimeError(f"Memory operation failed: {failure_reason}")

                time.sleep(poll_interval)

            except ClientError as e:
                logger.error("Error checking memory status: %s", e)
                raise

        elapsed = time.time() - start_time
        raise TimeoutError(
            f"Memory {memory_id} did not reach status ACTIVE within {max_wait} seconds "
            f"(elapsed: {elapsed:.1f}s)"
        )
