"""AgentCore Memory SDK - Control Plane Client.

This module provides a simplified interface for Bedrock AgentCore Memory control plane operations.
It handles memory resource management, strategy operations, and status monitoring.
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from .constants import (
    MemoryStatus,
)

logger = logging.getLogger(__name__)


class MemoryControlPlaneClient:
    """Client for Bedrock AgentCore Memory control plane operations."""

    def __init__(self, region_name: str = "us-west-2", environment: str = "prod"):
        """Initialize the Memory Control Plane client.

        Args:
            region_name: AWS region name
            environment: Environment name (prod, gamma, etc.)
        """
        self.region_name = region_name
        self.environment = environment

        self.endpoint = os.getenv(
            "BEDROCK_AGENTCORE_CONTROL_ENDPOINT", f"https://bedrock-agentcore-control.{region_name}.amazonaws.com"
        )

        service_name = os.getenv("BEDROCK_AGENTCORE_CONTROL_SERVICE", "bedrock-agentcore-control")
        self.client = boto3.client(service_name, region_name=self.region_name, endpoint_url=self.endpoint)

        logger.info("Initialized MemoryControlPlaneClient for %s in %s", environment, region_name)

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
        params = {
            "name": name,
            "eventExpiryDuration": event_expiry_days,
            "clientToken": str(uuid.uuid4()),
        }

        if description:
            params["description"] = description

        if memory_execution_role_arn:
            params["memoryExecutionRoleArn"] = memory_execution_role_arn

        if strategies:
            params["memoryStrategies"] = strategies

        try:
            response = self.client.create_memory(**params)
            memory = response["memory"]
            memory_id = memory["id"]

            logger.info("Created memory: %s", memory_id)

            if wait_for_active:
                return self._wait_for_memory_active(memory_id, max_wait, poll_interval)

            return memory

        except ClientError as e:
            logger.error("Failed to create memory: %s", e)
            raise

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
        try:
            memories = []
            next_token = None

            while len(memories) < max_results:
                params = {"maxResults": min(100, max_results - len(memories))}
                if next_token:
                    params["nextToken"] = next_token

                response = self.client.list_memories(**params)
                batch = response.get("memories", [])
                memories.extend(batch)

                next_token = response.get("nextToken")
                if not next_token or len(memories) >= max_results:
                    break

            # Add strategy count to each memory summary
            for memory in memories:
                memory["strategyCount"] = 0  # List memories doesn't include strategies

            return memories[:max_results]

        except ClientError as e:
            logger.error("Failed to list memories: %s", e)
            raise

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
        wait_for_strategies: bool = False,  # Changed default to False
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
                        self._wait_for_status(
                            memory_id=memory_id,
                            target_status=MemoryStatus.ACTIVE.value,
                            max_wait=max_wait,
                            poll_interval=poll_interval,
                            check_strategies=True,
                        )
                except Exception as e:
                    logger.warning("Error waiting for strategies to become ACTIVE: %s", e)

            # Now delete the memory
            response = self.client.delete_memory(memoryId=memory_id, clientToken=str(uuid.uuid4()))

            logger.info("Initiated deletion of memory: %s", memory_id)

            if not wait_for_deletion:
                return response

            # Wait for deletion to complete
            start_time = time.time()
            while time.time() - start_time < max_wait:
                try:
                    self.client.get_memory(memoryId=memory_id)
                    time.sleep(poll_interval)
                except ClientError as e:
                    if e.response["Error"]["Code"] == "ResourceNotFoundException":
                        logger.info("Memory %s successfully deleted", memory_id)
                        return response
                    raise

            raise TimeoutError(f"Memory {memory_id} was not deleted within {max_wait} seconds")

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
            Updated memory object with strategyId field
        """
        # Get the strategy type and name for identification
        strategy_type = list(strategy.keys())[0]  # e.g., 'semanticMemoryStrategy'
        strategy_name = strategy[strategy_type].get("name")

        logger.info("Adding strategy %s of type %s to memory %s", strategy_name, strategy_type, memory_id)

        # Use update_memory with add_strategies parameter but don't wait for memory
        memory = self.update_memory(
            memory_id=memory_id,
            add_strategies=[strategy],
            wait_for_active=False,  # Don't wait for memory, we'll check strategy specifically
        )

        # If we need to wait for the strategy to become active
        if wait_for_active:
            # First, get the memory again to ensure we have the latest state
            memory = self.get_memory(memory_id)

            # Find the newly added strategy by matching name
            strategies = memory.get("strategies", [])
            strategy_id = None

            for s in strategies:
                # Match by name since that's unique within a memory
                if s.get("name") == strategy_name:
                    strategy_id = s.get("strategyId")
                    logger.info("Found newly added strategy %s with ID %s", strategy_name, strategy_id)
                    break

            if strategy_id:
                return self._wait_for_strategy_active(memory_id, strategy_id, max_wait, poll_interval)
            else:
                logger.warning("Could not identify newly added strategy %s to wait for activation", strategy_name)

        return memory

    def get_strategy(self, memory_id: str, strategy_id: str) -> Dict[str, Any]:
        """Get a specific strategy from a memory resource.

        Args:
            memory_id: Memory resource ID
            strategy_id: Strategy ID

        Returns:
            Strategy details
        """
        try:
            memory = self.get_memory(memory_id)
            strategies = memory.get("strategies", [])

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
        # Note: API expects memoryStrategyId for input but returns strategyId in response
        modify_config: Dict = {"memoryStrategyId": strategy_id}

        if description is not None:
            modify_config["description"] = description

        if namespaces is not None:
            modify_config["namespaces"] = namespaces

        if configuration is not None:
            modify_config["configuration"] = configuration

        # Use update_memory with modify_strategies parameter but don't wait for memory
        memory = self.update_memory(
            memory_id=memory_id,
            modify_strategies=[modify_config],
            wait_for_active=False,  # Don't wait for memory, we'll check strategy specifically
        )

        # If we need to wait for the strategy to become active
        if wait_for_active:
            return self._wait_for_strategy_active(memory_id, strategy_id, max_wait, poll_interval)

        return memory

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
        # For remove_strategy, we only need to wait for memory to be active
        # since the strategy will be gone
        return self.update_memory(
            memory_id=memory_id,
            delete_strategy_ids=[strategy_id],
            wait_for_active=wait_for_active,
            max_wait=max_wait,
            poll_interval=poll_interval,
        )

    # ==================== HELPER METHODS ====================

    def _wait_for_memory_active(self, memory_id: str, max_wait: int, poll_interval: int) -> Dict[str, Any]:
        """Wait for memory to return to ACTIVE state."""
        logger.info("Waiting for memory %s to become ACTIVE...", memory_id)
        return self._wait_for_status(
            memory_id=memory_id, target_status=MemoryStatus.ACTIVE.value, max_wait=max_wait, poll_interval=poll_interval
        )

    def _wait_for_strategy_active(
        self, memory_id: str, strategy_id: str, max_wait: int, poll_interval: int
    ) -> Dict[str, Any]:
        """Wait for specific memory strategy to become ACTIVE."""
        logger.info("Waiting for strategy %s to become ACTIVE (max wait: %d seconds)...", strategy_id, max_wait)

        start_time = time.time()
        last_status = None

        while time.time() - start_time < max_wait:
            try:
                memory = self.get_memory(memory_id)
                strategies = memory.get("strategies", [])

                for strategy in strategies:
                    if strategy.get("strategyId") == strategy_id:
                        status = strategy["status"]

                        # Log status changes
                        if status != last_status:
                            logger.info("Strategy %s status: %s", strategy_id, status)
                            last_status = status

                        if status == MemoryStatus.ACTIVE.value:
                            elapsed = time.time() - start_time
                            logger.info("Strategy %s is now ACTIVE (took %.1f seconds)", strategy_id, elapsed)
                            return memory
                        elif status == MemoryStatus.FAILED.value:
                            failure_reason = strategy.get("failureReason", "Unknown")
                            raise RuntimeError(f"Strategy {strategy_id} failed to activate: {failure_reason}")

                        break
                else:
                    logger.warning("Strategy %s not found in memory %s", strategy_id, memory_id)

                # Wait before checking again
                time.sleep(poll_interval)

            except ClientError as e:
                logger.error("Error checking strategy status: %s", e)
                raise

        elapsed = time.time() - start_time
        raise TimeoutError(
            f"Strategy {strategy_id} did not become ACTIVE within {max_wait} seconds (last status: {last_status})"
        )

    def _wait_for_status(
        self, memory_id: str, target_status: str, max_wait: int, poll_interval: int, check_strategies: bool = True
    ) -> Dict[str, Any]:
        """Generic method to wait for a memory to reach a specific status.

        Args:
            memory_id: The ID of the memory to check
            target_status: The status to wait for (e.g., "ACTIVE")
            max_wait: Maximum time to wait in seconds
            poll_interval: Time between status checks in seconds
            check_strategies: Whether to also check that all strategies are in the target status

        Returns:
            The memory object once it reaches the target status

        Raises:
            TimeoutError: If the memory doesn't reach the target status within max_wait
            RuntimeError: If the memory or any strategy reaches a FAILED state
        """
        logger.info("Waiting for memory %s to reach status %s...", memory_id, target_status)

        start_time = time.time()
        last_memory_status = None
        strategy_statuses = {}

        while time.time() - start_time < max_wait:
            try:
                memory = self.get_memory(memory_id)
                status = memory.get("status")

                # Log status changes for memory
                if status != last_memory_status:
                    logger.info("Memory %s status: %s", memory_id, status)
                    last_memory_status = status

                if status == target_status:
                    # Check if all strategies are also in the target status
                    if check_strategies and target_status == MemoryStatus.ACTIVE.value:
                        strategies = memory.get("strategies", [])
                        all_strategies_active = True

                        for strategy in strategies:
                            strategy_id = strategy.get("strategyId")
                            strategy_status = strategy.get("status")

                            # Log strategy status changes
                            if (
                                strategy_id not in strategy_statuses
                                or strategy_statuses[strategy_id] != strategy_status
                            ):
                                logger.info("Strategy %s status: %s", strategy_id, strategy_status)
                                strategy_statuses[strategy_id] = strategy_status

                            if strategy_status != target_status:
                                if strategy_status == MemoryStatus.FAILED.value:
                                    failure_reason = strategy.get("failureReason", "Unknown")
                                    raise RuntimeError(f"Strategy {strategy_id} failed: {failure_reason}")

                                all_strategies_active = False

                        if not all_strategies_active:
                            logger.info(
                                "Memory %s is %s but %d strategies are still processing",
                                memory_id,
                                target_status,
                                len([s for s in strategies if s.get("status") != target_status]),
                            )
                            time.sleep(poll_interval)
                            continue

                    elapsed = time.time() - start_time
                    logger.info(
                        "Memory %s and all strategies are now %s (took %.1f seconds)", memory_id, target_status, elapsed
                    )
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
            f"Memory {memory_id} did not reach status {target_status} within {max_wait} seconds "
            f"(elapsed: {elapsed:.1f}s)"
        )
