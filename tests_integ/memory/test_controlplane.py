"""Integration tests for MemoryControlPlaneClient."""

import os
import time

import pytest

from bedrock_agentcore.memory.controlplane import MemoryControlPlaneClient


@pytest.mark.integration
class TestMemoryControlPlaneClient:
    """Integration tests for MemoryControlPlaneClient."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.client = MemoryControlPlaneClient(region_name=cls.region)
        cls.test_prefix = f"test_cp_{int(time.time())}"
        cls.memory_ids = []

    @classmethod
    def teardown_class(cls):
        for mid in cls.memory_ids:
            try:
                cls.client.delete_memory(memory_id=mid)
            except Exception as e:
                print(f"Failed to delete memory {mid}: {e}")

    @pytest.mark.order(1)
    def test_create_memory_with_strategies(self):
        memory = self.client.create_memory(
            name=f"{self.test_prefix}_strategies",
            strategies=[{"semanticMemoryStrategy": {"name": "TestSemantic", "description": "semantic strategy"}}],
            wait_for_active=True,
            max_wait=300,
            poll_interval=10,
        )
        self.__class__.memory_ids.append(memory["id"])
        assert memory["name"] == f"{self.test_prefix}_strategies"
        assert memory["status"] == "ACTIVE"
        assert len(memory.get("strategies", [])) > 0

    @pytest.mark.order(2)
    def test_update_memory_description(self):
        if len(self.memory_ids) < 1:
            pytest.skip("prerequisite test did not create memory")
        memory_id = self.memory_ids[0]
        self.client.update_memory(memory_id=memory_id, description="updated description")
        details = self.client.get_memory(memory_id)
        assert details["description"] == "updated description"

    @pytest.mark.order(3)
    def test_add_strategy_to_existing_memory(self):
        # Needs a memory with no strategies to test adding the first strategy
        memory = self.client.create_memory(
            name=f"{self.test_prefix}_addstrat",
            wait_for_active=True,
            max_wait=300,
            poll_interval=10,
        )
        self.__class__.memory_ids.append(memory["id"])

        self.client.add_strategy(
            memory_id=memory["id"],
            strategy={"semanticMemoryStrategy": {"name": "AddedSemantic", "description": "added strategy"}},
            wait_for_active=True,
            max_wait=300,
            poll_interval=10,
        )

        details = self.client.get_memory(memory["id"])
        strategies = details.get("strategies", [])
        active = [s for s in strategies if s.get("name") == "AddedSemantic" and s.get("status") == "ACTIVE"]
        assert len(active) == 1

    @pytest.mark.order(4)
    def test_add_strategy_immediately_after_another(self):
        if len(self.memory_ids) < 2:
            pytest.skip("prerequisite test did not create memory")
        memory_id = self.memory_ids[1]  # memory from test 3

        self.client.add_strategy(
            memory_id=memory_id,
            strategy={"userPreferenceMemoryStrategy": {"name": "UserPref", "description": "user preference strategy"}},
            wait_for_active=True,
            max_wait=300,
            poll_interval=10,
        )

        details = self.client.get_memory(memory_id)
        strategies = details.get("strategies", [])
        active = [s for s in strategies if s.get("name") == "UserPref" and s.get("status") == "ACTIVE"]
        assert len(active) == 1

    @pytest.mark.order(5)
    def test_list_memories(self):
        if len(self.memory_ids) < 1:
            pytest.skip("prerequisite test did not create memory")
        memories = self.client.list_memories()
        test_memories = [m for m in memories if m.get("id", "").startswith(self.test_prefix)]
        assert len(test_memories) >= 2

    @pytest.mark.order(6)
    def test_delete_memory(self):
        if len(self.memory_ids) < 1:
            pytest.skip("prerequisite test did not create memory")
        memory_id = self.memory_ids[0]
        self.client.delete_memory(memory_id=memory_id, wait_for_deletion=True, max_wait=120, poll_interval=5)
