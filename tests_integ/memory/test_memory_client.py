"""Integration tests for MemoryClient.

Requires environment variables:
    MEMORY_KINESIS_ARN: ARN of a pre-existing Kinesis stream
    MEMORY_ROLE_ARN: ARN of an IAM role the memory service can assume
    MEMORY_PREPOPULATED_ID: ID of a memory with pre-extracted data (for retrieval tests)
    MEMORY_PREPOPULATED_ACTOR_ID: Actor ID in the pre-populated memory (default: integ-test-actor)
    MEMORY_PREPOPULATED_SESSION_ID: Session ID in the pre-populated memory (default: integ-test-session)

Run with:
    pytest tests_integ/memory/test_memory_client.py -xvs --log-cli-level=INFO
"""

import os
import time
import uuid

import pytest

from bedrock_agentcore.memory import MemoryClient

from .helpers import poll_until


@pytest.mark.integration
class TestMemoryClient:
    """Integration tests for MemoryClient."""

    @classmethod
    def setup_class(cls):
        cls.kinesis_stream_arn = os.environ.get("MEMORY_KINESIS_ARN")
        cls.execution_role_arn = os.environ.get("MEMORY_ROLE_ARN")

        if not cls.kinesis_stream_arn or not cls.execution_role_arn:
            pytest.fail("MEMORY_KINESIS_ARN and MEMORY_ROLE_ARN must be set")

        cls.prepopulated_memory_id = os.environ.get("MEMORY_PREPOPULATED_ID")
        cls.prepopulated_actor_id = os.environ.get("MEMORY_PREPOPULATED_ACTOR_ID", "integ-test-actor")
        cls.prepopulated_session_id = os.environ.get("MEMORY_PREPOPULATED_SESSION_ID", "integ-test-session")
        if not cls.prepopulated_memory_id:
            pytest.fail("MEMORY_PREPOPULATED_ID must be set")

        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.client = MemoryClient(region_name=cls.region)
        cls.test_prefix = f"test_mc_{int(time.time())}"
        cls.id_prefix = f"test-mc-{int(time.time())}"
        cls.memory_ids = []

        # Create memory WITHOUT strategies for data plane tests (avoids flakiness with memory extraction latency)
        memory = cls.client.create_memory_and_wait(
            name=f"{cls.test_prefix}_dataplane",
            strategies=[],
            event_expiry_days=7,
        )
        cls.memory_id = memory.get("memoryId") or memory.get("id")
        cls.memory_ids.append(cls.memory_id)

    @classmethod
    def teardown_class(cls):
        for memory_id in cls.memory_ids:
            try:
                cls.client.delete_memory(memory_id)
            except Exception as e:
                print(f"Failed to delete memory {memory_id}: {e}")

    # --- Stream Delivery Test (independent, no ordering) ---

    def _make_delivery_config(self, level="FULL_CONTENT"):
        return {
            "resources": [
                {
                    "kinesis": {
                        "dataStreamArn": self.kinesis_stream_arn,
                        "contentConfigurations": [{"type": "MEMORY_RECORDS", "level": level}],
                    }
                }
            ]
        }

    def test_stream_delivery_create_and_update(self):
        """Create memory with FULL_CONTENT delivery, update to METADATA_ONLY via passthrough."""
        delivery_config = self._make_delivery_config("FULL_CONTENT")

        memory = self.client.create_memory_and_wait(
            name=f"{self.test_prefix}_stream",
            strategies=[],
            memory_execution_role_arn=self.execution_role_arn,
            stream_delivery_resources=delivery_config,
        )

        memory_id = memory["id"]
        self.__class__.memory_ids.append(memory_id)

        assert memory["streamDeliveryResources"] == delivery_config

        # Test update via MemoryClient.__getattr__ passthrough to boto3 client.
        # Uses camelCase params because the passthrough forwards directly to boto3
        # without the snake_case translation that explicit SDK methods provide.
        updated_config = self._make_delivery_config("METADATA_ONLY")
        response = self.client.update_memory(
            memoryId=memory_id,
            clientToken=str(uuid.uuid4()),
            streamDeliveryResources=updated_config,
        )
        assert response["memory"]["streamDeliveryResources"] == updated_config

    # --- Data Plane Tests (order 1-10) ---

    @pytest.mark.order(1)
    # save_conversation returns an event with a valid eventId
    def test_save_conversation(self):
        actor_id = f"{self.id_prefix}-save-actor"
        session_id = f"{self.id_prefix}-save-session"
        result = self.client.save_conversation(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=[("Hello", "USER"), ("Hi there", "ASSISTANT")],
        )
        assert "eventId" in result

    @pytest.mark.order(2)
    # list_events returns all events saved to a session
    def test_list_events(self):
        actor_id = f"{self.id_prefix}-list-actor"
        session_id = f"{self.id_prefix}-list-session"
        for i in range(3):
            self.client.save_conversation(
                memory_id=self.memory_id,
                actor_id=actor_id,
                session_id=session_id,
                messages=[(f"msg {i}", "USER"), (f"reply {i}", "ASSISTANT")],
            )
        events = poll_until(
            fn=lambda: self.client.list_events(self.memory_id, actor_id, session_id),
            predicate=lambda e: len(e) >= 3,
        )
        assert len(events) >= 3

    @pytest.mark.order(3)
    # list_events with branch_name only returns events on that branch
    def test_list_events_with_branch_filter(self):
        actor_id = f"{self.id_prefix}-brfilt-actor"
        session_id = f"{self.id_prefix}-brfilt-session"
        base = self.client.save_conversation(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=[("base msg", "USER"), ("base reply", "ASSISTANT")],
        )
        poll_until(
            fn=lambda: self.client.list_events(self.memory_id, actor_id, session_id),
            predicate=lambda e: len(e) >= 1,
        )
        self.client.fork_conversation(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            root_event_id=base["eventId"],
            branch_name="filter-branch",
            new_messages=[("forked msg", "USER"), ("forked reply", "ASSISTANT")],
        )
        events = poll_until(
            fn=lambda: self.client.list_events(self.memory_id, actor_id, session_id, branch_name="filter-branch"),
            predicate=lambda e: len(e) >= 1,
        )
        assert len(events) >= 1

    @pytest.mark.order(4)
    # list_events respects max_results limit
    def test_list_events_with_max_results(self):
        actor_id = f"{self.id_prefix}-max-actor"
        session_id = f"{self.id_prefix}-max-session"
        for i in range(3):
            self.client.save_conversation(
                memory_id=self.memory_id,
                actor_id=actor_id,
                session_id=session_id,
                messages=[(f"msg {i}", "USER"), (f"reply {i}", "ASSISTANT")],
            )
        poll_until(
            fn=lambda: self.client.list_events(self.memory_id, actor_id, session_id),
            predicate=lambda e: len(e) >= 3,
        )
        limited = self.client.list_events(self.memory_id, actor_id, session_id, max_results=2)
        assert len(limited) == 2

    @pytest.mark.order(5)
    # get_last_k_turns returns exactly k turns
    def test_get_last_k_turns(self):
        actor_id = f"{self.id_prefix}-turns-actor"
        session_id = f"{self.id_prefix}-turns-session"
        messages = []
        for i in range(5):
            messages.extend([(f"user msg {i}", "USER"), (f"assistant msg {i}", "ASSISTANT")])
        self.client.save_conversation(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=messages,
        )
        turns = poll_until(
            fn=lambda: self.client.get_last_k_turns(
                memory_id=self.memory_id, actor_id=actor_id, session_id=session_id, k=2
            ),
            predicate=lambda t: len(t) == 2,
        )
        assert len(turns) == 2

    @pytest.mark.order(6)
    # get_last_k_turns works with branch_name="main"
    def test_get_last_k_turns_with_branch(self):
        actor_id = f"{self.id_prefix}-turnsbr-actor"
        session_id = f"{self.id_prefix}-turnsbr-session"
        messages = []
        for i in range(5):
            messages.extend([(f"user msg {i}", "USER"), (f"assistant msg {i}", "ASSISTANT")])
        self.client.save_conversation(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=messages,
        )
        turns = poll_until(
            fn=lambda: self.client.get_last_k_turns(
                memory_id=self.memory_id, actor_id=actor_id, session_id=session_id, k=2, branch_name="main"
            ),
            predicate=lambda t: len(t) == 2,
        )
        assert len(turns) == 2

    @pytest.mark.order(7)
    # get_last_k_turns returns the most recent turns, not the earliest
    def test_get_last_k_turns_returns_last_not_first(self):
        actor_id = f"{self.id_prefix}-order-actor"
        session_id = f"{self.id_prefix}-order-session"
        messages = []
        for i in range(5):
            messages.extend([(f"user msg {i}", "USER"), (f"assistant msg {i}", "ASSISTANT")])
        self.client.save_conversation(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=messages,
        )
        turns = poll_until(
            fn=lambda: self.client.get_last_k_turns(
                memory_id=self.memory_id, actor_id=actor_id, session_id=session_id, k=10
            ),
            predicate=lambda t: len(t) >= 1,
        )
        last_turn = turns[-1]
        first_msg_text = last_turn[0].get("content", {}).get("text", "")
        assert "user msg 4" in first_msg_text
        last_msg_text = last_turn[-1].get("content", {}).get("text", "")
        assert "assistant msg 4" in last_msg_text

    @pytest.mark.order(8)
    # fork_conversation creates a new branch from an existing event
    def test_fork_conversation(self):
        actor_id = f"{self.id_prefix}-fork-actor"
        session_id = f"{self.id_prefix}-fork-session"
        base = self.client.save_conversation(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=[("base msg", "USER"), ("base reply", "ASSISTANT")],
        )
        self.__class__.fork_actor_id = actor_id
        self.__class__.fork_session_id = session_id

        poll_until(
            fn=lambda: self.client.list_events(self.memory_id, actor_id, session_id),
            predicate=lambda e: len(e) >= 1,
        )
        result = self.client.fork_conversation(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            root_event_id=base["eventId"],
            branch_name="test-fork",
            new_messages=[("forked msg", "USER"), ("forked reply", "ASSISTANT")],
        )
        assert "eventId" in result
        branches = poll_until(
            fn=lambda: self.client.list_branches(self.memory_id, actor_id, session_id),
            predicate=lambda b: any(br["name"] == "test-fork" for br in b),
        )
        assert any(br["name"] == "test-fork" for br in branches)

    @pytest.mark.order(9)
    # list_branches returns all branches including forks
    def test_list_branches(self):
        if not getattr(self, "fork_actor_id", None):
            pytest.skip("fork test did not run")
        branches = poll_until(
            fn=lambda: self.client.list_branches(self.memory_id, self.fork_actor_id, self.fork_session_id),
            predicate=lambda b: len(b) >= 2,
        )
        assert len(branches) >= 2

    @pytest.mark.order(10)
    # 5 branches created in quick succession all persist
    def test_rapid_branch_creation(self):
        actor_id = f"{self.id_prefix}-rapid-actor"
        session_id = f"{self.id_prefix}-rapid-session"
        base = self.client.save_conversation(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=[("base", "USER"), ("reply", "ASSISTANT")],
        )
        poll_until(
            fn=lambda: self.client.list_events(self.memory_id, actor_id, session_id),
            predicate=lambda e: len(e) >= 1,
        )
        results = []
        for i in range(5):
            try:
                result = self.client.fork_conversation(
                    memory_id=self.memory_id,
                    actor_id=actor_id,
                    session_id=session_id,
                    root_event_id=base["eventId"],
                    branch_name=f"rapid-branch-{i}",
                    new_messages=[(f"fork {i}", "USER"), (f"reply {i}", "ASSISTANT")],
                )
                results.append(result)
            except Exception as e:
                results.append(e)
        assert all(not isinstance(r, Exception) for r in results), f"Some forks failed: {results}"
        branches = poll_until(
            fn=lambda: self.client.list_branches(self.memory_id, actor_id, session_id),
            predicate=lambda b: len(b) >= 6,  # main + 5 forks
        )
        branch_names = [b["name"] for b in branches]
        for i in range(5):
            assert f"rapid-branch-{i}" in branch_names

    # --- Retrieval Tests (order 11-15) ---
    # Uses a pre-populated memory (MEMORY_PREPOPULATED_ID) to test retrieve_memories
    # without depending on flaky extraction timing. See TESTING.md for setup.

    @pytest.mark.order(11)
    # retrieve_memories returns semantic facts from /facts/ namespace
    def test_retrieve_semantic_memories(self):
        results = self.client.retrieve_memories(
            memory_id=self.prepopulated_memory_id,
            namespace=f"/facts/{self.prepopulated_actor_id}/",
            query="developer preferences",
        )
        assert len(results) > 0, "Expected semantic memories in /facts/ namespace"

    @pytest.mark.order(12)
    # retrieve_memories returns user preferences from /preferences/ namespace
    def test_retrieve_preference_memories(self):
        results = self.client.retrieve_memories(
            memory_id=self.prepopulated_memory_id,
            namespace=f"/preferences/{self.prepopulated_actor_id}/",
            query="programming language",
        )
        assert len(results) > 0, "Expected preference memories in /preferences/ namespace"

    @pytest.mark.order(13)
    # retrieve_memories returns summaries from /summaries/ namespace
    def test_retrieve_summary_memories(self):
        results = self.client.retrieve_memories(
            memory_id=self.prepopulated_memory_id,
            namespace=f"/summaries/{self.prepopulated_actor_id}/{self.prepopulated_session_id}/",
            query="distributed systems",
        )
        assert len(results) > 0, "Expected summary memories in /summaries/ namespace"

    @pytest.mark.order(14)
    # retrieve_memories with a prefix namespace matches broader results
    def test_retrieve_memories_prefix_namespace(self):
        results = self.client.retrieve_memories(
            memory_id=self.prepopulated_memory_id,
            namespace="/facts/",
            query="developer preferences",
        )
        assert len(results) > 0, "Expected prefix namespace to match"

    @pytest.mark.order(15)
    # retrieve_memories with wildcard "*" returns empty (client-side guard)
    def test_retrieve_memories_wildcard_rejected(self):
        results = self.client.retrieve_memories(
            memory_id=self.prepopulated_memory_id,
            namespace="*",
            query="anything",
        )
        assert len(results) == 0, "Expected wildcard namespace to return empty results"

    # --- Lifecycle Tests (order 16-18) ---
    @pytest.mark.order(16)
    # create_memory_and_wait returns a memory in ACTIVE status
    def test_create_memory_and_wait(self):
        memory = self.client.create_memory_and_wait(
            name=f"{self.test_prefix}_lifecycle",
            strategies=[],
            event_expiry_days=7,
        )
        self.__class__.lifecycle_memory_id = memory.get("memoryId") or memory.get("id")
        self.memory_ids.append(self.lifecycle_memory_id)
        assert memory["status"] == "ACTIVE"

    @pytest.mark.order(17)
    # list_memories includes the newly created memory
    def test_list_memories(self):
        if not getattr(self, "lifecycle_memory_id", None):
            pytest.skip("create test did not run")
        memories = self.client.list_memories()
        assert any(
            m.get("memoryId") == self.lifecycle_memory_id or m.get("id") == self.lifecycle_memory_id for m in memories
        )

    @pytest.mark.order(18)
    # delete_memory_and_wait removes the memory
    def test_delete_memory_and_wait(self):
        if not getattr(self, "lifecycle_memory_id", None):
            pytest.skip("create test did not run")
        self.client.delete_memory_and_wait(self.lifecycle_memory_id)
        if self.lifecycle_memory_id in self.memory_ids:
            self.memory_ids.remove(self.lifecycle_memory_id)

    # --- Passthrough Tests (order 19-23) ---

    @pytest.mark.order(19)
    def test_list_actors(self):
        """list_actors returns actors that have created events in the memory."""
        response = self.client.list_actors(memoryId=self.memory_id)
        assert "actorSummaries" in response
        assert len(response["actorSummaries"]) > 0

    @pytest.mark.order(20)
    def test_list_sessions(self):
        """list_sessions returns sessions for a known actor."""
        actor_id = f"{self.id_prefix}-save-actor"
        response = self.client.list_sessions(memoryId=self.memory_id, actorId=actor_id)
        assert "sessionSummaries" in response
        assert len(response["sessionSummaries"]) > 0

    @pytest.mark.order(21)
    def test_get_event(self):
        """get_event retrieves an event by ID after creation."""
        actor_id = f"{self.id_prefix}-getev-actor"
        session_id = f"{self.id_prefix}-getev-session"
        created = self.client.save_conversation(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=[("test get event", "USER"), ("acknowledged", "ASSISTANT")],
        )
        event_id = created["eventId"]
        self.__class__.get_event_actor_id = actor_id
        self.__class__.get_event_session_id = session_id
        self.__class__.get_event_event_id = event_id

        poll_until(
            fn=lambda: self.client.list_events(self.memory_id, actor_id, session_id),
            predicate=lambda e: len(e) >= 1,
        )

        response = self.client.get_event(
            memoryId=self.memory_id,
            sessionId=session_id,
            actorId=actor_id,
            eventId=event_id,
        )
        assert response["event"]["eventId"] == event_id

    @pytest.mark.order(22)
    def test_delete_event(self):
        """delete_event removes an event and subsequent get_event raises ResourceNotFoundException."""
        event_id = getattr(self, "get_event_event_id", None)
        if not event_id:
            pytest.skip("test_get_event did not run")
        response = self.client.delete_event(
            memoryId=self.memory_id,
            sessionId=self.get_event_session_id,
            eventId=event_id,
            actorId=self.get_event_actor_id,
        )
        assert response["eventId"] == event_id

        with pytest.raises(self.client.gmdp_client.exceptions.ResourceNotFoundException):
            self.client.get_event(
                memoryId=self.memory_id,
                sessionId=self.get_event_session_id,
                actorId=self.get_event_actor_id,
                eventId=event_id,
            )

    @pytest.mark.order(23)
    def test_list_memory_records(self):
        """list_memory_records returns extracted records from a prepopulated memory."""
        response = self.client.list_memory_records(
            memoryId=self.prepopulated_memory_id,
            namespace=f"/facts/{self.prepopulated_actor_id}/",
        )
        assert "memoryRecordSummaries" in response
        assert len(response["memoryRecordSummaries"]) > 0
