"""Integration tests for MemoryClient — full public-method coverage.

Covers all 37 public methods of MemoryClient across 11 workflow tests.
Every test is fully independent — no inter-test dependencies.

Run with: pytest -xvs tests_integ/memory/test_memory_client.py
Parallel: pytest -xvs tests_integ/memory/test_memory_client.py -n auto --dist loadscope
"""

import logging
import os
from datetime import datetime

import pytest

from bedrock_agentcore.memory import MemoryClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestMemoryClient:
    """Integration tests for MemoryClient."""

    MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

    RESTAURANT_CONVERSATION = [
        ("I'm vegetarian and I prefer restaurants with a quiet atmosphere.", "USER"),
        (
            "Thank you for letting me know. I'll recommend vegetarian-friendly, "
            "quiet restaurants. Any specific cuisine?",
            "ASSISTANT",
        ),
        ("I'm in the mood for Italian cuisine.", "USER"),
        ("Great choice! Do you have a preferred price range or location?", "ASSISTANT"),
        ("I'd prefer something mid-range and located downtown.", "USER"),
        (
            "I'll search for mid-range, vegetarian-friendly Italian restaurants "
            "downtown. Would you like me to book a table?",
            "ASSISTANT",
        ),
        ("Yes, please book for 7 PM.", "USER"),
        (
            "I'll find a suitable restaurant and make a reservation for 7 PM. Anything else?",
            "ASSISTANT",
        ),
        ("That's all for now. Thank you!", "USER"),
    ]

    @pytest.fixture(scope="class", autouse=True)
    def setup_client(self, request):
        """Provision MemoryClient and per-worker test prefix."""
        region = os.environ.get("AWS_REGION", "us-west-2")

        role_arn = os.environ.get("MEMORY_ROLE_ARN")
        if not role_arn:
            pytest.fail("MEMORY_ROLE_ARN environment variable is not set")

        request.cls.role_arn = role_arn
        request.cls.client = MemoryClient(region_name=region)

        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
        request.cls.test_prefix = "mc_%s_%s" % (datetime.now().strftime("%Y%m%d%H%M%S"), worker_id)
        request.cls.memory_ids = []

        yield

        # Teardown — delete control-plane test memories
        for memory_id in request.cls.memory_ids:
            try:
                request.cls.client.delete_memory(memory_id)
                logger.info("Deleting test memory: %s", memory_id)
            except Exception as e:
                logger.error("Failed to delete test memory %s: %s", memory_id, e)

    @pytest.fixture(scope="class")
    def shared_memory(self, request):
        """Create or get the shared memory for data-plane tests."""
        logger.info("Creating/getting shared memory for tests...")
        memory = request.cls.client.create_or_get_memory(
            name="SharedDataPlane",
            strategies=[
                {
                    "semanticMemoryStrategy": {
                        "name": "TestStrategy",
                        "namespaces": ["test/{actorId}/{sessionId}/"],
                    }
                }
            ],
            event_expiry_days=7,
            memory_execution_role_arn=request.cls.role_arn,
        )
        request.cls.memory_id = memory["memoryId"]
        request.cls.client._wait_for_memory_active(request.cls.memory_id, max_wait=300, poll_interval=10)
        logger.info("Shared test memory: %s", request.cls.memory_id)

        yield

        # Teardown — delete shared memory
        try:
            request.cls.client.delete_memory(request.cls.memory_id)
            logger.info("Deleted shared memory: %s", request.cls.memory_id)
        except Exception as e:
            logger.error("Failed to delete shared memory %s: %s", request.cls.memory_id, e)

    @pytest.fixture()
    def ids(self, request):
        """Generate unique actor_id and session_id from the test name."""
        suffix = request.node.name.replace("test_", "", 1)
        request.cls.actor_id = "actor_%s_%s" % (self.test_prefix, suffix)
        request.cls.session_id = "session_%s_%s" % (self.test_prefix, suffix)

    # ------------------------------------------------------------------
    # Data-plane tests (use the shared memory)
    # ------------------------------------------------------------------

    def test_create_event(self, shared_memory, ids):
        """create_event: store a USER + ASSISTANT conversation event and verify via list_events."""
        event = self.client.create_event(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            messages=self.RESTAURANT_CONVERSATION,
        )
        self.assert_created_event(event)

        # Verify content persisted via list_events
        events = self.client.list_events(self.memory_id, self.actor_id, self.session_id)
        assert len(events) >= 1, "Expected at least 1 event"
        self.assert_listed_event(events[0], self.RESTAURANT_CONVERSATION)
        logger.info("Created and verified event: %s", event["eventId"])

    def test_create_blob_event(self, shared_memory, ids):
        """create_blob_event: store a blob payload and verify via list_events."""
        blob_data = {"type": "tool_output", "result": {"temperature": 72, "unit": "F"}}

        event = self.client.create_blob_event(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            blob_data=blob_data,
        )
        self.assert_created_event(event)

        # Verify blob payload persisted via list_events
        events = self.client.list_events(self.memory_id, self.actor_id, self.session_id)
        assert len(events) >= 1, "Expected at least 1 event"
        self.assert_blob_event(events[0], blob_data)
        logger.info("Created and verified blob event: %s", event["eventId"])

    def test_branching(self, shared_memory, ids):
        """fork_conversation, list_branches, list_branch_events, merge_branch_context, get_conversation_tree."""
        branch_messages = [
            ("Actually, can we switch to Japanese cuisine instead?", "USER"),
            ("Of course! I'll look for mid-range Japanese restaurants downtown.", "ASSISTANT"),
        ]

        # Create a root event to fork from
        root_event = self.client.create_event(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            messages=self.RESTAURANT_CONVERSATION,
        )
        root_event_id = root_event["eventId"]

        # Fork from the root event
        branch_name = "alt_branch"
        fork_event = self.client.fork_conversation(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            root_event_id=root_event_id,
            branch_name=branch_name,
            new_messages=branch_messages,
        )
        self.assert_created_event(fork_event)
        logger.info("Forked conversation: %s", fork_event["eventId"])

        # list_branches
        branches = self.client.list_branches(
            memory_id=self.memory_id, actor_id=self.actor_id, session_id=self.session_id
        )
        assert isinstance(branches, list), "list_branches must return a list"
        branch_names = [b.get("branchName", b.get("name", "")) for b in branches]
        assert branch_name in branch_names, "Branch %r not found in %s" % (branch_name, branch_names)
        logger.info("Branches: %s", branch_names)

        # list_branch_events — verify fork messages appear
        branch_events = self.client.list_branch_events(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            branch_name=branch_name,
        )
        assert len(branch_events) >= 1, "Branch should have at least one event"
        self.assert_listed_event(branch_events[0], branch_messages)
        logger.info("Branch events count: %d", len(branch_events))

        # merge_branch_context — should contain messages from both root and branch
        context = self.client.merge_branch_context(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            branch_name=branch_name,
        )
        expected_merged = self.RESTAURANT_CONVERSATION + branch_messages
        assert len(context) == len(expected_merged), "Expected %d merged messages (9 root + 2 branch), got %d" % (
            len(expected_merged),
            len(context),
        )
        for i, (text, role) in enumerate(expected_merged):
            assert context[i]["content"] == text, "Merged message %d content mismatch: expected %r, got %r" % (
                i,
                text,
                context[i]["content"],
            )
            assert context[i]["role"] == role, "Merged message %d role mismatch: expected %s, got %s" % (
                i,
                role,
                context[i]["role"],
            )
        logger.info("Merged context: %d messages", len(context))

        # get_conversation_tree — verify structure
        tree = self.client.get_conversation_tree(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
        )
        self.assert_tree_has_branch(tree, branch_name, min_main_events=0)
        logger.info(
            "Conversation tree: %d main events, %d branches",
            len(tree["main_branch"]["events"]),
            len(tree["main_branch"]["branches"]),
        )

    def test_list_events(self, shared_memory, ids):
        """list_events: all, branch filter, max_results."""
        # Create 3 events
        for i in range(3):
            self.client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[
                    ("Question %d" % (i + 1), "USER"),
                    ("Answer %d" % (i + 1), "ASSISTANT"),
                ],
            )

        # All events
        all_events = self.client.list_events(self.memory_id, self.actor_id, self.session_id)
        assert len(all_events) >= 3, "Expected >=3 events, got %d" % len(all_events)
        logger.info("Total events: %d", len(all_events))

        # Main branch filter
        main_events = self.client.list_events(self.memory_id, self.actor_id, self.session_id, branch_name="main")
        assert isinstance(main_events, list)
        logger.info("Main branch events: %d", len(main_events))

        # max_results
        limited = self.client.list_events(self.memory_id, self.actor_id, self.session_id, max_results=2)
        assert len(limited) <= 2, "max_results=2 returned %d events" % len(limited)
        logger.info("Limited events (max_results=2): %d", len(limited))

        # Verify all 3 events (list_events returns newest first)
        self.assert_listed_event(all_events[0], [("Question 3", "USER"), ("Answer 3", "ASSISTANT")])
        self.assert_listed_event(all_events[1], [("Question 2", "USER"), ("Answer 2", "ASSISTANT")])
        self.assert_listed_event(all_events[2], [("Question 1", "USER"), ("Answer 1", "ASSISTANT")])

    def test_get_last_k_turns(self, shared_memory, ids):
        """get_last_k_turns: main branch and named branch after fork."""
        # 9-message restaurant conversation (5 USER, 4 ASSISTANT)
        self.client.create_event(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            messages=self.RESTAURANT_CONVERSATION,
        )

        # Verify the event was stored with all 9 messages
        events = self.client.list_events(self.memory_id, self.actor_id, self.session_id)
        assert len(events) >= 1, "Expected at least 1 event"
        self.assert_listed_event(events[0], self.RESTAURANT_CONVERSATION)

        # Main branch — all 9 messages
        all_turns = self.client.get_last_k_turns(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            k=10,
        )
        self.assert_turns(all_turns, self.RESTAURANT_CONVERSATION)

        # Fork into INDIAN_FOOD branch from the root event
        root_event_id = events[0]["eventId"]
        self.client.fork_conversation(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            root_event_id=root_event_id,
            branch_name="INDIAN_FOOD",
            new_messages=[
                ("Actually, I changed my mind. How about Indian food?", "USER"),
                ("Indian cuisine is a great choice! Do you like spicy food?", "ASSISTANT"),
            ],
        )

        # Get turns from the INDIAN_FOOD branch (only branch events, no parent)
        branch_turns = self.client.get_last_k_turns(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            branch_name="INDIAN_FOOD",
            k=10,
        )
        assert branch_turns, "get_last_k_turns on INDIAN_FOOD branch returned no turns"
        self.assert_turns(
            branch_turns,
            [
                ("Actually, I changed my mind. How about Indian food?", "USER"),
                ("Indian cuisine is a great choice! Do you like spicy food?", "ASSISTANT"),
            ],
        )
        logger.info("Verified INDIAN_FOOD branch: 2 messages")

    def test_retrieve_memories(self, shared_memory, ids):
        """wait_for_memories, retrieve_memories, wildcard namespace rejection."""
        self.client.create_event(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            messages=self.RESTAURANT_CONVERSATION,
        )

        namespace = "test/%s/%s/" % (self.actor_id, self.session_id)
        query = "vegetarian Italian restaurants downtown"

        # Wait for memory extraction
        logger.info("Waiting for memory extraction (up to 180s)...")
        extraction_done = self.client.wait_for_memories(
            memory_id=self.memory_id,
            namespace=namespace,
            test_query=query,
            max_wait=180,
            poll_interval=15,
        )

        results = self.client.retrieve_memories(
            memory_id=self.memory_id,
            namespace=namespace,
            query=query,
        )
        assert isinstance(results, list), "retrieve_memories must return a list"
        if extraction_done and len(results) > 0:
            logger.info("Retrieved %d memories", len(results))
        else:
            logger.warning(
                "Memory extraction done=%s, results=%d — extraction may need more time",
                extraction_done,
                len(results),
            )

        # Wildcard namespace should return empty (wildcards not supported)
        wildcard_results = self.client.retrieve_memories(
            memory_id=self.memory_id,
            namespace="test/*/",
            query="anything",
        )
        assert wildcard_results == [], "Wildcard namespace should return empty, got %d results" % len(wildcard_results)
        logger.info("Wildcard namespace correctly returned empty")

    def test_process_turn_with_llm(self, shared_memory, ids):
        """process_turn_with_llm: mock callback echoes input."""
        user_input = "Tell me about quantum computing"

        def mock_llm_callback(prompt, memories):
            return "Echo: %s (memories=%d)" % (prompt, len(memories))

        memories, response, event = self.client.process_turn_with_llm(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            user_input=user_input,
            llm_callback=mock_llm_callback,
        )

        assert isinstance(memories, list), "memories should be a list"
        assert isinstance(response, str), "response should be a string"
        assert user_input in response, "response should contain the user input"
        assert "eventId" in event, "event should have eventId"
        logger.info("process_turn_with_llm returned response: %s", response[:80])

    # ------------------------------------------------------------------
    # Control-plane tests (create their own memories)
    # ------------------------------------------------------------------

    def test_create_or_get_memory(self):
        """create_or_get_memory, list_memories."""
        name = "%s_create_or_get" % self.test_prefix

        # First call — creates memory
        memory1 = self.client.create_or_get_memory(
            name=name,
            strategies=[
                {
                    "semanticMemoryStrategy": {
                        "name": "Sem1",
                        "namespaces": ["test/{actorId}/"],
                    }
                }
            ],
            event_expiry_days=7,
            memory_execution_role_arn=self.role_arn,
        )
        memory_id = memory1["memoryId"]
        self.__class__.memory_ids.append(memory_id)
        logger.info("create_or_get_memory created: %s", memory_id)

        # Second call — should return existing memory
        memory2 = self.client.create_or_get_memory(
            name=name,
            strategies=[
                {
                    "semanticMemoryStrategy": {
                        "name": "Sem1",
                        "namespaces": ["test/{actorId}/"],
                    }
                }
            ],
            event_expiry_days=7,
            memory_execution_role_arn=self.role_arn,
        )
        assert memory2["memoryId"] == memory_id, "Second call should return the same memory"
        logger.info("create_or_get_memory returned existing: %s", memory2["memoryId"])

        # list_memories
        all_memories = self.client.list_memories()
        memory_ids = [m["memoryId"] for m in all_memories]
        assert memory_id in memory_ids, "list_memories should include the created memory"
        logger.info("list_memories returned %d memories (includes ours)", len(all_memories))

    def test_add_builtin_strategies(self):
        """Test add_semantic/summary/user_preference/episodic_strategy_and_wait and get_memory_strategies."""
        # Create a fresh memory for strategy tests
        memory = self.client.create_memory_and_wait(
            name="%s_builtin_strat" % self.test_prefix,
            strategies=[],
            event_expiry_days=7,
            memory_execution_role_arn=self.role_arn,
        )
        memory_id = memory["memoryId"]
        self.__class__.memory_ids.append(memory_id)
        logger.info("Created strategy test memory: %s", memory_id)

        # 1) Semantic
        result = self.client.add_semantic_strategy_and_wait(
            memory_id=memory_id,
            name="BuiltinSemantic",
            namespaces=["sem/{actorId}/"],
        )
        self.assert_memory_active(result)
        logger.info("Added semantic strategy")

        # 2) Summary
        result = self.client.add_summary_strategy_and_wait(
            memory_id=memory_id,
            name="BuiltinSummary",
            namespaces=["sum/{sessionId}/"],
        )
        self.assert_memory_active(result)
        logger.info("Added summary strategy")

        # 3) User preference
        result = self.client.add_user_preference_strategy_and_wait(
            memory_id=memory_id,
            name="BuiltinUserPref",
            namespaces=["pref/{actorId}/"],
        )
        self.assert_memory_active(result)
        logger.info("Added user preference strategy")

        # 4) Episodic (reflection namespace must be same as or prefix of episodic namespace)
        result = self.client.add_episodic_strategy_and_wait(
            memory_id=memory_id,
            name="BuiltinEpisodic",
            reflection_namespaces=["ep/{actorId}/"],
            namespaces=["ep/{actorId}/"],
        )
        self.assert_memory_active(result)
        logger.info("Added episodic strategy")

        # Verify all 4 strategies via get_memory_strategies
        strategies = self.client.get_memory_strategies(memory_id)
        self.assert_strategies_by_name(
            strategies,
            ["BuiltinSemantic", "BuiltinSummary", "BuiltinUserPref", "BuiltinEpisodic"],
        )
        logger.info("Verified 4 built-in strategies")

    def test_custom_strategies_and_modification(self):
        """Test custom semantic/episodic strategies, modify_strategy, delete_strategy."""
        # Create its own memory with a seed strategy so we have something to work with
        memory = self.client.create_memory_and_wait(
            name="%s_custom_strat" % self.test_prefix,
            strategies=[
                {
                    "semanticMemoryStrategy": {
                        "name": "SeedStrategy",
                        "namespaces": ["seed/{actorId}/"],
                    }
                }
            ],
            event_expiry_days=7,
            memory_execution_role_arn=self.role_arn,
        )
        memory_id = memory["memoryId"]
        self.__class__.memory_ids.append(memory_id)
        logger.info("Created custom strategy test memory: %s", memory_id)

        # Add custom semantic strategy
        result = self.client.add_custom_semantic_strategy_and_wait(
            memory_id=memory_id,
            name="CustomSemantic",
            extraction_config={"prompt": "Extract key facts.", "modelId": self.MODEL_ID},
            consolidation_config={"prompt": "Consolidate facts.", "modelId": self.MODEL_ID},
            namespaces=["custom_sem/{actorId}/"],
        )
        self.assert_memory_active(result)
        logger.info("Added custom semantic strategy")

        # Add custom episodic strategy
        result = self.client.add_custom_episodic_strategy_and_wait(
            memory_id=memory_id,
            name="CustomEpisodic",
            extraction_config={"prompt": "Extract episodes.", "modelId": self.MODEL_ID},
            consolidation_config={"prompt": "Consolidate episodes.", "modelId": self.MODEL_ID},
            reflection_config={
                "prompt": "Reflect on episodes.",
                "modelId": self.MODEL_ID,
                "namespaces": ["custom_ep/{actorId}/"],
            },
            namespaces=["custom_ep/{actorId}/"],
        )
        self.assert_memory_active(result)
        logger.info("Added custom episodic strategy")

        # Get all strategies to find IDs
        strategies = self.client.get_memory_strategies(memory_id)
        strategy_map = {s["name"]: s for s in strategies}
        self.assert_strategies_by_name(strategies, ["SeedStrategy", "CustomSemantic", "CustomEpisodic"])

        # modify_strategy — update description on custom semantic
        custom_sem_id = strategy_map["CustomSemantic"]["strategyId"]
        self.client.modify_strategy(
            memory_id=memory_id,
            strategy_id=custom_sem_id,
            description="Updated custom semantic description",
        )
        logger.info("Modified custom semantic strategy description")

        # Wait for memory to become ACTIVE after modification
        self.client._wait_for_memory_active(memory_id, max_wait=300, poll_interval=10)

        # delete_strategy — delete the custom episodic strategy
        custom_ep_id = strategy_map["CustomEpisodic"]["strategyId"]
        self.client.delete_strategy(memory_id=memory_id, strategy_id=custom_ep_id)
        logger.info("Deleted custom episodic strategy")

        # Wait for memory to settle
        self.client._wait_for_memory_active(memory_id, max_wait=300, poll_interval=10)

        # update_memory_strategies_and_wait — delete the seed strategy
        seed_id = strategy_map["SeedStrategy"]["strategyId"]
        result = self.client.update_memory_strategies_and_wait(
            memory_id=memory_id,
            delete_strategy_ids=[seed_id],
        )
        self.assert_memory_active(result)
        logger.info("Deleted SeedStrategy via update_memory_strategies_and_wait")

        # Verify remaining strategies
        remaining = self.client.get_memory_strategies(memory_id)
        remaining_names = {s["name"] for s in remaining}
        assert "CustomEpisodic" not in remaining_names, "CustomEpisodic should have been deleted"
        assert "SeedStrategy" not in remaining_names, "SeedStrategy should have been deleted"
        assert "CustomSemantic" in remaining_names, "CustomSemantic should still exist"
        logger.info("Verified remaining strategies: %s", remaining_names)

    def test_delete_memory_and_wait(self):
        """delete_memory_and_wait: create throwaway memory, delete it, verify gone."""
        memory = self.client.create_memory_and_wait(
            name="%s_throwaway" % self.test_prefix,
            strategies=[],
            event_expiry_days=7,
            memory_execution_role_arn=self.role_arn,
        )
        memory_id = memory["memoryId"]
        logger.info("Created throwaway memory: %s", memory_id)

        # Delete and wait
        self.client.delete_memory_and_wait(memory_id=memory_id)
        logger.info("delete_memory_and_wait completed for: %s", memory_id)

        # Verify it's gone — get_memory_status should raise
        with pytest.raises((Exception,), match="ResourceNotFoundException|not found"):
            self.client.get_memory_status(memory_id)
        logger.info("Confirmed memory %s is deleted (get_memory_status raised)", memory_id)

    # ------------------------------------------------------------------
    # Assertion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def assert_created_event(actual: dict[str, object]) -> None:
        """Assert a create_event/fork_conversation response has eventId and eventTimestamp."""
        assert "eventId" in actual, "Event must have eventId"
        assert "eventTimestamp" in actual, "Event must have eventTimestamp"

    @staticmethod
    def assert_listed_event(
        actual: dict[str, object],
        expected: list[tuple[str, str]],
    ) -> None:
        """Assert an event from list_events has full structure and matches expected messages."""
        assert "eventId" in actual, "Event must have eventId"
        assert "eventTimestamp" in actual, "Event must have eventTimestamp"
        assert "payload" in actual, "Listed event must have payload"
        assert len(actual["payload"]) == len(expected), "Expected %d payload items, got %d" % (
            len(expected),
            len(actual["payload"]),
        )
        for i, (text, role) in enumerate(expected):
            msg = actual["payload"][i]["conversational"]
            assert msg["role"] == role, "Message %d role: expected %s, got %s" % (i, role, msg["role"])
            assert msg["content"]["text"] == text, "Message %d text: expected %r, got %r" % (
                i,
                text,
                msg["content"]["text"],
            )

    @staticmethod
    def assert_blob_event(actual: dict[str, object], expected_blob: dict) -> None:
        """Assert an event has a valid blob payload matching expected data."""
        assert "eventId" in actual, "Event must have eventId"
        assert "eventTimestamp" in actual, "Event must have eventTimestamp"
        assert "payload" in actual, "Blob event must have payload"
        assert len(actual["payload"]) == 1, "Blob event should have 1 payload item"
        assert "blob" in actual["payload"][0], "Payload item should contain a blob key"
        actual_blob = str(actual["payload"][0]["blob"])

        # Service returns blob as a stringified Java-style map; check all leaf values are present
        def _leaf_values(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    yield from _leaf_values(v)
            else:
                yield obj

        for leaf in _leaf_values(expected_blob):
            assert str(leaf) in actual_blob, "Blob missing leaf value %r in %s" % (leaf, actual_blob)

    @staticmethod
    def assert_tree_has_branch(
        tree: dict[str, object],
        branch_name: str,
        min_main_events: int = 1,
        min_branch_events: int = 1,
    ) -> None:
        """Assert a conversation tree has the expected structure and branch."""
        assert "main_branch" in tree, "Tree must have a main_branch"
        main = tree["main_branch"]
        assert "events" in main, "main_branch must have events"
        assert len(main["events"]) >= min_main_events, "main_branch should have >= %d events, got %d" % (
            min_main_events,
            len(main["events"]),
        )
        assert "branches" in main, "main_branch must have branches"
        assert branch_name in main["branches"], "Branch %r not in tree" % branch_name
        branch = main["branches"][branch_name]
        assert len(branch["events"]) >= min_branch_events, "Branch %r should have >= %d events, got %d" % (
            branch_name,
            min_branch_events,
            len(branch["events"]),
        )

    @staticmethod
    def assert_turns(
        actual: list[list[dict[str, object]]],
        expected: list[tuple[str, str]],
    ) -> None:
        """Assert flattened turn messages match expected (text, role) pairs."""
        messages = [msg for turn in actual for msg in turn]
        assert len(messages) == len(expected), "Expected %d messages, got %d" % (len(expected), len(messages))
        for i, (text, role) in enumerate(expected):
            assert messages[i]["role"] == role, "Message %d role: expected %s, got %s" % (i, role, messages[i]["role"])
            assert messages[i]["content"]["text"] == text, "Message %d text: expected %r, got %r" % (
                i,
                text,
                messages[i]["content"]["text"],
            )

    @staticmethod
    def assert_memory_active(result: dict[str, object]) -> None:
        """Assert a memory response shows ACTIVE status."""
        assert result["status"] == "ACTIVE", "Expected ACTIVE, got %s" % result["status"]

    @staticmethod
    def assert_strategies_by_name(strategies: list[dict[str, object]], expected_names: list[str]) -> None:
        """Assert a list of strategies contains exactly the expected names."""
        actual = {s["name"] for s in strategies}
        for name in expected_names:
            assert name in actual, "Missing strategy: %s (have %s)" % (name, actual)
        assert len(strategies) == len(expected_names), "Expected %d strategies, got %d: %s" % (
            len(expected_names),
            len(strategies),
            actual,
        )


if __name__ == "__main__":
    pytest.main(["-xvs", "test_memory_client.py"])
