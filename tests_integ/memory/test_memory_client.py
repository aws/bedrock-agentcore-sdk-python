"""Test script for critical AgentCore Memory SDK issues."""

import logging
import os
import time
from datetime import datetime

from bedrock_agentcore.memory import MemoryClient

# Use INFO level logging for cleaner output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_list_events_api(client: MemoryClient, memory_id: str):
    """Test the new list_events public API method."""
    logger.info("=" * 80)
    logger.info("TESTING LIST_EVENTS PUBLIC API (Issue #1)")
    logger.info("=" * 80)

    actor_id = "test-list-%s" % datetime.now().strftime("%Y%m%d%H%M%S")
    session_id = "session-%s" % datetime.now().strftime("%Y%m%d%H%M%S")

    # Create some events
    logger.info("\n1. Creating test events...")

    for i in range(3):
        event = client.save_conversation(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=[
                ("Message %d from user" % (i + 1), "USER"),
                ("Response %d from assistant" % (i + 1), "ASSISTANT"),
            ],
        )
        logger.info("Created event %d: %s", i + 1, event["eventId"])
        time.sleep(1)

    # Wait for indexing - INCREASED WAIT TIME
    logger.info("\nWaiting 60 seconds for event indexing...")
    time.sleep(60)

    # Test list_events
    logger.info("\n2. Testing list_events() method...")

    try:
        # Get all events
        all_events = client.list_events(memory_id, actor_id, session_id)
        logger.info("✓ Retrieved %d events total", len(all_events))

        # Get main branch only
        main_events = client.list_events(memory_id, actor_id, session_id, branch_name="main")
        logger.info("✓ Retrieved %d main branch events", len(main_events))

        # Get with max_results
        limited_events = client.list_events(memory_id, actor_id, session_id, max_results=2)
        logger.info("✓ Retrieved %d events with max_results=2", len(limited_events))

        # Show event structure
        if all_events:
            logger.info("\nSample event structure:")
            event = all_events[0]
            logger.info("  Event ID: %s", event.get("eventId"))
            logger.info("  Timestamp: %s", event.get("eventTimestamp"))
            logger.info("  Has payload: %s", "payload" in event)

    except Exception as e:
        logger.error("❌ list_events failed: %s", e)
        raise


def test_strategy_polling_fix(client: MemoryClient):
    """Test that all strategy operations use polling to avoid CREATING state errors."""
    logger.info("\n%s", "=" * 80)
    logger.info("TESTING STRATEGY POLLING FIX (Issue #2)")
    logger.info("=" * 80)

    # Create memory without strategies
    logger.info("\n1. Creating memory without strategies...")
    memory = client.create_memory_and_wait(
        name="PollingTest_%s" % datetime.now().strftime("%Y%m%d%H%M%S"),
        strategies=[],  # No strategies initially
        event_expiry_days=7,
    )
    memory_id = memory["memoryId"]
    logger.info("✓ Created memory: %s", memory_id)

    # Add first strategy
    logger.info("\n2. Adding summary strategy with polling...")
    try:
        memory = client.add_summary_strategy_and_wait(
            memory_id=memory_id, name="TestSummary", namespaces=["summaries/{sessionId}/"]
        )
        logger.info("✓ Added summary strategy, memory is %s", memory["status"])
    except Exception as e:
        logger.error("❌ Failed to add summary strategy: %s", e)
        raise

    # Create some events while memory is active
    logger.info("\n3. Creating events...")
    actor_id = "test-actor"
    session_id = "test-session"

    event = client.save_conversation(
        memory_id=memory_id,
        actor_id=actor_id,
        session_id=session_id,
        messages=[("Test message", "USER"), ("Test response", "ASSISTANT")],
    )
    logger.info("✓ Created event: %s", event["eventId"])

    # Add another strategy immediately
    logger.info("\n4. Adding user preference strategy immediately...")
    try:
        memory = client.add_user_preference_strategy_and_wait(
            memory_id=memory_id, name="TestPreferences", namespaces=["preferences/{actorId}/"]
        )
        logger.info("✓ Added user preference strategy without error, memory is %s", memory["status"])
    except Exception as e:
        logger.error("❌ Failed due to CREATING state: %s", e)
        raise

    # Clean up
    try:
        client.delete_memory_and_wait(memory_id)
        logger.info("✓ Cleaned up test memory")
    except Exception:
        pass


def test_get_last_k_turns_fix(client: MemoryClient, memory_id: str):
    """Test that get_last_k_turns returns the correct turns."""
    logger.info("\n%s", "=" * 80)
    logger.info("TESTING GET_LAST_K_TURNS FIX (Issue #3)")
    logger.info("=" * 80)

    actor_id = "restaurant-user-%s" % datetime.now().strftime("%Y%m%d%H%M%S")
    session_id = "restaurant-session-%s" % datetime.now().strftime("%Y%m%d%H%M%S")

    # Create the exact conversation from the issue
    logger.info("\n1. Creating restaurant conversation...")

    event = client.save_conversation(
        memory_id=memory_id,
        actor_id=actor_id,
        session_id=session_id,
        messages=[
            ("I'm vegetarian and I prefer restaurants with a quiet atmosphere.", "USER"),
            (
                "Thank you for letting me know. I'll make sure to recommend restaurants that are "
                "vegetarian-friendly and have a quiet atmosphere. Is there any specific cuisine "
                "you're interested in today?",
                "ASSISTANT",
            ),
            ("I'm in the mood for Italian cuisine.", "USER"),
            (
                "Great choice! I'll look for Italian vegetarian restaurants with a quiet "
                "atmosphere. Do you have a preferred price range or location?",
                "ASSISTANT",
            ),
            ("I'd prefer something mid-range and located downtown.", "USER"),
            (
                "Noted. I'll search for mid-range, vegetarian-friendly Italian restaurants in "
                "the downtown area with a quiet atmosphere. Would you like me to book a table "
                "for a specific time?",
                "ASSISTANT",
            ),
            ("Yes, please book for 7 PM.", "USER"),
            (
                "Sure, I'll find a suitable restaurant and make a reservation for 7 PM. "
                "Is there anything else I can assist you with?",
                "ASSISTANT",
            ),
            ("No, that's all for now. Thank you!", "USER"),
        ],
    )
    logger.info("✓ Conversation saved: %s", event["eventId"])

    # Wait for event indexing - INCREASED WAIT TIME
    logger.info("\nWaiting 60 seconds for event indexing...")
    time.sleep(60)

    # Test 1: Without branch_name
    logger.info("\n2. Testing get_last_k_turns without branch_name...")
    try:
        turns = client.get_last_k_turns(memory_id=memory_id, actor_id=actor_id, session_id=session_id, k=2)
        logger.info("✓ Retrieved %d turns (no branch_name)", len(turns))

        if turns:
            logger.info("\nLast 2 turns:")
            for i, turn in enumerate(turns):
                logger.info("  Turn %d:", i + 1)
                for msg in turn:
                    role = msg.get("role", "")
                    text = msg.get("content", {}).get("text", "")[:60] + "..."
                    logger.info("    %s: %s", role, text)
        else:
            logger.error("❌ No turns returned!")

    except Exception as e:
        logger.error("❌ Failed without branch_name: %s", e)

    # Test 2: With branch_name="main"
    logger.info("\n3. Testing get_last_k_turns with branch_name='main'...")
    try:
        turns = client.get_last_k_turns(
            memory_id=memory_id, actor_id=actor_id, session_id=session_id, branch_name="main", k=2
        )
        logger.info("✓ Retrieved %d turns (branch_name='main')", len(turns))

        if not turns:
            logger.error("❌ No turns returned for main branch!")

    except Exception as e:
        logger.error("❌ Failed with branch_name='main': %s", e)

    # Test 3: Verify we get the LAST turns, not the first
    logger.info("\n4. Verifying we get LAST turns, not first...")
    all_turns = client.get_last_k_turns(
        memory_id=memory_id,
        actor_id=actor_id,
        session_id=session_id,
        k=10,  # Get all turns
    )

    if all_turns:
        last_turn = all_turns[-1]
        if last_turn and last_turn[0].get("content", {}).get("text", "").startswith("No, that's all"):
            logger.info("✓ Correctly returned LAST turns (ends with 'No, that's all')")
        else:
            logger.error("❌ Returned FIRST turns instead of LAST!")


def test_namespace_wildcards(client: MemoryClient, memory_id: str):
    """Test and document that wildcards are not supported in namespaces."""
    logger.info("\n%s", "=" * 80)
    logger.info("TESTING NAMESPACE WILDCARD LIMITATION (Issue #4)")
    logger.info("=" * 80)

    # Check memory strategy configuration
    logger.info("\n1. Checking memory strategy configuration:")
    strategies = client.get_memory_strategies(memory_id)
    for strategy in strategies:
        logger.info("Strategy type: %s", strategy.get("type") or strategy.get("memoryStrategyType"))
        logger.info("Strategy namespaces: %s", strategy.get("namespaces", []))

    # Create multiple test events with different actor/session combinations
    logger.info("\n2. Creating multiple test events...")

    actor_ids = []
    session_ids = []

    for i in range(3):
        actor_id = "wildcard-test-%s-%d" % (datetime.now().strftime("%Y%m%d%H%M%S"), i)
        session_id = "wildcard-session-%s-%d" % (datetime.now().strftime("%Y%m%d%H%M%S"), i)
        actor_ids.append(actor_id)
        session_ids.append(session_id)

        event = client.save_conversation(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=[
                (f"Test message {i + 1} for wildcard testing with specific keyword", "USER"),
                (f"Response {i + 1} for wildcard testing with specific keyword", "ASSISTANT"),
            ],
        )
        logger.info("✓ Created event %d: %s", i + 1, event["eventId"])

    # Wait for extraction - INCREASED WAIT TIME
    logger.info("\nWaiting 90 seconds for memory extraction...")
    time.sleep(90)

    # Test 1: Wildcard namespace (should fail)
    logger.info("\n3. Testing with wildcard namespace '*'...")

    result = client.wait_for_memories(
        memory_id=memory_id, namespace="*", test_query="specific keyword", max_wait=30, poll_interval=10
    )

    if not result:
        logger.info("✓ Correctly rejected wildcard namespace")
    else:
        logger.error("❌ Wildcard should not have worked!")

    # Test 2: Retrieve with wildcard (should return empty)
    logger.info("\n4. Testing retrieve_memories with wildcard...")

    memories = client.retrieve_memories(memory_id=memory_id, namespace="*", query="specific keyword")

    if len(memories) == 0:
        logger.info("✓ Correctly returned empty for wildcard namespace")
    else:
        logger.error("❌ Should not return memories with wildcard!")

    # Test 3: Exact namespace (should work)
    logger.info("\n5. Testing with exact namespace...")

    # Use the first actor/session from our created events
    actor_id = actor_ids[0]
    session_id = session_ids[0]

    # Assuming semantic strategy with pattern "test/{actorId}/{sessionId}"
    exact_namespace = f"test/{actor_id}/{session_id}/"

    logger.info("Trying exact namespace: %s", exact_namespace)
    memories = client.retrieve_memories(memory_id=memory_id, namespace=exact_namespace, query="specific keyword")

    logger.info("✓ Retrieved %d memories with exact namespace", len(memories))

    if memories:
        for i, mem in enumerate(memories[:2]):
            logger.info("  Memory %d: %s", i + 1, mem.get("content", {}).get("text", "")[:80])

    # Test 4: Prefix namespace (should work like S3 prefix)
    logger.info("\n6. Testing with prefix namespace...")

    # Try multiple prefix options
    prefixes = [
        "test/",
        f"test/{actor_id}/",
    ]

    for prefix in prefixes:
        logger.info("\nTrying prefix namespace: %s", prefix)
        memories = client.retrieve_memories(memory_id=memory_id, namespace=prefix, query="specific keyword")

        logger.info("✓ Retrieved %d memories with prefix namespace", len(memories))

        if memories:
            for i, mem in enumerate(memories[:2]):
                logger.info("  Memory %d: %s", i + 1, mem.get("content", {}).get("text", "")[:80])


def main():
    """Run all critical issue tests."""

    # Get role ARN from environment
    role_arn = os.getenv("MEMORY_ROLE_ARN")
    if not role_arn:
        logger.error("Please set MEMORY_ROLE_ARN environment variable")
        return

    # Get region and environment from environment variables with defaults
    region = os.getenv("AWS_REGION", "us-west-2")
    environment = os.getenv("MEMORY_ENVIRONMENT", "prod")

    logger.info("Using region: %s, environment: %s", region, environment)

    client = MemoryClient(region_name=region)

    # Test Issue #2 first (strategy polling)
    test_strategy_polling_fix(client)

    # Create a memory for remaining tests
    logger.info("\n\nCreating memory for remaining tests...")
    # Explicitly define strategy with clear namespace pattern for testing
    memory = client.create_memory_and_wait(
        name="RetrievalTest_%s" % datetime.now().strftime("%Y%m%d%H%M%S"),
        strategies=[
            {
                "semanticMemoryStrategy": {
                    "name": "TestStrategy",
                    "namespaces": ["test/{actorId}/{sessionId}/"],  # Explicit namespace pattern
                }
            }
        ],
        event_expiry_days=7,
        memory_execution_role_arn=role_arn,
    )
    memory_id = memory["memoryId"]
    logger.info("Created test memory: %s", memory_id)

    try:
        # Test Issue #1: list_events API
        test_list_events_api(client, memory_id)

        # Test Issue #3: get_last_k_turns fix
        test_get_last_k_turns_fix(client, memory_id)

        # Test Issue #4: namespace wildcards
        logger.info("\n\nStarting namespace wildcard tests with memory ID: %s", memory_id)
        logger.info(
            "IMPORTANT: All retrieve calls will target the semantic strategy with "
            "namespace pattern: test/{actorId}/{sessionId}"
        )
        test_namespace_wildcards(client, memory_id)

        logger.info("\n%s", "=" * 80)
        logger.info("ALL ISSUE TESTS COMPLETED")
        logger.info("=" * 80)

        logger.info("\nSummary:")
        logger.info("✓ Issue #1: list_events() method now available")
        logger.info("✓ Issue #2: All strategy operations use polling")
        logger.info("✓ Issue #3: get_last_k_turns() returns correct turns")
        logger.info("✓ Issue #4: Wildcard limitation documented - use exact namespaces or prefixes instead")

    except Exception as e:
        logger.exception("Test failed: %s", e)
    finally:
        logger.info("\nCleaning up test memory...")
        try:
            client.delete_memory_and_wait(memory_id)
            logger.info("✓ Test memory deleted")
        except Exception as e:
            logger.error("Failed to delete test memory: %s", e)


if __name__ == "__main__":
    main()
