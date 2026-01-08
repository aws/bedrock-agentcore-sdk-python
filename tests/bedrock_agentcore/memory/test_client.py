"""Unit tests for Memory Client - no external connections."""

import time
import uuid
import warnings
from datetime import datetime
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.constants import StrategyType


def test_client_initialization():
    """Test client initialization."""
    with patch("boto3.client") as mock_boto_client:
        # Setup the mock to return a consistent region_name
        mock_client_instance = MagicMock()
        mock_client_instance.meta.region_name = "us-west-2"
        mock_boto_client.return_value = mock_client_instance

        client = MemoryClient(region_name="us-west-2")

        # Check that the region was set correctly and boto3.client was called twice
        assert client.region_name == "us-west-2"
        assert mock_boto_client.call_count == 2


def test_client_initialization_region_mismatch():
    """Test client initialization with region mismatch warning."""

    with patch("boto3.client") as mock_boto_client:
        # First test - environment variable takes precedence
        with patch("boto3.Session") as mock_session:
            # Mock the session instance to simulate AWS_REGION=us-east-1
            mock_session_instance = MagicMock()
            mock_session_instance.region_name = "us-east-1"
            mock_session.return_value = mock_session_instance

            # Mock the boto client
            mock_client_instance = MagicMock()
            mock_client_instance.meta.region_name = "us-east-1"
            mock_boto_client.return_value = mock_client_instance

            # When region_name is provided, environment variable should still take precedence
            client1 = MemoryClient(region_name="us-west-2")
            assert client1.region_name == "us-west-2"

        # Second test - no environment variable, explicit param is used
        with patch("boto3.Session") as mock_session:
            # Mock the session instance to simulate no AWS_REGION set
            mock_session_instance = MagicMock()
            mock_session_instance.region_name = None
            mock_session.return_value = mock_session_instance

            # Mock the boto client
            mock_client_instance = MagicMock()
            mock_client_instance.meta.region_name = "us-west-2"
            mock_boto_client.return_value = mock_client_instance

            # When AWS_REGION is not set, explicitly provided region should be used
            client2 = MemoryClient(region_name="us-west-2")
            assert client2.region_name == "us-west-2"


def test_namespace_defaults():
    """Test namespace defaults."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Test strategy without namespace
        strategies = [{StrategyType.SEMANTIC.value: {"name": "TestStrategy"}}]
        processed = client._add_default_namespaces(strategies)

        assert "namespaces" in processed[0][StrategyType.SEMANTIC.value]


def test_create_memory():
    """Test create_memory."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock UUID generation to ensure deterministic test
        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Mock the gmcp_client
            mock_gmcp = MagicMock()
            client.gmcp_client = mock_gmcp

            # Mock successful response
            mock_gmcp.create_memory.return_value = {"memory": {"memoryId": "test-memory-123", "status": "CREATING"}}

            result = client.create_memory(
                name="TestMemory", strategies=[{StrategyType.SEMANTIC.value: {"name": "TestStrategy"}}]
            )

            assert result["memoryId"] == "test-memory-123"
            assert mock_gmcp.create_memory.called

            # Verify the client token was passed
            args, kwargs = mock_gmcp.create_memory.call_args
            assert kwargs.get("clientToken") == "12345678-1234-5678-1234-567812345678"


def test_save_conversation_and_retrieve_memories():
    """Test save_conversation and retrieve_memories."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the clients
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock retrieval response
        mock_gmdp.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [{"content": {"text": "Previous memory"}, "memoryRecordId": "rec-123"}]
        }

        # Mock event creation response
        mock_gmdp.create_event.return_value = {"event": {"eventId": "event-123", "memoryId": "mem-123"}}

        # Test UUID patch for deterministic testing
        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test retrieve_memories
            memories = client.retrieve_memories(memory_id="mem-123", namespace="test/namespace", query="Hello", top_k=3)

            assert len(memories) == 1
            assert memories[0]["memoryRecordId"] == "rec-123"

            # Test save_conversation
            event = client.save_conversation(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                messages=[("Hello", "USER"), ("Hi there", "ASSISTANT")],
            )

            assert event["eventId"] == "event-123"

            # Verify correct parameters were passed to create_event
            args, kwargs = mock_gmdp.create_event.call_args
            assert kwargs.get("clientToken") == "12345678-1234-5678-1234-567812345678"
            assert len(kwargs.get("payload", [])) == 2


def test_error_handling():
    """Test error handling."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client to raise an error
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid parameter"}}
        mock_gmcp.create_memory.side_effect = ClientError(error_response, "CreateMemory")

        try:
            client.create_memory(name="TestMemory", strategies=[{StrategyType.SEMANTIC.value: {"name": "Test"}}])
            raise AssertionError("Error was not raised as expected")
        except ClientError as e:
            assert "ValidationException" in str(e)


def test_branch_operations():
    """Test branch operations."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the clients
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock list_events response
        mock_gmdp.list_events.return_value = {
            "events": [
                {
                    "eventId": "event-1",
                    "eventTimestamp": datetime.now(),
                    "payload": [{"conversational": {"role": "USER", "content": {"text": "Hello"}}}],
                },
                {
                    "eventId": "event-2",
                    "eventTimestamp": datetime.now(),
                    "branch": {"name": "test-branch", "rootEventId": "event-1"},
                    "payload": [{"conversational": {"role": "USER", "content": {"text": "Branched message"}}}],
                },
            ]
        }

        # Test fork_conversation
        mock_gmdp.create_event.return_value = {"event": {"eventId": "event-3", "memoryId": "mem-123"}}

        # Test list_branches
        branches = client.list_branches(memory_id="mem-123", actor_id="user-123", session_id="session-456")
        assert len(branches) == 2

        # Test fork_conversation
        forked_event = client.fork_conversation(
            memory_id="mem-123",
            actor_id="user-123",
            session_id="session-456",
            root_event_id="event-1",
            branch_name="new-branch",
            new_messages=[("Fork message", "USER")],
        )

        assert forked_event["eventId"] == "event-3"


def test_memory_strategy_management():
    """Test memory strategy management."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the clients
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock get_memory response for strategy listing
        mock_gmcp.get_memory.return_value = {
            "memory": {
                "memoryId": "mem-123",
                "status": "ACTIVE",
                "memoryStrategies": [
                    {"memoryStrategyId": "strat-123", "memoryStrategyType": "SEMANTIC", "name": "Test Strategy"}
                ],
            }
        }

        # Mock update_memory response for strategy modifications
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-123", "status": "ACTIVE"}}

        # Test get_memory_strategies
        strategies = client.get_memory_strategies("mem-123")
        assert len(strategies) == 1
        assert strategies[0]["memoryStrategyId"] == "strat-123"

        # Test add_semantic_strategy
        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            client.add_semantic_strategy(memory_id="mem-123", name="New Semantic Strategy", description="Test strategy")

            assert mock_gmcp.update_memory.called
            args, kwargs = mock_gmcp.update_memory.call_args
            assert "memoryStrategies" in kwargs
            assert "addMemoryStrategies" in kwargs["memoryStrategies"]


def test_timestamp_and_advanced_message_handling():
    """Test timestamp and advanced message handling."""
    with patch("boto3.client"):
        client = MemoryClient()
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp
        mock_gmdp.create_event.return_value = {"event": {"eventId": "event-ts-1", "memoryId": "mem-123"}}

        custom_timestamp = datetime(2023, 1, 15, 12, 30, 45)

        # Test save_conversation with custom timestamps
        event = client.save_conversation(
            memory_id="mem-123",
            actor_id="user-123",
            session_id="session-456",
            messages=[("Hello", "USER"), ("Hi there", "ASSISTANT")],
            event_timestamp=custom_timestamp,
        )

        assert event["eventId"] == "event-ts-1"

        # Check timestamp was passed correctly
        args, kwargs = mock_gmdp.create_event.call_args
        assert kwargs.get("eventTimestamp") == custom_timestamp


def test_deprecated_methods():
    """Test deprecated methods with warnings."""
    with patch("boto3.client"):
        client = MemoryClient()
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Create responses for deprecated methods
        mock_gmdp.create_event.return_value = {"event": {"eventId": "event-dep-1", "memoryId": "mem-123"}}
        mock_gmdp.retrieve_memory_records.return_value = {"memoryRecordSummaries": []}

        # Use warnings.catch_warnings to verify deprecation warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Test deprecated save_turn method
            event = client.save_turn(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                user_input="Hello",
                agent_response="Hi",
            )

            # Test deprecated process_turn method
            memories, event = client.process_turn(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                user_input="Hello",
                agent_response="Hi",
                retrieval_namespace="test/ns",
            )

            assert len(w) >= 2
            assert any("save_turn() is deprecated" in str(warning.message) for warning in w)
            assert any("process_turn() is deprecated" in str(warning.message) for warning in w)


def test_create_memory_and_wait_success():
    """Test successful create_memory_and_wait scenario."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock both clients
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock create_memory response
        mock_gmcp.create_memory.return_value = {"memory": {"memoryId": "test-mem-456", "status": "CREATING"}}

        # Mock get_memory to return ACTIVE immediately (simulate quick activation)
        mock_gmcp.get_memory.return_value = {
            "memory": {"memoryId": "test-mem-456", "status": "ACTIVE", "name": "TestMemory"}
        }

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                    result = client.create_memory_and_wait(
                        name="TestMemory",
                        strategies=[{StrategyType.SEMANTIC.value: {"name": "TestStrategy"}}],
                        max_wait=300,
                        poll_interval=10,
                    )

                    assert result["memoryId"] == "test-mem-456"
                    assert result["status"] == "ACTIVE"


def test_create_memory_and_wait_timeout():
    """Test timeout scenario for create_memory_and_wait."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock both clients
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock create_memory response
        mock_gmcp.create_memory.return_value = {"memory": {"memoryId": "test-mem-timeout", "status": "CREATING"}}

        # Mock get_memory to always return CREATING (never becomes ACTIVE)
        mock_gmcp.get_memory.return_value = {"memory": {"memoryId": "test-mem-timeout", "status": "CREATING"}}

        # Mock time to simulate timeout
        # Provide enough values: start_time=0, then loop checks (0,0,0), then timeout (301,301,301)
        with patch("time.time", side_effect=[0, 0, 0, 301, 301, 301]):
            with patch("time.sleep"):
                with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                    try:
                        client.create_memory_and_wait(
                            name="TimeoutMemory",
                            strategies=[{StrategyType.SEMANTIC.value: {"name": "TestStrategy"}}],
                            max_wait=300,
                            poll_interval=10,
                        )
                        raise AssertionError("TimeoutError was not raised")
                    except TimeoutError as e:
                        assert "did not become ACTIVE within 300 seconds" in str(e)


def test_create_memory_and_wait_failure():
    """Test failure scenario for create_memory_and_wait."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock both clients
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock create_memory response
        mock_gmcp.create_memory.return_value = {"memory": {"memoryId": "test-mem-failed", "status": "CREATING"}}

        # Mock get_memory to return FAILED status
        mock_gmcp.get_memory.return_value = {
            "memory": {"memoryId": "test-mem-failed", "status": "FAILED", "failureReason": "Configuration error"}
        }

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                    try:
                        client.create_memory_and_wait(
                            name="FailedMemory",
                            strategies=[{StrategyType.SEMANTIC.value: {"name": "TestStrategy"}}],
                            max_wait=300,
                            poll_interval=10,
                        )
                        raise AssertionError("RuntimeError was not raised")
                    except RuntimeError as e:
                        assert "Memory creation failed: Configuration error" in str(e)


def test_process_turn_with_llm_success_with_retrieval():
    """Test successful process_turn_with_llm with memory retrieval."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the clients
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock retrieval response
        mock_memories = [
            {"content": {"text": "Previous context"}, "memoryRecordId": "rec-123"},
            {"content": {"text": "More context"}, "memoryRecordId": "rec-456"},
        ]
        mock_gmdp.retrieve_memory_records.return_value = {"memoryRecordSummaries": mock_memories}

        # Mock event creation response
        mock_gmdp.create_event.return_value = {"event": {"eventId": "event-llm-123", "memoryId": "mem-123"}}

        # Define a simple LLM callback
        def mock_llm_callback(user_input: str, memories: list) -> str:
            context = " | ".join([m["content"]["text"] for m in memories])
            return f"Based on context: {context}, response to: {user_input}"

        # Test process_turn_with_llm
        memories, response, event = client.process_turn_with_llm(
            memory_id="mem-123",
            actor_id="user-123",
            session_id="session-456",
            user_input="What did we discuss before?",
            llm_callback=mock_llm_callback,
            retrieval_namespace="support/facts/session-456",
            retrieval_query="previous discussion",
            top_k=5,
        )

        # Verify results
        assert len(memories) == 2
        assert memories[0]["memoryRecordId"] == "rec-123"
        assert "Previous context | More context" in response
        assert "What did we discuss before?" in response
        assert event["eventId"] == "event-llm-123"

        # Verify retrieval was called with correct parameters
        retrieve_args, retrieve_kwargs = mock_gmdp.retrieve_memory_records.call_args
        assert retrieve_kwargs["memoryId"] == "mem-123"
        assert retrieve_kwargs["namespace"] == "support/facts/session-456"
        assert retrieve_kwargs["searchCriteria"]["searchQuery"] == "previous discussion"
        assert retrieve_kwargs["searchCriteria"]["topK"] == 5

        # Verify event creation was called with correct parameters
        event_args, event_kwargs = mock_gmdp.create_event.call_args
        assert event_kwargs["memoryId"] == "mem-123"
        assert event_kwargs["actorId"] == "user-123"
        assert event_kwargs["sessionId"] == "session-456"
        assert len(event_kwargs["payload"]) == 2
        assert event_kwargs["payload"][0]["conversational"]["role"] == "USER"
        assert event_kwargs["payload"][0]["conversational"]["content"]["text"] == "What did we discuss before?"
        assert event_kwargs["payload"][1]["conversational"]["role"] == "ASSISTANT"
        assert "Previous context | More context" in event_kwargs["payload"][1]["conversational"]["content"]["text"]


def test_list_events_with_pagination():
    """Test list_events with pagination support."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock paginated responses
        first_batch = [
            {"eventId": f"event-{i}", "eventTimestamp": datetime(2023, 1, 1, 10, i % 60, i % 60)} for i in range(100)
        ]
        second_batch = [
            {"eventId": f"event-{i}", "eventTimestamp": datetime(2023, 1, 1, 11, (i - 100) % 60, (i - 100) % 60)}
            for i in range(100, 150)
        ]

        # Setup side effects for multiple calls
        mock_gmdp.list_events.side_effect = [
            {"events": first_batch, "nextToken": "token-123"},
            {"events": second_batch, "nextToken": None},
        ]

        # Test with max_results that requires pagination
        events = client.list_events(memory_id="mem-123", actor_id="user-123", session_id="session-456", max_results=150)

        assert len(events) == 150
        assert events[0]["eventId"] == "event-0"
        assert events[99]["eventId"] == "event-99"
        assert events[149]["eventId"] == "event-149"

        # Verify two API calls were made
        assert mock_gmdp.list_events.call_count == 2

        # Check first call parameters
        first_call = mock_gmdp.list_events.call_args_list[0]
        assert first_call[1]["maxResults"] == 100
        assert "nextToken" not in first_call[1]

        # Check second call parameters
        second_call = mock_gmdp.list_events.call_args_list[1]
        assert second_call[1]["nextToken"] == "token-123"
        assert second_call[1]["maxResults"] == 50


def test_list_events_with_branch_filter():
    """Test list_events with branch filtering."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock response with branched events
        mock_events = [
            {
                "eventId": "event-branch-1",
                "eventTimestamp": datetime(2023, 1, 1, 10, 0, 0),
                "branch": {"name": "test-branch", "rootEventId": "event-0"},
                "payload": [{"conversational": {"role": "USER", "content": {"text": "Branch message"}}}],
            }
        ]
        mock_gmdp.list_events.return_value = {"events": mock_events, "nextToken": None}

        # Test with branch filter
        events = client.list_events(
            memory_id="mem-123",
            actor_id="user-123",
            session_id="session-456",
            branch_name="test-branch",
            include_parent_branches=True,
        )

        assert len(events) == 1
        assert events[0]["eventId"] == "event-branch-1"
        assert events[0]["branch"]["name"] == "test-branch"

        # Verify filter was applied correctly
        args, kwargs = mock_gmdp.list_events.call_args
        assert "filter" in kwargs
        assert kwargs["filter"]["branch"]["name"] == "test-branch"
        assert kwargs["filter"]["branch"]["includeParentBranches"] is True


def test_list_events_max_results_limit():
    """Test list_events respects max_results limit."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock response with more events than requested
        large_batch = [
            {"eventId": f"event-{i}", "eventTimestamp": datetime(2023, 1, 1, 10, 0, i % 60)} for i in range(100)
        ]
        mock_gmdp.list_events.return_value = {"events": large_batch, "nextToken": "has-more"}

        # Test with small max_results
        events = client.list_events(memory_id="mem-123", actor_id="user-123", session_id="session-456", max_results=25)

        # Should only return 25 events, not all 100
        assert len(events) == 25
        assert events[0]["eventId"] == "event-0"
        assert events[24]["eventId"] == "event-24"

        # Should only make one API call
        assert mock_gmdp.list_events.call_count == 1

        # Verify API was called with correct max_results
        args, kwargs = mock_gmdp.list_events.call_args
        assert kwargs["maxResults"] == 25


def test_list_memories():
    """Test list_memories functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock response
        mock_memories = [
            {"memoryId": "mem-1", "name": "Memory 1", "status": "ACTIVE"},
            {"memoryId": "mem-2", "name": "Memory 2", "status": "ACTIVE"},
        ]
        mock_gmcp.list_memories.return_value = {"memories": mock_memories, "nextToken": None}

        # Test list_memories
        memories = client.list_memories(max_results=50)

        assert len(memories) == 2
        assert memories[0]["memoryId"] == "mem-1"
        assert memories[1]["memoryId"] == "mem-2"

        # Verify API call
        args, kwargs = mock_gmcp.list_memories.call_args
        assert kwargs["maxResults"] == 50


def test_list_memories_with_pagination():
    """Test list_memories with pagination support."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock paginated responses
        first_batch = [{"memoryId": f"mem-{i}", "name": f"Memory {i}", "status": "ACTIVE"} for i in range(1, 101)]
        second_batch = [{"memoryId": f"mem-{i}", "name": f"Memory {i}", "status": "ACTIVE"} for i in range(101, 151)]

        # Setup side effects for multiple calls
        mock_gmcp.list_memories.side_effect = [
            {"memories": first_batch, "nextToken": "pagination-token-123"},
            {"memories": second_batch, "nextToken": None},
        ]

        # Test with max_results that requires pagination
        memories = client.list_memories(max_results=150)

        assert len(memories) == 150
        assert memories[0]["memoryId"] == "mem-1"
        assert memories[0]["name"] == "Memory 1"
        assert memories[99]["memoryId"] == "mem-100"
        assert memories[149]["memoryId"] == "mem-150"

        # Verify two API calls were made
        assert mock_gmcp.list_memories.call_count == 2

        # Check first call parameters
        first_call = mock_gmcp.list_memories.call_args_list[0]
        assert first_call[1]["maxResults"] == 100
        assert "nextToken" not in first_call[1]

        # Check second call parameters
        second_call = mock_gmcp.list_memories.call_args_list[1]
        assert second_call[1]["nextToken"] == "pagination-token-123"
        assert second_call[1]["maxResults"] == 50  # Remaining results needed

        # Verify normalization was applied (both old and new field names should exist)
        for memory in memories:
            assert "memoryId" in memory
            assert "id" in memory
            assert memory["memoryId"] == memory["id"]


def test_delete_memory():
    """Test delete_memory functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock response
        mock_gmcp.delete_memory.return_value = {"status": "DELETING"}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test delete_memory
            result = client.delete_memory("mem-123")

            assert result["status"] == "DELETING"

            # Verify API call
            args, kwargs = mock_gmcp.delete_memory.call_args
            assert kwargs["memoryId"] == "mem-123"
            assert kwargs["clientToken"] == "12345678-1234-5678-1234-567812345678"


def test_get_memory_status():
    """Test get_memory_status functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock response
        mock_gmcp.get_memory.return_value = {"memory": {"memoryId": "mem-123", "status": "ACTIVE"}}

        # Test get_memory_status
        status = client.get_memory_status("mem-123")

        assert status == "ACTIVE"

        # Verify API call
        args, kwargs = mock_gmcp.get_memory.call_args
        assert kwargs["memoryId"] == "mem-123"


def test_add_summary_strategy():
    """Test add_summary_strategy functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock response
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-123", "status": "CREATING"}}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test add_summary_strategy
            client.add_summary_strategy(
                memory_id="mem-123", name="Test Summary Strategy", description="Test description"
            )

            assert mock_gmcp.update_memory.called

            # Verify strategy was added correctly
            args, kwargs = mock_gmcp.update_memory.call_args
            assert "memoryStrategies" in kwargs
            assert "addMemoryStrategies" in kwargs["memoryStrategies"]


def test_add_user_preference_strategy():
    """Test add_user_preference_strategy functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock response
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-456", "status": "CREATING"}}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test add_user_preference_strategy
            client.add_user_preference_strategy(
                memory_id="mem-456",
                name="Test User Preference Strategy",
                description="User preference test description",
                namespaces=["preferences/{actorId}"],
            )

            assert mock_gmcp.update_memory.called

            # Verify strategy was added correctly
            args, kwargs = mock_gmcp.update_memory.call_args
            assert "memoryStrategies" in kwargs
            assert "addMemoryStrategies" in kwargs["memoryStrategies"]

            # Verify the strategy configuration
            add_strategies = kwargs["memoryStrategies"]["addMemoryStrategies"]
            assert len(add_strategies) == 1

            strategy = add_strategies[0]
            assert "userPreferenceMemoryStrategy" in strategy

            user_pref_config = strategy["userPreferenceMemoryStrategy"]
            assert user_pref_config["name"] == "Test User Preference Strategy"
            assert user_pref_config["description"] == "User preference test description"
            assert user_pref_config["namespaces"] == ["preferences/{actorId}"]

            # Verify client token and memory ID
            assert kwargs["memoryId"] == "mem-456"
            assert kwargs["clientToken"] == "12345678-1234-5678-1234-567812345678"


def test_add_custom_semantic_strategy():
    """Test add_custom_semantic_strategy functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock response
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-789", "status": "CREATING"}}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test add_custom_semantic_strategy
            extraction_config = {
                "prompt": "Extract key information from the conversation",
                "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            }
            consolidation_config = {
                "prompt": "Consolidate extracted information into coherent summaries",
                "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            }

            client.add_custom_semantic_strategy(
                memory_id="mem-789",
                name="Test Custom Semantic Strategy",
                extraction_config=extraction_config,
                consolidation_config=consolidation_config,
                description="Custom semantic strategy test description",
                namespaces=["custom/{actorId}/{sessionId}"],
            )

            assert mock_gmcp.update_memory.called

            # Verify strategy was added correctly
            args, kwargs = mock_gmcp.update_memory.call_args
            assert "memoryStrategies" in kwargs
            assert "addMemoryStrategies" in kwargs["memoryStrategies"]

            # Verify the strategy configuration
            add_strategies = kwargs["memoryStrategies"]["addMemoryStrategies"]
            assert len(add_strategies) == 1

            strategy = add_strategies[0]
            assert "customMemoryStrategy" in strategy

            custom_config = strategy["customMemoryStrategy"]
            assert custom_config["name"] == "Test Custom Semantic Strategy"
            assert custom_config["description"] == "Custom semantic strategy test description"
            assert custom_config["namespaces"] == ["custom/{actorId}/{sessionId}"]

            # Verify the semantic override configuration
            assert "configuration" in custom_config
            assert "semanticOverride" in custom_config["configuration"]

            semantic_override = custom_config["configuration"]["semanticOverride"]

            # Verify extraction configuration
            assert "extraction" in semantic_override
            extraction = semantic_override["extraction"]
            assert extraction["appendToPrompt"] == "Extract key information from the conversation"
            assert extraction["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"

            # Verify consolidation configuration
            assert "consolidation" in semantic_override
            consolidation = semantic_override["consolidation"]
            assert consolidation["appendToPrompt"] == "Consolidate extracted information into coherent summaries"
            assert consolidation["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"

            # Verify client token and memory ID
            assert kwargs["memoryId"] == "mem-789"
            assert kwargs["clientToken"] == "12345678-1234-5678-1234-567812345678"


def test_merge_branch_context():
    """Test merge_branch_context functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock events response
        mock_events = [
            {
                "eventId": "event-1",
                "eventTimestamp": datetime(2023, 1, 1, 10, 0, 0),
                "payload": [{"conversational": {"role": "USER", "content": {"text": "First message"}}}],
            },
            {
                "eventId": "event-2",
                "eventTimestamp": datetime(2023, 1, 1, 10, 5, 0),
                "payload": [{"conversational": {"role": "ASSISTANT", "content": {"text": "Second message"}}}],
            },
        ]
        mock_gmdp.list_events.return_value = {"events": mock_events, "nextToken": None}

        # Test merge_branch_context
        messages = client.merge_branch_context(
            memory_id="mem-123", actor_id="user-123", session_id="session-456", branch_name="test-branch"
        )

        assert len(messages) == 2
        assert messages[0]["content"] == "First message"
        assert messages[0]["role"] == "USER"
        assert messages[1]["content"] == "Second message"
        assert messages[1]["role"] == "ASSISTANT"


def test_wait_for_memories():
    """Test wait_for_memories functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock retrieval response (simulate memories found)
        mock_gmdp.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [{"content": {"text": "Found memory"}, "memoryRecordId": "rec-1"}]
        }

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                # Test wait_for_memories (should return True when memories found)
                result = client.wait_for_memories(
                    memory_id="mem-123", namespace="test/namespace", test_query="test", max_wait=30, poll_interval=5
                )

                assert result

                # Verify retrieval was called
                assert mock_gmdp.retrieve_memory_records.called


def test_wait_for_memories_wildcard_namespace():
    """Test wait_for_memories rejects wildcard namespaces."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client (shouldn't be called)
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Test with wildcard namespace - should return False immediately
        result = client.wait_for_memories(
            memory_id="mem-123", namespace="test/namespace/*", test_query="test", max_wait=30, poll_interval=5
        )

        assert not result

        # Should not make any API calls due to wildcard rejection
        assert not mock_gmdp.retrieve_memory_records.called


def test_get_last_k_turns():
    """Test get_last_k_turns functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock events response with conversation turns
        mock_events = [
            {
                "eventId": "event-1",
                "eventTimestamp": datetime(2023, 1, 1, 10, 0, 0),
                "payload": [
                    {"conversational": {"role": "USER", "content": {"text": "Hello"}}},
                    {"conversational": {"role": "ASSISTANT", "content": {"text": "Hi there"}}},
                ],
            },
            {
                "eventId": "event-2",
                "eventTimestamp": datetime(2023, 1, 1, 10, 5, 0),
                "payload": [
                    {"conversational": {"role": "USER", "content": {"text": "How are you?"}}},
                    {"conversational": {"role": "ASSISTANT", "content": {"text": "I'm doing well"}}},
                ],
            },
        ]
        mock_gmdp.list_events.return_value = {"events": mock_events, "nextToken": None}

        # Test get_last_k_turns
        turns = client.get_last_k_turns(memory_id="mem-123", actor_id="user-123", session_id="session-456", k=2)

        assert len(turns) == 2
        assert len(turns[0]) == 2  # First turn has 2 messages
        assert turns[0][0]["role"] == "USER"
        assert turns[0][1]["role"] == "ASSISTANT"


def test_delete_memory_and_wait():
    """Test delete_memory_and_wait functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock delete response
        mock_gmcp.delete_memory.return_value = {"status": "DELETING"}

        # Mock get_memory to raise ResourceNotFoundException (memory deleted)
        error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Memory not found"}}
        mock_gmcp.get_memory.side_effect = ClientError(error_response, "GetMemory")

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                    # Test delete_memory_and_wait
                    result = client.delete_memory_and_wait("mem-123", max_wait=60, poll_interval=5)

                    assert result["status"] == "DELETING"

                    # Verify delete was called
                    assert mock_gmcp.delete_memory.called
                    args, kwargs = mock_gmcp.delete_memory.call_args
                    assert kwargs["memoryId"] == "mem-123"


def test_update_memory_strategies():
    """Test update_memory_strategies functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock response
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-123", "status": "CREATING"}}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test adding strategies
            add_strategies = [{StrategyType.SEMANTIC.value: {"name": "New Strategy"}}]
            client.update_memory_strategies(memory_id="mem-123", add_strategies=add_strategies)

            assert mock_gmcp.update_memory.called

            # Verify correct parameters
            args, kwargs = mock_gmcp.update_memory.call_args
            assert kwargs["memoryId"] == "mem-123"
            assert "memoryStrategies" in kwargs
            assert "addMemoryStrategies" in kwargs["memoryStrategies"]


def test_update_memory_strategies_modify():
    """Test update_memory_strategies with modify_strategies."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock get_memory_strategies to return existing strategies
        mock_gmcp.get_memory.return_value = {
            "memory": {
                "memoryId": "mem-123",
                "status": "ACTIVE",
                "memoryStrategies": [
                    {"memoryStrategyId": "strat-456", "memoryStrategyType": "SEMANTIC", "name": "Existing Strategy"}
                ],
            }
        }

        # Mock update_memory response
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-123", "status": "CREATING"}}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test modifying strategies
            modify_strategies = [{"memoryStrategyId": "strat-456", "description": "Updated description"}]
            client.update_memory_strategies(memory_id="mem-123", modify_strategies=modify_strategies)

            assert mock_gmcp.update_memory.called

            # Verify correct parameters
            args, kwargs = mock_gmcp.update_memory.call_args
            assert kwargs["memoryId"] == "mem-123"
            assert "memoryStrategies" in kwargs
            assert "modifyMemoryStrategies" in kwargs["memoryStrategies"]

            # Verify the modified strategy has the correct ID
            modified_strategy = kwargs["memoryStrategies"]["modifyMemoryStrategies"][0]
            assert modified_strategy["memoryStrategyId"] == "strat-456"
            assert modified_strategy["description"] == "Updated description"


def test_normalize_memory_response():
    """Test _normalize_memory_response functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Test memory with new field names only
        memory_with_new_fields = {
            "id": "mem-123",
            "name": "Test Memory",
            "strategies": [{"strategyId": "strat-1", "type": "SEMANTIC", "name": "Test Strategy"}],
        }

        # Test normalization
        normalized = client._normalize_memory_response(memory_with_new_fields)

        # Should have both old and new field names
        assert normalized["id"] == "mem-123"
        assert normalized["memoryId"] == "mem-123"
        assert "strategies" in normalized
        assert "memoryStrategies" in normalized

        # Check strategy normalization
        strategy = normalized["strategies"][0]
        assert strategy["strategyId"] == "strat-1"
        assert strategy["memoryStrategyId"] == "strat-1"
        assert strategy["type"] == "SEMANTIC"
        assert strategy["memoryStrategyType"] == "SEMANTIC"


def test_wait_for_memory_active():
    """Test _wait_for_memory_active functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock get_memory responses
        mock_gmcp.get_memory.return_value = {
            "memory": {"memoryId": "mem-123", "status": "ACTIVE", "name": "Test Memory"}
        }

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                # Test _wait_for_memory_active
                result = client._wait_for_memory_active("mem-123", max_wait=60, poll_interval=5)

                assert result["memoryId"] == "mem-123"
                assert result["status"] == "ACTIVE"

                # Verify get_memory was called
                assert mock_gmcp.get_memory.called


def test_wait_for_memory_active_failed_status():
    """Test _wait_for_memory_active when memory status becomes FAILED."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock get_memory to return FAILED status
        mock_gmcp.get_memory.return_value = {
            "memory": {"memoryId": "mem-failed", "status": "FAILED", "failureReason": "Strategy configuration error"}
        }

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                # Test _wait_for_memory_active with FAILED status
                try:
                    client._wait_for_memory_active("mem-failed", max_wait=60, poll_interval=5)
                    raise AssertionError("RuntimeError was not raised")
                except RuntimeError as e:
                    assert "Memory update failed: Strategy configuration error" in str(e)

                # Verify get_memory was called
                assert mock_gmcp.get_memory.called


def test_wait_for_memory_active_client_error():
    """Test _wait_for_memory_active when ClientError is raised."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock get_memory to raise ClientError
        error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid memory ID"}}
        mock_gmcp.get_memory.side_effect = ClientError(error_response, "GetMemory")

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                # Test _wait_for_memory_active with ClientError
                try:
                    client._wait_for_memory_active("mem-invalid", max_wait=60, poll_interval=5)
                    raise AssertionError("ClientError was not raised")
                except ClientError as e:
                    assert "ValidationException" in str(e)

                # Verify get_memory was called
                assert mock_gmcp.get_memory.called


def test_wrap_configuration():
    """Test _wrap_configuration functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Test basic configuration wrapping
        config = {
            "extraction": {"appendToPrompt": "Custom prompt", "modelId": "test-model"},
            "consolidation": {"appendToPrompt": "Consolidation prompt", "modelId": "test-model"},
        }

        # Test wrapping for CUSTOM strategy with semantic override
        wrapped = client._wrap_configuration(config, "CUSTOM", "SEMANTIC_OVERRIDE")

        # Should wrap in custom configuration structure
        assert "extraction" in wrapped
        assert "consolidation" in wrapped


def test_wrap_configuration_basic():
    """Test _wrap_configuration with basic config."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Test config that doesn't need wrapping
        simple_config = {"extraction": {"modelId": "test-model"}}

        # Test with SEMANTIC strategy
        wrapped = client._wrap_configuration(simple_config, "SEMANTIC", None)

        # Should pass through unchanged
        assert wrapped["extraction"]["modelId"] == "test-model"


def test_wrap_configuration_semantic_strategy():
    """Test _wrap_configuration with SEMANTIC strategy type."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Test extraction configuration that needs wrapping
        config = {
            "extraction": {"triggerEveryNMessages": 5, "historicalContextWindowSize": 10, "modelId": "semantic-model"}
        }

        wrapped = client._wrap_configuration(config, "SEMANTIC", None)

        # Should wrap in semanticExtractionConfiguration
        assert "extraction" in wrapped
        assert "semanticExtractionConfiguration" in wrapped["extraction"]
        assert wrapped["extraction"]["semanticExtractionConfiguration"]["triggerEveryNMessages"] == 5
        assert wrapped["extraction"]["semanticExtractionConfiguration"]["historicalContextWindowSize"] == 10
        assert wrapped["extraction"]["semanticExtractionConfiguration"]["modelId"] == "semantic-model"


def test_wrap_configuration_user_preference_strategy():
    """Test _wrap_configuration with USER_PREFERENCE strategy type."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Test extraction configuration that needs wrapping for user preferences
        config = {
            "extraction": {"triggerEveryNMessages": 3, "historicalContextWindowSize": 20, "preferenceType": "dietary"}
        }

        wrapped = client._wrap_configuration(config, "USER_PREFERENCE", None)

        # Should wrap in userPreferenceExtractionConfiguration
        assert "extraction" in wrapped
        assert "userPreferenceExtractionConfiguration" in wrapped["extraction"]
        assert wrapped["extraction"]["userPreferenceExtractionConfiguration"]["triggerEveryNMessages"] == 3
        assert wrapped["extraction"]["userPreferenceExtractionConfiguration"]["historicalContextWindowSize"] == 20
        assert wrapped["extraction"]["userPreferenceExtractionConfiguration"]["preferenceType"] == "dietary"


def test_wrap_configuration_custom_semantic_override():
    """Test _wrap_configuration with CUSTOM strategy and SEMANTIC_OVERRIDE."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Test custom semantic override configuration
        config = {
            "extraction": {
                "triggerEveryNMessages": 2,
                "historicalContextWindowSize": 15,
                "appendToPrompt": "Extract key insights",
                "modelId": "custom-semantic-model",
            },
            "consolidation": {"appendToPrompt": "Consolidate insights", "modelId": "consolidation-model"},
        }

        wrapped = client._wrap_configuration(config, "CUSTOM", "SEMANTIC_OVERRIDE")

        # Should wrap extraction in customExtractionConfiguration with semanticExtractionOverride
        assert "extraction" in wrapped
        assert "customExtractionConfiguration" in wrapped["extraction"]
        assert "semanticExtractionOverride" in wrapped["extraction"]["customExtractionConfiguration"]

        semantic_config = wrapped["extraction"]["customExtractionConfiguration"]["semanticExtractionOverride"]
        assert semantic_config["triggerEveryNMessages"] == 2
        assert semantic_config["historicalContextWindowSize"] == 15
        assert semantic_config["appendToPrompt"] == "Extract key insights"
        assert semantic_config["modelId"] == "custom-semantic-model"

        # Should wrap consolidation in customConsolidationConfiguration with semanticConsolidationOverride
        assert "consolidation" in wrapped
        assert "customConsolidationConfiguration" in wrapped["consolidation"]
        assert "semanticConsolidationOverride" in wrapped["consolidation"]["customConsolidationConfiguration"]

        consolidation_config = wrapped["consolidation"]["customConsolidationConfiguration"][
            "semanticConsolidationOverride"
        ]
        assert consolidation_config["appendToPrompt"] == "Consolidate insights"
        assert consolidation_config["modelId"] == "consolidation-model"


def test_list_branch_events_pagination():
    """Test list_branch_events with pagination."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock paginated responses
        first_batch = [
            {"eventId": f"branch-event-{i}", "eventTimestamp": datetime(2023, 1, 1, 10, i % 60, 0)} for i in range(100)
        ]
        second_batch = [
            {"eventId": f"branch-event-{i}", "eventTimestamp": datetime(2023, 1, 1, 11, (i - 100) % 60, 0)}
            for i in range(100, 130)
        ]

        # Setup side effects for multiple calls
        mock_gmdp.list_events.side_effect = [
            {"events": first_batch, "nextToken": "branch-token-123"},
            {"events": second_batch, "nextToken": None},
        ]

        # Test list_branch_events with pagination
        events = client.list_branch_events(
            memory_id="mem-123",
            actor_id="user-123",
            session_id="session-456",
            branch_name="test-branch",
            max_results=130,
        )

        assert len(events) == 130
        assert events[0]["eventId"] == "branch-event-0"
        assert events[99]["eventId"] == "branch-event-99"
        assert events[129]["eventId"] == "branch-event-129"

        # Verify two API calls were made
        assert mock_gmdp.list_events.call_count == 2

        # Check first call parameters
        first_call = mock_gmdp.list_events.call_args_list[0]
        assert first_call[1]["memoryId"] == "mem-123"
        assert first_call[1]["actorId"] == "user-123"
        assert first_call[1]["sessionId"] == "session-456"
        assert first_call[1]["maxResults"] == 100
        assert first_call[1]["filter"]["branch"]["name"] == "test-branch"
        assert "nextToken" not in first_call[1]

        # Check second call parameters
        second_call = mock_gmdp.list_events.call_args_list[1]
        assert second_call[1]["nextToken"] == "branch-token-123"
        assert second_call[1]["maxResults"] == 30


def test_modify_strategy():
    """Test modify_strategy convenience method."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock get_memory_strategies to return existing strategies (needed by update_memory_strategies)
        mock_gmcp.get_memory.return_value = {
            "memory": {
                "memoryId": "mem-123",
                "status": "ACTIVE",
                "memoryStrategies": [
                    {"memoryStrategyId": "strat-789", "memoryStrategyType": "SEMANTIC", "name": "Test Strategy"}
                ],
            }
        }

        # Mock update_memory response
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-123", "status": "CREATING"}}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test modify_strategy
            client.modify_strategy(
                memory_id="mem-123",
                strategy_id="strat-789",
                description="Modified description",
                namespaces=["custom/namespace"],
            )

            assert mock_gmcp.update_memory.called

            # Verify correct parameters were passed to update_memory_strategies
            args, kwargs = mock_gmcp.update_memory.call_args
            assert kwargs["memoryId"] == "mem-123"
            assert "memoryStrategies" in kwargs
            assert "modifyMemoryStrategies" in kwargs["memoryStrategies"]

            # Verify the modified strategy has correct details
            modified_strategy = kwargs["memoryStrategies"]["modifyMemoryStrategies"][0]
            assert modified_strategy["memoryStrategyId"] == "strat-789"
            assert modified_strategy["description"] == "Modified description"
            assert modified_strategy["namespaces"] == ["custom/namespace"]


def test_retrieve_memories_resource_not_found_error():
    """Test retrieve_memories with ResourceNotFoundException."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ResourceNotFoundException
        error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Memory not found"}}
        mock_gmdp.retrieve_memory_records.side_effect = ClientError(error_response, "RetrieveMemoryRecords")

        # Test retrieve_memories - should return empty list and log warning
        result = client.retrieve_memories(
            memory_id="nonexistent-mem-123", namespace="test/namespace", query="test query", top_k=5
        )

        # Should return empty list instead of raising exception
        assert result == []

        # Verify API was called with correct parameters
        args, kwargs = mock_gmdp.retrieve_memory_records.call_args
        assert kwargs["memoryId"] == "nonexistent-mem-123"
        assert kwargs["namespace"] == "test/namespace"
        assert kwargs["searchCriteria"]["searchQuery"] == "test query"
        assert kwargs["searchCriteria"]["topK"] == 5


def test_retrieve_memories_validation_error():
    """Test retrieve_memories with ValidationException."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ValidationException
        error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid search parameters"}}
        mock_gmdp.retrieve_memory_records.side_effect = ClientError(error_response, "RetrieveMemoryRecords")

        # Test retrieve_memories - should return empty list and log warning
        result = client.retrieve_memories(
            memory_id="mem-123",
            namespace="invalid/namespace",
            query="",
            top_k=-1,  # Invalid parameters
        )

        # Should return empty list instead of raising exception
        assert result == []

        # Verify API was called
        assert mock_gmdp.retrieve_memory_records.called


def test_retrieve_memories_service_error():
    """Test retrieve_memories with ServiceException."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ServiceException
        error_response = {"Error": {"Code": "ServiceException", "Message": "Internal service error"}}
        mock_gmdp.retrieve_memory_records.side_effect = ClientError(error_response, "RetrieveMemoryRecords")

        # Test retrieve_memories - should return empty list and log warning
        result = client.retrieve_memories(memory_id="mem-123", namespace="test/namespace", query="test query", top_k=3)

        # Should return empty list instead of raising exception
        assert result == []

        # Verify API was called with correct parameters
        args, kwargs = mock_gmdp.retrieve_memory_records.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert kwargs["namespace"] == "test/namespace"
        assert kwargs["searchCriteria"]["searchQuery"] == "test query"
        assert kwargs["searchCriteria"]["topK"] == 3


def test_retrieve_memories_unknown_error():
    """Test retrieve_memories with unknown ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock unknown error
        error_response = {"Error": {"Code": "UnknownException", "Message": "Something unexpected happened"}}
        mock_gmdp.retrieve_memory_records.side_effect = ClientError(error_response, "RetrieveMemoryRecords")

        # Test retrieve_memories - should return empty list and log warning
        result = client.retrieve_memories(memory_id="mem-123", namespace="test/namespace", query="test query", top_k=3)

        # Should return empty list instead of raising exception
        assert result == []

        # Verify API was called
        assert mock_gmdp.retrieve_memory_records.called


def test_retrieve_memories_wildcard_namespace():
    """Test retrieve_memories rejects wildcard namespaces."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client (shouldn't be called)
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Test with wildcard namespace - should return empty list without API call
        result = client.retrieve_memories(
            memory_id="mem-123", namespace="test/namespace/*", query="test query", top_k=3
        )

        # Should return empty list
        assert result == []

        # Should not make API call due to wildcard rejection
        assert not mock_gmdp.retrieve_memory_records.called


def test_add_semantic_strategy_and_wait():
    """Test add_semantic_strategy_and_wait functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock update_memory response
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-123", "status": "CREATING"}}

        # Mock get_memory response (simulating ACTIVE status)
        mock_gmcp.get_memory.return_value = {"memory": {"memoryId": "mem-123", "status": "ACTIVE"}}

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                    # Test add_semantic_strategy_and_wait
                    result = client.add_semantic_strategy_and_wait(
                        memory_id="mem-123", name="Test Strategy", description="Test description"
                    )

                    assert result["memoryId"] == "mem-123"
                    assert result["status"] == "ACTIVE"

                    # Verify update_memory was called
                    assert mock_gmcp.update_memory.called

                    # Verify get_memory was called (for waiting)
                    assert mock_gmcp.get_memory.called


def test_add_summary_strategy_and_wait():
    """Test add_summary_strategy_and_wait functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock update_memory response
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-456", "status": "CREATING"}}

        # Mock get_memory response (simulating ACTIVE status)
        mock_gmcp.get_memory.return_value = {"memory": {"memoryId": "mem-456", "status": "ACTIVE"}}

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                    # Test add_summary_strategy_and_wait
                    result = client.add_summary_strategy_and_wait(
                        memory_id="mem-456", name="Test Summary Strategy", description="Test description"
                    )

                    assert result["memoryId"] == "mem-456"
                    assert result["status"] == "ACTIVE"

                    # Verify update_memory was called
                    assert mock_gmcp.update_memory.called

                    # Verify get_memory was called (for waiting)
                    assert mock_gmcp.get_memory.called


def test_add_user_preference_strategy_and_wait():
    """Test add_user_preference_strategy_and_wait functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock update_memory response
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-789", "status": "CREATING"}}

        # Mock get_memory response (simulating ACTIVE status)
        mock_gmcp.get_memory.return_value = {"memory": {"memoryId": "mem-789", "status": "ACTIVE"}}

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                    # Test add_user_preference_strategy_and_wait
                    result = client.add_user_preference_strategy_and_wait(
                        memory_id="mem-789", name="Test User Preference Strategy", description="Test description"
                    )

                    assert result["memoryId"] == "mem-789"
                    assert result["status"] == "ACTIVE"

                    # Verify update_memory was called
                    assert mock_gmcp.update_memory.called

                    # Verify get_memory was called (for waiting)
                    assert mock_gmcp.get_memory.called


def test_add_custom_semantic_strategy_and_wait():
    """Test add_custom_semantic_strategy_and_wait functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock update_memory response
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-999", "status": "CREATING"}}

        # Mock get_memory response (simulating ACTIVE status)
        mock_gmcp.get_memory.return_value = {"memory": {"memoryId": "mem-999", "status": "ACTIVE"}}

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                    # Test add_custom_semantic_strategy_and_wait
                    extraction_config = {"prompt": "Extract key info", "modelId": "claude-3-sonnet"}
                    consolidation_config = {"prompt": "Consolidate info", "modelId": "claude-3-haiku"}

                    result = client.add_custom_semantic_strategy_and_wait(
                        memory_id="mem-999",
                        name="Test Custom Strategy",
                        extraction_config=extraction_config,
                        consolidation_config=consolidation_config,
                        description="Test description",
                    )

                    assert result["memoryId"] == "mem-999"
                    assert result["status"] == "ACTIVE"

                    # Verify update_memory was called
                    assert mock_gmcp.update_memory.called

                    # Verify get_memory was called (for waiting)
                    assert mock_gmcp.get_memory.called


def test_update_memory_strategies_and_wait():
    """Test update_memory_strategies_and_wait functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock get_memory for strategy retrieval and waiting
        def mock_get_memory_response(*args, **kwargs):
            # Return ACTIVE status for waiting calls
            return {"memory": {"memoryId": "mem-123", "status": "ACTIVE", "memoryStrategies": []}}

        mock_gmcp.get_memory.side_effect = mock_get_memory_response

        # Mock update_memory response
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-123", "status": "CREATING"}}

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                    # Test update_memory_strategies_and_wait
                    add_strategies = [{StrategyType.SEMANTIC.value: {"name": "New Strategy"}}]
                    result = client.update_memory_strategies_and_wait(
                        memory_id="mem-123", add_strategies=add_strategies
                    )

                    assert result["memoryId"] == "mem-123"
                    assert result["status"] == "ACTIVE"

                    # Verify update_memory was called
                    assert mock_gmcp.update_memory.called

                    # Verify get_memory was called multiple times
                    assert mock_gmcp.get_memory.call_count >= 2


def test_fork_conversation():
    """Test fork_conversation functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock create_event response
        mock_gmdp.create_event.return_value = {"event": {"eventId": "event-fork-123", "memoryId": "mem-123"}}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test fork_conversation
            result = client.fork_conversation(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                root_event_id="event-root-456",
                branch_name="test-branch",
                new_messages=[("Forked message", "USER"), ("Forked response", "ASSISTANT")],
            )

            assert result["eventId"] == "event-fork-123"

            # Verify create_event was called with branch info
            args, kwargs = mock_gmdp.create_event.call_args
            assert kwargs["memoryId"] == "mem-123"
            assert kwargs["actorId"] == "user-123"
            assert kwargs["sessionId"] == "session-456"
            assert "branch" in kwargs
            assert kwargs["branch"]["rootEventId"] == "event-root-456"
            assert kwargs["branch"]["name"] == "test-branch"
            assert len(kwargs["payload"]) == 2


def test_delete_strategy():
    """Test delete_strategy functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock get_memory for strategy retrieval
        mock_gmcp.get_memory.return_value = {"memory": {"memoryId": "mem-123", "memoryStrategies": []}}

        # Mock update_memory response
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-123", "status": "ACTIVE"}}

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            # Test delete_strategy
            result = client.delete_strategy(memory_id="mem-123", strategy_id="strat-456")

            assert result["memoryId"] == "mem-123"

            # Verify update_memory was called with delete operation
            args, kwargs = mock_gmcp.update_memory.call_args
            assert "memoryStrategies" in kwargs
            assert "deleteMemoryStrategies" in kwargs["memoryStrategies"]
            assert kwargs["memoryStrategies"]["deleteMemoryStrategies"][0]["memoryStrategyId"] == "strat-456"


def test_add_strategy_warning():
    """Test add_strategy shows deprecation warning."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock get_memory for strategy retrieval
        mock_gmcp.get_memory.return_value = {"memory": {"memoryId": "mem-123", "memoryStrategies": []}}

        # Mock update_memory response
        mock_gmcp.update_memory.return_value = {"memory": {"memoryId": "mem-123", "status": "CREATING"}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                # Test add_strategy (should show warning)
                strategy = {StrategyType.SEMANTIC.value: {"name": "Test Strategy"}}
                client.add_strategy(memory_id="mem-123", strategy=strategy)

                # Should have shown a warning
                assert len(w) >= 1
                assert any("may leave memory in CREATING state" in str(warning.message) for warning in w)

                # Verify update_memory was called
                assert mock_gmcp.update_memory.called


def test_create_event():
    """Test create_event functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock create_event response
        mock_gmdp.create_event.return_value = {"event": {"eventId": "event-create-123", "memoryId": "mem-123"}}

        # Test create_event
        result = client.create_event(
            memory_id="mem-123",
            actor_id="user-123",
            session_id="session-456",
            messages=[("Hello", "USER"), ("Hi there", "ASSISTANT")],
        )

        assert result["eventId"] == "event-create-123"

        # Verify create_event was called with correct parameters
        args, kwargs = mock_gmdp.create_event.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert kwargs["actorId"] == "user-123"
        assert kwargs["sessionId"] == "session-456"
        assert len(kwargs["payload"]) == 2
        assert kwargs["payload"][0]["conversational"]["role"] == "USER"
        assert kwargs["payload"][1]["conversational"]["role"] == "ASSISTANT"


def test_create_event_with_branch():
    """Test create_event with branch parameter."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock create_event response
        mock_gmdp.create_event.return_value = {"event": {"eventId": "event-branch-123", "memoryId": "mem-123"}}

        # Test create_event with branch
        branch = {"name": "test-branch", "rootEventId": "event-root-123"}
        result = client.create_event(
            memory_id="mem-123",
            actor_id="user-123",
            session_id="session-456",
            messages=[("Branch message", "USER")],
            branch=branch,
        )

        assert result["eventId"] == "event-branch-123"

        # Verify branch was passed correctly
        args, kwargs = mock_gmdp.create_event.call_args
        assert kwargs["branch"] == branch


def test_create_memory_and_wait_client_error():
    """Test create_memory_and_wait with ClientError during status check."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock both clients
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock create_memory response
        mock_gmcp.create_memory.return_value = {"memory": {"memoryId": "test-mem-error", "status": "CREATING"}}

        # Mock get_memory to raise ClientError
        error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid memory ID"}}
        mock_gmcp.get_memory.side_effect = ClientError(error_response, "GetMemory")

        with patch("time.time", return_value=0):
            with patch("time.sleep"):
                with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                    try:
                        client.create_memory_and_wait(
                            name="ErrorMemory",
                            strategies=[{StrategyType.SEMANTIC.value: {"name": "TestStrategy"}}],
                            max_wait=300,
                            poll_interval=10,
                        )
                        raise AssertionError("ClientError was not raised")
                    except ClientError as e:
                        assert "ValidationException" in str(e)


def test_create_event_client_error():
    """Test create_event with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ClientError
        error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid event parameters"}}
        mock_gmdp.create_event.side_effect = ClientError(error_response, "CreateEvent")

        try:
            client.create_event(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                messages=[("Hello", "USER")],
            )
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ValidationException" in str(e)


def test_create_event_no_messages_error():
    """Test create_event with no messages raises ValueError."""
    with patch("boto3.client"):
        client = MemoryClient()

        try:
            client.create_event(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                messages=[],  # Empty messages list
            )
            raise AssertionError("ValueError was not raised")
        except ValueError as e:
            assert "At least one message is required" in str(e)


def test_create_event_invalid_message_format_error():
    """Test create_event with invalid message format raises ValueError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Test with message that doesn't have exactly 2 elements
        try:
            client.create_event(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                messages=[("Hello",)],  # Missing role  # type: ignore
            )
            raise AssertionError("ValueError was not raised")
        except ValueError as e:
            assert "Each message must be (text, role)" in str(e)

        # Test with message that has too many elements
        try:
            client.create_event(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                messages=[("Hello", "USER", "extra")],  # Too many elements  # type: ignore
            )
            raise AssertionError("ValueError was not raised")
        except ValueError as e:
            assert "Each message must be (text, role)" in str(e)


def test_create_event_invalid_role_error():
    """Test create_event with invalid role raises ValueError."""
    with patch("boto3.client"):
        client = MemoryClient()

        try:
            client.create_event(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                messages=[("Hello", "INVALID_ROLE")],  # Invalid role
            )
            raise AssertionError("ValueError was not raised")
        except ValueError as e:
            assert "Invalid role 'INVALID_ROLE'" in str(e)
            assert "Must be one of:" in str(e)


def test_save_conversation_client_error():
    """Test save_conversation with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ClientError
        error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Memory not found"}}
        mock_gmdp.create_event.side_effect = ClientError(error_response, "CreateEvent")

        try:
            client.save_conversation(
                memory_id="nonexistent-mem",
                actor_id="user-123",
                session_id="session-456",
                messages=[("Hello", "USER"), ("Hi", "ASSISTANT")],
            )
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ResourceNotFoundException" in str(e)


def test_list_events_client_error():
    """Test list_events with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ClientError
        error_response = {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}}
        mock_gmdp.list_events.side_effect = ClientError(error_response, "ListEvents")

        try:
            client.list_events(memory_id="mem-123", actor_id="user-123", session_id="session-456", max_results=50)
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "AccessDeniedException" in str(e)


def test_list_branches_client_error():
    """Test list_branches with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ClientError
        error_response = {"Error": {"Code": "ServiceException", "Message": "Internal service error"}}
        mock_gmdp.list_events.side_effect = ClientError(error_response, "ListEvents")

        try:
            client.list_branches(memory_id="mem-123", actor_id="user-123", session_id="session-456")
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ServiceException" in str(e)


def test_list_branch_events_client_error():
    """Test list_branch_events with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ClientError
        error_response = {"Error": {"Code": "ThrottlingException", "Message": "Request throttled"}}
        mock_gmdp.list_events.side_effect = ClientError(error_response, "ListEvents")

        try:
            client.list_branch_events(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                branch_name="test-branch",
                max_results=100,
            )
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ThrottlingException" in str(e)


def test_get_event():
    """Test get_event functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock response
        mock_gmdp.get_event.return_value = {
            "event": {
                "eventId": "123#abc123",
                "memoryId": "mem-123",
                "actorId": "user-123",
                "sessionId": "session-456",
                "eventTimestamp": datetime.now(),
                "payload": [{"conversational": {"role": "USER", "content": {"text": "Hello"}}}],
            }
        }

        # Test get_event
        response = client.get_event(
            memoryId="mem-123",
            actorId="user-123",
            sessionId="session-456",
            eventId="123#abc123",
        )

        assert response["event"]["eventId"] == "123#abc123"
        assert response["event"]["memoryId"] == "mem-123"
        assert response["event"]["actorId"] == "user-123"
        assert response["event"]["sessionId"] == "session-456"
        assert len(response["event"]["payload"]) == 1
        assert response["event"]["payload"][0]["conversational"]["role"] == "USER"

        # Verify API call
        args, kwargs = mock_gmdp.get_event.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert kwargs["actorId"] == "user-123"
        assert kwargs["sessionId"] == "session-456"
        assert kwargs["eventId"] == "123#abc123"


def test_get_event_client_error():
    """Test get_event with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ClientError
        error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Event not found"}}
        mock_gmdp.get_event.side_effect = ClientError(error_response, "GetEvent")

        try:
            client.get_event(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                event_id="invalid-event",
            )
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ResourceNotFoundException" in str(e)


def test_delete_event():
    """Test delete_event functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Test delete_event
        client.delete_event(
            memoryId="mem-123",
            actorId="user-123",
            sessionId="session-456",
            eventId="123#abc123",
        )

        # Verify API call
        args, kwargs = mock_gmdp.delete_event.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert kwargs["actorId"] == "user-123"
        assert kwargs["sessionId"] == "session-456"
        assert kwargs["eventId"] == "123#abc123"


def test_delete_event_client_error():
    """Test delete_event with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ClientError
        error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Event not found"}}
        mock_gmdp.delete_event.side_effect = ClientError(error_response, "DeleteEvent")

        try:
            client.delete_event(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                event_id="invalid-event",
            )
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ResourceNotFoundException" in str(e)


def test_get_memory_record():
    """Test get_memory_record functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock response
        mock_gmdp.get_memory_record.return_value = {
            "memoryRecord": {
                "memoryRecordId": "rec-123",
                "memoryStrategyId": "strat-456",
                "content": {"text": "Memory record content"},
                "createdAt": int(time.time()),
                "namespaces": ["test/namespace"],
            }
        }

        # Test get_memory_record
        response = client.get_memory_record(
            memoryId="mem-123",
            memoryRecordId="rec-123",
        )

        assert response["memoryRecord"]["memoryRecordId"] == "rec-123"
        assert response["memoryRecord"]["memoryStrategyId"] == "strat-456"
        assert response["memoryRecord"]["content"]["text"] == "Memory record content"
        assert "createdAt" in response["memoryRecord"]
        assert response["memoryRecord"]["namespaces"] == ["test/namespace"]

        # Verify API call
        args, kwargs = mock_gmdp.get_memory_record.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert kwargs["memoryRecordId"] == "rec-123"


def test_get_memory_record_client_error():
    """Test get_memory_record with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ClientError
        error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Memory record not found"}}
        mock_gmdp.get_memory_record.side_effect = ClientError(error_response, "GetMemoryRecord")

        try:
            client.get_memory_record(
                memory_id="mem-123",
                memory_record_id="invalid-record",
            )
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ResourceNotFoundException" in str(e)


def test_delete_memory_record():
    """Test delete_memory_record functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock response
        mock_gmdp.delete_memory_record.return_value = {"memoryRecordId": "rec-123"}

        # Test delete_memory_record
        response = client.delete_memory_record(
            memoryId="mem-123",
            memoryRecordId="rec-123",
        )

        assert response == {"memoryRecordId": "rec-123"}

        # Verify API call
        args, kwargs = mock_gmdp.delete_memory_record.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert kwargs["memoryRecordId"] == "rec-123"


def test_delete_memory_record_client_error():
    """Test delete_memory_record with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ClientError
        error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Memory record not found"}}
        mock_gmdp.delete_memory_record.side_effect = ClientError(error_response, "DeleteMemoryRecord")

        try:
            client.delete_memory_record(
                memory_id="mem-123",
                memory_record_id="invalid-record",
            )
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ResourceNotFoundException" in str(e)


def test_list_memory_records():
    """Test list_memory_records functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock response
        mock_gmdp.list_memory_records.return_value = {
            "memoryRecordSummaries": [
                {
                    "memoryRecordId": "rec-1",
                    "memoryStrategyId": "strat-456",
                    "content": {"text": "Memory record 1"},
                    "createdAt": int(time.time()),
                    "namespaces": ["test/namespace"],
                    "score": 0.95,
                },
                {
                    "memoryRecordId": "rec-2",
                    "memoryStrategyId": "strat-456",
                    "content": {"text": "Memory record 2"},
                    "createdAt": int(time.time()),
                    "namespaces": ["test/namespace"],
                    "score": 0.85,
                },
            ],
            "nextToken": "next-page-token",
        }

        # Test list_memory_records
        response = client.list_memory_records(memoryId="mem-123", namespace="test/namespace", maxResults=10)

        assert response["memoryRecordSummaries"]
        assert len(response["memoryRecordSummaries"]) == 2
        assert response["memoryRecordSummaries"][0]["memoryRecordId"] == "rec-1"
        assert response["memoryRecordSummaries"][1]["memoryRecordId"] == "rec-2"
        assert response["memoryRecordSummaries"][0]["score"] == 0.95
        assert response["memoryRecordSummaries"][1]["score"] == 0.85
        assert response["nextToken"] == "next-page-token"

        # Verify API call
        args, kwargs = mock_gmdp.list_memory_records.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert kwargs["namespace"] == "test/namespace"
        assert kwargs["maxResults"] == 10


def test_list_memory_records_with_strategy_filter():
    """Test list_memory_records with strategy filter."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock response
        mock_gmdp.list_memory_records.return_value = {
            "memoryRecordSummaries": [
                {
                    "memoryRecordId": "rec-1",
                    "memoryStrategyId": "strat-123",
                    "content": {"text": "Memory record 1"},
                    "createdAt": int(time.time()),
                    "namespaces": ["test/namespace"],
                }
            ],
            "nextToken": None,
        }

        # Test list_memory_records with strategy filter
        response = client.list_memory_records(
            memoryId="mem-123", namespace="test/namespace", memoryStrategyId="strat-123", maxResults=10
        )

        assert response["memoryRecordSummaries"]
        assert len(response["memoryRecordSummaries"]) == 1
        assert response["memoryRecordSummaries"][0]["memoryRecordId"] == "rec-1"
        assert response["memoryRecordSummaries"][0]["memoryStrategyId"] == "strat-123"
        assert response["nextToken"] is None

        # Verify API call
        args, kwargs = mock_gmdp.list_memory_records.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert kwargs["namespace"] == "test/namespace"
        assert kwargs["memoryStrategyId"] == "strat-123"
        assert kwargs["maxResults"] == 10


def test_list_memory_records_pagination():
    """Test list_memory_records pagination."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock response for first page
        mock_gmdp.list_memory_records.side_effect = [
            {"memoryRecordSummaries": [{"memoryRecordId": "rec-1"}], "nextToken": "page2-token"},
            {"memoryRecordSummaries": [{"memoryRecordId": "rec-2"}], "nextToken": None},
        ]

        # Get first page
        response1 = client.list_memory_records(memoryId="mem-123", namespace="test/namespace")

        assert len(response1["memoryRecordSummaries"]) == 1
        assert response1["memoryRecordSummaries"][0]["memoryRecordId"] == "rec-1"
        assert response1["nextToken"] == "page2-token"

        # Get second page
        response2 = client.list_memory_records(
            memoryId="mem-123", namespace="test/namespace", nextToken=response1["nextToken"]
        )

        assert len(response2["memoryRecordSummaries"]) == 1
        assert response2["memoryRecordSummaries"][0]["memoryRecordId"] == "rec-2"
        assert response2["nextToken"] is None

        # Verify API calls
        assert mock_gmdp.list_memory_records.call_count == 2
        second_call = mock_gmdp.list_memory_records.call_args_list[1]
        assert second_call[1]["nextToken"] == "page2-token"


def test_list_memory_records_client_error():
    """Test list_memory_records with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Test various error types
        error_cases = [
            {"code": "ResourceNotFoundException", "message": "Memory not found"},
            {"code": "ValidationException", "message": "Invalid parameters"},
            {"code": "ServiceException", "message": "Internal service error"},
            {"code": "UnknownException", "message": "Unknown error"},
        ]

        for error in error_cases:
            # Mock ClientError
            error_response = {"Error": {"Code": error["code"], "Message": error["message"]}}
            mock_gmdp.list_memory_records.side_effect = ClientError(error_response, "ListMemoryRecords")

            # Test error handling
            try:
                client.list_memory_records(memoryId="mem-123", namespace="test/namespace")
                raise AssertionError("ClientError was not raised")
            except ClientError as e:
                assert error["code"] in str(e)


def test_get_last_k_turns_client_error():
    """Test get_last_k_turns with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ClientError
        error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Session not found"}}
        mock_gmdp.list_events.side_effect = ClientError(error_response, "ListEvents")

        try:
            client.get_last_k_turns(memory_id="mem-123", actor_id="user-123", session_id="nonexistent-session", k=5)
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ResourceNotFoundException" in str(e)


def test_fork_conversation_client_error():
    """Test fork_conversation with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ClientError
        error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid root event ID"}}
        mock_gmdp.create_event.side_effect = ClientError(error_response, "CreateEvent")

        try:
            client.fork_conversation(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                root_event_id="invalid-event-id",
                branch_name="test-branch",
                new_messages=[("Fork message", "USER")],
            )
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ValidationException" in str(e)


def test_get_memory_strategies_client_error():
    """Test get_memory_strategies with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock ClientError
        error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Memory not found"}}
        mock_gmcp.get_memory.side_effect = ClientError(error_response, "GetMemory")

        try:
            client.get_memory_strategies("nonexistent-mem-123")
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ResourceNotFoundException" in str(e)


def test_list_memories_client_error():
    """Test list_memories with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock ClientError
        error_response = {"Error": {"Code": "AccessDeniedException", "Message": "Insufficient permissions"}}
        mock_gmcp.list_memories.side_effect = ClientError(error_response, "ListMemories")

        try:
            client.list_memories(max_results=50)
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "AccessDeniedException" in str(e)


def test_delete_memory_client_error():
    """Test delete_memory with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock ClientError
        error_response = {"Error": {"Code": "ConflictException", "Message": "Memory is in use"}}
        mock_gmcp.delete_memory.side_effect = ClientError(error_response, "DeleteMemory")

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            try:
                client.delete_memory("mem-in-use")
                raise AssertionError("ClientError was not raised")
            except ClientError as e:
                assert "ConflictException" in str(e)


def test_update_memory_strategies_client_error():
    """Test update_memory_strategies with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock ClientError
        error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid strategy configuration"}}
        mock_gmcp.update_memory.side_effect = ClientError(error_response, "UpdateMemory")

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            try:
                add_strategies = [{StrategyType.SEMANTIC.value: {"name": "Invalid Strategy"}}]
                client.update_memory_strategies(memory_id="mem-123", add_strategies=add_strategies)
                raise AssertionError("ClientError was not raised")
            except ClientError as e:
                assert "ValidationException" in str(e)


def test_save_conversation_no_messages_error():
    """Test save_conversation with no messages raises ValueError."""
    with patch("boto3.client"):
        client = MemoryClient()

        try:
            client.save_conversation(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                messages=[],  # Empty messages list
            )
            raise AssertionError("ValueError was not raised")
        except ValueError as e:
            assert "At least one message is required" in str(e)


def test_save_conversation_invalid_message_format_error():
    """Test save_conversation with invalid message format raises ValueError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Test with message that doesn't have exactly 2 elements
        try:
            client.save_conversation(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                messages=[("Hello",)],  # Missing role  # type: ignore
            )
            raise AssertionError("ValueError was not raised")
        except ValueError as e:
            assert "Each message must be (text, role)" in str(e)

        # Test with message that has too many elements
        try:
            client.save_conversation(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                messages=[("Hello", "USER", "extra")],  # Too many elements  # type: ignore
            )
            raise AssertionError("ValueError was not raised")
        except ValueError as e:
            assert "Each message must be (text, role)" in str(e)


def test_save_conversation_invalid_role_error():
    """Test save_conversation with invalid role raises ValueError."""
    with patch("boto3.client"):
        client = MemoryClient()

        try:
            client.save_conversation(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                messages=[("Hello", "INVALID_ROLE")],  # Invalid role
            )
            raise AssertionError("ValueError was not raised")
        except ValueError as e:
            assert "Invalid role 'INVALID_ROLE'" in str(e)
            assert "Must be one of:" in str(e)


def test_create_blob_event():
    """Test create_blob_event functionality."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock create_event response
        mock_gmdp.create_event.return_value = {"event": {"eventId": "event-blob-123", "memoryId": "mem-123"}}

        # Test create_blob_event
        blob_data = {"file_content": "base64_encoded_data", "metadata": {"type": "image"}}
        result = client.create_blob_event(
            memory_id="mem-123",
            actor_id="user-123",
            session_id="session-456",
            blob_data=blob_data,
        )

        assert result["eventId"] == "event-blob-123"

        # Verify create_event was called with correct parameters
        args, kwargs = mock_gmdp.create_event.call_args
        assert kwargs["memoryId"] == "mem-123"
        assert kwargs["actorId"] == "user-123"
        assert kwargs["sessionId"] == "session-456"
        assert len(kwargs["payload"]) == 1
        assert "blob" in kwargs["payload"][0]
        assert kwargs["payload"][0]["blob"] == blob_data


def test_create_blob_event_with_branch():
    """Test create_blob_event with branch parameter."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock create_event response
        mock_gmdp.create_event.return_value = {"event": {"eventId": "event-blob-branch-123", "memoryId": "mem-123"}}

        # Test create_blob_event with branch
        blob_data = {"data": "test_data"}
        branch = {"name": "test-branch", "rootEventId": "event-root-123"}
        result = client.create_blob_event(
            memory_id="mem-123",
            actor_id="user-123",
            session_id="session-456",
            blob_data=blob_data,
            branch=branch,
        )

        assert result["eventId"] == "event-blob-branch-123"

        # Verify branch was passed correctly
        args, kwargs = mock_gmdp.create_event.call_args
        assert kwargs["branch"] == branch


def test_create_blob_event_client_error():
    """Test create_blob_event with ClientError."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the client
        mock_gmdp = MagicMock()
        client.gmdp_client = mock_gmdp

        # Mock ClientError
        error_response = {
            "Error": {
                "Code": "ValidationException",
                "Message": "Invalid blob data",
            }
        }
        mock_gmdp.create_event.side_effect = ClientError(error_response, "CreateEvent")

        try:
            client.create_blob_event(
                memory_id="mem-123",
                actor_id="user-123",
                session_id="session-456",
                blob_data={"invalid": "data"},
            )
            raise AssertionError("ClientError was not raised")
        except ClientError as e:
            assert "ValidationException" in str(e)


def test_create_or_get_memory_creates_new():
    """Test create_or_get_memory creates new memory when it doesn't exist."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock create_memory_and_wait to return successful result
        with patch.object(client, "create_memory_and_wait") as mock_create_and_wait:
            mock_create_and_wait.return_value = {"memoryId": "new-memory-123", "status": "ACTIVE"}

            result = client.create_or_get_memory(
                name="TestMemory",
                strategies=[{StrategyType.SEMANTIC.value: {"name": "TestStrategy"}}],
            )

            assert result["memoryId"] == "new-memory-123"
            assert mock_create_and_wait.called


def test_create_or_get_memory_gets_existing():
    """Test create_or_get_memory returns existing memory when it already exists."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the gmcp_client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock ValidationException for create_memory (memory already exists)
        error_response = {
            "Error": {
                "Code": "ValidationException",
                "Message": "Memory with name 'ExistingMemory' already exists",
            }
        }
        mock_gmcp.create_memory.side_effect = ClientError(error_response, "CreateMemory")

        # Mock list_memories response
        mock_gmcp.list_memories.return_value = {
            "memories": [{"id": "ExistingMemory-456", "name": "ExistingMemory", "status": "ACTIVE"}],
            "nextToken": None,
        }

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            result = client.create_or_get_memory(
                name="ExistingMemory",
                strategies=[{StrategyType.SEMANTIC.value: {"name": "TestStrategy"}}],
            )

            assert result["id"] == "ExistingMemory-456"
            assert mock_gmcp.create_memory.called
            assert mock_gmcp.list_memories.called


def test_create_or_get_memory_other_client_error():
    """Test create_or_get_memory raises other ClientErrors."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the gmcp_client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock different ClientError (not "already exists")
        error_response = {
            "Error": {
                "Code": "ValidationException",
                "Message": "Invalid parameters",
            }
        }
        mock_gmcp.create_memory.side_effect = ClientError(error_response, "CreateMemory")

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            try:
                client.create_or_get_memory(
                    name="TestMemory",
                    strategies=[{StrategyType.SEMANTIC.value: {"name": "TestStrategy"}}],
                )
                raise AssertionError("ClientError was not raised")
            except ClientError as e:
                assert "ValidationException" in str(e)
                assert "Invalid parameters" in str(e)


def test_create_or_get_memory_general_exception():
    """Test create_or_get_memory raises general exceptions."""
    with patch("boto3.client"):
        client = MemoryClient()

        # Mock the gmcp_client
        mock_gmcp = MagicMock()
        client.gmcp_client = mock_gmcp

        # Mock general exception
        mock_gmcp.create_memory.side_effect = Exception("Unexpected error")

        with patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
            try:
                client.create_or_get_memory(
                    name="TestMemory",
                    strategies=[{StrategyType.SEMANTIC.value: {"name": "TestStrategy"}}],
                )
                raise AssertionError("Exception was not raised")
            except Exception as e:
                assert "Unexpected error" in str(e)
