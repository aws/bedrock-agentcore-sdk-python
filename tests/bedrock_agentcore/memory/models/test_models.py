"""Unit tests for memory model classes."""

from bedrock_agentcore.memory.models import (
    ActorSummary,
    Branch,
    Event,
    EventMessage,
    MemoryRecord,
    SessionSummary,
)


class TestActorSummary:
    """Test cases for ActorSummary class."""

    def test_actor_summary_initialization(self):
        """Test ActorSummary initialization."""
        data = {
            "actorId": "user-123",
            "createdAt": "2023-01-01T00:00:00Z",
            "lastActiveAt": "2023-01-02T00:00:00Z",
        }
        actor_summary = ActorSummary(data)

        assert actor_summary._data == data
        assert actor_summary.actorId == "user-123"
        assert actor_summary["actorId"] == "user-123"
        assert actor_summary.get("actorId") == "user-123"

    def test_actor_summary_dict_access(self):
        """Test ActorSummary dictionary-like access."""
        data = {"actorId": "user-456", "metadata": {"role": "admin"}}
        actor_summary = ActorSummary(data)

        assert "actorId" in actor_summary
        assert "nonexistent" not in actor_summary
        assert list(actor_summary.keys()) == ["actorId", "metadata"]


class TestBranch:
    """Test cases for Branch class."""

    def test_branch_initialization(self):
        """Test Branch initialization."""
        data = {
            "name": "feature-branch",
            "rootEventId": "event-123",
            "firstEventId": "event-124",
            "eventCount": 5,
            "created": "2023-01-01T00:00:00Z",
        }
        branch = Branch(data)

        assert branch._data == data
        assert branch.name == "feature-branch"
        assert branch["rootEventId"] == "event-123"
        assert branch.get("eventCount") == 5

    def test_branch_dict_access(self):
        """Test Branch dictionary-like access."""
        data = {"name": "main", "rootEventId": None, "eventCount": 10}
        branch = Branch(data)

        assert "name" in branch
        assert "nonexistent" not in branch
        assert set(branch.keys()) == {"name", "rootEventId", "eventCount"}


class TestEvent:
    """Test cases for Event class."""

    def test_event_initialization(self):
        """Test Event initialization."""
        data = {
            "eventId": "event-123",
            "memoryId": "memory-456",
            "actorId": "user-789",
            "sessionId": "session-abc",
            "eventTimestamp": "2023-01-01T00:00:00Z",
            "payload": [{"conversational": {"role": "USER", "content": {"text": "Hello"}}}],
        }
        event = Event(data)

        assert event._data == data
        assert event.eventId == "event-123"
        assert event["memoryId"] == "memory-456"
        assert event.get("actorId") == "user-789"

    def test_event_dict_access(self):
        """Test Event dictionary-like access."""
        data = {"eventId": "event-456", "payload": []}
        event = Event(data)

        assert "eventId" in event
        assert "nonexistent" not in event
        assert set(event.keys()) == {"eventId", "payload"}


class TestEventMessage:
    """Test cases for EventMessage class."""

    def test_event_message_initialization(self):
        """Test EventMessage initialization."""
        data = {
            "role": "USER",
            "content": {"text": "Hello, how are you?"},
            "timestamp": "2023-01-01T00:00:00Z",
        }
        event_message = EventMessage(data)

        assert event_message._data == data
        assert event_message.role == "USER"
        assert event_message["content"]["text"] == "Hello, how are you?"
        assert event_message.get("timestamp") == "2023-01-01T00:00:00Z"

    def test_event_message_dict_access(self):
        """Test EventMessage dictionary-like access."""
        data = {"role": "ASSISTANT", "content": {"text": "I'm doing well, thank you!"}}
        event_message = EventMessage(data)

        assert "role" in event_message
        assert "nonexistent" not in event_message
        assert set(event_message.keys()) == {"role", "content"}


class TestMemoryRecord:
    """Test cases for MemoryRecord class."""

    def test_memory_record_initialization(self):
        """Test MemoryRecord initialization."""
        data = {
            "memoryRecordId": "record-123",
            "content": {"text": "This is a memory record"},
            "namespace": "user/preferences/",
            "relevanceScore": 0.95,
            "createdAt": "2023-01-01T00:00:00Z",
        }
        memory_record = MemoryRecord(data)

        assert memory_record._data == data
        assert memory_record.memoryRecordId == "record-123"
        assert memory_record["content"]["text"] == "This is a memory record"
        assert memory_record.get("relevanceScore") == 0.95

    def test_memory_record_dict_access(self):
        """Test MemoryRecord dictionary-like access."""
        data = {"memoryRecordId": "record-456", "namespace": "support/facts/"}
        memory_record = MemoryRecord(data)

        assert "memoryRecordId" in memory_record
        assert "nonexistent" not in memory_record
        assert set(memory_record.keys()) == {"memoryRecordId", "namespace"}


class TestSessionSummary:
    """Test cases for SessionSummary class."""

    def test_session_summary_initialization(self):
        """Test SessionSummary initialization."""
        data = {
            "sessionId": "session-123",
            "actorId": "user-456",
            "memoryId": "memory-789",
            "createdAt": "2023-01-01T00:00:00Z",
            "lastActiveAt": "2023-01-02T00:00:00Z",
            "eventCount": 25,
        }
        session_summary = SessionSummary(data)

        assert session_summary._data == data
        assert session_summary.sessionId == "session-123"
        assert session_summary["actorId"] == "user-456"
        assert session_summary.get("eventCount") == 25

    def test_session_summary_dict_access(self):
        """Test SessionSummary dictionary-like access."""
        data = {"sessionId": "session-789", "eventCount": 0}
        session_summary = SessionSummary(data)

        assert "sessionId" in session_summary
        assert "nonexistent" not in session_summary
        assert set(session_summary.keys()) == {"sessionId", "eventCount"}


class TestModelInheritance:
    """Test cases to verify all models inherit from DictWrapper correctly."""

    def test_all_models_inherit_dict_wrapper_methods(self):
        """Test that all model classes inherit DictWrapper functionality."""
        test_data = {"key": "value", "number": 42}

        models = [
            ActorSummary(test_data),
            Branch(test_data),
            Event(test_data),
            EventMessage(test_data),
            MemoryRecord(test_data),
            SessionSummary(test_data),
        ]

        for model in models:
            # Test attribute access
            assert model.key == "value"
            assert model.number == 42

            # Test dictionary access
            assert model["key"] == "value"
            assert model["number"] == 42

            # Test get method
            assert model.get("key") == "value"
            assert model.get("missing", "default") == "default"

            # Test contains
            assert "key" in model
            assert "missing" not in model

            # Test dict methods
            assert "key" in model.keys()
            assert "value" in model.values()
            assert ("key", "value") in model.items()

            # Test string representation
            assert str(model) == str(test_data)
            assert repr(model) == repr(test_data)

    def test_model_with_empty_data(self):
        """Test all models work with empty data."""
        empty_data = {}

        models = [
            ActorSummary(empty_data),
            Branch(empty_data),
            Event(empty_data),
            EventMessage(empty_data),
            MemoryRecord(empty_data),
            SessionSummary(empty_data),
        ]

        for model in models:
            assert model.nonexistent is None
            assert model.get("nonexistent") is None
            assert "nonexistent" not in model
            assert list(model.keys()) == []
            assert list(model.values()) == []
            assert list(model.items()) == []

    def test_model_with_complex_data(self):
        """Test all models work with complex nested data."""
        complex_data = {
            "simple": "value",
            "nested": {"inner": {"deep": "value"}},
            "list": [1, 2, 3],
            "mixed": {"list": [{"key": "value"}], "number": 42},
        }

        models = [
            ActorSummary(complex_data),
            Branch(complex_data),
            Event(complex_data),
            EventMessage(complex_data),
            MemoryRecord(complex_data),
            SessionSummary(complex_data),
        ]

        for model in models:
            assert model.simple == "value"
            assert model.nested == {"inner": {"deep": "value"}}
            assert model.list == [1, 2, 3]
            assert model.mixed["number"] == 42
            assert model["nested"]["inner"]["deep"] == "value"
