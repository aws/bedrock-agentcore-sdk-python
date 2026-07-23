"""Tests for the native Strands AgentCore memory store."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock

import pytest
from strands.memory import AddMessagesContext

from bedrock_agentcore.memory.integrations.strands.memorystore.store import AgentCoreMemoryStore
from bedrock_agentcore.memory.integrations.strands.memorystore.types import RESERVED_METADATA_PREFIX

from .test_sender import assistant_message, user_message


def record(
    record_id: str,
    text: str,
    score: float | None = None,
    namespaces: list[str] | None = None,
) -> dict[str, Any]:
    """Build a data-plane memory record summary."""
    result: dict[str, Any] = {
        "memoryRecordId": record_id,
        "content": {"text": text},
        "namespaces": namespaces or ["/ns/a"],
        "memoryStrategyId": "strategy",
        "createdAt": datetime.now(timezone.utc),
    }
    if score is not None:
        result["score"] = score
    return result


def client_returning(records: list[dict[str, Any]] | None) -> Mock:
    """Build a mock boto3 client returning record summaries."""
    client = Mock()
    client.retrieve_memory_records.return_value = {"memoryRecordSummaries": records}
    client.create_event.return_value = {}
    return client


def make_store(client: Mock, **overrides: Any) -> AgentCoreMemoryStore:
    """Build an exact-mode store with test identity."""
    config: dict[str, Any] = {
        "memory_id": "mem-1",
        "actor_id": "actor-1",
        "session_id": "sess-1",
        "namespace": "/strategy/s/actor/{actorId}/preferences",
        "name": "prefs",
        "writable": False,
        "client": client,
    }
    config.update(overrides)
    return AgentCoreMemoryStore(**config)


async def test_exact_namespace_resolves_actor_and_uses_namespace() -> None:
    """Exact mode emits the exact-prefix wire field."""
    client = client_returning([record("1", "a")])
    await make_store(client).search("q")
    kwargs = client.retrieve_memory_records.call_args.kwargs
    assert kwargs["namespace"] == "/strategy/s/actor/actor-1/preferences"
    assert "namespacePath" not in kwargs


async def test_subtree_mode_uses_namespace_path() -> None:
    """Subtree mode emits ``namespacePath`` instead of ``namespace``."""
    client = client_returning([record("1", "a")])
    store = make_store(
        client,
        namespace=None,
        namespace_path="/strategy/s/actor/{actorId}",
    )
    await store.search("q")
    kwargs = client.retrieve_memory_records.call_args.kwargs
    assert kwargs["namespacePath"] == "/strategy/s/actor/actor-1"
    assert "namespace" not in kwargs


async def test_maps_memory_record_summary_to_entry() -> None:
    """Map content and reserved metadata using Python datetimes."""
    created_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    client = client_returning(
        [
            {
                "memoryRecordId": "rec-9",
                "content": {"text": "dark mode"},
                "score": 0.8,
                "namespaces": ["/ns/x"],
                "createdAt": created_at,
            }
        ]
    )
    results = await make_store(client).search("q")
    assert len(results) == 1
    assert results[0].content == "dark mode"
    assert results[0].metadata == {
        "_id": "rec-9",
        "_score": 0.8,
        "_namespaces": ["/ns/x"],
        "_createdAt": "2026-01-02T03:04:05.000Z",
    }
    assert all(key.startswith(RESERVED_METADATA_PREFIX) for key in results[0].metadata or {})


async def test_top_k_equals_want_without_score_floor() -> None:
    """Do not overfetch when no client-side filter can remove results."""
    client = client_returning([])
    await make_store(client, max_search_results=3, over_fetch_factor=10).search("q")
    assert client.retrieve_memory_records.call_args.kwargs["searchCriteria"]["topK"] == 3


async def test_score_floor_overfetches_filters_and_trims() -> None:
    """Overfetch before applying the client-side relevance floor."""
    client = client_returning(
        [
            record("1", "a", 0.9),
            record("2", "b", 0.1),
            record("3", "c", 0.7),
            record("4", "d", 0.2),
            record("5", "e", 0.6),
        ]
    )
    results = await make_store(client, max_search_results=2, min_score=0.5).search("q")
    assert client.retrieve_memory_records.call_args.kwargs["searchCriteria"]["topK"] == 8
    assert [result.content for result in results] == ["a", "c"]


@pytest.mark.parametrize(
    ("want", "factor", "expected"),
    [(3, 10, 30), (5, 1.5, 8), (50, 3, 100)],
)
async def test_custom_overfetch_is_ceiled_and_capped(want: int, factor: float, expected: int) -> None:
    """Keep AgentCore ``topK`` integral and no larger than 100."""
    client = client_returning([])
    await make_store(
        client,
        max_search_results=want,
        min_score=0.5,
        over_fetch_factor=factor,
    ).search("q")
    assert client.retrieve_memory_records.call_args.kwargs["searchCriteria"]["topK"] == expected


@pytest.mark.parametrize("value", [0, -1, 2.5, float("nan")])
async def test_invalid_call_time_result_cap_fails_before_network(value: Any) -> None:
    """Validate the effective search option, not only constructor defaults."""
    client = client_returning([])
    with pytest.raises(ValueError, match="max_search_results must be a positive integer"):
        await make_store(client).search("q", {"max_search_results": value})
    client.retrieve_memory_records.assert_not_called()


async def test_unscored_records_are_zero_under_positive_floor() -> None:
    """Treat absent scores as zero for filtering."""
    client = client_returning([record("1", "scored", 0.9), record("2", "unscored")])
    results = await make_store(client, min_score=0.5).search("q")
    assert [result.content for result in results] == ["scored"]


async def test_non_text_content_maps_to_empty_string() -> None:
    """Unknown MemoryContent union members do not become text."""
    item = record("x", "ignored", 0.9)
    item["content"] = {"unknown": ["blob", {}]}
    results = await make_store(client_returning([item])).search("q")
    assert results[0].content == ""


async def test_empty_response_returns_empty_list() -> None:
    """An absent summary list is an empty search result."""
    assert await make_store(client_returning(None)).search("q") == []


async def test_retrieve_errors_propagate() -> None:
    """Let MemoryManager isolate and report store failures."""
    client = Mock()
    client.retrieve_memory_records.side_effect = RuntimeError("throttled")
    with pytest.raises(RuntimeError, match="throttled"):
        await make_store(client).search("q")


async def test_writable_store_sends_messages_and_sequence_numbers() -> None:
    """Use the sender's batched deterministic-token path."""
    client = client_returning([])
    store = make_store(client, writable=True)
    await store.add_messages(
        [user_message("first"), assistant_message("second")],
        AddMessagesContext(sequence_numbers=[41, 42]),
    )
    client.create_event.assert_called_once()
    kwargs = client.create_event.call_args.kwargs
    assert len(kwargs["payload"]) == 2
    assert kwargs["payload"][0]["conversational"]["role"] == "USER"
    assert kwargs["clientToken"].endswith("-41-42")


def test_unsupported_optional_methods_are_absent_at_runtime() -> None:
    """Keep Strands capability detection aligned with the methods actually supported."""
    store = make_store(client_returning([]))
    assert not hasattr(store, "add")
    assert not hasattr(store, "initialize")
    assert not hasattr(store, "get_tools")


async def test_non_writable_store_rejects_add_messages() -> None:
    """Guard direct misuse even though MemoryManager will not call this sink."""
    with pytest.raises(ValueError, match="not writable"):
        await make_store(client_returning([])).add_messages([user_message("x")])


def test_only_writable_store_carries_extraction(caplog: pytest.LogCaptureFixture) -> None:
    """Drop and warn about extraction configuration on a recall-only store."""
    trigger = Mock()
    config = {"trigger": trigger}
    writable = make_store(client_returning([]), writable=True, extraction=config)
    with caplog.at_level("WARNING"):
        readonly = make_store(client_returning([]), extraction=config)
    assert writable.extraction == config
    assert readonly.extraction is None
    assert "writable is false" in caplog.text


@pytest.mark.parametrize("extraction", [None, False])
def test_recall_only_store_does_not_warn(extraction: object, caplog: pytest.LogCaptureFixture) -> None:
    """No extraction and explicit opt-out are valid recall-only configurations."""
    with caplog.at_level("WARNING"):
        make_store(client_returning([]), extraction=extraction)
    assert "writable is false" not in caplog.text


@pytest.mark.parametrize("name", [None, "", "   "])
def test_store_self_names_from_namespace_when_name_absent(name: str | None) -> None:
    """Use a non-degenerate namespace slug."""
    store = make_store(
        client_returning([]),
        name=name,
        namespace="/users/{actorId}/facts",
    )
    assert store.name == "users-facts"


@pytest.mark.parametrize("field", ["memory_id", "actor_id", "session_id"])
def test_identity_uses_python_strip_semantics(field: str) -> None:
    """Reject Python whitespace-only identity and retain BOM content."""
    client = client_returning([])
    store = make_store(client, **{field: "\ufeff"})
    assert getattr(store, f"_{field}") == "\ufeff"
    with pytest.raises(ValueError, match=rf"{field} must be a non-empty"):
        make_store(client, **{field: "\u0085"})


def test_explicit_name_uses_python_strip_semantics() -> None:
    """Treat Python whitespace as absent and retain a BOM name."""
    client = client_returning([])
    assert make_store(client, name=" \u0085 ").name == "strategy-s-actor-preferences"
    assert make_store(client, name="\ufeff").name == "\ufeff"


def test_read_target_uses_python_strip_semantics() -> None:
    """Reject Python whitespace-only targets and retain BOM content."""
    client = client_returning([])
    assert make_store(client, namespace="\ufeff")._resolved_namespace == "\ufeff"
    with pytest.raises(ValueError, match="namespace must be a non-empty"):
        make_store(client, namespace="\u0085")


def test_writable_defaults_false() -> None:
    """A bare store is recall-safe."""
    store = AgentCoreMemoryStore(
        memory_id="mem-1",
        actor_id="actor-1",
        session_id="sess-1",
        namespace="/users/{actorId}/facts",
        client=client_returning([]),
    )
    assert store.writable is False


async def test_direct_store_stands_alone_without_factory() -> None:
    """Flat identity plus namespace is sufficient for read/write construction."""
    client = client_returning([])
    store = AgentCoreMemoryStore(
        memory_id="mem-1",
        actor_id="actor-1",
        session_id="sess-1",
        namespace="/users/{actorId}/facts",
        writable=True,
        extraction=True,
        client=client,
    )
    assert store.name == "users-facts"
    assert store.extraction is True
    await store.add_messages([user_message("hi")])
    assert client.create_event.call_args.kwargs["actorId"] == "actor-1"


@pytest.mark.parametrize(
    ("field", "value"),
    [("memory_id", ""), ("actor_id", "   "), ("session_id", "")],
)
def test_rejects_empty_identity(field: str, value: str) -> None:
    """Validate each flat identity field."""
    kwargs: dict[str, Any] = {
        "memory_id": "mem-1",
        "actor_id": "actor-1",
        "session_id": "sess-1",
        field: value,
    }
    with pytest.raises(ValueError, match=rf"{field} must be a non-empty"):
        AgentCoreMemoryStore(
            **kwargs,
            namespace="/users/{actorId}/facts",
            client=client_returning([]),
        )


def test_rejects_empty_or_ambiguous_read_target() -> None:
    """Require exactly one non-empty read target."""
    client = client_returning([])
    with pytest.raises(ValueError, match="namespace must be a non-empty"):
        make_store(client, namespace="   ")
    with pytest.raises(ValueError, match="exactly one"):
        make_store(client, namespace=None)
    with pytest.raises(ValueError, match="exactly one"):
        make_store(client, namespace="/a", namespace_path="/b")


@pytest.mark.parametrize(
    "target",
    [
        {"namespace": "/strategies/{memoryStrategyId}/actors/{actorId}/facts"},
        {"namespace": "/users/{actorId}/we{ird"},
        {"namespace": "/users/{actorId}/weird}"},
        {"namespace": "/a/{strategy/b"},
        {"namespace": None, "namespace_path": "/strategies/{memoryStrategyId}/actors/{actorId}"},
    ],
)
def test_rejects_unresolved_or_malformed_placeholders(target: dict[str, object]) -> None:
    """Fail before AgentCore rejects braces at first retrieval."""
    with pytest.raises(ValueError, match="still contains"):
        make_store(client_returning([]), **target)


def test_actor_dollar_sequences_are_inserted_verbatim() -> None:
    """Python substitution does not interpret JavaScript-style replacement syntax."""
    store = make_store(
        client_returning([]),
        actor_id="a$$b",
        namespace="/p/{actorId}/x",
    )
    assert store.name == "prefs"


@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.5, float("inf")])
def test_rejects_invalid_min_score(value: float) -> None:
    """The relevance floor must be finite and normalized."""
    with pytest.raises(ValueError, match="finite number between 0 and 1"):
        make_store(client_returning([]), min_score=value)


@pytest.mark.parametrize("value", [0, -1, 2.5])
def test_rejects_invalid_constructor_result_cap(value: Any) -> None:
    """The store-level result cap must be a positive integer."""
    with pytest.raises(ValueError, match="positive integer"):
        make_store(client_returning([]), max_search_results=value)


@pytest.mark.parametrize("value", [0, 0.5, float("nan"), float("inf")])
def test_rejects_invalid_overfetch_factor(value: float) -> None:
    """Overfetch factors must be finite and at least one."""
    with pytest.raises(ValueError, match="number >= 1"):
        make_store(client_returning([]), over_fetch_factor=value)


def test_direct_store_rejects_invalid_event_cap() -> None:
    """Writable direct construction delegates cap validation to its sender."""
    with pytest.raises(ValueError, match="positive integer"):
        make_store(client_returning([]), writable=True, max_turns_per_event=0)


async def test_ordered_substitution_rescans_actor_value_during_session_pass() -> None:
    """Match source runtime chaining when actor substitution introduces a session token."""
    client = client_returning([])
    store = make_store(
        client,
        actor_id="literal-{sessionId}",
        session_id="runtime-parity",
        namespace="/p/{actorId}/x",
    )
    await store.search("q")
    assert client.retrieve_memory_records.call_args.kwargs["namespace"] == "/p/literal-runtime-parity/x"


async def test_search_sends_only_source_equivalent_top_k() -> None:
    """Do not add the target-only top-level ``maxResults`` request field."""
    client = client_returning([])
    await make_store(client, max_search_results=10, min_score=0.5).search("q")
    kwargs = client.retrieve_memory_records.call_args.kwargs
    assert kwargs["searchCriteria"]["topK"] == 40
    assert "maxResults" not in kwargs


async def test_result_cap_above_100_is_allowed_without_score_floor() -> None:
    """Match source positive-integer validation; only overfetch topK is capped."""
    client = client_returning([])
    await make_store(client, max_search_results=101).search("q")
    assert client.retrieve_memory_records.call_args.kwargs["searchCriteria"]["topK"] == 101


async def test_result_cap_above_100_caps_only_overfetch_top_k() -> None:
    """A score-floor request keeps the source's MAX_TOPK overfetch cap."""
    client = client_returning([])
    await make_store(client, max_search_results=101, min_score=0.5).search("q")
    assert client.retrieve_memory_records.call_args.kwargs["searchCriteria"]["topK"] == 100


def test_client_region_prefers_explicit_session_without_loading_invalid_ambient_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit session bypasses an invalid ambient profile and supplies its region."""
    import boto3

    from bedrock_agentcore.memory.integrations.strands.memorystore import store as store_module

    supplied_session = boto3.Session(
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="session-region",
    )
    create_client = Mock(return_value=Mock())
    monkeypatch.setattr(supplied_session, "client", create_client)
    monkeypatch.setenv("AWS_PROFILE", "profile-that-does-not-exist")
    monkeypatch.setenv("AWS_REGION", "environment-region")

    store_module._create_data_plane_client(boto3_session=supplied_session)

    create_client.assert_called_once()
    assert create_client.call_args.kwargs["region_name"] == "session-region"


def test_explicit_region_overrides_explicit_session_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the caller's explicit region before the selected session region."""
    from bedrock_agentcore.memory.integrations.strands.memorystore import store as store_module

    supplied_session = Mock(region_name="session-region")
    supplied_session.client.return_value = Mock()
    monkeypatch.setenv("AWS_REGION", "environment-region")
    store_module._create_data_plane_client(region_name="explicit-region", boto3_session=supplied_session)
    assert supplied_session.client.call_args.kwargs["region_name"] == "explicit-region"


def test_client_region_falls_back_through_environment_default_and_us_west(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve the selected default session, environment, then SDK fallback region."""
    from bedrock_agentcore.memory.integrations.strands.memorystore import store as store_module

    default_session = Mock(region_name="default-region")
    default_session.client.return_value = Mock()
    session_factory = Mock(return_value=default_session)
    monkeypatch.setattr(store_module.boto3, "Session", session_factory)
    monkeypatch.setenv("AWS_REGION", "environment-region")
    store_module._create_data_plane_client()
    assert default_session.client.call_args.kwargs["region_name"] == "default-region"

    default_session.client.reset_mock()
    default_session.region_name = None
    store_module._create_data_plane_client()
    assert default_session.client.call_args.kwargs["region_name"] == "environment-region"

    default_session.client.reset_mock()
    monkeypatch.delenv("AWS_REGION")
    store_module._create_data_plane_client()
    assert default_session.client.call_args.kwargs["region_name"] == "us-west-2"


class _FalseyClient:
    """Client whose truth value is false but whose methods remain usable."""

    def __bool__(self) -> bool:
        return False

    def retrieve_memory_records(self, **_kwargs: Any) -> dict[str, Any]:
        return {"memoryRecordSummaries": []}

    def create_event(self, **_kwargs: Any) -> dict[str, Any]:
        return {}


def test_store_preserves_explicit_falsey_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use nullish client selection rather than truthiness."""
    from bedrock_agentcore.memory.integrations.strands.memorystore import store as store_module

    falsey = _FalseyClient()
    create = Mock(side_effect=AssertionError("must not construct a replacement client"))
    monkeypatch.setattr(store_module, "_create_data_plane_client", create)
    store = make_store(falsey)  # type: ignore[arg-type]
    assert store._client is falsey
    create.assert_not_called()
