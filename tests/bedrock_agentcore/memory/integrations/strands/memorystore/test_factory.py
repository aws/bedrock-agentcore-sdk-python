"""Tests for AgentCore multi-namespace store construction."""

from typing import Any
from unittest.mock import Mock

import pytest
from strands.memory import ExtractionTrigger, ExtractionTriggerContext, MemoryMessageFilter

from bedrock_agentcore.memory.integrations.strands.memorystore.factory import (
    assert_writable_topology,
    create_agentcore_memory_stores,
)
from bedrock_agentcore.memory.integrations.strands.memorystore.store import AgentCoreMemoryStore


class FakeTrigger(ExtractionTrigger):
    """Minimal custom Strands extraction cadence."""

    name = "fake"

    def attach(self, context: ExtractionTriggerContext) -> None:
        """Accept an extraction context without registering hooks."""


def base_input(**overrides: Any) -> dict[str, Any]:
    """Build a two-namespace writable factory input."""
    result: dict[str, Any] = {
        "memory_id": "mem-1",
        "actor_id": "actor-1",
        "session_id": "sess-1",
        "namespaces": [
            {"namespace": "/strategy/s/actor/{actorId}/facts"},
            {"namespace": "/strategy/s/actor/{actorId}/preferences"},
        ],
        "extraction": {"cadence": FakeTrigger()},
        "client": Mock(),
    }
    result.update(overrides)
    return result


def test_returns_one_store_per_namespace_with_one_default_writer() -> None:
    """The factory creates one store per read namespace and one write sink."""
    stores = create_agentcore_memory_stores(**base_input())
    assert len(stores) == 2
    writers = [store for store in stores if store.writable]
    assert len(writers) == 1
    assert writers[0].name == "strategy-s-actor-facts"
    assert writers[0].extraction is not None
    assert next(store for store in stores if not store.writable).extraction is None


def test_custom_cadence_and_filter_reach_writer() -> None:
    """Translate factory extraction options to Strands ``ExtractionConfig`` keys."""
    trigger = FakeTrigger()
    message_filter = MemoryMessageFilter(exclude=["toolUse", "toolResult", "image"])
    stores = create_agentcore_memory_stores(**base_input(extraction={"cadence": trigger, "filter": message_filter}))
    extraction = next(store.extraction for store in stores if store.writable)
    assert extraction == {"trigger": trigger, "filter": message_filter}


def test_explicit_writer_flag_selects_non_first_namespace() -> None:
    """Honor the namespace explicitly designated as the write sink."""
    stores = create_agentcore_memory_stores(
        **base_input(
            namespaces=[
                {"namespace": "/strategy/s/actor/{actorId}/facts"},
                {
                    "namespace": "/strategy/s/actor/{actorId}/preferences",
                    "writable": True,
                },
            ]
        )
    )
    writers = [store for store in stores if store.writable]
    assert [store.name for store in writers] == ["strategy-s-actor-preferences"]


def test_explicit_opt_out_skips_first_default_writer_candidate() -> None:
    """Do not override a namespace's ``writable=False`` opt-out."""
    stores = create_agentcore_memory_stores(
        **base_input(
            namespaces=[
                {
                    "namespace": "/strategy/s/actor/{actorId}/facts",
                    "writable": False,
                },
                {"namespace": "/strategy/s/actor/{actorId}/preferences"},
            ],
            extraction=True,
        )
    )
    assert [store.name for store in stores if store.writable] == ["strategy-s-actor-preferences"]


def test_all_explicit_opt_outs_reject_enabled_extraction() -> None:
    """Enabled extraction requires an eligible write sink."""
    with pytest.raises(ValueError, match="every namespace is marked writable: false"):
        create_agentcore_memory_stores(
            **base_input(
                namespaces=[
                    {"namespace": "/a/{actorId}", "writable": False},
                    {"namespace": "/b/{actorId}", "writable": False},
                ],
                extraction=True,
            )
        )


def test_multiple_explicit_writers_are_rejected() -> None:
    """Namespace-free ``create_event`` would otherwise duplicate writes."""
    with pytest.raises(ValueError, match="at most one store may be writable"):
        create_agentcore_memory_stores(
            **base_input(
                namespaces=[
                    {"namespace": "/a/{actorId}", "writable": True},
                    {"namespace": "/b/{actorId}", "writable": True},
                ]
            )
        )


@pytest.mark.parametrize("extraction", [None, False])
def test_recall_only_has_no_writer(extraction: object) -> None:
    """Omitted and false extraction both construct read-only stores."""
    input_data = base_input(extraction=extraction)
    if extraction is None:
        input_data.pop("extraction")
    stores = create_agentcore_memory_stores(**input_data)
    assert all(not store.writable for store in stores)
    assert all(store.extraction is None for store in stores)


def test_extraction_true_passes_framework_default_shorthand() -> None:
    """Let MemoryManager choose its standard cadence."""
    stores = create_agentcore_memory_stores(**base_input(extraction=True))
    assert next(store.extraction for store in stores if store.writable) is True


def test_names_are_derived_or_respected() -> None:
    """Use explicit names and a fallback for placeholder-only namespaces."""
    stores = create_agentcore_memory_stores(
        **base_input(
            namespaces=[
                {"namespace": "/a/{actorId}", "name": "alpha"},
                {"namespace": "/b/{actorId}", "name": "beta"},
            ]
        )
    )
    assert [store.name for store in stores] == ["alpha", "beta"]
    fallback = create_agentcore_memory_stores(**base_input(namespaces=[{"namespace": "{actorId}"}]))
    assert fallback[0].name == "agentcore-memory"


def test_factory_shares_one_client_across_stores() -> None:
    """Construct or accept one boto3 client for the complete topology."""
    client = Mock()
    stores = create_agentcore_memory_stores(**base_input(client=client))
    assert all(isinstance(store, AgentCoreMemoryStore) and store._client is client for store in stores)


@pytest.mark.parametrize(
    "namespaces",
    [[], [{"namespace": "   "}], [{}], [None]],
)
def test_rejects_missing_or_invalid_namespaces(namespaces: list[object]) -> None:
    """Require at least one non-empty namespace string."""
    expected = "at least one namespace" if not namespaces else r"namespaces\[0\]\.namespace"
    with pytest.raises(ValueError, match=expected):
        create_agentcore_memory_stores(**base_input(namespaces=namespaces))


def test_namespace_validation_uses_python_strip_semantics() -> None:
    """Reject Python whitespace-only namespaces and retain BOM content."""
    stores = create_agentcore_memory_stores(**base_input(namespaces=[{"namespace": "\ufeff"}]))
    assert len(stores) == 1
    with pytest.raises(ValueError, match=r"namespaces\[0\]\.namespace must be a non-empty"):
        create_agentcore_memory_stores(**base_input(namespaces=[{"namespace": "\u0085"}]))


@pytest.mark.parametrize(
    "override",
    [{"actor_id": ""}, {"session_id": "  "}, {"memory_id": ""}],
)
def test_identity_validation_propagates_from_store(override: dict[str, str]) -> None:
    """Keep flat identity validation consistent with direct construction."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        create_agentcore_memory_stores(**base_input(**override))


def test_unresolved_placeholder_validation_propagates() -> None:
    """Reject unsupported namespace placeholders in the factory path too."""
    with pytest.raises(ValueError, match=r"\{memoryStrategyId\}"):
        create_agentcore_memory_stores(
            **base_input(namespaces=[{"namespace": "/strategies/{memoryStrategyId}/actors/{actorId}"}])
        )


@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_factory_validates_event_cap_even_for_recall_only(value: object) -> None:
    """Validate tuning even when no sender is built."""
    with pytest.raises(ValueError, match="positive integer"):
        create_agentcore_memory_stores(**base_input(extraction=False, max_turns_per_event=value))


def hand_built_store(*, name: str = "facts", writable: bool = False) -> AgentCoreMemoryStore:
    """Build one store for topology assertions."""
    return AgentCoreMemoryStore(
        memory_id="mem-1",
        actor_id="actor-1",
        session_id="sess-1",
        namespace="/users/{actorId}/facts",
        name=name,
        writable=writable,
        client=Mock(),
    )


def test_assert_writable_topology_accepts_zero_or_one_writer() -> None:
    """Recall-only and exactly-one-writer topologies are valid."""
    assert_writable_topology([hand_built_store(), hand_built_store(name="prefs")])
    assert_writable_topology([hand_built_store(writable=True), hand_built_store(name="prefs")])


def test_assert_writable_topology_rejects_multiple_writers() -> None:
    """Hand-built store sets can use the same exported guard."""
    with pytest.raises(ValueError, match="at most one store may be writable"):
        assert_writable_topology([hand_built_store(name="a", writable=True), hand_built_store(name="b", writable=True)])


def test_assert_writable_topology_can_require_writer() -> None:
    """Expected extraction turns zero writers into an error."""
    with pytest.raises(ValueError, match="no store is writable"):
        assert_writable_topology([hand_built_store()], True)
    assert_writable_topology([hand_built_store()], False)


@pytest.mark.parametrize("value", [101, 1000])
def test_factory_accepts_event_caps_above_python_service_assumption(value: int) -> None:
    """Match source validation, which only requires a positive integer."""
    stores = create_agentcore_memory_stores(**base_input(extraction=True, max_turns_per_event=value))
    writer = next(store for store in stores if store.writable)
    assert writer._sender is not None
    assert writer._sender._max_turns_per_event == value


def test_factory_binds_actor_and_session_into_distinct_namespaces() -> None:
    """Each factory call resolves its own actor/session identity."""
    namespace = "/users/{actorId}/sessions/{sessionId}/facts"
    first = create_agentcore_memory_stores(
        **base_input(actor_id="actor-a", session_id="session-a", namespaces=[{"namespace": namespace}])
    )
    second = create_agentcore_memory_stores(
        **base_input(actor_id="actor-b", session_id="session-b", namespaces=[{"namespace": namespace}])
    )
    assert first[0]._resolved_namespace == "/users/actor-a/sessions/session-a/facts"
    assert second[0]._resolved_namespace == "/users/actor-b/sessions/session-b/facts"


def test_factory_preserves_explicit_falsey_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use nullish client selection rather than truthiness."""
    from bedrock_agentcore.memory.integrations.strands.memorystore import factory as factory_module

    class FalseyClient:
        def __bool__(self) -> bool:
            return False

        def create_event(self, **_kwargs: Any) -> dict[str, Any]:
            return {}

        def retrieve_memory_records(self, **_kwargs: Any) -> dict[str, Any]:
            return {"memoryRecordSummaries": []}

    client = FalseyClient()
    create = Mock(side_effect=AssertionError("must not construct a replacement client"))
    monkeypatch.setattr(factory_module, "_create_data_plane_client", create)
    stores = create_agentcore_memory_stores(**base_input(client=client))
    assert all(store._client is client for store in stores)
    create.assert_not_called()
