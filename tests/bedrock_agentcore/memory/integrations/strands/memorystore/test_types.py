"""Runtime type metadata tests for the public Strands memory-store types."""

from typing import get_args, get_type_hints

import boto3

from bedrock_agentcore.memory.integrations.strands.memorystore.types import (
    AgentCoreEventSenderConfig,
    AgentCoreExactNamespaceStoreConfig,
    AgentCoreExtractionConfig,
    AgentCoreNamespaceConfig,
    AgentCoreSubtreeStoreConfig,
    CreateAgentCoreMemoryStoresInput,
    _AgentCoreMemoryConnectionConfig,
)


def test_connection_config_runtime_typed_dict_keys_and_hints() -> None:
    """Required/optional keys and boto3 hints remain introspectable on Python 3.10+."""
    assert _AgentCoreMemoryConnectionConfig.__required_keys__ == frozenset({"memory_id", "actor_id", "session_id"})
    assert _AgentCoreMemoryConnectionConfig.__optional_keys__ == frozenset(
        {
            "metadata_provider",
            "max_turns_per_event",
            "extraction_mode",
            "region_name",
            "boto3_session",
            "client",
        }
    )
    assert get_args(get_type_hints(_AgentCoreMemoryConnectionConfig)["boto3_session"]) == (boto3.Session,)


def test_public_factory_typed_dict_keys_are_correct_at_runtime() -> None:
    """Public factory configuration exposes accurate required/optional metadata."""
    assert CreateAgentCoreMemoryStoresInput.__required_keys__ == frozenset(
        {"memory_id", "actor_id", "session_id", "namespaces"}
    )
    assert CreateAgentCoreMemoryStoresInput.__optional_keys__ == frozenset(
        {"extraction", "metadata_provider", "max_turns_per_event", "region_name", "boto3_session", "client"}
    )
    hints = get_type_hints(CreateAgentCoreMemoryStoresInput)
    assert get_args(hints["boto3_session"]) == (boto3.Session,)
    assert "extraction_mode" not in hints


def test_other_public_typed_dict_runtime_metadata() -> None:
    """NotRequired fields are optional without postponed annotations."""
    assert AgentCoreEventSenderConfig.__required_keys__ == frozenset({"client", "memory_id", "actor_id", "session_id"})
    assert AgentCoreEventSenderConfig.__optional_keys__ == frozenset(
        {"metadata_provider", "run_id", "max_turns_per_event", "extraction_mode"}
    )
    assert AgentCoreNamespaceConfig.__required_keys__ == frozenset({"namespace"})
    assert AgentCoreNamespaceConfig.__optional_keys__ == frozenset(
        {"name", "description", "max_search_results", "min_score", "over_fetch_factor", "writable"}
    )
    assert AgentCoreExtractionConfig.__required_keys__ == frozenset()
    assert AgentCoreExtractionConfig.__optional_keys__ == frozenset({"cadence", "filter"})
    exact_hints = get_type_hints(AgentCoreExactNamespaceStoreConfig)
    subtree_hints = get_type_hints(AgentCoreSubtreeStoreConfig)
    assert exact_hints["boto3_session"] == get_type_hints(_AgentCoreMemoryConnectionConfig)["boto3_session"]
    assert subtree_hints["boto3_session"] == get_type_hints(_AgentCoreMemoryConnectionConfig)["boto3_session"]
    assert AgentCoreExactNamespaceStoreConfig.__required_keys__ == frozenset(
        {"memory_id", "actor_id", "session_id", "namespace"}
    )
    assert AgentCoreSubtreeStoreConfig.__required_keys__ == frozenset(
        {"memory_id", "actor_id", "session_id", "namespace_path"}
    )
    get_type_hints(AgentCoreEventSenderConfig)
    get_type_hints(AgentCoreNamespaceConfig)
    get_type_hints(AgentCoreExtractionConfig)
