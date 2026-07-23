"""Factory helpers for multi-namespace AgentCore memory topologies."""

from __future__ import annotations

from collections.abc import Sequence

import boto3
from strands.memory import ExtractionConfig, MemoryStore

from .store import AgentCoreMemoryStore, _create_data_plane_client
from .types import (
    AgentCoreDataPlaneClient,
    AgentCoreExtractionConfig,
    AgentCoreNamespaceConfig,
    MetadataProvider,
)


def assert_writable_topology(stores: Sequence[MemoryStore], expect_extraction: bool = False) -> None:
    """Require at most one writable store, and optionally require one writer.

    Args:
        stores: AgentCore stores sharing one identity stream.
        expect_extraction: Whether a writer is required.

    Raises:
        ValueError: If the topology would duplicate writes or cannot extract.
    """
    # create_event writes to the identity stream rather than a namespace. Multiple
    # writers would therefore duplicate the same conversation events.
    writers = [store for store in stores if store.writable]
    if len(writers) > 1:
        names = ", ".join(f'"{store.name}"' for store in writers)
        raise ValueError(
            f"AgentCore memory: at most one store may be writable, but {len(writers)} are ({names}). "
            "create_event is namespace-free, so multiple writable stores would write duplicate events to the "
            "same (memory_id, actor_id, session_id) stream. Mark exactly one namespace writable."
        )
    if expect_extraction and not writers:
        raise ValueError(
            "AgentCore memory: extraction is enabled but no store is writable. Mark one namespace writable "
            "(or omit extraction for recall-only)."
        )


def create_agentcore_memory_stores(
    *,
    memory_id: str,
    actor_id: str,
    session_id: str,
    namespaces: list[AgentCoreNamespaceConfig],
    extraction: bool | AgentCoreExtractionConfig | None = None,
    metadata_provider: MetadataProvider | None = None,
    max_turns_per_event: int | None = None,
    region_name: str | None = None,
    boto3_session: boto3.Session | None = None,
    client: AgentCoreDataPlaneClient | None = None,
) -> list[MemoryStore]:
    """Build one store per exact namespace with one shared boto3 client.

    Args:
        memory_id: AgentCore Memory resource identifier.
        actor_id: Actor identifier.
        session_id: Session identifier.
        namespaces: Per-namespace store configuration dictionaries.
        extraction: Recall-only switch, or custom cadence/filter configuration.
        metadata_provider: Optional per-message event metadata callback.
        max_turns_per_event: Maximum turns packed into one event.
        region_name: Region used when constructing the shared client.
        boto3_session: Session used when constructing the shared client.
        client: Preconstructed shared data-plane client.

    Returns:
        One store per namespace.

    Raises:
        ValueError: If namespace or writer configuration is invalid.
    """
    if not isinstance(namespaces, list) or not namespaces:
        raise ValueError("create_agentcore_memory_stores: at least one namespace is required")
    for index, namespace_config in enumerate(namespaces):
        namespace = namespace_config.get("namespace") if isinstance(namespace_config, dict) else None
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError(
                f"create_agentcore_memory_stores: namespaces[{index}].namespace must be a non-empty string"
            )
    if max_turns_per_event is not None and (type(max_turns_per_event) is not int or max_turns_per_event < 1):
        raise ValueError(
            f"create_agentcore_memory_stores: max_turns_per_event must be a positive integer, got {max_turns_per_event}"
        )

    # ``True`` leaves cadence to MemoryManager; only the object form builds a
    # custom Strands extraction configuration.
    write_enabled = extraction not in (None, False)
    extraction_config: bool | ExtractionConfig | None
    if not write_enabled:
        extraction_config = None
    elif isinstance(extraction, dict) and ("cadence" in extraction or "filter" in extraction):
        extraction_config = ExtractionConfig()
        cadence = extraction.get("cadence")
        message_filter = extraction.get("filter")
        if cadence is not None:
            extraction_config["trigger"] = cadence
        if message_filter is not None:
            extraction_config["filter"] = message_filter
    else:
        extraction_config = True

    # Build one connection and reuse it for every namespace in this identity set.
    shared_client = (
        client
        if client is not None
        else _create_data_plane_client(region_name=region_name, boto3_session=boto3_session)
    )
    # The default writer skips explicit opt-outs. Keep multiple explicit writers
    # intact so the topology check fails loudly instead of silently choosing one.
    any_flagged = any(config.get("writable") is True for config in namespaces)
    default_writer_index = -1
    if write_enabled and not any_flagged:
        default_writer_index = next(
            (index for index, config in enumerate(namespaces) if config.get("writable") is not False), -1
        )
    if write_enabled and not any_flagged and default_writer_index == -1:
        raise ValueError(
            "create_agentcore_memory_stores: extraction is enabled but every namespace is marked writable: false; "
            "leave one namespace un-opted-out (or set writable: true on the intended writer)."
        )

    stores: list[AgentCoreMemoryStore] = []
    for index, namespace_config in enumerate(namespaces):
        is_writer = namespace_config.get("writable") is True or index == default_writer_index
        stores.append(
            AgentCoreMemoryStore(
                memory_id=memory_id,
                actor_id=actor_id,
                session_id=session_id,
                namespace=str(namespace_config["namespace"]),
                name=namespace_config.get("name"),
                description=namespace_config.get("description"),
                max_search_results=namespace_config.get("max_search_results"),
                min_score=namespace_config.get("min_score"),
                over_fetch_factor=namespace_config.get("over_fetch_factor", 4),
                writable=is_writer,
                extraction=extraction_config if is_writer else None,
                metadata_provider=metadata_provider,
                max_turns_per_event=max_turns_per_event,
                client=shared_client,
            )
        )
    # MemoryManager validates store-name uniqueness, so only write topology is checked here.
    assert_writable_topology(stores, write_enabled)
    return list(stores)
