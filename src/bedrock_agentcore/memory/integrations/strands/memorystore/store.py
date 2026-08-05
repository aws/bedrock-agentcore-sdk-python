"""AgentCore Memory implementation of the native Strands ``MemoryStore`` contract."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import boto3
from strands.memory import AddMessagesContext, ExtractionConfig, MemoryEntry, MemoryStore, SearchOptions
from strands.types.content import Message

from bedrock_agentcore.memory.client import MemoryClient

from .sender import AgentCoreEventSender
from .types import (
    DEFAULT_MAX_SEARCH_RESULTS,
    DEFAULT_MAX_TURNS_PER_EVENT,
    DEFAULT_OVERFETCH_FACTOR,
    DEFAULT_REGION,
    MAX_TOPK,
    RESERVED_METADATA_PREFIX,
    ExtractionMode,
    MetadataProvider,
    assert_non_empty,
    assert_resolved_namespace,
    resolve_namespace,
    slugify_namespace,
)

if TYPE_CHECKING:
    from typing_extensions import Never

    _MemoryStoreBase = MemoryStore
else:
    _MemoryStoreBase = object


logger = logging.getLogger(__name__)


def _create_memory_client(
    *, region_name: str | None = None, boto3_session: boto3.Session | None = None
) -> MemoryClient:
    """Build the default :class:`~bedrock_agentcore.memory.MemoryClient` for this integration.

    ``MemoryClient`` re-checks the session by truthiness, so a deliberately falsey
    session-like object would be replaced by a default session there.

    Args:
        region_name: Explicit region, preferred over the session and environment.
        boto3_session: Session used to construct the client.

    Returns:
        The client whose AgentCore data plane this integration calls.
    """
    session = boto3_session if boto3_session is not None else boto3.Session()
    region = region_name or session.region_name or os.environ.get("AWS_REGION") or DEFAULT_REGION
    return MemoryClient(region_name=region, boto3_session=session, integration_source="strands")


class AgentCoreMemoryStore(_MemoryStoreBase):
    """Expose AgentCore long-term memory through Strands' native memory interface.

    Identity and one exact or subtree read target are fixed at construction. Only a
    writable store carries ``add_messages`` and extraction configuration. The flat
    ``add`` method is intentionally absent because it would discard conversation roles.
    """

    # Strands models optional MemoryStore methods on its Protocol for type checking,
    # while runtime capability detection requires absent methods to stay absent.
    if TYPE_CHECKING:
        add: Never
        initialize: Never
        get_tools: Never

    def __init__(
        self,
        *,
        memory_id: str,
        actor_id: str,
        session_id: str,
        namespace: str | None = None,
        namespace_path: str | None = None,
        name: str | None = None,
        description: str | None = None,
        max_search_results: int | None = None,
        writable: bool = False,
        extraction: bool | ExtractionConfig | None = None,
        min_score: float | None = None,
        over_fetch_factor: float = DEFAULT_OVERFETCH_FACTOR,
        metadata_provider: MetadataProvider | None = None,
        max_turns_per_event: int | None = None,
        extraction_mode: ExtractionMode | None = None,
        region_name: str | None = None,
        boto3_session: boto3.Session | None = None,
        client: MemoryClient | None = None,
    ) -> None:
        """Initialize one exact-namespace or subtree store.

        Args:
            memory_id: AgentCore Memory resource identifier.
            actor_id: Actor identifier.
            session_id: Session identifier.
            namespace: Exact namespace template to retrieve.
            namespace_path: Parent namespace template for subtree retrieval.
            name: Strands store name; defaults to a namespace slug.
            description: Optional human-readable store description.
            max_search_results: Default result count for this store. AgentCore returns at most
                100 records per search, so larger values are clamped to that limit.
            writable: Whether ``add_messages`` may write conversation events.
            extraction: Strands automatic-extraction configuration.
            min_score: Optional client-side relevance threshold.
            over_fetch_factor: Retrieval multiplier used when ``min_score`` is set.
            metadata_provider: Optional per-message event metadata callback.
            max_turns_per_event: Maximum turns packed into one event.
            extraction_mode: Optional AgentCore extraction mode, including ``SKIP``.
            region_name: Region used when constructing the default ``MemoryClient``.
            boto3_session: Session used when constructing the default ``MemoryClient``.
            client: Preconstructed ``MemoryClient``; one is built when omitted.

        Raises:
            ValueError: If identity, read target, or numeric options are invalid.
        """
        if (namespace is None) == (namespace_path is None):
            raise ValueError("AgentCoreMemoryStore: exactly one of namespace or namespace_path is required")
        template = namespace_path if namespace_path is not None else namespace
        read_field = "namespace_path" if namespace_path is not None else "namespace"
        assert template is not None

        self._memory_id = assert_non_empty(memory_id, "memory_id")
        self._actor_id = assert_non_empty(actor_id, "actor_id")
        self._session_id = assert_non_empty(session_id, "session_id")
        assert_non_empty(template, read_field)
        self._resolved_namespace = resolve_namespace(template, self._actor_id, self._session_id)
        assert_resolved_namespace(self._resolved_namespace, template)
        self._read_mode = "subtree" if namespace_path is not None else "exact"

        explicit_name = name.strip() if name is not None else ""
        self.name = explicit_name or slugify_namespace(template)
        self.description = description
        if max_search_results is not None and (type(max_search_results) is not int or max_search_results < 1):
            raise ValueError(
                f"AgentCoreMemoryStore: max_search_results must be a positive integer, got {max_search_results}"
            )
        self.max_search_results = max_search_results
        self._warned_result_cap_clamped = False
        # Recall-safe default: a store never writes unless explicitly enabled.
        self.writable = writable
        self.extraction: bool | ExtractionConfig | None = extraction if writable else None
        if min_score is not None and (
            isinstance(min_score, bool)
            or not isinstance(min_score, (int, float))
            or not math.isfinite(min_score)
            or min_score < 0
            or min_score > 1
        ):
            raise ValueError(
                f"AgentCoreMemoryStore: min_score must be a finite number between 0 and 1, got {min_score}"
            )
        if (
            isinstance(over_fetch_factor, bool)
            or not isinstance(over_fetch_factor, (int, float))
            or not math.isfinite(over_fetch_factor)
            or over_fetch_factor < 1
        ):
            raise ValueError(f"AgentCoreMemoryStore: over_fetch_factor must be a number >= 1, got {over_fetch_factor}")
        self._min_score = min_score
        self._over_fetch_factor = over_fetch_factor
        self._memory_client = (
            client
            if client is not None
            else _create_memory_client(region_name=region_name, boto3_session=boto3_session)
        )
        # Call the data plane the client already vends: MemoryClient's own create_event is a
        # higher-level (text, role) API with no clientToken, and its retrieve_memories swallows
        # ClientError, while MemoryManager expects retrieval failures to surface.
        self._client = self._memory_client.gmdp_client
        self._sender = (
            AgentCoreEventSender(
                client=self._memory_client,
                memory_id=self._memory_id,
                actor_id=self._actor_id,
                session_id=self._session_id,
                metadata_provider=metadata_provider,
                max_turns_per_event=(
                    max_turns_per_event if max_turns_per_event is not None else DEFAULT_MAX_TURNS_PER_EVENT
                ),
                extraction_mode=extraction_mode,
            )
            if writable
            else None
        )
        if not writable and extraction not in (None, False):
            logger.warning(
                '[agentcore-memory] store "%s" has an extraction config but writable is false; extraction will not run',
                self.name,
            )

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        """Retrieve relevant AgentCore memory records.

        Args:
            query: Semantic search query.
            options: Optional Strands per-call result cap.

        Returns:
            Records mapped to Strands ``MemoryEntry`` objects.

        Raises:
            ValueError: If the effective result cap is invalid.
        """
        want = (
            options.get("max_search_results")
            if options is not None and "max_search_results" in options
            else self.max_search_results or DEFAULT_MAX_SEARCH_RESULTS
        )
        if type(want) is not int or want < 1:
            raise ValueError(f"AgentCoreMemoryStore.search: max_search_results must be a positive integer, got {want}")
        top_k = want
        if self._min_score is not None:
            # Clamp before converting to int: unlike JavaScript's Math.ceil, Python's
            # math.ceil cannot convert an overflowed infinite product.
            over_fetch = want * self._over_fetch_factor
            top_k = MAX_TOPK if over_fetch >= MAX_TOPK else math.ceil(over_fetch)
        # AgentCore rejects topK above MAX_TOPK, so clamp on every path rather than only the
        # score-filter one: a plausible max_search_results of 101 otherwise reached the wire
        # verbatim and failed as an opaque service ValidationException, while the same value
        # with min_score set succeeded. Warn once so the reduced cap is not silent.
        if top_k > MAX_TOPK:
            top_k = MAX_TOPK
        if want > MAX_TOPK and not self._warned_result_cap_clamped:
            self._warned_result_cap_clamped = True
            logger.warning(
                "AgentCoreMemoryStore %s: max_search_results=%d exceeds AgentCore's topK limit of %d; "
                "searches return at most %d records per request.",
                self.name,
                want,
                MAX_TOPK,
                MAX_TOPK,
            )
        kwargs: dict[str, Any] = {
            "memoryId": self._memory_id,
            "searchCriteria": {"searchQuery": query, "topK": top_k},
            "namespacePath" if self._read_mode == "subtree" else "namespace": self._resolved_namespace,
        }
        # Let retrieval errors propagate: MemoryManager isolates store failures and
        # applies its normal partial-failure behavior.
        response = await asyncio.to_thread(self._client.retrieve_memory_records, **kwargs)
        records = response.get("memoryRecordSummaries") or []
        filtered = [
            record
            for record in records
            if self._min_score is None or float(record.get("score", 0) or 0) >= self._min_score
        ][:want]
        return [self._to_entry(record) for record in filtered]

    async def add_messages(self, messages: list[Message], context: AddMessagesContext | None = None) -> None:
        """Write role-preserving conversation messages to AgentCore.

        Args:
            messages: Strands messages to ingest.
            context: Optional manager-provided sequence numbers.

        Raises:
            ValueError: If this store is recall-only.
        """
        if self._sender is None:
            raise ValueError(f'AgentCoreMemoryStore "{self.name}" is not writable; add_messages is unavailable')
        await self._sender.send_batch(messages, context.sequence_numbers if context else None)

    @staticmethod
    def _to_entry(record: dict[str, Any]) -> MemoryEntry:
        content = record.get("content")
        text = content.get("text", "") if isinstance(content, dict) and isinstance(content.get("text"), str) else ""
        # Reserved keys prevent store-supplied record fields colliding with user metadata.
        prefix = RESERVED_METADATA_PREFIX
        metadata: dict[str, Any] = {}
        if "memoryRecordId" in record:
            metadata[f"{prefix}id"] = record["memoryRecordId"]
        if "score" in record:
            metadata[f"{prefix}score"] = record["score"]
        if "namespaces" in record:
            metadata[f"{prefix}namespaces"] = record["namespaces"]
        if "createdAt" in record:
            created_at = record["createdAt"]
            metadata[f"{prefix}createdAt"] = _format_created_at(created_at)
        return MemoryEntry(content=text, metadata=metadata)


def _format_created_at(value: object) -> str:
    if not isinstance(value, datetime):
        return str(value)
    timestamp = value
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    utc = timestamp.astimezone(timezone.utc)
    return utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")
