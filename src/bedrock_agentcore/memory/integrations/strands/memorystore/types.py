"""Public types and namespace helpers for the Strands AgentCore memory store."""

import re
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

import boto3
from strands.memory import ExtractionConfig, ExtractionTrigger, MemoryMessageFilter
from strands.types.content import Message
from typing_extensions import Never, NotRequired, TypedDict

ExtractionMode = Literal["SKIP"]
"""Long-term extraction control accepted by AgentCore ``create_event``."""

# Defaults apply only when neither call-level nor store-level configuration overrides them.
DEFAULT_REGION = "us-west-2"
DEFAULT_MAX_SEARCH_RESULTS = 5
DEFAULT_OVERFETCH_FACTOR = 4
# Bound score-filter over-fetching even when callers configure a large multiplier.
MAX_TOPK = 100
# Packing turns lets extraction cadence control API volume instead of writing one event per message.
DEFAULT_MAX_TURNS_PER_EVENT = 50
# Store-supplied record fields use this prefix to avoid colliding with user metadata.
RESERVED_METADATA_PREFIX = "_"


class AgentCoreDataPlaneClient(Protocol):
    """Structural type for the boto3 AgentCore data-plane client."""

    def create_event(self, **kwargs: Any) -> dict[str, Any]:
        """Create an AgentCore memory event."""
        ...

    def retrieve_memory_records(self, **kwargs: Any) -> dict[str, Any]:
        """Retrieve AgentCore long-term memory records."""
        ...


MetadataValue = str | int | float | bool
"""Scalar metadata value accepted by AgentCore event metadata."""

MetadataProvider = Callable[[Message], Mapping[str, MetadataValue]]
"""Derive event metadata from one message.

Strings pass through; other finite scalars use :func:`json.dumps` formatting. AgentCore
accepts only letters, digits, whitespace, and ``._:/=+@-`` in the resulting value.
"""


class _AgentCoreMemoryConnectionConfig(TypedDict):
    """Connection and write identity shared internally by AgentCore memory stores."""

    memory_id: str
    actor_id: str
    session_id: str
    metadata_provider: NotRequired[MetadataProvider]
    max_turns_per_event: NotRequired[int]
    extraction_mode: NotRequired[ExtractionMode]
    region_name: NotRequired[str]
    boto3_session: NotRequired[boto3.Session]
    client: NotRequired[AgentCoreDataPlaneClient]


class _AgentCoreMemoryStoreOptions(_AgentCoreMemoryConnectionConfig, total=False):
    """Fields shared by exact-namespace and subtree store configurations."""

    name: str
    description: str
    max_search_results: int
    writable: bool
    extraction: bool | ExtractionConfig
    min_score: float
    over_fetch_factor: float


class AgentCoreExactNamespaceStoreConfig(_AgentCoreMemoryStoreOptions):
    """Read one exact namespace prefix after substituting actor/session placeholders."""

    namespace: str
    namespace_path: NotRequired[Never]


class AgentCoreSubtreeStoreConfig(_AgentCoreMemoryStoreOptions):
    """Read a parent namespace path and all of its child namespaces."""

    namespace_path: str
    namespace: NotRequired[Never]


AgentCoreMemoryStoreConfig = AgentCoreExactNamespaceStoreConfig | AgentCoreSubtreeStoreConfig
"""One flat store config with identity and exactly one read-target shape.

``writable`` defaults to false for recall-only behavior; a name defaults to a slug of
the namespace template.
"""


class AgentCoreEventSenderConfig(TypedDict):
    """Configuration for :class:`AgentCoreEventSender`."""

    client: AgentCoreDataPlaneClient
    memory_id: str
    actor_id: str
    session_id: str
    metadata_provider: NotRequired[MetadataProvider]
    run_id: NotRequired[str]
    max_turns_per_event: NotRequired[int]
    extraction_mode: NotRequired[ExtractionMode]


class AgentCoreNamespaceConfig(TypedDict):
    """Per-namespace read configuration used by the store factory."""

    namespace: str
    name: NotRequired[str]
    description: NotRequired[str]
    max_search_results: NotRequired[int]
    min_score: NotRequired[float]
    over_fetch_factor: NotRequired[float]
    writable: NotRequired[bool]


class AgentCoreExtractionConfig(TypedDict, total=False):
    """Writable-store extraction cadence and message filtering."""

    cadence: ExtractionTrigger | list[ExtractionTrigger]
    filter: MemoryMessageFilter


class CreateAgentCoreMemoryStoresInput(TypedDict):
    """Configuration accepted by :func:`create_agentcore_memory_stores`."""

    memory_id: str
    actor_id: str
    session_id: str
    namespaces: list[AgentCoreNamespaceConfig]
    extraction: NotRequired[bool | AgentCoreExtractionConfig]
    metadata_provider: NotRequired[MetadataProvider]
    max_turns_per_event: NotRequired[int]
    region_name: NotRequired[str]
    boto3_session: NotRequired[boto3.Session]
    client: NotRequired[AgentCoreDataPlaneClient]


_UNRESOLVED_PLACEHOLDER = re.compile(r"\{[^{}]*\}")
_ANY_BRACE = re.compile(r"[{}]")
_NAMESPACE_PLACEHOLDER = re.compile(r"\{[^{}]*\}")
_NON_ALPHANUMERIC = re.compile(r"[^a-zA-Z0-9]+")


def resolve_namespace(template: str, actor_id: str, session_id: str) -> str:
    """Resolve ``{actorId}`` and then ``{sessionId}`` in a namespace template.

    AgentCore resolves strategy placeholders when extracting records, but retrieval
    does not resolve placeholders and rejects braces. The store therefore substitutes
    its two known identity placeholders before reading and rejects anything left over.

    Args:
        template: Namespace template to resolve.
        actor_id: Actor identifier substituted for ``{actorId}``.
        session_id: Session identifier substituted for ``{sessionId}``.

    Returns:
        The resolved namespace.
    """
    return template.replace("{actorId}", actor_id).replace("{sessionId}", session_id)


def assert_resolved_namespace(resolved: str, template: str) -> None:
    """Reject unresolved placeholders and unmatched braces.

    Args:
        resolved: Namespace after supported substitutions.
        template: Original namespace template, used in the error message.

    Raises:
        ValueError: If a token or brace remains.
    """
    token = _UNRESOLVED_PLACEHOLDER.search(resolved)
    brace = _ANY_BRACE.search(resolved)
    offending = token.group(0) if token else brace.group(0) if brace else None
    if offending is not None:
        raise ValueError(
            f'AgentCoreMemoryStore: namespace "{template}" still contains "{offending}" after substitution. '
            "Only {actorId} and {sessionId} are resolved client-side; the AgentCore retrieve path does not "
            'resolve placeholders and rejects "{"/"}". Provide a namespace whose only placeholders are '
            "{actorId}/{sessionId} (and no stray braces), or pre-substitute the others (for example, a concrete "
            "strategy id) before constructing the store."
        )


def assert_non_empty(value: object, field: str) -> str:
    """Return a non-empty string or raise a field-specific error.

    Args:
        value: Value to validate.
        field: Public field name used in the error.

    Returns:
        The validated string.

    Raises:
        ValueError: If ``value`` is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"AgentCoreMemoryStore: {field} must be a non-empty string")
    return value


def slugify_namespace(namespace: str) -> str:
    """Derive a stable store name from a namespace template.

    Args:
        namespace: Namespace template.

    Returns:
        A hyphenated slug, or ``agentcore-memory`` when no usable text remains.
    """
    without_placeholders = _NAMESPACE_PLACEHOLDER.sub("", namespace)
    slug = _NON_ALPHANUMERIC.sub("-", without_placeholders).strip("-")
    return slug or "agentcore-memory"
