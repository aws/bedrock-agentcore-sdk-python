"""Abstract base class for framework-specific span mappers."""

import abc
from typing import Any, Dict, List, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.common import (
    SpanMapResult,
)


class BaseSpanMapper(abc.ABC):
    """Base class for framework-specific span mappers.

    Subclasses implement scope_name and map() to extract evaluation fields
    from spans produced by a specific OTel instrumentation scope.
    """

    @property
    @abc.abstractmethod
    def scope_name(self) -> str:
        """The primary OTel scope this mapper handles, e.g. 'strands.telemetry.tracer'."""

    @property
    def scope_names(self) -> List[str]:
        """All OTel scope names this mapper handles. Override to support multiple scopes."""
        return [self.scope_name]

    @abc.abstractmethod
    def map(self, session_spans: List[Dict[str, Any]]) -> Optional[SpanMapResult]:
        """Extract evaluation fields from spans. Return None if nothing found."""

    def _filter_scope_spans(self, session_spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter to only spans matching this mapper's scope_names."""
        names = set(self.scope_names)
        return [s for s in session_spans if s.get("scope", {}).get("name") in names]
