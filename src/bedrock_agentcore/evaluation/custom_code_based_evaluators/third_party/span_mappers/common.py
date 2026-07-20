"""Common span mapping types and utilities."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class FieldExtractionError(ValueError):
    """Raised when span field extraction fails (no valid AgentInvocationSpan, unsupported scope, etc.)."""


@dataclass
class SpanMapResult:
    """Extraction result from spans — only what metrics consume."""

    input: Optional[str] = None
    actual_output: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    context: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    expected_output: Optional[str] = None
    tools_called: Optional[List[Dict[str, Any]]] = None
    expected_tools: Optional[List[Dict[str, Any]]] = None
    assertions: Optional[List[str]] = None
