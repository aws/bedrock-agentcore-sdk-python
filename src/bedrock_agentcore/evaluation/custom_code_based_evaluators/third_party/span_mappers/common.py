"""Common span mapping types and utilities."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, omitting None values."""
        result: Dict[str, Any] = {}
        if self.input is not None:
            result["input"] = self.input
        if self.actual_output is not None:
            result["actual_output"] = self.actual_output
        if self.retrieval_context is not None:
            result["retrieval_context"] = self.retrieval_context
        if self.context is not None:
            result["context"] = self.context
        if self.system_prompt is not None:
            result["system_prompt"] = self.system_prompt
        if self.expected_output is not None:
            result["expected_output"] = self.expected_output
        if self.tools_called is not None:
            result["tools_called"] = self.tools_called
        if self.expected_tools is not None:
            result["expected_tools"] = self.expected_tools
        if self.assertions is not None:
            result["assertions"] = self.assertions
        return result


def _try_parse_json(val: str) -> Any:
    """Try to parse a JSON string, return None on failure."""
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None
