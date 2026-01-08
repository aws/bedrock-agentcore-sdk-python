"""Convert OTel spans to ADOT format for AgentCore Evaluation API.

Architecture:
    Raw OTel Spans → Parsed Data (domain models) → ADOT Documents

Layers:
    1. Domain Models: Framework-agnostic data structures (adot_models.py)
    2. Extraction: Parse raw OTel spans into structured data (framework-specific)
    3. Transformation: Convert structured data into ADOT format (adot_models.py)
    4. Orchestration: Coordinate the conversion pipeline (framework-specific)

Extensibility:
    To add support for new frameworks (e.g., LangGraph + OpenInference):
    - Reuse adot_models.py (domain models and ADOT builders) as-is
    - Implement new event extractors for the framework's telemetry format
    - Implement new converter that uses framework-specific extractors
    - See strands_converter.py as a reference implementation

Example:
    >>> from bedrock_agentcore.evaluation.span_to_adot_serializer import convert_strands_to_adot
    >>> adot_docs = convert_strands_to_adot(raw_spans)
"""

from .strands_converter import convert_strands_to_adot

__all__ = ["convert_strands_to_adot"]
