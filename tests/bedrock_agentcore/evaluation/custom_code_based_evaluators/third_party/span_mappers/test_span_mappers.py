"""Tests for span mappers using strands-evals integration."""

import pytest

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import ReferenceInput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers import (
    SpanMapResult,
    map_spans,
)


def _make_strands_cloudwatch_spans():
    """Build Strands CloudWatch format spans (merged: span metadata + body)."""
    return [
        {
            "traceId": "trace1",
            "spanId": "span1",
            "scope": {"name": "strands.telemetry.tracer"},
            "name": "invoke_agent",
            "kind": "INTERNAL",
            "startTimeUnixNano": 1000000000,
            "endTimeUnixNano": 2000000000,
            "attributes": {"gen_ai.operation.name": "invoke_agent"},
            "status": {"code": "UNSET"},
            "span_events": [
                {
                    "event_name": "strands.telemetry.tracer",
                    "body": {
                        "input": {
                            "messages": [
                                {"role": "system", "content": "You are helpful."},
                                {"role": "user", "content": {"content": '[{"text": "What is AI?"}]'}},
                            ]
                        },
                        "output": {
                            "messages": [
                                {"role": "assistant", "content": {"message": "AI is artificial intelligence."}}
                            ]
                        },
                    },
                }
            ],
        }
    ]


class TestMapSpans:
    def test_strands_cloudwatch_extraction(self):
        spans = _make_strands_cloudwatch_spans()
        result = map_spans(spans)

        assert isinstance(result, SpanMapResult)
        assert result.input is not None
        assert result.actual_output is not None

    def test_raises_on_empty_spans(self):
        with pytest.raises(ValueError):
            map_spans([])

    def test_raises_on_unsupported_scope(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "unknown.scope"},
                "attributes": {},
            }
        ]
        with pytest.raises(ValueError):
            map_spans(spans)

    def test_reference_inputs_expected_output(self):
        spans = _make_strands_cloudwatch_spans()
        ref = ReferenceInput(
            context={},
            expected_response={"text": "AI stands for artificial intelligence."},
        )
        result = map_spans(spans, reference_inputs=[ref])

        assert result.expected_output == "AI stands for artificial intelligence."

    def test_reference_inputs_expected_tools(self):
        spans = _make_strands_cloudwatch_spans()
        ref = ReferenceInput(
            context={},
            expected_trajectory={"toolNames": ["search", "calculate"]},
        )
        result = map_spans(spans, reference_inputs=[ref])

        assert result.expected_tools == [{"name": "search"}, {"name": "calculate"}]

    def test_reference_inputs_assertions(self):
        spans = _make_strands_cloudwatch_spans()
        ref = ReferenceInput(
            context={},
            assertions=[{"text": "Fact 1"}, {"text": "Fact 2"}],
        )
        result = map_spans(spans, reference_inputs=[ref])

        assert result.assertions == ["Fact 1", "Fact 2"]

    def test_span_map_result_fields(self):
        result = SpanMapResult(
            input="hello",
            actual_output="world",
            retrieval_context=["ctx1"],
            tools_called=[{"name": "tool1", "input_parameters": {"a": 1}, "output": "result"}],
        )
        assert result.input == "hello"
        assert result.actual_output == "world"
        assert result.retrieval_context == ["ctx1"]
        assert result.tools_called == [{"name": "tool1", "input_parameters": {"a": 1}, "output": "result"}]
        assert result.expected_output is None
        assert result.system_prompt is None
