"""Tests for span parsers."""

import pytest

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import ReferenceInput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_parsers import (
    SpanParseResult,
    parse_spans,
)


def _make_agent_span(input_messages=None, output_messages=None, span_id="span1"):
    """Build an agent-level span with span_events."""
    span_events = []
    body = {}
    if input_messages is not None:
        body["input"] = {"messages": input_messages}
    if output_messages is not None:
        body["output"] = {"messages": output_messages}
    if body:
        span_events.append({"body": body})

    return {
        "traceId": "abc123",
        "spanId": span_id,
        "attributes": {"gen_ai.operation.name": "invoke_agent"},
        "span_events": span_events,
    }


class TestParseSpansSuccess:
    def test_extracts_input_and_output(self):
        spans = [
            _make_agent_span(
                input_messages=[{"role": "user", "content": "What is AI?"}],
                output_messages=[{"role": "assistant", "content": "Artificial intelligence."}],
            )
        ]

        result = parse_spans(spans)

        assert result.input == "What is AI?"
        assert result.actual_output == "Artificial intelligence."

    def test_extracts_tool_messages_as_retrieval_context(self):
        spans = [
            _make_agent_span(
                input_messages=[{"role": "user", "content": "query"}],
                output_messages=[
                    {"role": "tool", "content": "doc chunk 1"},
                    {"role": "tool", "content": "doc chunk 2"},
                    {"role": "assistant", "content": "answer"},
                ],
            )
        ]

        result = parse_spans(spans)

        assert result.retrieval_context == ["doc chunk 1", "doc chunk 2"]
        assert result.context == ["doc chunk 1", "doc chunk 2"]
        assert result.actual_output == "answer"

    def test_uses_first_user_message_as_input(self):
        spans = [
            _make_agent_span(
                input_messages=[
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": "second"},
                ],
                output_messages=[{"role": "assistant", "content": "reply"}],
            )
        ]

        result = parse_spans(spans)

        assert result.input == "first"

    def test_uses_last_assistant_message_as_output(self):
        spans = [
            _make_agent_span(
                input_messages=[{"role": "user", "content": "q"}],
                output_messages=[
                    {"role": "assistant", "content": "first reply"},
                    {"role": "assistant", "content": "final reply"},
                ],
            )
        ]

        result = parse_spans(spans)

        assert result.actual_output == "final reply"

    def test_expected_output_from_reference_inputs(self):
        spans = [
            _make_agent_span(
                input_messages=[{"role": "user", "content": "q"}],
                output_messages=[{"role": "assistant", "content": "a"}],
            )
        ]
        refs = [ReferenceInput(expectedResponse={"text": "expected answer"})]

        result = parse_spans(spans, reference_inputs=refs)

        assert result.expected_output == "expected answer"

    def test_nested_content_dict(self):
        spans = [
            _make_agent_span(
                input_messages=[{"role": "user", "content": {"content": "nested"}}],
                output_messages=[{"role": "assistant", "content": {"content": "nested out"}}],
            )
        ]

        result = parse_spans(spans)

        assert result.input == "nested"
        assert result.actual_output == "nested out"

    def test_body_as_json_string(self):
        import json

        body = {
            "input": {"messages": [{"role": "user", "content": "hello"}]},
            "output": {"messages": [{"role": "assistant", "content": "hi"}]},
        }
        span = {
            "traceId": "t1",
            "spanId": "s1",
            "attributes": {"gen_ai.operation.name": "invoke_agent"},
            "span_events": [{"body": json.dumps(body)}],
        }

        result = parse_spans([span])

        assert result.input == "hello"
        assert result.actual_output == "hi"

    def test_to_dict_omits_none(self):
        result = SpanParseResult(input="q", actual_output="a")
        d = result.to_dict()

        assert d == {"input": "q", "actual_output": "a"}
        assert "retrieval_context" not in d


class TestParseSpansErrors:
    def test_no_agent_spans_raises(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"gen_ai.operation.name": "other_op"},
                "span_events": [],
            }
        ]

        with pytest.raises(ValueError, match="Could not extract evaluation fields"):
            parse_spans(spans)

    def test_empty_spans_raises(self):
        with pytest.raises(ValueError, match="Could not extract evaluation fields"):
            parse_spans([])

    def test_agent_span_without_events_raises(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
                "span_events": [],
            }
        ]

        with pytest.raises(ValueError, match="Could not extract evaluation fields"):
            parse_spans(spans)

    def test_non_agent_spans_ignored(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"gen_ai.operation.name": "chat"},
                "span_events": [
                    {
                        "body": {
                            "input": {"messages": [{"role": "user", "content": "q"}]},
                            "output": {"messages": [{"role": "assistant", "content": "a"}]},
                        }
                    }
                ],
            }
        ]

        with pytest.raises(ValueError, match="Could not extract evaluation fields"):
            parse_spans(spans)
