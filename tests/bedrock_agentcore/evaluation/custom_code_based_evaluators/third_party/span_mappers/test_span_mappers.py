"""Tests for span mappers."""

import json

import pytest

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import ReferenceInput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers import (
    SpanMapResult,
    map_spans,
)


def _make_strands_agent_span(span_events, span_id="span1", trace_id="abc123"):
    """Build a Strands invoke_agent span with given span_events."""
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": "parent1",
        "scope": {"name": "strands.telemetry.tracer", "version": ""},
        "attributes": {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "test_agent",
            "gen_ai.system": "strands-agents",
        },
        "span_events": span_events,
    }


def _make_span_event(input_messages=None, output_messages=None):
    """Build a span_event body matching real Strands format."""
    body = {}
    if input_messages is not None:
        body["input"] = {"messages": input_messages}
    if output_messages is not None:
        body["output"] = {"messages": output_messages}
    return {"event_name": "strands.telemetry.tracer", "body": body}


def _make_non_strands_span(operation_name="chat"):
    """Build a span from a different scope (e.g., botocore)."""
    return {
        "traceId": "abc123",
        "spanId": "other1",
        "scope": {"name": "opentelemetry.instrumentation.botocore.bedrock-runtime", "version": "0.54b1"},
        "attributes": {"gen_ai.operation.name": operation_name},
        "span_events": [],
    }


class TestMapSpansSuccess:
    def test_extracts_input_and_output_plain_strings(self):
        spans = [
            _make_strands_agent_span([
                _make_span_event(
                    input_messages=[{"role": "user", "content": "What is AI?"}],
                    output_messages=[{"role": "assistant", "content": "Artificial intelligence."}],
                )
            ])
        ]

        result = map_spans(spans)

        assert result.input == "What is AI?"
        assert result.actual_output == "Artificial intelligence."

    def test_extracts_from_real_strands_format(self):
        """Test with content format matching real parser_output.json."""
        spans = [
            _make_strands_agent_span([
                _make_span_event(
                    input_messages=[
                        {"role": "system", "content": "You are a travel assistant."},
                        {"role": "user", "content": {"content": '[{"text": "What is the weather in Tokyo?"}]'}},
                    ],
                    output_messages=[
                        {
                            "role": "assistant",
                            "content": {"message": "The weather in Tokyo is sunny.", "finish_reason": "end_turn"},
                        }
                    ],
                )
            ])
        ]

        result = map_spans(spans)

        assert result.input == "What is the weather in Tokyo?"
        assert result.actual_output == "The weather in Tokyo is sunny."

    def test_multi_turn_uses_first_user_last_assistant(self):
        """Multiple span_events (one per turn) — first user input, last assistant output."""
        spans = [
            _make_strands_agent_span([
                _make_span_event(
                    input_messages=[{"role": "user", "content": {"content": '[{"text": "Plan a trip to Japan"}]'}}],
                    output_messages=[{"role": "assistant", "content": {"message": "Sure! Let me check flights."}}],
                ),
                _make_span_event(
                    input_messages=[{"role": "user", "content": {"content": '[{"text": "What about hotels?"}]'}}],
                    output_messages=[{"role": "assistant", "content": {"message": "Here are some hotel options."}}],
                ),
                _make_span_event(
                    input_messages=[{"role": "user", "content": {"content": '[{"text": "Thanks!"}]'}}],
                    output_messages=[{"role": "assistant", "content": {"message": "You are welcome! Have a great trip."}}],
                ),
            ])
        ]

        result = map_spans(spans)

        assert result.input == "Plan a trip to Japan"
        assert result.actual_output == "You are welcome! Have a great trip."

    def test_extracts_tool_messages_as_retrieval_context(self):
        spans = [
            _make_strands_agent_span([
                _make_span_event(
                    input_messages=[{"role": "user", "content": "query"}],
                    output_messages=[
                        {"role": "tool", "content": "doc chunk 1"},
                        {"role": "tool", "content": "doc chunk 2"},
                        {"role": "assistant", "content": "answer"},
                    ],
                )
            ])
        ]

        result = map_spans(spans)

        assert result.retrieval_context == ["doc chunk 1", "doc chunk 2"]
        assert result.context == ["doc chunk 1", "doc chunk 2"]
        assert result.actual_output == "answer"

    def test_ignores_non_strands_spans(self):
        """Only processes spans with strands.telemetry.tracer scope."""
        spans = [
            _make_non_strands_span(),
            _make_strands_agent_span([
                _make_span_event(
                    input_messages=[{"role": "user", "content": "hello"}],
                    output_messages=[{"role": "assistant", "content": "hi"}],
                )
            ]),
        ]

        result = map_spans(spans)

        assert result.input == "hello"
        assert result.actual_output == "hi"

    def test_skips_system_messages(self):
        spans = [
            _make_strands_agent_span([
                _make_span_event(
                    input_messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hi"},
                    ],
                    output_messages=[{"role": "assistant", "content": "Hello!"}],
                )
            ])
        ]

        result = map_spans(spans)

        assert result.input == "Hi"
        assert result.actual_output == "Hello!"

    def test_expected_output_from_reference_inputs(self):
        spans = [
            _make_strands_agent_span([
                _make_span_event(
                    input_messages=[{"role": "user", "content": "q"}],
                    output_messages=[{"role": "assistant", "content": "a"}],
                )
            ])
        ]
        refs = [ReferenceInput(expectedResponse={"text": "expected answer"})]

        result = map_spans(spans, reference_inputs=refs)

        assert result.expected_output == "expected answer"

    def test_body_as_json_string(self):
        body = {
            "input": {"messages": [{"role": "user", "content": "hello"}]},
            "output": {"messages": [{"role": "assistant", "content": "hi"}]},
        }
        span = {
            "traceId": "t1",
            "spanId": "s1",
            "scope": {"name": "strands.telemetry.tracer", "version": ""},
            "attributes": {"gen_ai.operation.name": "invoke_agent"},
            "span_events": [{"event_name": "strands.telemetry.tracer", "body": json.dumps(body)}],
        }

        result = map_spans([span])

        assert result.input == "hello"
        assert result.actual_output == "hi"

    def test_to_dict_omits_none(self):
        result = SpanMapResult(input="q", actual_output="a")
        d = result.to_dict()

        assert d == {"input": "q", "actual_output": "a"}
        assert "retrieval_context" not in d


class TestMapSpansErrors:
    def test_no_strands_scope_raises(self):
        spans = [_make_non_strands_span()]

        with pytest.raises(ValueError, match="Could not extract evaluation fields"):
            map_spans(spans)

    def test_empty_spans_raises(self):
        with pytest.raises(ValueError, match="Could not extract evaluation fields"):
            map_spans([])

    def test_strands_scope_but_no_invoke_agent_raises(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer", "version": ""},
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
            map_spans(spans)

    def test_invoke_agent_with_empty_events_raises(self):
        spans = [_make_strands_agent_span(span_events=[])]

        with pytest.raises(ValueError, match="Could not extract evaluation fields"):
            map_spans(spans)


class TestMapSpansInlineEvents:
    """Tests for unified ADOT format (inline events[])."""

    def test_extracts_from_inline_events(self):
        """Real format from in_memory_spans test data."""
        span = {
            "traceId": "4ab9fca604243bbd9454c0a969732697",
            "spanId": "966a414a17031f25",
            "scope": {"name": "strands.telemetry.tracer", "version": None},
            "attributes": {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "TravelAgent",
            },
            "events": [
                {
                    "name": "gen_ai.user.message",
                    "attributes": {"content": '[{"text": "Hey, how can you help me"}]'},
                },
                {
                    "name": "gen_ai.choice",
                    "attributes": {
                        "message": "Hello! I'm a travel planning assistant.",
                        "finish_reason": "end_turn",
                    },
                },
            ],
            "span_events": [],
        }

        result = map_spans([span])

        assert result.input == "Hey, how can you help me"
        assert result.actual_output == "Hello! I'm a travel planning assistant."

    def test_inline_events_preferred_over_span_body(self):
        """If both events[] and span_events[] exist, inline events win."""
        span = {
            "traceId": "t1",
            "spanId": "s1",
            "scope": {"name": "strands.telemetry.tracer", "version": ""},
            "attributes": {"gen_ai.operation.name": "invoke_agent"},
            "events": [
                {"name": "gen_ai.user.message", "attributes": {"content": '[{"text": "inline input"}]'}},
                {"name": "gen_ai.choice", "attributes": {"message": "inline output"}},
            ],
            "span_events": [
                {
                    "body": {
                        "input": {"messages": [{"role": "user", "content": "body input"}]},
                        "output": {"messages": [{"role": "assistant", "content": "body output"}]},
                    }
                }
            ],
        }

        result = map_spans([span])

        assert result.input == "inline input"
        assert result.actual_output == "inline output"

    def test_falls_back_to_span_body_when_no_inline_events(self):
        """Empty events[] -> uses span_events[].body."""
        span = {
            "traceId": "t1",
            "spanId": "s1",
            "scope": {"name": "strands.telemetry.tracer", "version": ""},
            "attributes": {"gen_ai.operation.name": "invoke_agent"},
            "events": [],
            "span_events": [
                {
                    "body": {
                        "input": {"messages": [{"role": "user", "content": "body input"}]},
                        "output": {"messages": [{"role": "assistant", "content": "body output"}]},
                    }
                }
            ],
        }

        result = map_spans([span])

        assert result.input == "body input"
        assert result.actual_output == "body output"

    def test_inline_events_only_response(self):
        """Only gen_ai.choice, no gen_ai.user.message -> still returns output."""
        span = {
            "traceId": "t1",
            "spanId": "s1",
            "scope": {"name": "strands.telemetry.tracer", "version": ""},
            "attributes": {"gen_ai.operation.name": "invoke_agent"},
            "events": [
                {"name": "gen_ai.choice", "attributes": {"message": "response only"}},
            ],
            "span_events": [],
        }

        result = map_spans([span])

        assert result.input is None
        assert result.actual_output == "response only"

    def test_inline_events_multi_turn(self):
        """Multi-turn inline events: first user input, last agent output."""
        span = {
            "traceId": "t1",
            "spanId": "s1",
            "scope": {"name": "strands.telemetry.tracer", "version": ""},
            "attributes": {
                "gen_ai.operation.name": "invoke_agent",
                "system_prompt": "You are a helpful assistant.",
            },
            "events": [
                {"name": "gen_ai.user.message", "attributes": {"content": '[{"text": "Hello"}]'}},
                {"name": "gen_ai.choice", "attributes": {"message": "Hi there!"}},
                {"name": "gen_ai.user.message", "attributes": {"content": '[{"text": "What is 2+2?"}]'}},
                {"name": "gen_ai.choice", "attributes": {"message": "The answer is 4."}},
            ],
            "span_events": [],
        }

        result = map_spans([span])

        assert result.input == "Hello"
        assert result.actual_output == "The answer is 4."
        assert result.system_prompt == "You are a helpful assistant."

    def test_span_body_multi_turn(self):
        """Multi-turn span body: first user input, last assistant output, tool outputs as retrieval_context."""
        spans = [
            _make_strands_agent_span([
                _make_span_event(
                    input_messages=[
                        {"role": "system", "content": "You are a travel agent."},
                        {"role": "user", "content": "Plan a trip"},
                    ],
                    output_messages=[{"role": "assistant", "content": "Sure! Where to?"}],
                ),
                _make_span_event(
                    input_messages=[{"role": "user", "content": "Tokyo"}],
                    output_messages=[
                        {"role": "tool", "content": "Flight info: $500"},
                        {"role": "assistant", "content": "Found flights to Tokyo."},
                    ],
                ),
            ])
        ]

        result = map_spans(spans)

        assert result.input == "Plan a trip"
        assert result.actual_output == "Found flights to Tokyo."
        assert result.system_prompt == "You are a travel agent."
        assert result.retrieval_context == ["Flight info: $500"]

    def test_to_dict_includes_system_prompt(self):
        """to_dict() includes system_prompt when present."""
        spans = [
            _make_strands_agent_span([
                _make_span_event(
                    input_messages=[
                        {"role": "system", "content": "System prompt here."},
                        {"role": "user", "content": "Hi"},
                    ],
                    output_messages=[
                        {"role": "tool", "content": "tool result"},
                        {"role": "assistant", "content": "Hello!"},
                    ],
                ),
            ])
        ]

        result = map_spans(spans)
        d = result.to_dict()

        assert d["input"] == "Hi"
        assert d["actual_output"] == "Hello!"
        assert d["system_prompt"] == "System prompt here."
        assert d["retrieval_context"] == ["tool result"]
