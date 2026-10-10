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


def _make_service_normalized_session_spans(num_turns=3):
    """Build service-normalized SESSION format spans (span_events[*].body).

    This is the format the AgentCore service sends to Lambda for SESSION-level
    evaluators: one span with multiple span_events, each representing a turn.

    Uses the REAL Strands body shape as emitted by ``invoke_agent`` and normalized
    by the service: message content is DOUBLE-ENCODED JSON under ``content`` /
    ``message`` (a JSON string, not an already-parsed list). This is the same
    shape the single-turn ``_valid_strands_spans`` fixtures use. Tests that assert
    on turn content therefore also guard against the "raw JSON leaks into the turn
    text" regression.
    """
    import json

    span_events = []
    for i in range(num_turns):
        span_events.append({
            "event_name": "strands.telemetry.tracer",
            "body": {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": {"content": json.dumps([{"text": f"User message {i + 1}"}])},
                        }
                    ]
                },
                "output": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": {"message": json.dumps([{"text": f"Assistant response {i + 1}"}])},
                        }
                    ]
                },
            },
        })
    return [
        {
            "traceId": "trace-multi",
            "spanId": "span-multi",
            "source": "adot_cw",
            "scope": {"name": "strands.telemetry.tracer"},
            "attributes": {"session.id": "session-1"},
            "span_events": span_events,
        }
    ]


class TestServiceNormalizedMultiTurn:
    """Tests for multi-turn extraction from service-normalized SESSION format."""

    def test_extracts_all_turns_from_span_events(self):
        spans = _make_service_normalized_session_spans(num_turns=3)
        result = map_spans(spans)

        assert result.turns is not None
        assert len(result.turns) == 6  # 3 user + 3 assistant
        assert result.turns[0] == {"role": "user", "content": "User message 1"}
        assert result.turns[1] == {"role": "assistant", "content": "Assistant response 1"}
        assert result.turns[4] == {"role": "user", "content": "User message 3"}
        assert result.turns[5] == {"role": "assistant", "content": "Assistant response 3"}

    def test_turn_content_has_no_json_syntax(self):
        """Regression guard: turn content must be decoded plain text, not raw JSON.

        Real service bodies double-encode content as a JSON string
        ('[{"text": ...}]'). If the extractor returns that string verbatim, the
        turn content would contain JSON punctuation. This asserts the double
        encoding is decoded to plain text.
        """
        spans = _make_service_normalized_session_spans(num_turns=3)
        result = map_spans(spans)

        assert result.turns is not None
        for turn in result.turns:
            content = turn["content"]
            assert "text" not in content or content in (
                "User message 1",
                "User message 2",
                "User message 3",
                "Assistant response 1",
                "Assistant response 2",
                "Assistant response 3",
            )
            assert not any(ch in content for ch in ("{", "}", "[", "]", '"')), (
                f"turn content still contains raw JSON syntax: {content!r}"
            )

    def test_input_and_output_are_last_turn(self):
        spans = _make_service_normalized_session_spans(num_turns=3)
        result = map_spans(spans)

        assert result.input == "User message 3"
        assert result.actual_output == "Assistant response 3"

    def test_single_span_event_returns_none_turns(self):
        """Single span_event should NOT populate turns (not multi-turn)."""
        spans = _make_service_normalized_session_spans(num_turns=1)
        result = map_spans(spans)

        # With only 1 turn (2 entries: user+assistant), turns should be None
        assert result.turns is None
        # But input/output should still be extracted
        assert result.input == "User message 1"
        assert result.actual_output == "Assistant response 1"

    def test_handles_string_content_variant(self):
        """Test spans where content is a plain string instead of list of dicts."""
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"session.id": "sess"},
                "span_events": [
                    {
                        "body": {
                            "input": {"messages": [{"role": "user", "content": "Hello plain"}]},
                            "output": {"messages": [{"role": "assistant", "content": "Hi plain"}]},
                        }
                    },
                    {
                        "body": {
                            "input": {"messages": [{"role": "user", "content": "Follow up"}]},
                            "output": {"messages": [{"role": "assistant", "content": "Got it"}]},
                        }
                    },
                ],
            }
        ]
        result = map_spans(spans)

        assert result.turns is not None
        assert len(result.turns) == 4
        assert result.turns[0] == {"role": "user", "content": "Hello plain"}
        assert result.turns[3] == {"role": "assistant", "content": "Got it"}

    def test_handles_nested_content_dict_variant(self):
        """Test the {content: {content: [{text: ...}]}} nesting."""
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"session.id": "sess"},
                "span_events": [
                    {
                        "body": {
                            "input": {
                                "messages": [
                                    {"role": "user", "content": {"content": [{"text": "Turn 1 input"}]}}
                                ]
                            },
                            "output": {
                                "messages": [
                                    {"role": "assistant", "content": {"message": [{"text": "Turn 1 output"}]}}
                                ]
                            },
                        }
                    },
                    {
                        "body": {
                            "input": {
                                "messages": [
                                    {"role": "user", "content": {"content": [{"text": "Turn 2 input"}]}}
                                ]
                            },
                            "output": {
                                "messages": [
                                    {"role": "assistant", "content": {"message": [{"text": "Turn 2 output"}]}}
                                ]
                            },
                        }
                    },
                ],
            }
        ]
        result = map_spans(spans)

        assert result.turns is not None
        assert len(result.turns) == 4
        assert result.turns[0]["content"] == "Turn 1 input"
        assert result.turns[1]["content"] == "Turn 1 output"
        assert result.turns[2]["content"] == "Turn 2 input"
        assert result.turns[3]["content"] == "Turn 2 output"

    def test_five_turns_for_session_evaluation(self):
        """Realistic test: 5-turn conversation as sent by the service."""
        spans = _make_service_normalized_session_spans(num_turns=5)
        result = map_spans(spans)

        assert result.turns is not None
        assert len(result.turns) == 10
        assert result.input == "User message 5"
        assert result.actual_output == "Assistant response 5"
