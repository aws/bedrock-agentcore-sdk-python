"""Tests for OpenInference instrumentation LangChain span mapper."""

import json

import pytest

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers import (
    map_spans,
)


SCOPE = "openinference.instrumentation.langchain"


def _make_agent_span(input_value=None, output_value=None, span_events=None, name="LangGraph"):
    """Build an OpenInference CHAIN agent span."""
    attrs = {"openinference.span.kind": "CHAIN"}
    if input_value is not None:
        attrs["input.value"] = input_value
    if output_value is not None:
        attrs["output.value"] = output_value
    return {
        "traceId": "t1",
        "spanId": "agent1",
        "name": name,
        "scope": {"name": SCOPE, "version": ""},
        "attributes": attrs,
        "span_events": span_events or [],
    }


def _make_tool_span(output_value, span_id="tool1"):
    """Build an OpenInference TOOL span."""
    return {
        "traceId": "t1",
        "spanId": span_id,
        "name": "search_tool",
        "scope": {"name": SCOPE, "version": ""},
        "attributes": {
            "openinference.span.kind": "TOOL",
            "output.value": output_value,
        },
        "span_events": [],
    }


def _make_llm_span(span_events=None, span_id="llm1"):
    """Build an OpenInference LLM span."""
    return {
        "traceId": "t1",
        "spanId": span_id,
        "name": "ChatModel",
        "scope": {"name": SCOPE, "version": ""},
        "attributes": {"openinference.span.kind": "LLM"},
        "span_events": span_events or [],
    }


class TestOpenInferenceAttributesPath:
    """Tests for extraction from span attributes (input.value / output.value)."""

    def test_basic_extraction(self):
        input_val = json.dumps({"messages": [["user", "What is AI?"]]})
        output_val = json.dumps({"messages": [["assistant", "Artificial intelligence."]]})
        spans = [_make_agent_span(input_value=input_val, output_value=output_val)]

        result = map_spans(spans)

        assert result.input == "What is AI?"
        assert result.actual_output == "Artificial intelligence."

    def test_dict_format_messages(self):
        input_val = json.dumps({
            "messages": [{"type": "human", "data": {"content": "Hello"}}]
        })
        output_val = json.dumps({
            "messages": [{"type": "ai", "data": {"content": "Hi there!"}}]
        })
        spans = [_make_agent_span(input_value=input_val, output_value=output_val)]

        result = map_spans(spans)

        assert result.input == "Hello"
        assert result.actual_output == "Hi there!"

    def test_raw_string_fallback(self):
        """When input.value/output.value are plain strings, use them directly."""
        spans = [_make_agent_span(input_value="plain question", output_value="plain answer")]

        result = map_spans(spans)

        assert result.input == "plain question"
        assert result.actual_output == "plain answer"

    def test_skips_tool_use_only_ai_messages(self):
        input_val = json.dumps({"messages": [["user", "Search for flights"]]})
        output_val = json.dumps({
            "messages": [
                {"type": "ai", "data": {"content": [{"type": "tool_use", "name": "search", "input": {}}]}},
                {"type": "ai", "data": {"content": "Found 3 flights to Tokyo."}},
            ]
        })
        spans = [_make_agent_span(input_value=input_val, output_value=output_val)]

        result = map_spans(spans)

        assert result.actual_output == "Found 3 flights to Tokyo."


class TestOpenInferenceLogEventFallback:
    """Tests for extraction from span_events[].body (CloudWatch ADOT format)."""

    def test_extracts_from_span_body_when_no_attributes(self):
        """When input.value/output.value are absent, falls back to span_events body."""
        body = {
            "input": {
                "messages": [{"content": json.dumps({"messages": [["user", "What is 2+2?"]]}), "role": "user"}]
            },
            "output": {
                "messages": [{"content": json.dumps({"messages": [["assistant", "4"]]}), "role": "assistant"}]
            },
        }
        span_events = [{"body": body}]
        spans = [_make_agent_span(span_events=span_events)]

        result = map_spans(spans)

        assert result.input == "What is 2+2?"
        assert result.actual_output == "4"

    def test_span_body_with_generations_format(self):
        """Output in generations format."""
        body = {
            "input": {
                "messages": [{"content": "Tell me a joke", "role": "user"}]
            },
            "output": {
                "messages": [{
                    "content": json.dumps({"generations": [[{"text": "Why did the chicken cross the road?"}]]}),
                    "role": "assistant",
                }]
            },
        }
        span_events = [{"body": body}]
        spans = [_make_agent_span(span_events=span_events)]

        result = map_spans(spans)

        assert result.input == "Tell me a joke"
        assert result.actual_output == "Why did the chicken cross the road?"

    def test_span_body_as_json_string(self):
        """Body serialized as a JSON string."""
        body = {
            "input": {"messages": [{"content": "hello", "role": "user"}]},
            "output": {"messages": [{"content": "hi", "role": "assistant"}]},
        }
        span_events = [{"body": json.dumps(body)}]
        spans = [_make_agent_span(span_events=span_events)]

        result = map_spans(spans)

        assert result.input == "hello"
        assert result.actual_output == "hi"

    def test_attributes_preferred_over_span_body(self):
        """When both attribute and body data exist, attributes win."""
        input_val = json.dumps({"messages": [["user", "from attributes"]]})
        output_val = json.dumps({"messages": [["assistant", "attr answer"]]})
        body = {
            "input": {"messages": [{"content": "from body", "role": "user"}]},
            "output": {"messages": [{"content": "body answer", "role": "assistant"}]},
        }
        span_events = [{"body": body}]
        spans = [_make_agent_span(input_value=input_val, output_value=output_val, span_events=span_events)]

        result = map_spans(spans)

        assert result.input == "from attributes"
        assert result.actual_output == "attr answer"

    def test_span_body_with_langgraph_dict_messages(self):
        """Body with LangGraph dict-style messages."""
        messages_input = [{"type": "human", "kwargs": {"content": "Plan a trip"}}]
        messages_output = [{"type": "ai", "kwargs": {"content": "Here's your itinerary."}}]
        body = {
            "input": {
                "messages": [{"content": json.dumps({"messages": messages_input}), "role": "user"}]
            },
            "output": {
                "messages": [{"content": json.dumps({"messages": messages_output}), "role": "assistant"}]
            },
        }
        span_events = [{"body": body}]
        spans = [_make_agent_span(span_events=span_events)]

        result = map_spans(spans)

        assert result.input == "Plan a trip"
        assert result.actual_output == "Here's your itinerary."


class TestOpenInferenceToolOutputParsing:
    """Tests for tool output content extraction (nested JSON handling)."""

    def test_plain_string_tool_output(self):
        spans = [
            _make_agent_span(input_value="q", output_value="a"),
            _make_tool_span("The weather is sunny."),
        ]

        result = map_spans(spans)

        assert result.retrieval_context == ["The weather is sunny."]

    def test_nested_content_field(self):
        """Tool output with {"content": "...", "tool_call_id": "...", "status": "success"}."""
        tool_output = json.dumps({
            "content": "Tokyo: sunny, 25°C",
            "tool_call_id": "call_123",
            "status": "success",
            "name": "get_weather",
        })
        spans = [
            _make_agent_span(input_value="q", output_value="a"),
            _make_tool_span(tool_output),
        ]

        result = map_spans(spans)

        assert result.retrieval_context == ["Tokyo: sunny, 25°C"]

    def test_nested_data_content_field(self):
        """Tool output with {"data": {"content": "...", ...}} (openinference 0.1.62+)."""
        tool_output = json.dumps({
            "data": {
                "content": "Flight: $500 round trip",
                "tool_call_id": "call_456",
                "status": "success",
            }
        })
        spans = [
            _make_agent_span(input_value="q", output_value="a"),
            _make_tool_span(tool_output),
        ]

        result = map_spans(spans)

        assert result.retrieval_context == ["Flight: $500 round trip"]

    def test_text_blocks_format(self):
        """Tool output as JSON text blocks: [{"text": "..."}, {"text": "..."}]."""
        tool_output = json.dumps([{"text": "Result 1"}, {"text": "Result 2"}])
        spans = [
            _make_agent_span(input_value="q", output_value="a"),
            _make_tool_span(tool_output),
        ]

        result = map_spans(spans)

        assert result.retrieval_context == ["Result 1\nResult 2"]

    def test_content_as_list_of_text_blocks(self):
        """Tool output: {"content": [{"text": "block1"}, {"text": "block2"}]}."""
        tool_output = json.dumps({
            "content": [{"text": "chunk A"}, {"text": "chunk B"}],
            "name": "retriever",
        })
        spans = [
            _make_agent_span(input_value="q", output_value="a"),
            _make_tool_span(tool_output),
        ]

        result = map_spans(spans)

        assert result.retrieval_context == ["chunk A\nchunk B"]

    def test_multiple_tool_spans(self):
        spans = [
            _make_agent_span(input_value="q", output_value="a"),
            _make_tool_span("result 1", span_id="tool1"),
            _make_tool_span(json.dumps({"content": "result 2"}), span_id="tool2"),
        ]

        result = map_spans(spans)

        assert result.retrieval_context == ["result 1", "result 2"]
        assert result.context == ["result 1", "result 2"]

    def test_plain_dict_tool_output_returned_as_json(self):
        """Dict without 'content' key is returned as raw JSON string."""
        tool_output = json.dumps({"temperature": 25, "unit": "celsius"})
        spans = [
            _make_agent_span(input_value="q", output_value="a"),
            _make_tool_span(tool_output),
        ]

        result = map_spans(spans)

        # Falls through to raw string since no "content"/"data" key
        assert result.retrieval_context == [tool_output]


class TestOpenInferenceSystemPrompt:
    """Tests for system prompt extraction from LLM spans."""

    def test_extracts_system_prompt_from_llm_span_body(self):
        """System prompt in LLM span body input messages."""
        messages_list = [
            {"type": "system", "kwargs": {"content": "You are a helpful travel assistant."}},
            {"type": "human", "kwargs": {"content": "Plan a trip"}},
        ]
        body = {
            "input": {
                "messages": [{"content": json.dumps({"messages": messages_list}), "role": "user"}]
            },
            "output": {
                "messages": [{"content": json.dumps({"generations": [[{"text": "response"}]]}), "role": "assistant"}]
            },
        }
        llm_span = _make_llm_span(span_events=[{"body": body}])
        agent_span = _make_agent_span(input_value="Plan a trip", output_value="Here's your plan.")

        result = map_spans([agent_span, llm_span])

        assert result.system_prompt == "You are a helpful travel assistant."

    def test_system_prompt_with_list_content(self):
        """System prompt content as list of text items."""
        messages_list = [
            {"type": "system", "kwargs": {"content": [{"text": "Rule 1"}, {"text": "Rule 2"}]}},
            {"type": "human", "kwargs": {"content": "question"}},
        ]
        body = {
            "input": {
                "messages": [{"content": json.dumps({"messages": messages_list}), "role": "user"}]
            },
            "output": {
                "messages": [{"content": "answer", "role": "assistant"}]
            },
        }
        llm_span = _make_llm_span(span_events=[{"body": body}])
        agent_span = _make_agent_span(input_value="q", output_value="a")

        result = map_spans([agent_span, llm_span])

        assert result.system_prompt == "Rule 1\n\nRule 2"

    def test_no_system_prompt_returns_none(self):
        """When no system message exists, system_prompt is None."""
        spans = [_make_agent_span(input_value="q", output_value="a")]

        result = map_spans(spans)

        assert result.system_prompt is None

    def test_system_prompt_with_constructor_type(self):
        """System message with type=constructor pattern."""
        messages_list = [
            {"type": "constructor", "kwargs": {"type": "system", "content": "Be concise."}},
            {"type": "human", "kwargs": {"content": "Hi"}},
        ]
        body = {
            "input": {
                "messages": [{"content": json.dumps({"messages": messages_list}), "role": "user"}]
            },
            "output": {
                "messages": [{"content": "Hello!", "role": "assistant"}]
            },
        }
        llm_span = _make_llm_span(span_events=[{"body": body}])
        agent_span = _make_agent_span(input_value="Hi", output_value="Hello!")

        result = map_spans([agent_span, llm_span])

        assert result.system_prompt == "Be concise."

    def test_system_prompt_from_id_based_classification(self):
        """System message identified by ID array containing 'SystemMessage'."""
        messages_list = [
            {"id": ["langchain", "schema", "messages", "SystemMessage"], "kwargs": {"content": "You are expert."}},
            {"type": "human", "kwargs": {"content": "Help"}},
        ]
        body = {
            "input": {
                "messages": [{"content": json.dumps({"messages": messages_list}), "role": "user"}]
            },
            "output": {
                "messages": [{"content": "Sure!", "role": "assistant"}]
            },
        }
        llm_span = _make_llm_span(span_events=[{"body": body}])
        agent_span = _make_agent_span(input_value="Help", output_value="Sure!")

        result = map_spans([agent_span, llm_span])

        assert result.system_prompt == "You are expert."


class TestOpenInferenceAgentSpanDetection:
    """Tests for agent span finding logic."""

    def test_chain_langgraph_span(self):
        spans = [_make_agent_span(input_value="q", output_value="a", name="LangGraph")]

        result = map_spans(spans)

        assert result.input == "q"

    def test_agent_kind_span(self):
        span = {
            "traceId": "t1",
            "spanId": "s1",
            "name": "MyCustomAgent",
            "scope": {"name": SCOPE, "version": ""},
            "attributes": {
                "openinference.span.kind": "AGENT",
                "input.value": "question",
                "output.value": "answer",
            },
            "span_events": [],
        }

        result = map_spans([span])

        assert result.input == "question"
        assert result.actual_output == "answer"

    def test_skips_route_spans(self):
        """AGENT spans named route_* are skipped."""
        route_span = {
            "traceId": "t1",
            "spanId": "s1",
            "name": "route_after_agent",
            "scope": {"name": SCOPE, "version": ""},
            "attributes": {
                "openinference.span.kind": "AGENT",
                "input.value": "wrong",
                "output.value": "wrong",
            },
            "span_events": [],
        }
        real_span = _make_agent_span(input_value="correct q", output_value="correct a")
        spans = [route_span, real_span]

        result = map_spans(spans)

        assert result.input == "correct q"

    def test_fallback_to_any_chain_span(self):
        """If no LangGraph-named CHAIN, falls back to first CHAIN."""
        span = {
            "traceId": "t1",
            "spanId": "s1",
            "name": "CustomChain",
            "scope": {"name": SCOPE, "version": ""},
            "attributes": {
                "openinference.span.kind": "CHAIN",
                "input.value": "question",
                "output.value": "answer",
            },
            "span_events": [],
        }

        result = map_spans([span])

        assert result.input == "question"

    def test_no_agent_span_returns_none(self):
        """Only TOOL spans → raises ValueError (no mapper can extract)."""
        spans = [_make_tool_span("output")]

        with pytest.raises(ValueError, match="Could not extract"):
            map_spans(spans)
