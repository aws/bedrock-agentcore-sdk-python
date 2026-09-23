"""Tests for public evaluation span helpers."""

import pytest

from bedrock_agentcore.evaluation.spans import is_tool_span, tool_span_ids, trace_ids


@pytest.mark.parametrize(
    "span,expected",
    [
        ({"attributes": {"gen_ai.operation.name": "execute_tool"}}, True),
        ({"attributes": {"openinference.span.kind": "TOOL"}}, True),
        ({"attributes": {"traceloop.span.kind": "tool"}}, True),
        ({"attributes": {"gen_ai.operation.name": "chat", "openinference.span.kind": "TOOL"}}, True),
        ({"attributes": {"gen_ai.operation.name": "chat"}}, False),
        ({"attributes": {"openinference.span.kind": "tool"}}, False),
        ({}, False),
        ({"attributes": None}, False),
        ({"attributes": []}, False),
        ({"attributes": "execute_tool"}, False),
    ],
)
def test_is_tool_span(span, expected):
    assert is_tool_span(span) is expected


@pytest.mark.parametrize("name", ["is_tool_span", "tool_span_ids", "trace_ids"])
def test_public_package_exports(name):
    from bedrock_agentcore import evaluation
    from bedrock_agentcore.evaluation import spans

    assert name in evaluation.__all__
    assert getattr(evaluation, name) is getattr(spans, name)


@pytest.mark.parametrize("trace_id", [None, "", "t1", "missing"])
def test_tool_span_ids_missing_ids_duplicates_and_trace_filter(trace_id):
    spans = [
        {"spanId": "s1", "traceId": "t1", "attributes": {"gen_ai.operation.name": "execute_tool"}},
        {"spanId": "s1", "traceId": "t1", "attributes": {"openinference.span.kind": "TOOL"}},
        {"spanId": "s2", "attributes": {"traceloop.span.kind": "tool"}},
        {"spanId": "", "attributes": {"traceloop.span.kind": "tool"}},
        {"spanId": None, "attributes": {"traceloop.span.kind": "tool"}},
        {"attributes": {"traceloop.span.kind": "tool"}},
    ]
    expected = {None: ["s1", "s1", "s2"], "": ["s1", "s1", "s2"], "t1": ["s1", "s1"], "missing": []}
    assert tool_span_ids(spans, trace_id=trace_id) == expected[trace_id]


def test_trace_ids_skips_empty_and_preserves_first_appearance():
    spans = [{"traceId": value} for value in ["t2", None, "", "t1", "t2", "t3", "t1"]]
    assert trace_ids(spans) == ["t2", "t1", "t3"]


SAMPLE_SPANS = [
    {
        "scope": {"name": "agent"},
        "traceId": "trace-1",
        "spanId": "span-1",
        "name": "Agent.invoke",
        "kind": "SPAN_KIND_SERVER",
        "attributes": {"gen_ai.operation.name": "invoke_agent"},
    },
    {
        "scope": {"name": "agent"},
        "traceId": "trace-1",
        "spanId": "span-2",
        "name": "Tool:search",
        "kind": "SPAN_KIND_INTERNAL",
        "attributes": {"gen_ai.operation.name": "execute_tool"},
    },
    {
        "scope": {"name": "agent"},
        "traceId": "trace-1",
        "spanId": "span-3",
        "name": "Tool:calculator",
        "kind": "SPAN_KIND_INTERNAL",
        "attributes": {"gen_ai.operation.name": "execute_tool"},
    },
    {
        "scope": {"name": "agent"},
        "traceId": "trace-2",
        "spanId": "span-4",
        "name": "Agent.invoke",
        "kind": "SPAN_KIND_SERVER",
        "attributes": {"gen_ai.operation.name": "invoke_agent"},
    },
    {
        "scope": {"name": "agent"},
        "traceId": "trace-2",
        "spanId": "span-5",
        "name": "Tool:search",
        "kind": "SPAN_KIND_INTERNAL",
        "attributes": {"gen_ai.operation.name": "execute_tool"},
    },
]


# --- Static helper tests ---


class TestExtractTraceIds:
    def test_extracts_unique_ordered(self):
        ids = trace_ids(SAMPLE_SPANS)
        assert ids == ["trace-1", "trace-2"]

    def test_empty_spans(self):
        assert trace_ids([]) == []

    def test_skips_missing_trace_id(self):
        spans = [{"spanId": "s1"}, {"traceId": "t1", "spanId": "s2"}]
        assert trace_ids(spans) == ["t1"]


class TestExtractToolSpanIds:
    def test_extracts_tool_spans(self):
        ids = tool_span_ids(SAMPLE_SPANS)
        assert ids == ["span-2", "span-3", "span-5"]

    def test_ignores_non_tool_spans(self):
        spans = [
            {"name": "Agent.invoke", "kind": "SPAN_KIND_SERVER", "spanId": "s1"},
            {"name": "LLM.call", "kind": "SPAN_KIND_INTERNAL", "spanId": "s2"},
        ]
        assert tool_span_ids(spans) == []

    def test_empty_spans(self):
        assert tool_span_ids([]) == []

    def test_filters_by_trace_id(self):
        ids = tool_span_ids(SAMPLE_SPANS, trace_id="trace-1")
        assert ids == ["span-2", "span-3"]

    def test_filters_by_trace_id_no_match(self):
        ids = tool_span_ids(SAMPLE_SPANS, trace_id="trace-999")
        assert ids == []

    def test_extracts_langgraph_tool_spans(self):
        spans = [
            {"spanId": "s1", "traceId": "t1", "attributes": {"openinference.span.kind": "TOOL"}},
            {"spanId": "s2", "traceId": "t1", "attributes": {"openinference.span.kind": "LLM"}},
        ]
        assert tool_span_ids(spans) == ["s1"]

    def test_extracts_traceloop_tool_spans(self):
        spans = [
            {"spanId": "s1", "traceId": "t1", "attributes": {"traceloop.span.kind": "tool"}},
            {"spanId": "s2", "traceId": "t1", "attributes": {"traceloop.span.kind": "workflow"}},
        ]
        assert tool_span_ids(spans) == ["s1"]

    def test_ignores_span_without_tool_attributes(self):
        spans = [
            {"spanId": "s1", "traceId": "t1", "attributes": {"gen_ai.operation.name": "invoke_agent"}},
            {"spanId": "s2", "traceId": "t1", "attributes": {"some.other.attr": "value"}},
            {"spanId": "s3", "traceId": "t1"},
        ]
        assert tool_span_ids(spans) == []
