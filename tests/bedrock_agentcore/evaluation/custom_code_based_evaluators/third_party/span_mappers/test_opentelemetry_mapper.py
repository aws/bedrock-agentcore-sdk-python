"""Tests for OpenTelemetry instrumentation LangChain span mapper."""

import json

import pytest

from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers import (
    map_spans,
)
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.span_mappers.langgraph import (
    SCOPE_AMAZON_OPENTELEMETRY_DISTRO_INSTRUMENTATION_LANGCHAIN,
    SCOPE_OPENTELEMETRY_INSTRUMENTATION_LANGCHAIN,
    OpenTelemetryInstrumentationLangchainMapper,
)

TRACELOOP_SCOPE = SCOPE_OPENTELEMETRY_INSTRUMENTATION_LANGCHAIN
ADOT_NATIVE_SCOPE = SCOPE_AMAZON_OPENTELEMETRY_DISTRO_INSTRUMENTATION_LANGCHAIN


def _make_workflow_span(scope_name, input_value=None, output_value=None, span_events=None):
    """Create a workflow span with the given scope."""
    span = {
        "scope": {"name": scope_name},
        "name": "LangGraph",
        "attributes": {
            "traceloop.span.kind": "workflow",
        },
    }
    if input_value is not None:
        span["attributes"]["gen_ai.task.input"] = input_value
    if output_value is not None:
        span["attributes"]["gen_ai.task.output"] = output_value
    if span_events is not None:
        span["span_events"] = span_events
    return span


def _make_tool_span(scope_name, tool_result=None, task_output=None, span_events=None,
                    span_kind_attr="traceloop", operation_name=None):
    """Create a tool span."""
    span = {
        "scope": {"name": scope_name},
        "name": "calculate_bmi",
        "attributes": {},
    }
    if span_kind_attr == "traceloop":
        span["attributes"]["traceloop.span.kind"] = "tool"
    if operation_name:
        span["attributes"]["gen_ai.operation.name"] = operation_name
    if tool_result is not None:
        span["attributes"]["gen_ai.tool.call.result"] = tool_result
    if task_output is not None:
        span["attributes"]["gen_ai.task.output"] = task_output
    if span_events is not None:
        span["span_events"] = span_events
    return span


def _make_llm_span(scope_name, prompts=None):
    """Create an LLM span with gen_ai.prompt attributes."""
    span = {
        "scope": {"name": scope_name},
        "name": "ChatBedrock",
        "attributes": {
            "llm.request.type": "chat",
        },
    }
    if prompts:
        for i, (role, content) in enumerate(prompts):
            span["attributes"][f"gen_ai.prompt.{i}.role"] = role
            span["attributes"][f"gen_ai.prompt.{i}.content"] = content
    return span


# ─── Test: Amazon OTEL Distro scope support ───


class TestAdotNativeScopeSupport:
    """Tests that spans from amazon.opentelemetry.distro.instrumentation.langchain are handled."""

    def test_mapper_supports_both_scopes(self):
        mapper = OpenTelemetryInstrumentationLangchainMapper()
        assert TRACELOOP_SCOPE in mapper.scope_names
        assert ADOT_NATIVE_SCOPE in mapper.scope_names

    def test_adot_native_workflow_span_extraction(self):
        """Spans with ADOT native scope should be processed."""
        input_val = json.dumps({"inputs": {"messages": [("user", "What is 2+2?")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "4")]}})
        spans = [_make_workflow_span(ADOT_NATIVE_SCOPE, input_val, output_val)]

        result = map_spans(spans)
        assert result.input == "What is 2+2?"
        assert result.actual_output == "4"

    def test_adot_native_invoke_agent_span(self):
        """ADOT native uses gen_ai.operation.name == invoke_agent for agent spans."""
        input_val = json.dumps({"inputs": {"messages": [("user", "Hello")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "Hi there!")]}})
        span = {
            "scope": {"name": ADOT_NATIVE_SCOPE},
            "name": "agent",
            "attributes": {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.task.input": input_val,
                "gen_ai.task.output": output_val,
            },
        }
        result = map_spans([span])
        assert result.input == "Hello"
        assert result.actual_output == "Hi there!"

    def test_mixed_scopes_in_same_trace(self):
        """Workflow span from one scope + tool span from another should both be processed."""
        input_val = json.dumps({"inputs": {"messages": [("user", "Calculate BMI")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "Your BMI is 22.9")]}})
        workflow = _make_workflow_span(ADOT_NATIVE_SCOPE, input_val, output_val)
        tool = _make_tool_span(
            ADOT_NATIVE_SCOPE,
            tool_result=json.dumps({"output": "BMI: 22.9 (Normal)"}),
            span_kind_attr="none",
            operation_name="execute_tool",
        )
        result = map_spans([workflow, tool])
        assert result.input == "Calculate BMI"
        assert result.retrieval_context == ["BMI: 22.9 (Normal)"]


# ─── Test: Tool span detection with gen_ai.operation.name ───


class TestToolSpanDetection:
    """Tests that tool spans are detected via both traceloop.span.kind and gen_ai.operation.name."""

    def test_traceloop_tool_span_detected(self):
        """Traditional traceloop.span.kind == tool detection still works."""
        input_val = json.dumps({"inputs": {"messages": [("user", "BMI check")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "Done")]}})
        workflow = _make_workflow_span(TRACELOOP_SCOPE, input_val, output_val)
        tool = _make_tool_span(TRACELOOP_SCOPE, task_output="BMI: 22.9")
        result = map_spans([workflow, tool])
        assert result.retrieval_context == ["BMI: 22.9"]

    def test_execute_tool_operation_name_detected(self):
        """gen_ai.operation.name == execute_tool is detected as a tool span."""
        input_val = json.dumps({"inputs": {"messages": [("user", "BMI check")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "Done")]}})
        workflow = _make_workflow_span(TRACELOOP_SCOPE, input_val, output_val)
        tool = _make_tool_span(
            TRACELOOP_SCOPE,
            task_output="BMI: 22.9",
            span_kind_attr="none",
            operation_name="execute_tool",
        )
        result = map_spans([workflow, tool])
        assert result.retrieval_context == ["BMI: 22.9"]

    def test_adot_native_execute_tool_detected(self):
        """ADOT native scope with execute_tool is detected."""
        input_val = json.dumps({"inputs": {"messages": [("user", "BMI check")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "Done")]}})
        workflow = _make_workflow_span(ADOT_NATIVE_SCOPE, input_val, output_val)
        tool = _make_tool_span(
            ADOT_NATIVE_SCOPE,
            tool_result=json.dumps({"output": "BMI: 22.9"}),
            span_kind_attr="none",
            operation_name="execute_tool",
        )
        result = map_spans([workflow, tool])
        assert result.retrieval_context == ["BMI: 22.9"]


# ─── Test: gen_ai.tool.call.result attribute extraction ───


class TestToolResultAttributeExtraction:
    """Tests for extracting tool output from gen_ai.tool.call.result attribute."""

    def test_langchain_tool_message_wrapper(self):
        """Handles {"output": {"kwargs": {"content": "text"}}} format."""
        tool_result = json.dumps({
            "output": {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "ToolMessage"],
                "kwargs": {
                    "content": "BMI: 22.9 (Normal weight)",
                    "type": "tool",
                    "name": "calculate_bmi",
                    "tool_call_id": "call_123",
                    "status": "success",
                },
            }
        })
        input_val = json.dumps({"inputs": {"messages": [("user", "BMI check")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "Done")]}})
        workflow = _make_workflow_span(TRACELOOP_SCOPE, input_val, output_val)
        tool = _make_tool_span(TRACELOOP_SCOPE, tool_result=tool_result)
        result = map_spans([workflow, tool])
        assert result.retrieval_context == ["BMI: 22.9 (Normal weight)"]

    def test_simple_string_output(self):
        """Handles {"output": "plain text"} format."""
        tool_result = json.dumps({"output": "The weather is 72°F"})
        input_val = json.dumps({"inputs": {"messages": [("user", "Weather?")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "72°F")]}})
        workflow = _make_workflow_span(TRACELOOP_SCOPE, input_val, output_val)
        tool = _make_tool_span(TRACELOOP_SCOPE, tool_result=tool_result)
        result = map_spans([workflow, tool])
        assert result.retrieval_context == ["The weather is 72°F"]

    def test_list_content_blocks(self):
        """Handles {"output": {"kwargs": {"content": [{"text": "a"}, {"text": "b"}]}}}."""
        tool_result = json.dumps({
            "output": {
                "kwargs": {
                    "content": [{"text": "Line 1"}, {"text": "Line 2"}],
                    "tool_call_id": "call_456",
                }
            }
        })
        input_val = json.dumps({"inputs": {"messages": [("user", "Data?")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "Here")]}})
        workflow = _make_workflow_span(TRACELOOP_SCOPE, input_val, output_val)
        tool = _make_tool_span(TRACELOOP_SCOPE, tool_result=tool_result)
        result = map_spans([workflow, tool])
        assert result.retrieval_context == ["Line 1\nLine 2"]

    def test_plain_dict_output(self):
        """Handles {"output": {"key": "value"}} without kwargs → JSON dump."""
        tool_result = json.dumps({"output": {"temperature": 72, "unit": "F"}})
        input_val = json.dumps({"inputs": {"messages": [("user", "Weather?")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "72°F")]}})
        workflow = _make_workflow_span(TRACELOOP_SCOPE, input_val, output_val)
        tool = _make_tool_span(TRACELOOP_SCOPE, tool_result=tool_result)
        result = map_spans([workflow, tool])
        assert result.retrieval_context == ['{"temperature": 72, "unit": "F"}']

    def test_non_json_result(self):
        """Non-JSON tool result is returned as-is."""
        input_val = json.dumps({"inputs": {"messages": [("user", "Hello")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "Hi")]}})
        workflow = _make_workflow_span(TRACELOOP_SCOPE, input_val, output_val)
        tool = _make_tool_span(TRACELOOP_SCOPE, tool_result="plain text result")
        result = map_spans([workflow, tool])
        assert result.retrieval_context == ["plain text result"]

    def test_priority_over_span_events(self):
        """gen_ai.tool.call.result takes priority over span_events body."""
        tool_result = json.dumps({"output": "from attribute"})
        span_events = [{"body": json.dumps({"output": "from body"})}]
        input_val = json.dumps({"inputs": {"messages": [("user", "Hi")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "Hello")]}})
        workflow = _make_workflow_span(TRACELOOP_SCOPE, input_val, output_val)
        tool = _make_tool_span(TRACELOOP_SCOPE, tool_result=tool_result, span_events=span_events)
        result = map_spans([workflow, tool])
        assert result.retrieval_context == ["from attribute"]

    def test_fallback_to_span_events_when_no_attribute(self):
        """Falls back to span_events body when gen_ai.tool.call.result is absent."""
        span_events = [{"body": json.dumps({"output": "from body"})}]
        input_val = json.dumps({"inputs": {"messages": [("user", "Hi")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "Hello")]}})
        workflow = _make_workflow_span(TRACELOOP_SCOPE, input_val, output_val)
        tool = _make_tool_span(TRACELOOP_SCOPE, span_events=span_events)
        result = map_spans([workflow, tool])
        assert result.retrieval_context == ["from body"]

    def test_fallback_to_task_output_when_no_body(self):
        """Falls back to gen_ai.task.output when both attribute and body are absent."""
        input_val = json.dumps({"inputs": {"messages": [("user", "Hi")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "Hello")]}})
        workflow = _make_workflow_span(TRACELOOP_SCOPE, input_val, output_val)
        tool = _make_tool_span(TRACELOOP_SCOPE, task_output="fallback output")
        result = map_spans([workflow, tool])
        assert result.retrieval_context == ["fallback output"]


# ─── Test: System prompt still works ───


class TestSystemPromptExtraction:
    """Verify system prompt extraction remains functional."""

    def test_system_prompt_from_llm_span(self):
        """System prompt extracted from gen_ai.prompt.0.role == system."""
        input_val = json.dumps({"inputs": {"messages": [("user", "Hello")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "Hi")]}})
        workflow = _make_workflow_span(TRACELOOP_SCOPE, input_val, output_val)
        llm = _make_llm_span(TRACELOOP_SCOPE, prompts=[
            ("system", "You are a helpful assistant"),
            ("user", "Hello"),
        ])
        result = map_spans([workflow, llm])
        assert result.system_prompt == "You are a helpful assistant"

    def test_system_prompt_with_adot_native_scope(self):
        """System prompt works with ADOT native scope."""
        input_val = json.dumps({"inputs": {"messages": [("user", "Hello")]}})
        output_val = json.dumps({"outputs": {"messages": [("ai", "Hi")]}})
        workflow = _make_workflow_span(ADOT_NATIVE_SCOPE, input_val, output_val)
        llm = _make_llm_span(ADOT_NATIVE_SCOPE, prompts=[
            ("system", "Be concise"),
            ("user", "Hello"),
        ])
        result = map_spans([workflow, llm])
        assert result.system_prompt == "Be concise"
