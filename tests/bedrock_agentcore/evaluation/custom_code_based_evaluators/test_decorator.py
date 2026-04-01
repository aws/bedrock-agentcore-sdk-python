"""Tests for the @custom_code_based_evaluator decorator."""

import pytest

from bedrock_agentcore.evaluation.custom_code_based_evaluators import (
    EvaluatorInput,
    EvaluatorOutput,
    custom_code_based_evaluator,
)


def _make_event(level="TRACE", trace_ids=None, span_ids=None):
    """Build a raw Lambda event dict."""
    event = {
        "schemaVersion": "1.0",
        "evaluationLevel": level,
        "evaluationInput": {
            "sessionSpans": [
                {"traceId": "abc123", "spanId": "span1", "name": "Agent", "attributes": {}},
            ]
        },
        "evaluationTarget": {},
    }
    if trace_ids is not None:
        event["evaluationTarget"]["traceIds"] = trace_ids
    if span_ids is not None:
        event["evaluationTarget"]["spanIds"] = span_ids
    return event


class TestDecoratorWithRawEvent:
    def test_trace_level(self):
        @custom_code_based_evaluator()
        def handler(inp, context):
            return EvaluatorOutput(value=1.0, label="Pass", explanation="Valid")

        event = _make_event(level="TRACE", trace_ids=["abc123"])
        result = handler(event)

        assert result["value"] == 1.0
        assert result["label"] == "Pass"
        assert result["explanation"] == "Valid"

    def test_session_level(self):
        @custom_code_based_evaluator()
        def handler(inp, context):
            return EvaluatorOutput(value=0.5, label="Partial")

        event = _make_event(level="SESSION")
        result = handler(event)

        assert result["value"] == 0.5
        assert result["label"] == "Partial"
        assert result["explanation"] is None

    def test_tool_call_level(self):
        @custom_code_based_evaluator()
        def handler(inp, context):
            return EvaluatorOutput(value=0.0, label="Fail")

        event = _make_event(level="TOOL_CALL", span_ids=["span1"])
        result = handler(event)

        assert result["value"] == 0.0
        assert result["label"] == "Fail"

    def test_default_schema_version(self):
        @custom_code_based_evaluator()
        def handler(inp, context):
            assert inp.schema_version == "1.0"
            return EvaluatorOutput(value=1.0, label="Pass")

        event = _make_event()
        del event["schemaVersion"]
        handler(event)


class TestExceptionPropagation:
    def test_exception_propagates(self):
        @custom_code_based_evaluator()
        def handler(inp, context):
            raise RuntimeError("boom")

        event = _make_event()
        with pytest.raises(RuntimeError, match="boom"):
            handler(event)


class TestFunctoolsWraps:
    def test_preserves_name(self):
        @custom_code_based_evaluator()
        def my_evaluator(inp, context):
            return EvaluatorOutput(value=1.0, label="Pass")

        assert my_evaluator.__name__ == "my_evaluator"

    def test_preserves_module(self):
        @custom_code_based_evaluator()
        def my_evaluator(inp, context):
            return EvaluatorOutput(value=1.0, label="Pass")

        assert my_evaluator.__module__ == __name__


class TestContextPassthrough:
    def test_context_passed_to_function(self):
        received_context = []

        @custom_code_based_evaluator()
        def handler(inp, context):
            received_context.append(context)
            return EvaluatorOutput(value=1.0, label="Pass")

        mock_context = {"function_name": "my-lambda"}
        event = _make_event()
        handler(event, mock_context)

        assert received_context == [mock_context]

    def test_context_defaults_to_none(self):
        received_context = []

        @custom_code_based_evaluator()
        def handler(inp, context):
            received_context.append(context)
            return EvaluatorOutput(value=1.0, label="Pass")

        event = _make_event()
        handler(event)

        assert received_context == [None]


class TestReturnTypeValidation:
    def test_rejects_dict_return(self):
        @custom_code_based_evaluator()
        def handler(inp, context):
            return {"value": 1.0, "label": "Pass"}

        event = _make_event()
        with pytest.raises(TypeError, match="Evaluator must return an EvaluatorOutput, got dict"):
            handler(event)

    def test_rejects_none_return(self):
        @custom_code_based_evaluator()
        def handler(inp, context):
            return None

        event = _make_event()
        with pytest.raises(TypeError, match="Evaluator must return an EvaluatorOutput, got NoneType"):
            handler(event)


class TestUnwrapped:
    def test_unwrapped_returns_evaluator_output(self):
        @custom_code_based_evaluator()
        def handler(inp, context):
            return EvaluatorOutput(value=1.0, label="Pass")

        inp = EvaluatorInput(
            evaluation_level="TRACE",
            session_spans=[],
            target_trace_id="t1",
            target_span_id=None,
            schema_version="1.0",
        )
        result = handler.unwrapped(inp, None)
        assert isinstance(result, EvaluatorOutput)
        assert result.value == 1.0
        assert result.label == "Pass"

    def test_unwrapped_is_original_function(self):
        def my_eval(inp, context):
            return EvaluatorOutput(value=0.5, label="Partial")

        handler = custom_code_based_evaluator()(my_eval)
        assert handler.unwrapped is my_eval
