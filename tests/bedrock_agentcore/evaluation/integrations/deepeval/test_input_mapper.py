"""Tests for deepeval input_mapper module."""

import json
from unittest.mock import MagicMock

import pytest
from deepeval.test_case import LLMTestCaseParams

from bedrock_agentcore.evaluation.integrations.deepeval.input_mapper import (
    ParsedEvaluationEvent,
    _extract_fields_from_spans,
    _get_required_params,
    build_test_case,
)


def _make_log_record(
    input_messages=None,
    output_messages=None,
    trace_id=None,
):
    """Build a single log record dict."""
    record = {"body": {}}
    if input_messages is not None:
        record["body"]["input"] = {"messages": input_messages}
    if output_messages is not None:
        record["body"]["output"] = {"messages": output_messages}
    if trace_id is not None:
        record["traceId"] = trace_id
    return record


def _make_span_with_log_records(log_records, span_id="span1", as_json_string=True):
    """Build a span dict with _eval_log_records in attributes."""
    value = json.dumps(log_records) if as_json_string else log_records
    return {
        "traceId": "abc123",
        "spanId": span_id,
        "attributes": {"_eval_log_records": value},
    }


def _make_event(
    level="TRACE",
    trace_ids=None,
    span_ids=None,
    spans=None,
    reference_inputs=None,
):
    """Build a raw Lambda event dict for testing."""
    if spans is None:
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "What is the capital of France?"}],
                output_messages=[{"role": "assistant", "content": "The capital of France is Paris."}],
            )
        ]
        spans = [_make_span_with_log_records(log_records)]

    event = {
        "schemaVersion": "1.0",
        "evaluationLevel": level,
        "evaluationInput": {"sessionSpans": spans},
        "evaluationTarget": {},
    }
    if trace_ids is not None:
        event["evaluationTarget"]["traceIds"] = trace_ids
    if span_ids is not None:
        event["evaluationTarget"]["spanIds"] = span_ids
    if reference_inputs is not None:
        event["evaluationReferenceInputs"] = reference_inputs
    return event


def _mock_metric(name="MockMetric", required_params=None, evaluation_params=None, threshold=0.5):
    """Create a mock DeepEval metric."""
    metric = MagicMock()
    type(metric).__name__ = name
    metric.threshold = threshold

    if required_params is not None:
        metric._required_params = required_params
    else:
        del metric._required_params

    if evaluation_params is not None:
        metric.evaluation_params = evaluation_params
    else:
        del metric.evaluation_params

    return metric


class TestParsedEvaluationEvent:
    def test_from_lambda_event_trace_level(self):
        event = _make_event(level="TRACE", trace_ids=["trace-1"])
        parsed = ParsedEvaluationEvent.from_lambda_event(event)

        assert parsed.evaluation_level == "TRACE"
        assert parsed.target_trace_id == "trace-1"
        assert parsed.target_span_id is None
        assert len(parsed.session_spans) == 1

    def test_from_lambda_event_tool_call_level(self):
        event = _make_event(level="TOOL_CALL", span_ids=["span-42"])
        parsed = ParsedEvaluationEvent.from_lambda_event(event)

        assert parsed.evaluation_level == "TOOL_CALL"
        assert parsed.target_span_id == "span-42"
        assert parsed.target_trace_id is None

    def test_from_lambda_event_session_level(self):
        event = _make_event(level="SESSION")
        parsed = ParsedEvaluationEvent.from_lambda_event(event)

        assert parsed.evaluation_level == "SESSION"
        assert parsed.target_trace_id is None
        assert parsed.target_span_id is None

    def test_from_lambda_event_with_reference_inputs(self):
        refs = [{"expectedResponse": "Paris is the capital of France."}]
        event = _make_event(reference_inputs=refs)
        parsed = ParsedEvaluationEvent.from_lambda_event(event)

        assert parsed.reference_inputs == refs

    def test_from_lambda_event_missing_reference_inputs(self):
        event = _make_event()
        parsed = ParsedEvaluationEvent.from_lambda_event(event)

        assert parsed.reference_inputs == []

    def test_from_lambda_event_missing_evaluation_level_raises(self):
        event = _make_event()
        del event["evaluationLevel"]

        with pytest.raises(KeyError):
            ParsedEvaluationEvent.from_lambda_event(event)

    def test_from_lambda_event_missing_evaluation_input_raises(self):
        event = _make_event()
        del event["evaluationInput"]

        with pytest.raises(KeyError):
            ParsedEvaluationEvent.from_lambda_event(event)

    def test_from_lambda_event_missing_target_key_defaults(self):
        event = _make_event()
        del event["evaluationTarget"]
        parsed = ParsedEvaluationEvent.from_lambda_event(event)

        assert parsed.target_trace_id is None
        assert parsed.target_span_id is None


class TestGetRequiredParams:
    def test_uses_required_params_attribute(self):
        metric = _mock_metric(
            required_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        )
        result = _get_required_params(metric)

        assert result == ["input", "actual_output"]

    def test_falls_back_to_static_registry(self):
        metric = _mock_metric(name="FaithfulnessMetric")
        result = _get_required_params(metric)

        assert result == ["input", "actual_output", "retrieval_context"]

    def test_falls_back_to_evaluation_params(self):
        metric = _mock_metric(
            name="UnknownMetric",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        )
        result = _get_required_params(metric)

        assert result == ["input", "retrieval_context"]

    def test_defaults_to_input_and_actual_output(self):
        metric = _mock_metric(name="UnknownMetric")
        result = _get_required_params(metric)

        assert result == ["input", "actual_output"]

    def test_empty_required_params_falls_through(self):
        metric = _mock_metric(name="UnknownMetric", required_params=[])
        result = _get_required_params(metric)

        assert result == ["input", "actual_output"]


class TestExtractFieldsFromSpans:
    def test_basic_extraction(self):
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "hello"}],
                output_messages=[{"role": "assistant", "content": "world"}],
            )
        ]
        spans = [_make_span_with_log_records(log_records)]
        parsed = ParsedEvaluationEvent(
            evaluation_level="TRACE", session_spans=spans
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields["input"] == "hello"
        assert fields["actual_output"] == "world"

    def test_tool_messages_become_retrieval_context(self):
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "query"}],
                output_messages=[
                    {"role": "tool", "content": "doc chunk 1"},
                    {"role": "tool", "content": "doc chunk 2"},
                    {"role": "assistant", "content": "answer"},
                ],
            )
        ]
        spans = [_make_span_with_log_records(log_records)]
        parsed = ParsedEvaluationEvent(
            evaluation_level="TRACE", session_spans=spans
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields["retrieval_context"] == ["doc chunk 1", "doc chunk 2"]
        assert fields["actual_output"] == "answer"

    def test_tool_messages_also_set_context_for_hallucination_metric(self):
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "query"}],
                output_messages=[
                    {"role": "tool", "content": "context chunk"},
                    {"role": "assistant", "content": "answer"},
                ],
            )
        ]
        spans = [_make_span_with_log_records(log_records)]
        parsed = ParsedEvaluationEvent(
            evaluation_level="TRACE", session_spans=spans
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields["context"] == ["context chunk"]
        assert fields["context"] == fields["retrieval_context"]

    def test_message_content_as_dict_with_content_key(self):
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "content": {"content": "nested content"}}],
                output_messages=[{"role": "assistant", "content": {"content": "nested output"}}],
            )
        ]
        spans = [_make_span_with_log_records(log_records)]
        parsed = ParsedEvaluationEvent(
            evaluation_level="TRACE", session_spans=spans
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields["input"] == "nested content"
        assert fields["actual_output"] == "nested output"

    def test_message_content_as_dict_with_message_key(self):
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "message": "msg key input"}],
                output_messages=[{"role": "assistant", "message": "msg key output"}],
            )
        ]
        spans = [_make_span_with_log_records(log_records)]
        parsed = ParsedEvaluationEvent(
            evaluation_level="TRACE", session_spans=spans
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields["input"] == "msg key input"
        assert fields["actual_output"] == "msg key output"

    def test_message_content_as_plain_string_in_content_field(self):
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "plain string"}],
                output_messages=[{"role": "assistant", "content": "plain response"}],
            )
        ]
        spans = [_make_span_with_log_records(log_records)]
        parsed = ParsedEvaluationEvent(
            evaluation_level="TRACE", session_spans=spans
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields["input"] == "plain string"
        assert fields["actual_output"] == "plain response"

    def test_target_trace_id_filters_records(self):
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "relevant"}],
                output_messages=[{"role": "assistant", "content": "relevant answer"}],
                trace_id="target-trace",
            ),
            _make_log_record(
                input_messages=[{"role": "user", "content": "irrelevant"}],
                output_messages=[{"role": "assistant", "content": "irrelevant answer"}],
                trace_id="other-trace",
            ),
        ]
        spans = [_make_span_with_log_records(log_records)]
        parsed = ParsedEvaluationEvent(
            evaluation_level="TRACE",
            session_spans=spans,
            target_trace_id="target-trace",
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields["input"] == "relevant"
        assert fields["actual_output"] == "relevant answer"

    def test_no_target_trace_id_includes_all_records(self):
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "first"}],
                output_messages=[{"role": "assistant", "content": "first answer"}],
                trace_id="trace-1",
            ),
            _make_log_record(
                input_messages=[{"role": "user", "content": "second"}],
                output_messages=[{"role": "assistant", "content": "second answer"}],
                trace_id="trace-2",
            ),
        ]
        spans = [_make_span_with_log_records(log_records)]
        parsed = ParsedEvaluationEvent(
            evaluation_level="SESSION", session_spans=spans
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields["input"] == "first\nsecond"
        assert fields["actual_output"] == "first answer\nsecond answer"

    def test_log_records_as_parsed_list(self):
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "from list"}],
                output_messages=[{"role": "assistant", "content": "from list answer"}],
            )
        ]
        spans = [_make_span_with_log_records(log_records, as_json_string=False)]
        parsed = ParsedEvaluationEvent(
            evaluation_level="TRACE", session_spans=spans
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields["input"] == "from list"
        assert fields["actual_output"] == "from list answer"

    def test_invalid_json_log_records_skipped(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"_eval_log_records": "not valid json{{{"},
            }
        ]
        parsed = ParsedEvaluationEvent(
            evaluation_level="TRACE", session_spans=spans
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields == {}

    def test_span_without_log_records_skipped(self):
        spans = [{"traceId": "t1", "spanId": "s1", "attributes": {}}]
        parsed = ParsedEvaluationEvent(
            evaluation_level="TRACE", session_spans=spans
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields == {}

    def test_multiple_spans_aggregated(self):
        log_records_1 = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "q1"}],
                output_messages=[{"role": "assistant", "content": "a1"}],
            )
        ]
        log_records_2 = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "q2"}],
                output_messages=[{"role": "assistant", "content": "a2"}],
            )
        ]
        spans = [
            _make_span_with_log_records(log_records_1, span_id="s1"),
            _make_span_with_log_records(log_records_2, span_id="s2"),
        ]
        parsed = ParsedEvaluationEvent(
            evaluation_level="SESSION", session_spans=spans
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields["input"] == "q1\nq2"
        assert fields["actual_output"] == "a1\na2"

    def test_reference_inputs_expected_output(self):
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "q"}],
                output_messages=[{"role": "assistant", "content": "a"}],
            )
        ]
        spans = [_make_span_with_log_records(log_records)]
        parsed = ParsedEvaluationEvent(
            evaluation_level="TRACE",
            session_spans=spans,
            reference_inputs=[{"expectedResponse": "expected answer"}],
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields["expected_output"] == "expected answer"

    def test_record_without_matching_trace_id_key_included(self):
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "no trace id record"}],
                output_messages=[{"role": "assistant", "content": "response"}],
            ),
        ]
        spans = [_make_span_with_log_records(log_records)]
        parsed = ParsedEvaluationEvent(
            evaluation_level="TRACE",
            session_spans=spans,
            target_trace_id="target-trace",
        )

        fields = _extract_fields_from_spans(parsed)

        assert fields["input"] == "no trace id record"


class TestBuildTestCase:
    def test_basic_span_extraction(self):
        event = _make_event()
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="AnswerRelevancyMetric")

        test_case = build_test_case(parsed, metric)

        assert test_case.input == "What is the capital of France?"
        assert test_case.actual_output == "The capital of France is Paris."

    def test_retrieval_context_from_tool_messages(self):
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "query"}],
                output_messages=[
                    {"role": "tool", "content": "doc chunk 1"},
                    {"role": "tool", "content": "doc chunk 2"},
                    {"role": "assistant", "content": "answer"},
                ],
            )
        ]
        spans = [_make_span_with_log_records(log_records)]
        event = _make_event(spans=spans)
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="FaithfulnessMetric")

        test_case = build_test_case(parsed, metric)

        assert test_case.input == "query"
        assert test_case.actual_output == "answer"
        assert test_case.retrieval_context == ["doc chunk 1", "doc chunk 2"]

    def test_expected_output_from_reference_inputs(self):
        refs = [{"expectedResponse": "Paris"}]
        event = _make_event(reference_inputs=refs)
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="AnswerRelevancyMetric")

        test_case = build_test_case(parsed, metric)

        assert test_case.expected_output == "Paris"

    def test_missing_required_field_raises_value_error(self):
        log_records = [
            _make_log_record(
                input_messages=[{"role": "user", "content": "query"}],
                output_messages=[{"role": "assistant", "content": "answer"}],
            )
        ]
        spans = [_make_span_with_log_records(log_records)]
        event = _make_event(spans=spans)
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="FaithfulnessMetric")

        with pytest.raises(ValueError, match="retrieval_context"):
            build_test_case(parsed, metric)

    def test_custom_field_mapper_bypasses_extraction(self):
        event = _make_event()
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="AnswerRelevancyMetric")

        def custom_mapper(raw_event):
            return {
                "input": "custom input",
                "actual_output": "custom output",
            }

        test_case = build_test_case(parsed, metric, field_mapper=custom_mapper)

        assert test_case.input == "custom input"
        assert test_case.actual_output == "custom output"

    def test_field_mapper_receives_reconstructed_event(self):
        refs = [{"expectedResponse": "expected"}]
        event = _make_event(level="TRACE", trace_ids=["t1"], reference_inputs=refs)
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="AnswerRelevancyMetric")

        received_events = []

        def capture_mapper(raw_event):
            received_events.append(raw_event)
            return {"input": "x", "actual_output": "y"}

        build_test_case(parsed, metric, field_mapper=capture_mapper)

        raw = received_events[0]
        assert raw["evaluationLevel"] == "TRACE"
        assert raw["evaluationTarget"]["traceIds"] == ["t1"]
        assert raw["evaluationReferenceInputs"] == refs

    def test_multiple_user_messages_concatenated(self):
        log_records = [
            _make_log_record(
                input_messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "user", "content": "world"},
                ],
                output_messages=[{"role": "assistant", "content": "hi"}],
            )
        ]
        spans = [_make_span_with_log_records(log_records)]
        event = _make_event(spans=spans)
        parsed = ParsedEvaluationEvent.from_lambda_event(event)
        metric = _mock_metric(name="AnswerRelevancyMetric")

        test_case = build_test_case(parsed, metric)

        assert test_case.input == "hello\nworld"
