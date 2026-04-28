"""Unit tests for BatchEvaluationRunner."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from bedrock_agentcore.evaluation.runner.batch.batch_evaluation_models import (
    BatchEvaluationRunConfig,
    BatchEvaluatorConfig,
    CloudWatchOutputDataConfig,
    CloudWatchDataSourceConfig,
)
from bedrock_agentcore.evaluation.runner.batch.batch_evaluation_runner import (
    BatchEvaluationRunner,
)
from bedrock_agentcore.evaluation.runner.dataset_types import Dataset, PredefinedScenario, Turn
from bedrock_agentcore.evaluation.runner.invoker_types import AgentInvokerInput, AgentInvokerOutput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_T0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2024, 1, 1, 0, 1, 0, tzinfo=timezone.utc)

_SCENARIO = PredefinedScenario(
    scenario_id="s1",
    turns=[Turn(input="hello", expected_response="world")],
    assertions=["Be helpful"],
)

_DATASET = Dataset(scenarios=[_SCENARIO])


def _make_invoker():
    def invoker(inp: AgentInvokerInput) -> AgentInvokerOutput:
        return AgentInvokerOutput(agent_output="ok")

    return invoker


def _make_cw_source():
    return CloudWatchDataSourceConfig(
        service_names=["my-service"],
        log_group_names=["/aws/my-log-group"],
        ingestion_delay_seconds=0,
    )


def _make_config(source=None):
    return BatchEvaluationRunConfig(
        batch_evaluation_name="test-eval",
        evaluator_config=BatchEvaluatorConfig(evaluator_ids=["Builtin.Helpfulness"]),
        data_source=source or _make_cw_source(),
        max_concurrent_scenarios=2,
        polling_timeout_seconds=60,
        polling_interval_seconds=5,
    )


def _make_start_response(batch_evaluation_id="eval-001"):
    return {
        "batchEvaluationId": batch_evaluation_id,
        "batchEvaluationArn": f"arn:aws:bedrock-agentcore:us-west-2:123:batch-evaluation/{batch_evaluation_id}",
        "batchEvaluationName": "test-eval",
        "status": "PENDING",
        "createdAt": _T0,
    }


def _make_completed_response(batch_evaluation_id="eval-001", with_output_data=False, with_eval_results=False):
    response = {
        "batchEvaluationId": batch_evaluation_id,
        "batchEvaluationArn": f"arn:aws:bedrock-agentcore:us-west-2:123:batch-evaluation/{batch_evaluation_id}",
        "batchEvaluationName": "test-eval",
        "status": "COMPLETED",
        "createdAt": _T0,
    }
    if with_output_data:
        response["outputConfig"] = {
            "cloudWatchConfig": {
                "logGroupName": "/aws/agentcore/evaluation/results",
                "logStreamName": "batch-eval-001",
            }
        }
    if with_eval_results:
        response["evaluationResults"] = {
            "numberOfSessionsCompleted": 1,
            "numberOfSessionsInProgress": 0,
            "numberOfSessionsFailed": 0,
            "totalNumberOfSessions": 1,
            "evaluatorSummaries": [
                {
                    "evaluatorId": "Builtin.Helpfulness",
                    "statistics": {
                        "averageScore": 0.9,
                    },
                    "totalEvaluated": 1,
                    "totalFailed": 0,
                }
            ],
        }
    return response


# ---------------------------------------------------------------------------
# Helpers: patch runner internals
# ---------------------------------------------------------------------------


def _make_runner():
    with patch.object(BatchEvaluationRunner, "__init__", lambda self, **kw: None):
        runner = BatchEvaluationRunner.__new__(BatchEvaluationRunner)
    runner.region = "us-east-1"
    runner.data_plane_client = MagicMock()
    runner._logs_client = MagicMock()
    return runner


# ---------------------------------------------------------------------------
# BatchEvaluationRunConfig validators (#13)
# ---------------------------------------------------------------------------


def test_run_config_polling_timeout_must_exceed_interval():
    # Equal counts as invalid (validator uses <=)
    with pytest.raises(ValidationError, match="polling_timeout_seconds"):
        BatchEvaluationRunConfig(
            batch_evaluation_name="x",
            evaluator_config=BatchEvaluatorConfig(evaluator_ids=["Builtin.Helpfulness"]),
            data_source=_make_cw_source(),
            polling_timeout_seconds=5,
            polling_interval_seconds=5,
        )


def test_run_config_max_concurrent_must_be_positive():
    with pytest.raises(ValidationError, match="max_concurrent_scenarios"):
        BatchEvaluationRunConfig(
            batch_evaluation_name="x",
            evaluator_config=BatchEvaluatorConfig(evaluator_ids=["Builtin.Helpfulness"]),
            data_source=_make_cw_source(),
            max_concurrent_scenarios=0,
            polling_timeout_seconds=60,
            polling_interval_seconds=5,
        )


# ---------------------------------------------------------------------------
# CloudWatchDataSourceConfig.pre_evaluation_run_hook (#12)
# ---------------------------------------------------------------------------


@patch("time.sleep")
def test_pre_evaluation_run_hook_sleeps_when_delay_nonzero(mock_sleep):
    source = CloudWatchDataSourceConfig(
        service_names=["svc"],
        log_group_names=["/aws/logs"],
        ingestion_delay_seconds=30,
    )
    source.pre_evaluation_run_hook()
    mock_sleep.assert_called_once_with(30)


@patch("time.sleep")
def test_pre_evaluation_run_hook_skips_sleep_when_delay_zero(mock_sleep):
    source = CloudWatchDataSourceConfig(
        service_names=["svc"],
        log_group_names=["/aws/logs"],
        ingestion_delay_seconds=0,
    )
    source.pre_evaluation_run_hook()
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# BatchEvaluationRunner.run_dataset_evaluation() (#11)
# ---------------------------------------------------------------------------


def test_run_empty_dataset_raises():
    runner = _make_runner()
    # Bypass Dataset validator by using a MagicMock with empty scenarios
    empty_dataset = MagicMock()
    empty_dataset.scenarios = []
    with pytest.raises(ValueError, match="at least one scenario"):
        runner.run_dataset_evaluation(config=_make_config(), dataset=empty_dataset, agent_invoker=_make_invoker())


def test_run_all_scenarios_fail_raises():
    runner = _make_runner()
    with patch.object(
        runner,
        "_execute_scenarios_parallel",
        return_value=(
            [],
            [
                MagicMock(scenario_id="s1"),
            ],
        ),
    ):
        with pytest.raises(ValueError, match="All 1 scenarios failed"):
            runner.run_dataset_evaluation(config=_make_config(), dataset=_DATASET, agent_invoker=_make_invoker())


def test_run_happy_path_returns_result():
    runner = _make_runner()
    runner.data_plane_client.start_batch_evaluation.return_value = _make_start_response()
    runner.data_plane_client.get_batch_evaluation.return_value = _make_completed_response()

    with patch.object(
        runner,
        "_execute_scenarios_parallel",
        return_value=(
            [
                MagicMock(
                    scenario_id="s1",
                    session_id="s1-session-abc",
                    start_time=_T0,
                    end_time=_T1,
                    ground_truth=None,
                )
            ],
            [],
        ),
    ):
        result = runner.run_dataset_evaluation(config=_make_config(), dataset=_DATASET, agent_invoker=_make_invoker())

    assert result.batch_evaluation_id == "eval-001"
    assert result.status == "COMPLETED"
    assert result.agent_invocation_failures == []


def test_run_partial_failure_included_in_result():
    from bedrock_agentcore.evaluation.runner.batch.batch_evaluation_models import FailedScenario

    runner = _make_runner()
    runner.data_plane_client.start_batch_evaluation.return_value = _make_start_response()
    runner.data_plane_client.get_batch_evaluation.return_value = _make_completed_response()

    failed = FailedScenario(scenario_id="s2", error_message="timeout")

    with patch.object(
        runner,
        "_execute_scenarios_parallel",
        return_value=(
            [MagicMock(scenario_id="s1", session_id="s1-session-abc", start_time=_T0, end_time=_T1, ground_truth=None)],
            [failed],
        ),
    ):
        result = runner.run_dataset_evaluation(config=_make_config(), dataset=_DATASET, agent_invoker=_make_invoker())

    assert len(result.agent_invocation_failures) == 1
    assert result.agent_invocation_failures[0].scenario_id == "s2"


def test_run_parses_output_data_config():
    runner = _make_runner()
    runner.data_plane_client.start_batch_evaluation.return_value = _make_start_response()
    runner.data_plane_client.get_batch_evaluation.return_value = _make_completed_response(with_output_data=True)

    with patch.object(
        runner,
        "_execute_scenarios_parallel",
        return_value=(
            [MagicMock(scenario_id="s1", session_id="s1-session-abc", start_time=_T0, end_time=_T1, ground_truth=None)],
            [],
        ),
    ):
        result = runner.run_dataset_evaluation(config=_make_config(), dataset=_DATASET, agent_invoker=_make_invoker())

    assert result.output_data_config is not None
    assert result.output_data_config.log_group_name == "/aws/agentcore/evaluation/results"
    assert result.output_data_config.log_stream_name == "batch-eval-001"


def test_run_parses_evaluation_results():
    runner = _make_runner()
    runner.data_plane_client.start_batch_evaluation.return_value = _make_start_response()
    runner.data_plane_client.get_batch_evaluation.return_value = _make_completed_response(with_eval_results=True)

    with patch.object(
        runner,
        "_execute_scenarios_parallel",
        return_value=(
            [MagicMock(scenario_id="s1", session_id="s1-session-abc", start_time=_T0, end_time=_T1, ground_truth=None)],
            [],
        ),
    ):
        result = runner.run_dataset_evaluation(config=_make_config(), dataset=_DATASET, agent_invoker=_make_invoker())

    assert result.evaluation_results is not None
    assert result.evaluation_results.number_of_sessions_completed == 1
    assert result.evaluation_results.total_number_of_sessions == 1
    assert len(result.evaluation_results.evaluator_summaries) == 1
    summary = result.evaluation_results.evaluator_summaries[0]
    assert summary.evaluator_id == "Builtin.Helpfulness"
    assert summary.statistics.average_score == 0.9


def test_run_start_batch_evaluation_failure_raises_runtime_error():
    runner = _make_runner()
    runner.data_plane_client.start_batch_evaluation.side_effect = Exception("service unavailable")

    with patch.object(
        runner,
        "_execute_scenarios_parallel",
        return_value=(
            [MagicMock(scenario_id="s1", session_id="s1-session-abc", start_time=_T0, end_time=_T1, ground_truth=None)],
            [],
        ),
    ):
        with pytest.raises(RuntimeError, match="StartBatchEvaluation failed"):
            runner.run_dataset_evaluation(config=_make_config(), dataset=_DATASET, agent_invoker=_make_invoker())


def test_run_polling_failed_status_raises_runtime_error():
    runner = _make_runner()
    runner.data_plane_client.start_batch_evaluation.return_value = _make_start_response()
    runner.data_plane_client.get_batch_evaluation.return_value = {
        **_make_completed_response(),
        "status": "FAILED",
        "errorDetails": ["Internal error"],
    }

    with patch.object(
        runner,
        "_execute_scenarios_parallel",
        return_value=(
            [MagicMock(scenario_id="s1", session_id="s1-session-abc", start_time=_T0, end_time=_T1, ground_truth=None)],
            [],
        ),
    ):
        with pytest.raises(RuntimeError, match="failed"):
            runner.run_dataset_evaluation(config=_make_config(), dataset=_DATASET, agent_invoker=_make_invoker())


def test_run_polling_stopped_status_raises_runtime_error():
    runner = _make_runner()
    runner.data_plane_client.start_batch_evaluation.return_value = _make_start_response()
    runner.data_plane_client.get_batch_evaluation.return_value = {
        **_make_completed_response(),
        "status": "STOPPED",
    }

    with patch.object(
        runner,
        "_execute_scenarios_parallel",
        return_value=(
            [MagicMock(scenario_id="s1", session_id="s1-session-abc", start_time=_T0, end_time=_T1, ground_truth=None)],
            [],
        ),
    ):
        with pytest.raises(RuntimeError, match="stopped"):
            runner.run_dataset_evaluation(config=_make_config(), dataset=_DATASET, agent_invoker=_make_invoker())


def test_poll_for_results_timeout_raises_timeout_error():
    # Test _poll_for_results directly to avoid logger.info calls in run()
    # consuming time.time() values before start_time is captured.
    runner = _make_runner()
    runner.data_plane_client.get_batch_evaluation.return_value = {
        **_make_completed_response(),
        "status": "IN_PROGRESS",
    }

    _call_count = 0

    def _time():
        nonlocal _call_count
        _call_count += 1
        return 0.0 if _call_count == 1 else 10.0

    with patch("bedrock_agentcore.evaluation.runner.batch.batch_evaluation_runner.time.time", side_effect=_time):
        with patch("bedrock_agentcore.evaluation.runner.batch.batch_evaluation_runner.time.sleep"):
            with pytest.raises(TimeoutError, match="Polling timeout exceeded"):
                runner._poll_for_results("eval-001", timeout=6, poll_interval=5)


def test_run_cloudwatch_source_passes_session_ids():
    """Runner passes session_ids to CloudWatchDataSourceConfig."""
    runner = _make_runner()
    runner.data_plane_client.start_batch_evaluation.return_value = _make_start_response()
    runner.data_plane_client.get_batch_evaluation.return_value = _make_completed_response()

    with patch.object(
        runner,
        "_execute_scenarios_parallel",
        return_value=(
            [MagicMock(scenario_id="s1", session_id="s1-session-abc", start_time=_T0, end_time=_T1, ground_truth=None)],
            [],
        ),
    ):
        runner.run_dataset_evaluation(
            config=_make_config(source=_make_cw_source()), dataset=_DATASET, agent_invoker=_make_invoker()
        )

    call_kwargs = runner.data_plane_client.start_batch_evaluation.call_args.kwargs
    filter_config = call_kwargs["dataSourceConfig"]["cloudWatchLogs"]["filterConfig"]
    assert filter_config["sessionIds"] == ["s1-session-abc"]


# ---------------------------------------------------------------------------
# fetch_evaluation_events (#11)
# ---------------------------------------------------------------------------


def test_fetch_evaluation_events_raises_when_no_output_data_config():
    runner = _make_runner()
    result = MagicMock()
    result.output_data_config = None
    result.batch_evaluation_id = "eval-001"

    with pytest.raises(ValueError, match="No output_data_config"):
        runner.fetch_evaluation_events(result)


def test_fetch_evaluation_events_returns_parsed_events():
    runner = _make_runner()
    result = MagicMock()
    result.output_data_config = CloudWatchOutputDataConfig(
        log_group_name="/aws/agentcore/evaluation/results",
        log_stream_name="batch-eval-abc123",
    )
    runner._logs_client.get_log_events.side_effect = [
        {"events": [{"message": '{"traceId": "t1"}'}], "nextForwardToken": "tok-1"},
        {"events": [], "nextForwardToken": "tok-1"},
    ]

    events = runner.fetch_evaluation_events(result)

    assert events == [{"traceId": "t1"}]


# ---------------------------------------------------------------------------
# _transform_ground_truth
# ---------------------------------------------------------------------------


def test_transform_ground_truth_returns_none_when_no_fields():
    # Use base Scenario (no turns/expected_trajectory fields) with no assertions
    from bedrock_agentcore.evaluation.runner.dataset_types import Scenario

    runner = _make_runner()
    scenario = Scenario(scenario_id="s1")
    assert runner._transform_ground_truth(scenario) is None


def test_transform_ground_truth_assertions_only():
    from bedrock_agentcore.evaluation.runner.dataset_types import Scenario

    runner = _make_runner()
    scenario = Scenario(scenario_id="s1", assertions=["Be concise", "Be helpful"])
    gt = runner._transform_ground_truth(scenario)
    assert gt == {"assertions": [{"text": "Be concise"}, {"text": "Be helpful"}]}


def test_transform_ground_truth_expected_trajectory():
    runner = _make_runner()
    scenario = PredefinedScenario(
        scenario_id="s1",
        turns=[Turn(input="hi")],
        expected_trajectory=["tool_a", "tool_b"],
    )
    gt = runner._transform_ground_truth(scenario)
    assert gt["expectedTrajectory"] == {"toolNames": ["tool_a", "tool_b"]}


def test_transform_ground_truth_turns_with_expected_response():
    runner = _make_runner()
    scenario = PredefinedScenario(
        scenario_id="s1",
        turns=[Turn(input="hi", expected_response="hello")],
    )
    gt = runner._transform_ground_truth(scenario)
    assert gt["turns"] == [{"input": {"prompt": "hi"}, "expectedResponse": {"text": "hello"}}]


def test_transform_ground_truth_turns_without_expected_response():
    runner = _make_runner()
    scenario = PredefinedScenario(
        scenario_id="s1",
        turns=[Turn(input="hi")],
    )
    gt = runner._transform_ground_truth(scenario)
    assert gt["turns"] == [{"input": {"prompt": "hi"}}]
    assert "expectedResponse" not in gt["turns"][0]


def test_transform_ground_truth_all_fields():
    runner = _make_runner()
    scenario = PredefinedScenario(
        scenario_id="s1",
        turns=[Turn(input="hi", expected_response="hello")],
        assertions=["Be helpful"],
        expected_trajectory=["tool_a"],
    )
    gt = runner._transform_ground_truth(scenario)
    assert "assertions" in gt
    assert "expectedTrajectory" in gt
    assert "turns" in gt


# ---------------------------------------------------------------------------
# _execute_scenario and _poll_for_results edge cases
# ---------------------------------------------------------------------------


def test_execute_scenario_unsupported_type_raises_type_error():
    from bedrock_agentcore.evaluation.runner.dataset_types import Scenario

    runner = _make_runner()

    class _CustomScenario(Scenario):
        pass

    scenario = _CustomScenario(scenario_id="s1")
    with pytest.raises(TypeError, match="Unsupported scenario type"):
        runner._execute_scenario(scenario, _make_invoker())


def test_poll_for_results_get_batch_evaluation_failure_raises_runtime_error():
    runner = _make_runner()
    runner.data_plane_client.get_batch_evaluation.side_effect = Exception("network error")

    with pytest.raises(RuntimeError, match="Failed to get batch evaluation result"):
        runner._poll_for_results("eval-001", timeout=60, poll_interval=5)


def test_poll_for_results_unknown_status_raises_runtime_error():
    runner = _make_runner()
    runner.data_plane_client.get_batch_evaluation.return_value = {
        **_make_completed_response(),
        "status": "UNKNOWN_STATE",
    }

    with pytest.raises(RuntimeError, match="Unknown batch evaluation status"):
        runner._poll_for_results("eval-001", timeout=60, poll_interval=5)


# ---------------------------------------------------------------------------
# fetch_evaluation_events: CloudWatch pagination (migrated from results reader)
# ---------------------------------------------------------------------------

_CW_OUTPUT_CONFIG = CloudWatchOutputDataConfig(
    log_group_name="/aws/agentcore/evaluation/results",
    log_stream_name="batch-eval-abc123",
)


def _make_log_event(message: str) -> dict:
    return {"timestamp": 1000, "message": message, "ingestionTime": 1000}


def _make_fetch_result(output_data_config):
    result = MagicMock()
    result.output_data_config = output_data_config
    result.batch_evaluation_id = "eval-001"
    return result


def test_fetch_evaluation_events_paginates_multiple_pages():
    runner = _make_runner()
    runner._logs_client.get_log_events.side_effect = [
        {"events": [_make_log_event('{"page": 1}')], "nextForwardToken": "tok-1"},
        {"events": [_make_log_event('{"page": 2}')], "nextForwardToken": "tok-2"},
        {"events": [], "nextForwardToken": "tok-2"},
    ]

    events = runner.fetch_evaluation_events(_make_fetch_result(_CW_OUTPUT_CONFIG))

    assert len(events) == 2
    assert events[0]["page"] == 1
    assert events[1]["page"] == 2


def test_fetch_evaluation_events_skips_non_json(caplog):
    import logging

    runner = _make_runner()
    runner._logs_client.get_log_events.side_effect = [
        {
            "events": [
                _make_log_event("not json at all"),
                _make_log_event('{"traceId": "t1"}'),
            ],
            "nextForwardToken": "tok-1",
        },
        {"events": [], "nextForwardToken": "tok-1"},
    ]

    with caplog.at_level(logging.WARNING):
        events = runner.fetch_evaluation_events(_make_fetch_result(_CW_OUTPUT_CONFIG))

    assert len(events) == 1
    assert events[0]["traceId"] == "t1"
    assert "Skipping non-JSON" in caplog.text


def test_fetch_evaluation_events_empty_stream():
    runner = _make_runner()
    runner._logs_client.get_log_events.side_effect = [
        {"events": [], "nextForwardToken": "tok-0"},
        {"events": [], "nextForwardToken": "tok-0"},
    ]

    events = runner.fetch_evaluation_events(_make_fetch_result(_CW_OUTPUT_CONFIG))

    assert events == []


def test_fetch_evaluation_events_start_from_head_only_on_first_call():
    runner = _make_runner()
    runner._logs_client.get_log_events.side_effect = [
        {"events": [_make_log_event('{"page": 1}')], "nextForwardToken": "tok-1"},
        {"events": [], "nextForwardToken": "tok-1"},
    ]

    runner.fetch_evaluation_events(_make_fetch_result(_CW_OUTPUT_CONFIG))

    calls = runner._logs_client.get_log_events.call_args_list
    assert len(calls) == 2
    assert "startFromHead" in calls[0][1]
    assert "startFromHead" not in calls[1][1]
    assert calls[1][1]["nextToken"] == "tok-1"


def test_fetch_evaluation_events_raises_lookup_error_when_stream_not_found():
    runner = _make_runner()

    class ResourceNotFoundException(Exception):
        pass

    runner._logs_client.exceptions.ResourceNotFoundException = ResourceNotFoundException
    runner._logs_client.get_log_events.side_effect = ResourceNotFoundException("stream not found")

    with pytest.raises(LookupError, match="CloudWatch log stream not found"):
        runner.fetch_evaluation_events(_make_fetch_result(_CW_OUTPUT_CONFIG))
