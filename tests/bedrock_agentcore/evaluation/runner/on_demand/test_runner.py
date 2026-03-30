"""Tests for the OnDemandEvaluationDatasetRunner and related types."""

from unittest.mock import MagicMock, patch

from bedrock_agentcore.evaluation.agent_span_collector import (
    AgentSpanCollector,
    CloudWatchAgentSpanCollector,
)
from bedrock_agentcore.evaluation.runner.dataset_types import (
    Dataset,
    PredefinedScenario,
    Scenario,
    Turn,
)
from bedrock_agentcore.evaluation.runner.invoker_types import (
    AgentInvokerInput,
    AgentInvokerOutput,
)
from bedrock_agentcore.evaluation.runner.on_demand.config import (
    EvaluationRunConfig,
    EvaluatorConfig,
)
from bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner import OnDemandEvaluationDatasetRunner
from bedrock_agentcore.evaluation.runner.on_demand.result import (
    EvaluationResult,
    EvaluatorResult,
    ScenarioResult,
)
from bedrock_agentcore.evaluation.runner.scenario_executor import (
    PredefinedScenarioExecutor,
    ScenarioExecutionResult,
)

# --- Helper tests ---


# --- Config construction tests ---


class TestEvaluatorConfig:
    def test_minimal(self):
        cfg = EvaluatorConfig(evaluator_ids=["accuracy"])
        assert cfg.evaluator_ids == ["accuracy"]

    def test_multiple_evaluators(self):
        cfg = EvaluatorConfig(evaluator_ids=["accuracy", "relevance"])
        assert len(cfg.evaluator_ids) == 2


class TestEvaluationRunConfig:
    def test_minimal(self):
        cfg = EvaluationRunConfig(
            evaluator_config=EvaluatorConfig(evaluator_ids=["accuracy"]),
        )
        assert cfg.evaluation_delay_seconds == 180

    def test_custom_evaluation_delay(self):
        cfg = EvaluationRunConfig(
            evaluator_config=EvaluatorConfig(evaluator_ids=["accuracy"]),
            evaluation_delay_seconds=60,
        )
        assert cfg.evaluation_delay_seconds == 60

    def test_default_max_concurrent_scenarios(self):
        cfg = EvaluationRunConfig(
            evaluator_config=EvaluatorConfig(evaluator_ids=["accuracy"]),
        )
        assert cfg.max_concurrent_scenarios == 5

    def test_custom_max_concurrent_scenarios(self):
        cfg = EvaluationRunConfig(
            evaluator_config=EvaluatorConfig(evaluator_ids=["accuracy"]),
            max_concurrent_scenarios=10,
        )
        assert cfg.max_concurrent_scenarios == 10


# --- Result dataclass tests ---


class TestResultTypes:
    def test_evaluator_result(self):
        er = EvaluatorResult(
            evaluator_id="accuracy",
            results=[{"value": 0.85, "explanation": "Good"}],
        )
        assert er.evaluator_id == "accuracy"
        assert er.results[0]["value"] == 0.85
        assert er.results[0]["explanation"] == "Good"

    def test_evaluator_result_with_error(self):
        er = EvaluatorResult(
            evaluator_id="trajectory",
            results=[
                {
                    "errorCode": "ValidationError",
                    "errorMessage": "Expected trajectory not provided",
                    "context": {"spanContext": {"sessionId": "s1"}},
                }
            ],
        )
        assert er.results[0]["errorCode"] == "ValidationError"
        assert er.results[0]["errorMessage"] == "Expected trajectory not provided"
        assert er.results[0]["context"]["spanContext"]["sessionId"] == "s1"

    def test_scenario_result(self):
        er = [EvaluatorResult(evaluator_id="accuracy", results=[{"value": 0.9}])]
        result = ScenarioResult(scenario_id="s1", session_id="sess-1", evaluator_results=er)
        assert result.scenario_id == "s1"
        assert result.session_id == "sess-1"
        assert result.status == "COMPLETED"
        assert result.error is None
        assert len(result.evaluator_results) == 1

    def test_scenario_result_failed(self):
        result = ScenarioResult(scenario_id="s1", session_id="sess-1", status="FAILED", error="timeout")
        assert result.status == "FAILED"
        assert result.error == "timeout"
        assert result.evaluator_results == []

    def test_evaluation_result(self):
        sr = ScenarioResult(scenario_id="s1", session_id="sess-1")
        result = EvaluationResult(scenario_results=[sr])
        assert len(result.scenario_results) == 1


# --- AgentSpanCollector tests ---


class TestAgentSpanCollector:
    def test_cloudwatch_is_span_collector(self):
        with patch("bedrock_agentcore.evaluation.utils.cloudwatch_span_helper.boto3"):
            collector = CloudWatchAgentSpanCollector(
                log_group_name="/aws/logs/my-agent",
                region="us-east-1",
            )
            assert isinstance(collector, AgentSpanCollector)


# --- OnDemandEvaluationDatasetRunner end-to-end tests ---


class TestOnDemandEvaluationDatasetRunnerRun:
    # --- Helper method tests ---

    def test_batch_exact(self):
        assert list(OnDemandEvaluationDatasetRunner._batch(["a", "b", "c", "d"], 2)) == [["a", "b"], ["c", "d"]]

    def test_batch_remainder(self):
        assert list(OnDemandEvaluationDatasetRunner._batch(["a", "b", "c"], 2)) == [["a", "b"], ["c"]]

    def test_batch_single(self):
        assert list(OnDemandEvaluationDatasetRunner._batch(["a", "b"], 10)) == [["a", "b"]]

    def test_batch_empty(self):
        assert list(OnDemandEvaluationDatasetRunner._batch([], 10)) == []

    def test_is_tool_span_strands(self):
        span = {"attributes": {"gen_ai.operation.name": "execute_tool"}}
        assert OnDemandEvaluationDatasetRunner._is_tool_span(span) is True

    def test_is_tool_span_langgraph_openinference(self):
        span = {"attributes": {"openinference.span.kind": "TOOL"}}
        assert OnDemandEvaluationDatasetRunner._is_tool_span(span) is True

    def test_is_tool_span_langgraph_otel(self):
        assert OnDemandEvaluationDatasetRunner._is_tool_span({"attributes": {"traceloop.span.kind": "tool"}}) is True

    def test_is_tool_span_not_tool(self):
        assert OnDemandEvaluationDatasetRunner._is_tool_span({"attributes": {"gen_ai.operation.name": "chat"}}) is False

    def test_is_tool_span_no_attributes(self):
        assert OnDemandEvaluationDatasetRunner._is_tool_span({}) is False
        assert OnDemandEvaluationDatasetRunner._is_tool_span({"attributes": None}) is False

    # --- Helpers ---

    def _make_dataset(self):
        scenario = PredefinedScenario(
            scenario_id="scenario-1",
            turns=[
                Turn(input="Hello"),
                Turn(input="What is 2+2?"),
            ],
        )
        return Dataset(scenarios=[scenario])

    def _make_config(self, evaluator_ids=None):
        return EvaluationRunConfig(
            evaluator_config=EvaluatorConfig(evaluator_ids=evaluator_ids or ["accuracy", "relevance"]),
            evaluation_delay_seconds=0,
        )

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_run_end_to_end(self, mock_boto3):
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {"evaluationResults": [{"value": 0.85, "explanation": "Looks good"}]}
        mock_cp_client.get_evaluator.return_value = {"level": "TRACE"}

        invoker_calls = []

        def mock_invoker(inp: AgentInvokerInput) -> AgentInvokerOutput:
            invoker_calls.append(inp)
            return AgentInvokerOutput(agent_output="response")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = [{"traceId": "t1", "spanId": "s1", "scope": {}}]

        runner = OnDemandEvaluationDatasetRunner(region="us-west-2")
        result = runner.run(
            config=self._make_config(),
            dataset=self._make_dataset(),
            agent_invoker=mock_invoker,
            span_collector=mock_collector,
        )

        assert isinstance(result, EvaluationResult)
        assert len(result.scenario_results) == 1

        sr = result.scenario_results[0]
        assert sr.scenario_id == "scenario-1"
        assert sr.status == "COMPLETED"
        assert sr.error is None
        assert len(sr.evaluator_results) == 2

        for er in sr.evaluator_results:
            assert len(er.results) == 1
            assert er.results[0]["value"] == 0.85
            assert er.results[0]["explanation"] == "Looks good"

        assert len(invoker_calls) == 2

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_scenario_id_and_session_id_tracked_independently(self, mock_boto3):
        """ScenarioResult preserves scenario_id from the dataset and a framework-generated session_id."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {"evaluationResults": [{"value": 0.9, "explanation": "Good"}]}
        mock_cp_client.get_evaluator.return_value = {"level": "SESSION"}

        def mock_invoker(inp: AgentInvokerInput) -> AgentInvokerOutput:
            return AgentInvokerOutput(agent_output="response")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = [{"traceId": "t1", "spanId": "s1"}]

        config = self._make_config(evaluator_ids=["eval1"])
        dataset = Dataset(scenarios=[PredefinedScenario(scenario_id="scenario-orig", turns=[Turn(input="hi")])])

        runner = OnDemandEvaluationDatasetRunner(region="us-west-2")
        result = runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        sr = result.scenario_results[0]
        assert sr.scenario_id == "scenario-orig"
        assert sr.session_id.startswith("scenario-orig-")
        assert sr.session_id != "scenario-orig"
        assert sr.status == "COMPLETED"

        # Span collector should be called with the framework-generated session_id
        mock_collector.collect.assert_called_once()
        collect_kwargs = mock_collector.collect.call_args[1]
        assert collect_kwargs["session_id"].startswith("scenario-orig-")

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_session_level_evaluator_no_target(self, mock_boto3):
        """Session-level evaluator sends no evaluationTarget."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {"evaluationResults": [{"value": 0.9, "explanation": "Good"}]}
        mock_cp_client.get_evaluator.return_value = {"level": "SESSION"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = [{"traceId": "t1", "spanId": "s1"}]

        config = self._make_config(evaluator_ids=["Builtin.GoalSuccessRate"])
        dataset = Dataset(scenarios=[PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])])

        runner = OnDemandEvaluationDatasetRunner()
        runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        call_kwargs = mock_dp_client.evaluate.call_args[1]
        assert "evaluationTarget" not in call_kwargs

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_trace_level_evaluator_sends_trace_ids(self, mock_boto3):
        """Trace-level evaluator sends evaluationTarget.traceIds."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {"evaluationResults": [{"value": 0.9, "explanation": "Good"}]}
        mock_cp_client.get_evaluator.return_value = {"level": "TRACE"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = [
            {"traceId": "trace-aaa", "spanId": "span-1"},
            {"traceId": "trace-bbb", "spanId": "span-2"},
        ]

        config = self._make_config(evaluator_ids=["Builtin.Helpfulness"])
        dataset = Dataset(scenarios=[PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])])

        runner = OnDemandEvaluationDatasetRunner()
        runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        call_kwargs = mock_dp_client.evaluate.call_args[1]
        assert call_kwargs["evaluationTarget"] == {"traceIds": ["trace-aaa", "trace-bbb"]}

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_tool_level_evaluator_sends_span_ids(self, mock_boto3):
        """Tool-level evaluator sends evaluationTarget.spanIds for tool spans only."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {"evaluationResults": [{"value": 0.9, "explanation": "Good"}]}
        mock_cp_client.get_evaluator.return_value = {"level": "TOOL_CALL"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = [
            {"traceId": "t1", "spanId": "span-chat", "attributes": {"gen_ai.operation.name": "chat"}},
            {"traceId": "t1", "spanId": "span-tool", "attributes": {"gen_ai.operation.name": "execute_tool"}},
        ]

        config = self._make_config(evaluator_ids=["Builtin.ToolSelectionAccuracy"])
        dataset = Dataset(scenarios=[PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])])

        runner = OnDemandEvaluationDatasetRunner()
        runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        call_kwargs = mock_dp_client.evaluate.call_args[1]
        assert call_kwargs["evaluationTarget"] == {"spanIds": ["span-tool"]}

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_trace_level_batching_over_10(self, mock_boto3):
        """Trace-level evaluator with >10 traces batches into multiple API calls."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {"evaluationResults": [{"value": 0.8, "explanation": "OK"}]}
        mock_cp_client.get_evaluator.return_value = {"level": "TRACE"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        # 15 unique traces
        spans = [{"traceId": f"trace-{i:03d}", "spanId": f"span-{i}"} for i in range(15)]
        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = spans

        config = self._make_config(evaluator_ids=["Builtin.Helpfulness"])
        dataset = Dataset(scenarios=[PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])])

        runner = OnDemandEvaluationDatasetRunner()
        result = runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        # Should be 2 API calls: batch of 10 + batch of 5
        assert mock_dp_client.evaluate.call_count == 2

        first_call = mock_dp_client.evaluate.call_args_list[0][1]
        assert len(first_call["evaluationTarget"]["traceIds"]) == 10

        second_call = mock_dp_client.evaluate.call_args_list[1][1]
        assert len(second_call["evaluationTarget"]["traceIds"]) == 5

        # Results merged: 2 calls × 1 result each = 2 results under 1 evaluator
        er = result.scenario_results[0].evaluator_results
        assert len(er) == 1
        assert len(er[0].results) == 2

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_evaluator_level_cached(self, mock_boto3):
        """get_evaluator is called once per evaluator ID, cached across scenarios."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {"evaluationResults": [{"value": 1.0, "explanation": "Perfect"}]}
        mock_cp_client.get_evaluator.return_value = {"level": "SESSION"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = []

        config = self._make_config(evaluator_ids=["Builtin.GoalSuccessRate"])
        dataset = Dataset(
            scenarios=[
                PredefinedScenario(scenario_id="s1", turns=[Turn(input="a")]),
                PredefinedScenario(scenario_id="s2", turns=[Turn(input="b")]),
            ],
        )

        runner = OnDemandEvaluationDatasetRunner()
        runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        # get_evaluator called only once despite 2 scenarios
        mock_cp_client.get_evaluator.assert_called_once_with(evaluatorId="Builtin.GoalSuccessRate")

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_get_evaluator_failure_fails_scenario(self, mock_boto3):
        """If get_evaluator fails, the scenario should be marked as FAILED."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_cp_client.get_evaluator.side_effect = Exception("Service unavailable")

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = [{"traceId": "t1", "spanId": "s1"}]

        config = self._make_config(evaluator_ids=["Builtin.Helpfulness"])
        dataset = Dataset(scenarios=[PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])])

        runner = OnDemandEvaluationDatasetRunner()
        result = runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        assert result.scenario_results[0].status == "FAILED"
        assert "Service unavailable" in result.scenario_results[0].error
        mock_dp_client.evaluate.assert_not_called()

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_trace_level_no_trace_ids_skips_evaluator(self, mock_boto3):
        """Trace-level evaluator with no trace IDs logs warning and skips (no failure)."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_cp_client.get_evaluator.return_value = {"level": "TRACE"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = []  # no spans at all

        config = self._make_config(evaluator_ids=["Builtin.Helpfulness"])
        dataset = Dataset(scenarios=[PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])])

        runner = OnDemandEvaluationDatasetRunner()
        result = runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        sr = result.scenario_results[0]
        assert sr.status == "COMPLETED"
        assert sr.error is None
        mock_dp_client.evaluate.assert_not_called()

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_tool_level_no_tool_spans_skips_evaluator(self, mock_boto3):
        """Tool-level evaluator with no tool spans logs warning and skips (no failure)."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_cp_client.get_evaluator.return_value = {"level": "TOOL_CALL"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = [
            {"traceId": "t1", "spanId": "s1", "attributes": {"gen_ai.operation.name": "chat"}},
        ]  # spans exist but none are tool spans

        config = self._make_config(evaluator_ids=["Builtin.ToolSelectionAccuracy"])
        dataset = Dataset(scenarios=[PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])])

        runner = OnDemandEvaluationDatasetRunner()
        result = runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        sr = result.scenario_results[0]
        assert sr.status == "COMPLETED"
        assert sr.error is None
        mock_dp_client.evaluate.assert_not_called()

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_run_evaluator_api_error(self, mock_boto3):
        """Evaluator API exception results in score=0, error fields set."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.side_effect = Exception("Service unavailable")
        mock_cp_client.get_evaluator.return_value = {"level": "SESSION"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = []

        config = self._make_config(evaluator_ids=["accuracy"])
        dataset = Dataset(scenarios=[PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])])

        runner = OnDemandEvaluationDatasetRunner()
        result = runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        er = result.scenario_results[0].evaluator_results[0]
        assert er.evaluator_id == "accuracy"
        assert er.results[0]["errorCode"] == "SDKError"
        assert "Service unavailable" in er.results[0]["errorMessage"]

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_run_evaluator_error_response(self, mock_boto3):
        """Evaluator returns errorCode/errorMessage in response."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {
            "evaluationResults": [
                {
                    "errorCode": "ValidationError",
                    "errorMessage": "Expected trajectory not provided",
                    "context": {"spanContext": {"sessionId": "s1"}},
                }
            ]
        }
        mock_cp_client.get_evaluator.return_value = {"level": "SESSION"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = []

        config = self._make_config(evaluator_ids=["Builtin.TrajectoryInOrderMatch"])
        dataset = Dataset(scenarios=[PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])])

        runner = OnDemandEvaluationDatasetRunner()
        result = runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        er = result.scenario_results[0].evaluator_results[0]
        assert er.evaluator_id == "Builtin.TrajectoryInOrderMatch"
        assert er.results[0]["errorCode"] == "ValidationError"
        assert er.results[0]["errorMessage"] == "Expected trajectory not provided"

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_run_scenario_failure_continues(self, mock_boto3):
        """A failed scenario does not abort the run."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {"evaluationResults": [{"value": 1.0, "explanation": "Perfect"}]}
        mock_cp_client.get_evaluator.return_value = {"level": "SESSION"}

        def mock_invoker(inp):
            if inp.session_id.startswith("s1-"):
                raise ConnectionError("timeout after 30s")
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = []

        dataset = Dataset(
            scenarios=[
                PredefinedScenario(scenario_id="s1", turns=[Turn(input="a")]),
                PredefinedScenario(scenario_id="s2", turns=[Turn(input="b")]),
            ],
        )
        config = self._make_config(evaluator_ids=["eval1"])

        runner = OnDemandEvaluationDatasetRunner()
        result = runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        assert len(result.scenario_results) == 2
        assert result.scenario_results[0].status == "FAILED"
        assert "timeout after 30s" in result.scenario_results[0].error
        assert result.scenario_results[1].status == "COMPLETED"

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_multiple_results_per_api_call(self, mock_boto3):
        """API returns multiple results (one per trace) and all are captured."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {
            "evaluationResults": [
                {"value": 0.9, "explanation": "Good", "context": {"spanContext": {"sessionId": "s1", "traceId": "t1"}}},
                {"value": 0.7, "explanation": "OK", "context": {"spanContext": {"sessionId": "s1", "traceId": "t2"}}},
            ]
        }
        mock_cp_client.get_evaluator.return_value = {"level": "TRACE"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = [
            {"traceId": "t1", "spanId": "s1"},
            {"traceId": "t2", "spanId": "s2"},
        ]

        config = self._make_config(evaluator_ids=["Builtin.Helpfulness"])
        dataset = Dataset(scenarios=[PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])])

        runner = OnDemandEvaluationDatasetRunner()
        result = runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        er = result.scenario_results[0].evaluator_results[0]
        assert er.evaluator_id == "Builtin.Helpfulness"
        assert len(er.results) == 2
        assert er.results[0]["value"] == 0.9
        assert er.results[0]["context"]["spanContext"]["traceId"] == "t1"
        assert er.results[1]["value"] == 0.7
        assert er.results[1]["context"]["spanContext"]["traceId"] == "t2"

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_reference_inputs_with_full_ground_truth(self, mock_boto3):
        """evaluationReferenceInputs includes expectedResponse, expectedTrajectory, and assertion."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {"evaluationResults": [{"value": 0.9, "explanation": "Good"}]}
        mock_cp_client.get_evaluator.return_value = {"level": "TRACE"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = [
            {"traceId": "t1", "spanId": "s1", "scope": {}},
            {"traceId": "t2", "spanId": "s2", "scope": {}},
        ]

        scenario = PredefinedScenario(
            scenario_id="ref-test",
            turns=[
                Turn(input="What is BMI?", expected_response="BMI is body mass index."),
                Turn(input="Calculate my BMI", expected_response="Your BMI is 22.9."),
            ],
            expected_trajectory=["calculate_bmi"],
            assertions=["Help user understand BMI"],
        )
        config = EvaluationRunConfig(
            evaluator_config=EvaluatorConfig(evaluator_ids=["Builtin.Helpfulness"]),
            evaluation_delay_seconds=0,
        )
        dataset = Dataset(scenarios=[scenario])

        runner = OnDemandEvaluationDatasetRunner()
        runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        call_kwargs = mock_dp_client.evaluate.call_args[1]
        ref_inputs = call_kwargs["evaluationReferenceInputs"]

        # Session-level: expectedTrajectory + assertion combined
        session_ref = ref_inputs[0]
        assert session_ref["expectedTrajectory"] == {"toolNames": ["calculate_bmi"]}
        assert session_ref["assertions"] == [{"text": "Help user understand BMI"}]
        assert "traceId" not in session_ref["context"]["spanContext"]

        # Per-trace: expectedResponse for each turn
        assert ref_inputs[1]["expectedResponse"] == {"text": "BMI is body mass index."}
        assert ref_inputs[1]["context"]["spanContext"]["traceId"] == "t1"
        assert ref_inputs[2]["expectedResponse"] == {"text": "Your BMI is 22.9."}
        assert ref_inputs[2]["context"]["spanContext"]["traceId"] == "t2"

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_reference_inputs_absent_when_no_ground_truth(self, mock_boto3):
        """evaluationReferenceInputs is absent when scenario has no ground truth."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {"evaluationResults": [{"value": 0.9, "explanation": "Good"}]}
        mock_cp_client.get_evaluator.return_value = {"level": "SESSION"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = [{"traceId": "t1", "spanId": "s1", "scope": {}}]

        scenario = PredefinedScenario(
            scenario_id="no-refs",
            turns=[Turn(input="Hello")],
        )
        config = EvaluationRunConfig(
            evaluator_config=EvaluatorConfig(evaluator_ids=["eval1"]),
            evaluation_delay_seconds=0,
        )
        dataset = Dataset(scenarios=[scenario])

        runner = OnDemandEvaluationDatasetRunner()
        runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        call_kwargs = mock_dp_client.evaluate.call_args[1]
        assert "evaluationReferenceInputs" not in call_kwargs

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_reference_inputs_partial_expected_responses(self, mock_boto3):
        """Only turns with expected_response get per-trace reference inputs."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {"evaluationResults": [{"value": 0.9, "explanation": "Good"}]}
        mock_cp_client.get_evaluator.return_value = {"level": "TRACE"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = [
            {"traceId": "t1", "spanId": "s1", "scope": {}},
            {"traceId": "t2", "spanId": "s2", "scope": {}},
            {"traceId": "t3", "spanId": "s3", "scope": {}},
        ]

        scenario = PredefinedScenario(
            scenario_id="partial-refs",
            turns=[
                Turn(input="Hello", expected_response="Hi there"),
                Turn(input="How are you?"),  # no expected_response
                Turn(input="Goodbye", expected_response="See you later"),
            ],
        )
        config = EvaluationRunConfig(
            evaluator_config=EvaluatorConfig(evaluator_ids=["Builtin.Helpfulness"]),
            evaluation_delay_seconds=0,
        )
        dataset = Dataset(scenarios=[scenario])

        runner = OnDemandEvaluationDatasetRunner()
        runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        call_kwargs = mock_dp_client.evaluate.call_args[1]
        ref_inputs = call_kwargs["evaluationReferenceInputs"]

        # Only 2 per-trace refs (turns 0 and 2), no session-level ref
        assert len(ref_inputs) == 2
        assert ref_inputs[0]["context"]["spanContext"]["traceId"] == "t1"
        assert ref_inputs[0]["expectedResponse"] == {"text": "Hi there"}
        assert ref_inputs[1]["context"]["spanContext"]["traceId"] == "t3"
        assert ref_inputs[1]["expectedResponse"] == {"text": "See you later"}

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_parallel_scenarios_run_concurrently(self, mock_boto3):
        """Multiple scenarios are processed concurrently and all results returned."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {"evaluationResults": [{"value": 0.9, "explanation": "Good"}]}
        mock_cp_client.get_evaluator.return_value = {"level": "SESSION"}

        def mock_invoker(inp):
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = []

        config = EvaluationRunConfig(
            evaluator_config=EvaluatorConfig(evaluator_ids=["eval1"]),
            evaluation_delay_seconds=0,
            max_concurrent_scenarios=3,
        )
        dataset = Dataset(
            scenarios=[
                PredefinedScenario(scenario_id="s1", turns=[Turn(input="a")]),
                PredefinedScenario(scenario_id="s2", turns=[Turn(input="b")]),
                PredefinedScenario(scenario_id="s3", turns=[Turn(input="c")]),
            ]
        )

        runner = OnDemandEvaluationDatasetRunner()
        result = runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        assert len(result.scenario_results) == 3
        scenario_ids = {sr.scenario_id for sr in result.scenario_results}
        assert scenario_ids == {"s1", "s2", "s3"}
        assert all(sr.status == "COMPLETED" for sr in result.scenario_results)

    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_phase1_failure_does_not_block_others(self, mock_boto3):
        """A failed scenario in phase 1 does not prevent other scenarios from completing."""
        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        mock_dp_client.evaluate.return_value = {"evaluationResults": [{"value": 1.0, "explanation": "Perfect"}]}
        mock_cp_client.get_evaluator.return_value = {"level": "SESSION"}

        def mock_invoker(inp):
            if inp.session_id.startswith("fail-"):
                raise ConnectionError("timeout")
            return AgentInvokerOutput(agent_output="ok")

        mock_collector = MagicMock(spec=AgentSpanCollector)
        mock_collector.collect.return_value = []

        config = EvaluationRunConfig(
            evaluator_config=EvaluatorConfig(evaluator_ids=["eval1"]),
            evaluation_delay_seconds=0,
            max_concurrent_scenarios=2,
        )
        dataset = Dataset(
            scenarios=[
                PredefinedScenario(scenario_id="fail", turns=[Turn(input="a")]),
                PredefinedScenario(scenario_id="ok", turns=[Turn(input="b")]),
            ]
        )

        runner = OnDemandEvaluationDatasetRunner()
        result = runner.run(config=config, dataset=dataset, agent_invoker=mock_invoker, span_collector=mock_collector)

        assert len(result.scenario_results) == 2
        # Find results by scenario_id (order may vary with concurrency)
        results_by_id = {sr.scenario_id: sr for sr in result.scenario_results}
        assert results_by_id["fail"].status == "FAILED"
        assert "timeout" in results_by_id["fail"].error
        assert results_by_id["ok"].status == "COMPLETED"


# --- ScenarioExecutor unit tests ---


class TestPredefinedScenarioExecutor:
    def test_invokes_each_turn(self):
        scenario = PredefinedScenario(
            scenario_id="tbt-1",
            turns=[Turn(input="Hello"), Turn(input="World")],
        )
        calls = []

        def mock_invoker(inp: AgentInvokerInput) -> AgentInvokerOutput:
            calls.append(inp)
            return AgentInvokerOutput(agent_output="ok")

        executor = PredefinedScenarioExecutor(agent_invoker=mock_invoker)
        result = executor.run_scenario(scenario)

        assert isinstance(result, ScenarioExecutionResult)
        assert result.scenario_id == "tbt-1"
        assert result.session_id.startswith("tbt-1-")
        assert result.session_id != "tbt-1"  # should have UUID suffix
        assert result.status == "COMPLETED"
        assert result.error is None
        assert len(calls) == 2
        # All turns receive the same session_id
        assert calls[0].session_id == calls[1].session_id

    def test_stable_session_id_across_turns(self):
        """Framework generates a stable session_id passed to every turn."""
        scenario = PredefinedScenario(
            scenario_id="tbt-orig",
            turns=[Turn(input="Hello"), Turn(input="World"), Turn(input="!")],
        )
        received_session_ids = []

        def mock_invoker(inp: AgentInvokerInput) -> AgentInvokerOutput:
            received_session_ids.append(inp.session_id)
            return AgentInvokerOutput(agent_output="ok")

        executor = PredefinedScenarioExecutor(agent_invoker=mock_invoker)
        result = executor.run_scenario(scenario)

        assert result.scenario_id == "tbt-orig"
        assert result.session_id.startswith("tbt-orig-")
        assert result.status == "COMPLETED"
        # All turns receive the same framework-generated session_id
        assert len(set(received_session_ids)) == 1
        assert received_session_ids[0] == result.session_id

    def test_invoker_failure_returns_failed(self):
        scenario = PredefinedScenario(
            scenario_id="tbt-3",
            turns=[Turn(input="hi"), Turn(input="bye")],
        )

        def failing_invoker(inp):
            if inp.payload == "bye":
                raise ConnectionError("timeout")
            return AgentInvokerOutput(agent_output="ok")

        executor = PredefinedScenarioExecutor(agent_invoker=failing_invoker)
        result = executor.run_scenario(scenario)

        assert result.status == "FAILED"
        assert "timeout" in result.error


class TestOnDemandEvaluationDatasetRunnerUnknownScenario:
    @patch("bedrock_agentcore.evaluation.runner.on_demand.on_demand_runner.boto3")
    def test_unknown_scenario_type_returns_failed(self, mock_boto3):
        class CustomScenario(Scenario):
            scenario_id: str = "custom-1"

        mock_dp_client = MagicMock()
        mock_cp_client = MagicMock()
        mock_boto3.client.side_effect = [mock_dp_client, mock_cp_client]

        dataset = Dataset(scenarios=[CustomScenario()])
        config = EvaluationRunConfig(
            evaluator_config=EvaluatorConfig(evaluator_ids=["accuracy"]), evaluation_delay_seconds=0
        )
        mock_collector = MagicMock(spec=AgentSpanCollector)

        runner = OnDemandEvaluationDatasetRunner()
        result = runner.run(
            config=config,
            dataset=dataset,
            agent_invoker=lambda inp: None,
            span_collector=mock_collector,
        )

        assert len(result.scenario_results) == 1
        assert result.scenario_results[0].status == "FAILED"
        assert "No runner registered" in result.scenario_results[0].error
