"""Batch Evaluation Runner for AgentCore Evaluation Service.

This module provides the BatchEvaluationRunner class that leverages the AgentCore
Evaluation Service's batch evaluation API to run evaluations asynchronously.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import boto3

from bedrock_agentcore._utils.endpoints import DEFAULT_REGION, get_data_plane_endpoint
from bedrock_agentcore.evaluation.runner.batch.batch_evaluation_models import (
    BatchEvaluationResult,
    BatchEvaluationRunConfig,
    BatchEvaluationSummary,
    CloudWatchOutputDataConfig,
    FailedScenario,
)
from bedrock_agentcore.evaluation.runner.dataset_types import Dataset, PredefinedScenario, Scenario, SimulatedScenario
from bedrock_agentcore.evaluation.runner.invoker_types import AgentInvokerFn
from bedrock_agentcore.evaluation.runner.scenario_executor import (
    PredefinedScenarioExecutor,
    ScenarioExecutionResult,
    ScenarioExecutor,
    SimulatedScenarioExecutor,
)

logger = logging.getLogger(__name__)


class _SessionExecutionMetadata(NamedTuple):
    """Internal carrier for per-scenario execution results passed between private methods."""

    scenario_id: str
    session_id: str
    start_time: datetime
    end_time: datetime
    ground_truth: Optional[Dict[str, Any]]


# States where the batch evaluation is still making progress
_RUNNING_STATES = frozenset({"PENDING", "IN_PROGRESS", "STOPPING"})


class BatchEvaluationRunner:
    """Runs evaluation using the AgentCore Batch Evaluation API.

    Starts a batch evaluation via StartBatchEvaluation, and polls GetBatchEvaluation for results.
    """

    _SCENARIO_EXECUTORS: Dict[type, type[ScenarioExecutor]] = {
        PredefinedScenario: PredefinedScenarioExecutor,
        SimulatedScenario: SimulatedScenarioExecutor,
    }

    def __init__(self, region: Optional[str] = None):
        """Initialize the batch evaluation runner.

        Args:
            region: AWS region. Defaults to boto3 session region or DEFAULT_REGION.
        """

        session = boto3.Session()
        self.region = region or session.region_name or DEFAULT_REGION
        self.data_plane_client = session.client(
            "bedrock-agentcore",
            region_name=self.region,
            endpoint_url=get_data_plane_endpoint(self.region),
        )
        self._logs_client = session.client("logs", region_name=self.region)

    @staticmethod
    def _get_boto3_error_code(e: Exception) -> Optional[str]:
        """Extract the error code from a boto3 ClientError, or None."""
        if hasattr(e, "response") and isinstance(e.response, dict):
            code = e.response.get("Error", {}).get("Code")
            return str(code) if code is not None else None
        return None

    def _transform_ground_truth(self, scenario: Scenario) -> Optional[dict]:
        """Transform scenario ground truth into InlineGroundTruth format.

        Includes turns, assertions, expectedTrajectory when present.

        Args:
            scenario: Scenario with optional ground truth fields.

        Returns:
            InlineGroundTruth dict or None if no GT evaluation fields are present.
        """
        ground_truth: Dict[str, Any] = {}

        if scenario.assertions:
            ground_truth["assertions"] = [{"text": a} for a in scenario.assertions]

        if isinstance(scenario, PredefinedScenario):
            if scenario.expected_trajectory is not None:
                ground_truth["expectedTrajectory"] = {"toolNames": scenario.expected_trajectory}

            if scenario.turns:
                ground_truth["turns"] = [
                    {
                        "input": {"prompt": turn.input if isinstance(turn.input, str) else json.dumps(turn.input)},
                        **({"expectedResponse": {"text": turn.expected_response}} if turn.expected_response else {}),
                    }
                    for turn in scenario.turns
                ]

        if not ground_truth:
            logger.debug(
                "No ground truth fields found for scenario %s (%s)",
                scenario.scenario_id,
                type(scenario).__name__,
            )
            return None

        return ground_truth

    def _execute_scenario(
        self,
        config: BatchEvaluationRunConfig,
        scenario: Scenario,
        agent_invoker: AgentInvokerFn,
    ) -> ScenarioExecutionResult:
        """Execute a single scenario and return the execution result.

        Args:
            config: Batch evaluation run configuration.
            scenario: Scenario to execute.
            agent_invoker: Agent invocation function.

        Returns:
            ScenarioExecutionResult with status "COMPLETED" or "FAILED".

        Raises:
            TypeError: If the scenario type is not supported.
        """
        executor_cls = self._SCENARIO_EXECUTORS.get(type(scenario))
        if executor_cls is None:
            raise TypeError(f"Unsupported scenario type: {type(scenario).__name__}")

        kwargs: Dict[str, Any] = {"agent_invoker": agent_invoker}
        if isinstance(scenario, SimulatedScenario):
            kwargs["simulation_config"] = config.simulation_config

        executor = executor_cls(**kwargs)
        return executor.run_scenario(scenario)

    def _execute_scenarios_parallel(
        self,
        config: BatchEvaluationRunConfig,
        dataset: Dataset,
        agent_invoker: AgentInvokerFn,
        max_workers: int,
    ) -> Tuple[List[_SessionExecutionMetadata], List[FailedScenario]]:
        """Execute all scenarios in parallel using ThreadPoolExecutor.

        Args:
            config: Batch evaluation run configuration forwarded to each executor.
            dataset: Collection of scenarios.
            agent_invoker: Agent invocation function.
            max_workers: Maximum concurrent executions.

        Returns:
            Tuple of (successful_sessions, failed_scenarios).
        """
        successful_sessions: List[_SessionExecutionMetadata] = []
        failed_scenarios: List[FailedScenario] = []

        workers = min(max_workers, len(dataset.scenarios)) if dataset.scenarios else 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_scenario = {
                executor.submit(self._execute_scenario, config, scenario, agent_invoker): scenario
                for scenario in dataset.scenarios
            }
            for future in as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                try:
                    result = future.result()
                    if result.status == "FAILED":
                        logger.warning(
                            "Scenario %s failed during invocation: %s",
                            scenario.scenario_id,
                            result.error,
                        )
                        failed_scenarios.append(
                            FailedScenario(
                                scenario_id=scenario.scenario_id,
                                error_message=result.error or "",
                            )
                        )
                    else:
                        successful_sessions.append(
                            _SessionExecutionMetadata(
                                scenario_id=result.scenario_id,
                                session_id=result.session_id,
                                start_time=result.start_time,
                                end_time=result.end_time,
                                ground_truth=self._transform_ground_truth(scenario),
                            )
                        )
                except Exception as e:
                    logger.exception(
                        "Scenario %s failed during execution: %s",
                        scenario.scenario_id,
                        e,
                    )
                    failed_scenarios.append(
                        FailedScenario(
                            scenario_id=scenario.scenario_id,
                            error_message=str(e),
                        )
                    )

        logger.info(
            "Scenario execution complete: %d successful, %d failed",
            len(successful_sessions),
            len(failed_scenarios),
        )
        if failed_scenarios:
            logger.warning(
                "Partial failure: %d/%d scenarios failed: %s",
                len(failed_scenarios),
                len(dataset.scenarios),
                [fs.scenario_id for fs in failed_scenarios],
            )

        return successful_sessions, failed_scenarios

    def _poll_for_results(
        self,
        batch_evaluation_id: str,
        timeout: int,
        poll_interval: int,
    ) -> Dict[str, Any]:
        """Poll GetBatchEvaluation until a terminal state is reached.

        Args:
            batch_evaluation_id: Batch evaluation ID returned by StartBatchEvaluation.
            timeout: Maximum polling time in seconds.
            poll_interval: Fixed interval between polls in seconds.

        Returns:
            dict containing GetBatchEvaluation API response.

        Raises:
            TimeoutError: If polling exceeds timeout.
            RuntimeError: If the evaluation fails, is stopped, reaches an unknown status,
                or the API call fails.
        """
        start_time = time.monotonic()
        logger.info(
            "Polling for batch evaluation %s (timeout=%ds, interval=%ds)",
            batch_evaluation_id,
            timeout,
            poll_interval,
        )

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                logger.error(
                    "Polling timeout exceeded for batch evaluation %s (elapsed=%.1fs, timeout=%ds)",
                    batch_evaluation_id,
                    elapsed,
                    timeout,
                )
                raise TimeoutError(f"Polling timeout exceeded ({timeout}s) for batch evaluation {batch_evaluation_id}")

            try:
                response: Dict[str, Any] = self.data_plane_client.get_batch_evaluation(
                    batchEvaluationId=batch_evaluation_id,
                )
            except Exception as e:
                error_code = self._get_boto3_error_code(e)
                logger.exception(
                    "GetBatchEvaluation failed (error_code=%s): %s",
                    error_code,
                    e,
                )
                raise RuntimeError(f"Failed to get batch evaluation result: {e} (error_code={error_code})") from e

            status = response.get("status")
            logger.info(
                "Batch evaluation %s status: %s (elapsed: %.1fs)",
                batch_evaluation_id,
                status,
                elapsed,
            )

            if status == "COMPLETED":
                return response

            if status == "FAILED":
                error_details = response.get("errorDetails", [])
                logger.error("Batch evaluation %s failed: %s", batch_evaluation_id, error_details)
                raise RuntimeError(f"Batch evaluation {batch_evaluation_id} failed: {error_details}")

            if status == "STOPPED":
                raise RuntimeError(f"Batch evaluation {batch_evaluation_id} was stopped")

            if status in _RUNNING_STATES:
                time.sleep(poll_interval)
                continue

            raise RuntimeError(f"Unknown batch evaluation status: {status}")

    def run_dataset_evaluation(
        self,
        config: BatchEvaluationRunConfig,
        dataset: Dataset,
        agent_invoker: AgentInvokerFn,
    ) -> BatchEvaluationResult:
        """Run a batch evaluation on a Dataset.

        Executes all scenarios in parallel via ``agent_invoker``, transforms ground
        truth data, submits the collected sessions to ``StartBatchEvaluation``,
        and polls until the job reaches a terminal state.

        The returned :class:`BatchEvaluationResult` contains two levels of data:

        * ``result.evaluation_results`` — aggregate per-evaluator statistics
          (average scores, session counts). Available immediately.
        * Call :py:meth:`fetch_evaluation_events` for individual per-turn scores
          with explanations (``gen_ai.evaluation.explanation``).

        Args:
            config: Evaluation name, evaluator IDs, session source,
                concurrency, and polling timeouts.
            dataset: Scenarios to evaluate, with optional ground truth
                (``assertions``, ``expected_trajectory``, per-turn
                ``expected_response``).
            agent_invoker: Called once per turn per scenario. Must be
                thread-safe — up to ``config.max_concurrent_scenarios`` threads
                invoke it concurrently.

        Returns:
            :class:`BatchEvaluationResult` with job IDs, status,
            ``evaluation_results`` (:class:`BatchEvaluationSummary`),
            ``agent_invocation_failures``, and ``output_data_config``.

        Raises:
            ValueError: If ``dataset`` is empty or all scenarios fail during
                agent invocation.
            RuntimeError: If API calls fail or the job reaches ``FAILED``/``STOPPED``.
            TimeoutError: If the job exceeds ``config.polling_timeout_seconds``.
        """
        if not dataset.scenarios:
            raise ValueError("Dataset must contain at least one scenario")
        logger.info(
            "Starting batch evaluation: %d scenarios, max_concurrent=%d, timeout=%ds",
            len(dataset.scenarios),
            config.max_concurrent_scenarios,
            config.polling_timeout_seconds,
        )

        successful_sessions, failed_scenarios = self._execute_scenarios_parallel(
            config,
            dataset,
            agent_invoker,
            config.max_concurrent_scenarios,
        )

        if not successful_sessions:
            raise ValueError(
                f"All {len(dataset.scenarios)} scenarios failed during execution. "
                f"Failed scenario IDs: {[fs.scenario_id for fs in failed_scenarios]}"
            )

        config.data_source.pre_evaluation_run_hook()

        session_metadata_list = [
            {
                "sessionId": session.session_id,
                "testScenarioId": session.scenario_id,
                **({"groundTruth": {"inline": session.ground_truth}} if session.ground_truth else {}),
            }
            for session in successful_sessions
        ]

        logger.info("Calling StartBatchEvaluation (name=%s)", config.batch_evaluation_name)
        try:
            start_kwargs: Dict[str, Any] = dict(
                batchEvaluationName=config.batch_evaluation_name,
                evaluators=[{"evaluatorId": eid} for eid in config.evaluator_config.evaluator_ids],
                dataSourceConfig=config.data_source.to_data_source_config(
                    [s.session_id for s in successful_sessions],
                    min(s.start_time for s in successful_sessions),
                    max(s.end_time for s in successful_sessions),
                ),
                evaluationMetadata={"sessionMetadata": session_metadata_list},
            )
            if config.description is not None:
                start_kwargs["description"] = config.description
            start_response = self.data_plane_client.start_batch_evaluation(**start_kwargs)
        except Exception as e:
            error_code = self._get_boto3_error_code(e)
            logger.exception(
                "StartBatchEvaluation failed (name=%s, error_code=%s): %s",
                config.batch_evaluation_name,
                error_code,
                e,
            )
            raise RuntimeError(f"StartBatchEvaluation failed: {e} (error_code={error_code})") from e

        batch_evaluation_id: str = start_response["batchEvaluationId"]
        batch_evaluation_arn: str = start_response["batchEvaluationArn"]
        logger.info("Started batch evaluation: %s", batch_evaluation_id)

        response = self._poll_for_results(
            batch_evaluation_id,
            config.polling_timeout_seconds,
            config.polling_interval_seconds,
        )

        evaluation_results = None
        if "evaluationResults" in response:
            evaluation_results = BatchEvaluationSummary.model_validate(response["evaluationResults"])

        output_data_config = None
        if "outputConfig" in response:
            odc = response["outputConfig"].get("cloudWatchConfig")
            if odc:
                output_data_config = CloudWatchOutputDataConfig(
                    log_group_name=odc["logGroupName"],
                    log_stream_name=odc["logStreamName"],
                )

        result = BatchEvaluationResult(
            batch_evaluation_id=batch_evaluation_id,
            batch_evaluation_arn=batch_evaluation_arn,
            batch_evaluation_name=response["batchEvaluationName"],
            status=response["status"],
            created_at=response["createdAt"],
            description=response.get("description"),
            agent_invocation_failures=failed_scenarios,
            evaluation_results=evaluation_results,
            error_details=response.get("errorDetails"),
            output_data_config=output_data_config,
        )

        logger.info(
            "Batch evaluation complete: batch_evaluation_id=%s, status=%s, sessions_completed=%s, sessions_failed=%s",
            result.batch_evaluation_id,
            result.status,
            result.evaluation_results.number_of_sessions_completed if result.evaluation_results else None,
            result.evaluation_results.number_of_sessions_failed if result.evaluation_results else None,
        )
        return result

    def fetch_evaluation_events(self, result: BatchEvaluationResult) -> List[Dict[str, Any]]:
        """Fetch per-turn evaluation events from CloudWatch.

        Complements ``result.evaluation_results`` (:class:`BatchEvaluationSummary`),
        which contains aggregate average scores. This method returns one OTel event
        per turn per evaluator, each with an individual score and a natural-language
        explanation (``gen_ai.evaluation.explanation``).

        Args:
            result: Completed :class:`BatchEvaluationResult` from
                :py:meth:`run_dataset_evaluation`.

        Returns:
            List of event dicts, one per turn per evaluator, containing
            ``gen_ai.evaluation.name``, ``gen_ai.evaluation.score.value``,
            ``gen_ai.evaluation.score.label``, ``gen_ai.evaluation.explanation``,
            and trace context (``traceId``, ``gen_ai.response.id``).

        Raises:
            ValueError: If ``result.output_data_config`` is ``None`` (job did
                not produce a CloudWatch destination).
            LookupError: If the log stream does not exist yet; retry after a
                short delay.
        """
        if result.output_data_config is None:
            raise ValueError(
                f"No output_data_config on batch evaluation {result.batch_evaluation_id}. "
                "The service did not return a CloudWatch destination for this evaluation."
            )
        output_data_config = result.output_data_config
        results: List[Dict[str, Any]] = []
        kwargs: Dict[str, Any] = {
            "logGroupName": output_data_config.log_group_name,
            "logStreamName": output_data_config.log_stream_name,
            "startFromHead": True,
        }
        try:
            while True:
                response = self._logs_client.get_log_events(**kwargs)
                for event in response.get("events", []):
                    message = event.get("message", "")
                    try:
                        results.append(json.loads(message))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping non-JSON log event in stream %s: %r",
                            output_data_config.log_stream_name,
                            message[:200],
                        )
                next_token = response.get("nextForwardToken")
                if next_token == kwargs.get("nextToken"):
                    break
                kwargs.pop("startFromHead", None)
                kwargs["nextToken"] = next_token
        except self._logs_client.exceptions.ResourceNotFoundException as e:
            raise LookupError(
                f"CloudWatch log stream not found: group={output_data_config.log_group_name!r}, "
                f"stream={output_data_config.log_stream_name!r}. "
                "Evaluation results may not have been written yet."
            ) from e
        logger.info(
            "Fetched %d evaluation result events from %s/%s",
            len(results),
            output_data_config.log_group_name,
            output_data_config.log_stream_name,
        )
        return results
