"""OnDemandEvaluationDatasetRunner: orchestrates dataset scenarios, agent invocation, and evaluation."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Tuple

import boto3
from botocore.config import Config

from bedrock_agentcore.evaluation.agent_span_collector import AgentSpanCollector

from ..dataset_types import Dataset, PredefinedScenario, Scenario
from ..invoker_types import AgentInvokerFn
from ..scenario_executor import (
    PredefinedScenarioExecutor,
    ScenarioExecutionResult,
    ScenarioExecutor,
)
from .config import EvaluationRunConfig
from .result import EvaluationResult, EvaluatorResult, ScenarioResult

logger = logging.getLogger(__name__)

MAX_TARGET_IDS_PER_REQUEST = 10


class OnDemandEvaluationDatasetRunner:
    """Runs evaluation scenarios end-to-end.

    For each scenario in the dataset, the runner:
    1. Invokes the agent for each turn (run_scenario).
    2. Collects spans via the span collector (collect_spans).
    3. Builds the full evaluate API requests with level-aware targeting (build_evaluate_requests).
    4. Sends the requests and collects results (run_evaluations).
    """

    def __init__(self, region: Optional[str] = None):
        """Initialize the evaluation runner with AWS clients."""
        self.region = region or boto3.Session().region_name or "us-west-2"
        self._client = self._create_data_plane_client()
        self._control_client = self._create_control_plane_client()
        self._evaluator_level_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._scenario_executors: Dict[type, type[ScenarioExecutor]] = {
            PredefinedScenario: PredefinedScenarioExecutor,
        }

    def run(
        self,
        config: EvaluationRunConfig,
        dataset: Dataset,
        agent_invoker: AgentInvokerFn,
        span_collector: AgentSpanCollector,
    ) -> EvaluationResult:
        """Run evaluation across all scenarios in the dataset.

        Scenarios are processed in three batched phases:
          Phase 1: Invoke agents for all scenarios concurrently.
          Phase 2: Wait for CloudWatch span ingestion (evaluation_delay_seconds).
          Phase 3: Collect spans and evaluate all scenarios concurrently.

        Args:
            config: Evaluation runner configuration.
            dataset: The dataset containing scenarios to evaluate.
            agent_invoker: Callable that invokes the agent for each turn.
            span_collector: Collector for retrieving spans after invocation.

        Returns:
            EvaluationResult with scores for every scenario and evaluator.
        """
        num_scenarios = len(dataset.scenarios)
        logger.info(
            "Starting evaluation run: %d scenario(s), %d evaluator(s)",
            num_scenarios,
            len(config.evaluator_config.evaluator_ids),
        )

        results: List[Optional[ScenarioResult]] = [None] * num_scenarios
        max_workers = min(config.max_concurrent_scenarios, num_scenarios) if num_scenarios > 0 else 1

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Phase 1: Invoke all scenarios concurrently
            logger.info("Phase 1: Invoking %d scenario(s)", num_scenarios)
            invoke_futures = {}
            for idx, scenario in enumerate(dataset.scenarios):
                future = pool.submit(self._run_scenario, scenario, agent_invoker)
                invoke_futures[idx] = (scenario, future)

            # Collect invocation results
            execution_results: Dict[int, ScenarioExecutionResult] = {}
            for idx, (scenario, future) in invoke_futures.items():
                try:
                    exec_result = future.result()
                    if exec_result.status == "FAILED":
                        results[idx] = ScenarioResult(
                            scenario_id=exec_result.scenario_id,
                            session_id=exec_result.session_id,
                            status="FAILED",
                            error=exec_result.error,
                        )
                    else:
                        execution_results[idx] = exec_result
                except Exception as e:
                    logger.exception("Scenario %s invoke failed: %s", scenario.scenario_id, e)
                    results[idx] = ScenarioResult(
                        scenario_id=scenario.scenario_id,
                        session_id=scenario.scenario_id,
                        status="FAILED",
                        error=str(e),
                    )

            # Phase 2: Wait for CloudWatch span ingestion
            if execution_results and config.evaluation_delay_seconds > 0:
                logger.info("Phase 2: Waiting %ds for evaluation readiness...", config.evaluation_delay_seconds)
                time.sleep(config.evaluation_delay_seconds)

            # Phase 3: Collect spans + evaluate all successful scenarios concurrently
            logger.info("Phase 3: Collecting spans and evaluating %d scenario(s)", len(execution_results))
            eval_futures = {}
            for idx, exec_result in execution_results.items():
                scenario = dataset.scenarios[idx]
                future = pool.submit(
                    self._collect_and_evaluate,
                    scenario,
                    exec_result,
                    span_collector,
                    config.evaluator_config.evaluator_ids,
                )
                eval_futures[idx] = (scenario, exec_result, future)

            for idx, (scenario, exec_result, future) in eval_futures.items():
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.exception("Scenario %s collect+evaluate failed: %s", scenario.scenario_id, e)
                    results[idx] = ScenarioResult(
                        scenario_id=exec_result.scenario_id,
                        session_id=exec_result.session_id,
                        status="FAILED",
                        error=str(e),
                    )

        logger.info("Evaluation run complete: %d scenario(s) processed", num_scenarios)
        return EvaluationResult(scenario_results=results)

    def _collect_and_evaluate(
        self,
        scenario: Scenario,
        execution_result: ScenarioExecutionResult,
        span_collector: AgentSpanCollector,
        evaluator_ids: List[str],
    ) -> ScenarioResult:
        """Collect spans, build evaluate requests, and run evaluations for a single scenario."""
        spans = self._collect_spans(execution_result=execution_result, span_collector=span_collector)
        logger.debug("Collected %d span(s) for scenario %s", len(spans), scenario.scenario_id)

        evaluate_requests = self._build_evaluate_requests(
            scenario=scenario,
            spans=spans,
            evaluator_ids=evaluator_ids,
            session_id=execution_result.session_id,
        )
        logger.debug("Built %d evaluate request(s) for scenario %s", len(evaluate_requests), scenario.scenario_id)

        evaluator_results = self._run_evaluations(evaluate_requests)

        return ScenarioResult(
            scenario_id=execution_result.scenario_id,
            session_id=execution_result.session_id,
            status="COMPLETED",
            evaluator_results=evaluator_results,
        )

    # --- Step 1: Run Scenario ---

    def _run_scenario(
        self,
        scenario: Scenario,
        agent_invoker: AgentInvokerFn,
    ) -> ScenarioExecutionResult:
        """Invoke the agent for the given scenario."""
        executor_cls = self._scenario_executors.get(type(scenario))
        if executor_cls is None:
            raise TypeError(f"No runner registered for scenario type: {type(scenario).__name__}")
        executor = executor_cls(agent_invoker=agent_invoker)
        return executor.run_scenario(scenario)

    # --- Step 2: Collect Spans ---

    def _collect_spans(
        self,
        execution_result: ScenarioExecutionResult,
        span_collector: AgentSpanCollector,
    ) -> List:
        """Collect spans for a completed scenario execution."""
        return span_collector.collect(
            session_id=execution_result.session_id,
            start_time=execution_result.start_time,
            end_time=execution_result.end_time,
        )

    # --- Step 3: Build Evaluate Requests ---

    def _build_evaluate_requests(
        self,
        scenario: Scenario,
        spans: list,
        evaluator_ids: List[str],
        session_id: str = "",
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Build complete evaluate API requests for all evaluators.

        Constructs the base request (evaluationInput + evaluationReferenceInputs),
        looks up each evaluator's level, and builds the appropriate evaluationTarget
        with batching (max 10 IDs per request).

        Returns:
            List of (evaluator_id, request_dict) tuples.
        """
        base: Dict[str, Any] = {"evaluationInput": {"sessionSpans": spans}}
        trace_ids = self._extract_trace_ids(spans)
        tool_span_ids = self._extract_tool_span_ids(spans)

        reference_inputs = self._build_reference_inputs(scenario, trace_ids, session_id)
        if reference_inputs:
            base["evaluationReferenceInputs"] = reference_inputs

        requests: List[Tuple[str, Dict[str, Any]]] = []
        for evaluator_id in evaluator_ids:
            level = self._get_evaluator_level(evaluator_id)

            match level:
                case "SESSION":
                    requests.append((evaluator_id, base))

                case "TRACE":
                    if not trace_ids:
                        logger.warning(
                            "No trace IDs found in collected spans for trace-level evaluator %s. Skipping.",
                            evaluator_id,
                        )
                        continue
                    for batch in self._batch(trace_ids, MAX_TARGET_IDS_PER_REQUEST):
                        requests.append((evaluator_id, {**base, "evaluationTarget": {"traceIds": batch}}))

                case "TOOL_CALL":
                    if not tool_span_ids:
                        logger.warning(
                            "No tool span IDs found in collected spans for tool-level evaluator %s. Skipping.",
                            evaluator_id,
                        )
                        continue
                    for batch in self._batch(tool_span_ids, MAX_TARGET_IDS_PER_REQUEST):
                        requests.append((evaluator_id, {**base, "evaluationTarget": {"spanIds": batch}}))

                case _:
                    raise ValueError(f"Unknown evaluator level: {level}")

        return requests

    def _extract_trace_ids(self, spans: list) -> List[str]:
        """Extract unique trace IDs from spans, ordered by appearance."""
        trace_ids = []
        seen = set()
        for span in spans:
            trace_id = span.get("traceId")
            if trace_id and trace_id not in seen:
                trace_ids.append(trace_id)
                seen.add(trace_id)
        return trace_ids

    def _extract_tool_span_ids(self, spans: list) -> List[str]:
        """Extract span IDs for tool execution spans (supports Strands and LangGraph)."""
        tool_span_ids = []
        for span in spans:
            if self._is_tool_span(span):
                span_id = span.get("spanId")
                if span_id:
                    tool_span_ids.append(span_id)
        return tool_span_ids

    def _build_reference_inputs(
        self,
        scenario: Scenario,
        trace_ids: List[str],
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """Build evaluationReferenceInputs from scenario ground truth data.

        Returns a flat list containing:
        - One session-level object with expectedTrajectory and/or assertion (if present).
        - One per-trace object per turn that has expected_response (turn[i] maps to trace_ids[i]).
        """
        reference_inputs: List[Dict[str, Any]] = []

        # Session-level: combined expectedTrajectory + assertion
        session_ref: Dict[str, Any] = {"context": {"spanContext": {"sessionId": session_id}}}
        has_session_ref = False

        if hasattr(scenario, "expected_trajectory") and scenario.expected_trajectory:
            session_ref["expectedTrajectory"] = {"toolNames": scenario.expected_trajectory}
            has_session_ref = True

        if hasattr(scenario, "assertions") and scenario.assertions:
            session_ref["assertions"] = [{"text": a} for a in scenario.assertions]
            has_session_ref = True

        if has_session_ref:
            reference_inputs.append(session_ref)

        # Per-trace: expectedResponse (turn[i] → trace_ids[i])
        if hasattr(scenario, "turns"):
            for i, turn in enumerate(scenario.turns):
                if turn.expected_response and i < len(trace_ids):
                    reference_inputs.append(
                        {
                            "context": {"spanContext": {"sessionId": session_id, "traceId": trace_ids[i]}},
                            "expectedResponse": {"text": turn.expected_response},
                        }
                    )

        return reference_inputs

    def _get_evaluator_level(self, evaluator_id: str) -> str:
        """Look up an evaluator's level (SESSION/TRACE/TOOL_CALL) via the control plane API.

        Results are cached to avoid repeated API calls for the same evaluator.
        Thread-safe via double-check locking — the lock is not held during the API call.
        """
        with self._cache_lock:
            if evaluator_id in self._evaluator_level_cache:
                return self._evaluator_level_cache[evaluator_id]
        response = self._control_client.get_evaluator(evaluatorId=evaluator_id)
        level = response["level"]
        with self._cache_lock:
            self._evaluator_level_cache[evaluator_id] = level
        logger.debug("Fetched evaluator level: %s -> %s", evaluator_id, level)
        return level

    # --- Step 4: Run Evaluations ---

    def _run_evaluations(
        self,
        evaluate_requests: List[Tuple[str, Dict[str, Any]]],
    ) -> List[EvaluatorResult]:
        """Send pre-built evaluate requests and collect results, grouped by evaluator."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for evaluator_id, request in evaluate_requests:
            if evaluator_id not in grouped:
                grouped[evaluator_id] = []
            try:
                results = self._call_evaluator(evaluator_id, request)
                grouped[evaluator_id].extend(results)
            except Exception as e:
                logger.warning("Evaluator %s failed: %s", evaluator_id, e, exc_info=True)
                grouped[evaluator_id].append({"errorCode": "SDKError", "errorMessage": str(e)})

        return [EvaluatorResult(evaluator_id=eid, results=res) for eid, res in grouped.items()]

    def _call_evaluator(
        self,
        evaluator_id: str,
        evaluate_request: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Call a single evaluator via the AgentCore API. Returns all results from the response.

        Raises on API errors — caller is responsible for catching.
        """
        logger.debug("Calling evaluator %s with request: %s", evaluator_id, evaluate_request)
        response = self._client.evaluate(
            evaluatorId=evaluator_id,
            **evaluate_request,
        )
        results = response.get("evaluationResults", [])
        logger.debug("Evaluator %s returned %d result(s)", evaluator_id, len(results))

        return results

    # --- Client creation ---

    def _create_data_plane_client(self):
        """Create an bedrock-agentcore client."""
        client_config = Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=300,
        )
        return boto3.client(
            "bedrock-agentcore",
            region_name=self.region,
            config=client_config,
        )

    def _create_control_plane_client(self):
        """Create a boto3 bedrock-agentcore-control client for evaluator metadata."""
        client_config = Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=30,
        )
        return boto3.client(
            "bedrock-agentcore-control",
            region_name=self.region,
            config=client_config,
        )

    # --- Helpers ---

    @staticmethod
    def _batch(items: List[str], size: int) -> Iterator[List[str]]:
        """Yield successive batches of the given size from items."""
        for i in range(0, len(items), size):
            yield items[i : i + size]

    @staticmethod
    def _is_tool_span(span: Dict) -> bool:
        """Check if a span represents a tool execution (supports Strands and LangGraph)."""
        attrs = span.get("attributes", {})
        if not isinstance(attrs, dict):
            return False
        return (
            attrs.get("gen_ai.operation.name") == "execute_tool"
            or attrs.get("openinference.span.kind") == "TOOL"
            or attrs.get("traceloop.span.kind") == "tool"
        )
