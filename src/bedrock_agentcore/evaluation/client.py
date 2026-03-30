"""EvaluationClient for collecting spans and running evaluations."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.config import Config
from pydantic import BaseModel

from bedrock_agentcore._utils.user_agent import build_user_agent_suffix
from bedrock_agentcore.evaluation.agent_span_collector import CloudWatchAgentSpanCollector

logger = logging.getLogger(__name__)

MAX_TARGET_IDS_PER_REQUEST = 10
QUERY_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 2


class ReferenceInputs(BaseModel):
    """Ground truth inputs for evaluation.

    Attributes:
        assertions: Natural language assertions about expected behavior (session-level).
        expected_trajectory: Expected tool names in order (session-level).
        expected_response: Expected response text. A plain string applies to the
            last trace. A ``{trace_id: response}`` dict targets specific traces.
    """

    assertions: Optional[List[str]] = None
    expected_trajectory: Optional[List[str]] = None
    expected_response: Optional[Union[str, Dict[str, str]]] = None


class EvaluationClient:
    """Client for evaluating agent sessions.

    Collects spans from CloudWatch and calls the evaluation API with
    level-aware batching.

    Example::

        client = EvaluationClient(region_name="us-west-2")

        # Using agent_id (log group derived automatically)
        results = client.run(
            evaluator_ids=["accuracy", "toxicity"],
            session_id="sess-123",
            agent_id="my-agent",
        )

        # Using log_group_name directly
        results = client.run(
            evaluator_ids=["accuracy", "toxicity"],
            session_id="sess-123",
            log_group_name="/custom/my-log-group",
        )

        for r in results:
            print(f"{r['evaluatorId']}: {r.get('value')} - {r.get('explanation')}")
    """

    def __init__(
        self,
        region_name: Optional[str] = None,
        integration_source: Optional[str] = None,
    ):
        """Initialize the EvaluationClient.

        Args:
            region_name: AWS region. Falls back to boto3 session region or us-west-2.
            integration_source: Optional integration framework identifier for telemetry.
        """
        self.region_name = region_name or boto3.Session().region_name or "us-west-2"
        self.integration_source = integration_source

        user_agent_extra = build_user_agent_suffix(integration_source)
        client_config = Config(user_agent_extra=user_agent_extra)

        self._dp_client = boto3.client(
            "bedrock-agentcore",
            region_name=self.region_name,
            config=client_config,
        )
        self._cp_client = boto3.client(
            "bedrock-agentcore-control",
            region_name=self.region_name,
            config=client_config,
        )
        self._evaluator_level_cache: Dict[str, str] = {}

        logger.info("Initialized EvaluationClient in region %s", self.region_name)

    def run(
        self,
        evaluator_ids: List[str],
        session_id: str,
        agent_id: Optional[str] = None,
        look_back_time: timedelta = timedelta(days=7),
        log_group_name: Optional[str] = None,
        trace_id: Optional[str] = None,
        reference_inputs: Optional[ReferenceInputs] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate an agent session end-to-end.

        1. Collects spans from CloudWatch.
        2. For each evaluator, looks up its level (SESSION/TRACE/TOOL_CALL).
        3. Builds the appropriate evaluationTarget based on level.
        4. Calls evaluate() with auto-batching (max 10 target IDs per request).
        5. Returns combined evaluationResults from all evaluators.

        Either ``agent_id`` or ``log_group_name`` must be provided.
        When only ``agent_id`` is given, the log group name is derived as
        ``/aws/bedrock-agentcore/runtimes/{agent_id}-DEFAULT``.

        Args:
            evaluator_ids: List of evaluator IDs (built-in or custom ARNs).
            session_id: The session ID to evaluate.
            agent_id: The agent ID. Used to derive the log group when
                ``log_group_name`` is not provided.
            look_back_time: How far back to search for spans (default: 7 days).
            log_group_name: CloudWatch log group name. If provided, ``agent_id``
                is not required.
            trace_id: Optional trace ID to narrow evaluation to a single trace.
            reference_inputs: Optional ground truth for evaluation.

        Returns:
            List of evaluation result dicts from all evaluators.

        Raises:
            ValueError: If neither ``agent_id`` nor ``log_group_name`` is provided.
        """
        if not agent_id and not log_group_name:
            raise ValueError("Provide either agent_id or log_group_name.")

        if not log_group_name:
            log_group_name = f"/aws/bedrock-agentcore/runtimes/{agent_id}-DEFAULT"
            logger.debug("Derived log_group_name=%s from agent_id=%s", log_group_name, agent_id)

        end_time = datetime.now(timezone.utc)
        start_time = end_time - look_back_time

        logger.info(
            "Running evaluation for session=%s, log_group=%s, time_range=[%s, %s]",
            session_id,
            log_group_name,
            start_time,
            end_time,
        )

        # Step 1: Collect spans
        collector = CloudWatchAgentSpanCollector(
            log_group_name=log_group_name,
            region=self.region_name,
            max_wait_seconds=QUERY_TIMEOUT_SECONDS,
            poll_interval_seconds=POLL_INTERVAL_SECONDS,
        )
        spans = collector.collect(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
        )

        if not spans:
            logger.warning("No spans found for session %s", session_id)
            return []

        base_input: Dict[str, Any] = {"evaluationInput": {"sessionSpans": spans}}

        # Add reference inputs (ground truth) if provided
        if reference_inputs:
            all_trace_ids = self._extract_trace_ids(spans)
            ref_inputs = self._build_reference_inputs(
                session_id, reference_inputs, all_trace_ids, target_trace_id=trace_id
            )
            if ref_inputs:
                base_input["evaluationReferenceInputs"] = ref_inputs

        # Steps 2-4: For each evaluator, look up level, build targets, call API
        all_results: List[Dict[str, Any]] = []
        for evaluator_id in evaluator_ids:
            level = self._get_evaluator_level(evaluator_id)
            logger.info("Evaluating with %s (level=%s)", evaluator_id, level)
            requests = self._build_requests_for_level(evaluator_id, level, base_input, spans, trace_id)
            if len(requests) > 1:
                logger.debug("Split into %d batched request(s) for evaluator %s", len(requests), evaluator_id)
            for request in requests:
                response = self._dp_client.evaluate(evaluatorId=evaluator_id, **request)
                all_results.extend(response.get("evaluationResults", []))

        logger.info(
            "Evaluation complete: %d result(s) from %d evaluator(s)",
            len(all_results),
            len(evaluator_ids),
        )
        return all_results

    def _get_evaluator_level(self, evaluator_id: str) -> str:
        """Look up evaluator level with caching. Falls back to SESSION."""
        if evaluator_id not in self._evaluator_level_cache:
            try:
                response = self._cp_client.get_evaluator(evaluatorId=evaluator_id)
                self._evaluator_level_cache[evaluator_id] = response["level"]
            except Exception as e:
                logger.warning(
                    "Failed to get level for %s, defaulting to SESSION: %s",
                    evaluator_id,
                    e,
                )
                self._evaluator_level_cache[evaluator_id] = "SESSION"
        return self._evaluator_level_cache[evaluator_id]

    def _build_requests_for_level(
        self,
        evaluator_id: str,
        level: str,
        base_input: dict,
        spans: list,
        trace_id: Optional[str] = None,
    ) -> List[dict]:
        """Build one or more evaluate request payloads based on evaluator level.

        When ``trace_id`` is provided, TRACE-level evaluators target only that
        trace and TOOL_CALL-level evaluators are filtered to tool spans within
        that trace.
        """
        if level == "SESSION":
            return [base_input]

        if level == "TRACE":
            if trace_id:
                return [{**base_input, "evaluationTarget": {"traceIds": [trace_id]}}]
            trace_ids = self._extract_trace_ids(spans)
            logger.debug("Extracted %d unique trace ID(s) for evaluator %s", len(trace_ids), evaluator_id)
            if not trace_ids:
                logger.warning("No trace IDs found for trace-level evaluator %s, skipping", evaluator_id)
                return []
            return [
                {**base_input, "evaluationTarget": {"traceIds": trace_ids[i : i + MAX_TARGET_IDS_PER_REQUEST]}}
                for i in range(0, len(trace_ids), MAX_TARGET_IDS_PER_REQUEST)
            ]

        if level == "TOOL_CALL":
            tool_span_ids = self._extract_tool_span_ids(spans, trace_id=trace_id)
            logger.debug("Extracted %d tool span ID(s) for evaluator %s", len(tool_span_ids), evaluator_id)
            if not tool_span_ids:
                logger.warning("No tool span IDs found for tool-level evaluator %s, skipping", evaluator_id)
                return []
            return [
                {**base_input, "evaluationTarget": {"spanIds": tool_span_ids[i : i + MAX_TARGET_IDS_PER_REQUEST]}}
                for i in range(0, len(tool_span_ids), MAX_TARGET_IDS_PER_REQUEST)
            ]

        raise ValueError(f"Unknown evaluator level: {level}")

    @staticmethod
    def _extract_trace_ids(spans: list) -> List[str]:
        """Extract unique trace IDs from spans, ordered by appearance."""
        return list(dict.fromkeys(span.get("traceId") for span in spans if span.get("traceId")))

    @staticmethod
    def _is_tool_span(span: dict) -> bool:
        """Check if a span represents a tool execution (supports Strands, LangGraph, and Traceloop)."""
        attrs = span.get("attributes", {})
        if not isinstance(attrs, dict):
            return False
        return (
            attrs.get("gen_ai.operation.name") == "execute_tool"
            or attrs.get("openinference.span.kind") == "TOOL"
            or attrs.get("traceloop.span.kind") == "tool"
        )

    @staticmethod
    def _extract_tool_span_ids(spans: list, trace_id: Optional[str] = None) -> List[str]:
        """Extract span IDs for tool execution spans.

        Args:
            spans: List of span dicts.
            trace_id: If provided, only include tool spans with this trace ID.
        """
        tool_span_ids: List[str] = []
        for span in spans:
            if EvaluationClient._is_tool_span(span):
                if trace_id and span.get("traceId") != trace_id:
                    continue
                span_id = span.get("spanId")
                if span_id:
                    tool_span_ids.append(span_id)
        return tool_span_ids

    @staticmethod
    def _build_reference_inputs(
        session_id: str,
        reference_inputs: "ReferenceInputs",
        trace_ids: List[str],
        target_trace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build evaluationReferenceInputs from ReferenceInputs.

        Returns a list of reference input dicts scoped by spanContext:
        - Session-level entry for assertions and/or expected_trajectory.
        - Per-trace entries for expected_response.

        Args:
            session_id: The session ID for span context.
            reference_inputs: Ground truth inputs for evaluation.
            trace_ids: All trace IDs extracted from spans.
            target_trace_id: When provided and expected_response is a string,
                targets this trace instead of the last trace.
        """
        result: List[Dict[str, Any]] = []

        # Session-level: assertions and/or expected_trajectory
        session_ref: Dict[str, Any] = {"context": {"spanContext": {"sessionId": session_id}}}
        has_session_ref = False

        if reference_inputs.assertions:
            session_ref["assertions"] = [{"text": a} for a in reference_inputs.assertions]
            has_session_ref = True

        if reference_inputs.expected_trajectory:
            session_ref["expectedTrajectory"] = {"toolNames": reference_inputs.expected_trajectory}
            has_session_ref = True

        if has_session_ref:
            result.append(session_ref)

        # Trace-level: expected_response
        if reference_inputs.expected_response is not None:
            if isinstance(reference_inputs.expected_response, str):
                # Use explicit target_trace_id if provided, otherwise fall back to the last trace
                resolved_trace_id = target_trace_id if target_trace_id else (trace_ids[-1] if trace_ids else None)
                if resolved_trace_id:
                    result.append(
                        {
                            "context": {"spanContext": {"sessionId": session_id, "traceId": resolved_trace_id}},
                            "expectedResponse": {"text": reference_inputs.expected_response},
                        }
                    )
            elif isinstance(reference_inputs.expected_response, dict):
                # Dict maps trace_id -> response
                for tid, response_text in reference_inputs.expected_response.items():
                    result.append(
                        {
                            "context": {"spanContext": {"sessionId": session_id, "traceId": tid}},
                            "expectedResponse": {"text": response_text},
                        }
                    )

        return result
