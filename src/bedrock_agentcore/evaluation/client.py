"""EvaluationClient for collecting spans and running evaluations."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config

from bedrock_agentcore._utils.user_agent import build_user_agent_suffix
from bedrock_agentcore.evaluation._agent_span_collector import CloudWatchAgentSpanCollector

logger = logging.getLogger(__name__)

MAX_TARGET_IDS_PER_REQUEST = 10
QUERY_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 2


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

        base_input = {"evaluationInput": {"sessionSpans": spans}}

        # Steps 2-4: For each evaluator, look up level, build targets, call API
        all_results = []
        for evaluator_id in evaluator_ids:
            level = self._get_evaluator_level(evaluator_id)
            logger.info("Evaluating with %s (level=%s)", evaluator_id, level)
            requests = self._build_requests_for_level(evaluator_id, level, base_input, spans)
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
    ) -> List[dict]:
        """Build one or more evaluate request payloads based on evaluator level."""
        if level == "SESSION":
            return [base_input]

        if level == "TRACE":
            trace_ids = self._extract_trace_ids(spans)
            logger.debug("Extracted %d unique trace ID(s) for evaluator %s", len(trace_ids), evaluator_id)
            if not trace_ids:
                raise ValueError(f"No trace IDs found for trace-level evaluator {evaluator_id}")
            return [
                {**base_input, "evaluationTarget": {"traceIds": trace_ids[i : i + MAX_TARGET_IDS_PER_REQUEST]}}
                for i in range(0, len(trace_ids), MAX_TARGET_IDS_PER_REQUEST)
            ]

        if level == "TOOL_CALL":
            tool_span_ids = self._extract_tool_span_ids(spans)
            logger.debug("Extracted %d tool span ID(s) for evaluator %s", len(tool_span_ids), evaluator_id)
            if not tool_span_ids:
                raise ValueError(f"No tool span IDs found for tool-level evaluator {evaluator_id}")
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
    def _extract_tool_span_ids(spans: list) -> List[str]:
        """Extract span IDs for tool execution spans."""
        tool_span_ids: List[str] = []
        for span in spans:
            name = span.get("name", "")
            kind = span.get("kind")
            if kind == "SPAN_KIND_INTERNAL" and name.startswith("Tool:"):
                span_id = span.get("spanId")
                if span_id:
                    tool_span_ids.append(span_id)
        return tool_span_ids

