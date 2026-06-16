"""Fetch ADOT spans from CloudWatch for evaluation."""

import json
import logging
import time
from datetime import datetime
from typing import Any, List, Optional

import boto3

from bedrock_agentcore._utils.endpoints import DEFAULT_REGION

logger = logging.getLogger(__name__)


def _is_valid_adot_document(item: Any) -> bool:
    """Check if item is a valid ADOT document.

    Args:
        item: Potential ADOT document

    Returns:
        True if item has required ADOT fields
    """
    return isinstance(item, dict) and "scope" in item and "traceId" in item and "spanId" in item


class CloudWatchSpanHelper:
    """Fetches ADOT spans from CloudWatch for agent evaluation."""

    def __init__(self, region: str = DEFAULT_REGION):
        """Initialize the span fetcher.

        Args:
            region: AWS region for CloudWatch client
        """
        self.logs_client = boto3.client("logs", region_name=region)
        self.region = region

    def query_log_group(
        self,
        log_group_name: str,
        session_id: str,
        start_time: datetime,
        end_time: datetime,
        query_string: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[dict]:
        """Query a single CloudWatch log group for session data.

        Args:
            log_group_name: Name of the log group to query
            session_id: Session ID to filter by
            start_time: Query start time
            end_time: Query end time
            query_string: Optional custom query string. When provided, used instead
                of the default substring match query.
            agent_id: Optional agent ID to filter by (prevents cross-agent session collisions)

        Returns:
            List of parsed JSON log messages
        """
        if query_string is None:
            agent_filter = ""
            if agent_id is not None:
                agent_filter = (
                    f'\n        | parse resource.attributes.cloud.resource_id "runtime/*/" as parsedAgentId'
                    f"\n        | filter parsedAgentId = '{agent_id}'"
                )
            query_string = f"""fields @timestamp, @message
        | filter @message like "{session_id}"
        | filter ispresent(scope.name)
        | filter ispresent(traceId)
        | filter ispresent(spanId){agent_filter}
        | sort @timestamp asc"""

        max_attempts = 30
        initial_backoff = 0.5
        max_backoff = 5.0

        logger.debug(
            "Querying log group %s: start_time=%s, end_time=%s, query=%s",
            log_group_name,
            start_time,
            end_time,
            query_string,
        )

        try:
            response = self.logs_client.start_query(
                logGroupName=log_group_name,
                startTime=int(start_time.timestamp()),
                endTime=int(end_time.timestamp()),
                queryString=query_string,
            )

            query_id = response["queryId"]

            # Poll for completion with exponential backoff
            backoff = initial_backoff
            for _attempt in range(max_attempts):
                result = self.logs_client.get_query_results(queryId=query_id)

                if result["status"] == "Complete":
                    # Check if we hit the 10K result limit
                    statistics = result.get("statistics", {})
                    records_matched = statistics.get("recordsMatched", 0)
                    records_returned = len(result.get("results", []))

                    if records_matched > 10000:
                        logger.warning(
                            "CloudWatch query matched %d records but can only return 10,000. "
                            "Results may be incomplete for log group: %s. "
                            "Consider narrowing your time range or adding more specific filters.",
                            records_matched,
                            log_group_name,
                        )

                    logger.debug(
                        "CloudWatch query completed: %d results returned, %d records matched",
                        records_returned,
                        records_matched,
                    )
                    break
                elif result["status"] == "Failed":
                    logger.warning("CloudWatch query failed for log group: %s", log_group_name)
                    return []

                # Exponential backoff with cap
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            else:
                logger.warning(
                    "CloudWatch query timed out after %d attempts for log group: %s",
                    max_attempts,
                    log_group_name,
                )
                return []

            # Extract and parse messages
            items = []
            for row in result.get("results", []):
                for field in row:
                    if field["field"] == "@message":
                        try:
                            items.append(json.loads(field["value"]))
                        except json.JSONDecodeError:
                            continue
            return items
        except Exception as e:
            logger.warning("Error querying log group %s: %s", log_group_name, e)
            return []

    def fetch_spans(
        self,
        session_id: str,
        event_log_group: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        agent_id: Optional[str] = None,
    ) -> List[dict]:
        """Fetch ADOT spans from CloudWatch with configurable event log group.

        ADOT spans are always fetched from aws/spans. Event logs can be fetched from
        any configurable log group.

        Args:
            session_id: Session ID from agent execution
            event_log_group: CloudWatch log group name for event logs
                - For Runtime agents: "/aws/bedrock-agentcore/runtimes/{agent_id}-{endpoint}"
                - For custom agents: Any log group you configured (e.g., "/my-app/agent-events")
            start_time: Start time for log query
            end_time: End time for log query
            agent_id: Optional agent ID to filter by (prevents cross-agent session collisions)

        Returns:
            List of ADOT span and log record dictionaries

        Example (Runtime agent):
            >>> from datetime import datetime, timedelta, timezone
            >>> helper = CloudWatchSpanHelper(region="us-west-2")
            >>> start_time = datetime.now(timezone.utc) - timedelta(minutes=10)
            >>> end_time = datetime.now(timezone.utc)
            >>> spans = helper.fetch_spans(
            ...     session_id="abc-123",
            ...     event_log_group="/aws/bedrock-agentcore/runtimes/my-agent-ABC-DEFAULT",
            ...     start_time=start_time,
            ...     end_time=end_time,
            ... )

        Example (Custom agent):
            >>> spans = helper.fetch_spans(
            ...     session_id="abc-123",
            ...     event_log_group="/my-app/agent-events",
            ...     start_time=start_time,
            ...     end_time=end_time,
            ... )
        """
        if end_time is None:
            end_time = datetime.now()

        # Query both log groups
        aws_spans = self.query_log_group("aws/spans", session_id, start_time, end_time, agent_id=agent_id)
        event_logs = self.query_log_group(event_log_group, session_id, start_time, end_time, agent_id=agent_id)

        all_data = aws_spans + event_logs

        logger.info("Fetched %d span items from CloudWatch", len(all_data))
        return all_data

    def query_spans_by_trace(
        self,
        trace_id: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> List[dict]:
        """Query all spans for a trace from aws/spans log group.

        Note: Trace IDs are globally unique, so no agent_id filter needed to prevent cross-agent access

        Args:
            trace_id: The trace ID to query
            start_time_ms: Start time in milliseconds since epoch
            end_time_ms: End time in milliseconds since epoch

        Returns:
            List of result dictionaries from CloudWatch Logs Insights
        """
        query_string = f"""fields @timestamp, @message, traceId, spanId, name as spanName,
            kind, status.code as statusCode, status.message as statusMessage,
            durationNano/1000000 as durationMs, attributes.session.id as sessionId,
            startTimeUnixNano, endTimeUnixNano, parentSpanId, events,
            resource.attributes.service.name as serviceName
        | filter traceId = '{trace_id}'
        | sort startTimeUnixNano asc"""
        return self._execute_query(query_string, "aws/spans", start_time_ms, end_time_ms)

    def query_runtime_logs_by_traces(
        self,
        trace_ids: List[str],
        start_time_ms: int,
        end_time_ms: int,
        agent_id: str,
        endpoint_name: str = "DEFAULT",
    ) -> List[dict]:
        """Query runtime logs for multiple traces from agent-specific log group.

        Args:
            trace_ids: List of trace IDs to query
            start_time_ms: Start time in milliseconds since epoch
            end_time_ms: End time in milliseconds since epoch
            agent_id: Agent ID for constructing the log group name
            endpoint_name: Runtime endpoint name (default: DEFAULT)

        Returns:
            List of result dictionaries from CloudWatch Logs Insights
        """
        if not trace_ids:
            return []

        log_group = f"/aws/bedrock-agentcore/runtimes/{agent_id}-{endpoint_name}"
        trace_ids_quoted = ", ".join([f"'{tid}'" for tid in trace_ids])
        query_string = f"""fields @timestamp, @message, spanId, traceId, @logStream
        | filter traceId in [{trace_ids_quoted}]
        | sort @timestamp asc"""

        try:
            return self._execute_query(query_string, log_group, start_time_ms, end_time_ms)
        except Exception as e:
            logger.warning("Batch query failed, falling back to individual queries: %s", e)
            return self._query_runtime_logs_individually(trace_ids, log_group, start_time_ms, end_time_ms)

    def get_latest_session_id(
        self,
        start_time_ms: int,
        end_time_ms: int,
        agent_id: str,
    ) -> Optional[str]:
        """Get the most recent session ID for an agent.

        Args:
            start_time_ms: Start time in milliseconds since epoch
            end_time_ms: End time in milliseconds since epoch
            agent_id: Agent ID to query for

        Returns:
            Latest session ID or None if no sessions found
        """
        query_string = f"""filter resource.attributes.aws.service.type = "gen_ai_agent"
        | parse resource.attributes.cloud.resource_id "runtime/*/" as parsedAgentId
        | filter parsedAgentId = '{agent_id}'
        | stats max(endTimeUnixNano) as maxEnd by attributes.session.id
        | sort maxEnd desc
        | limit 1"""

        results = self._execute_query(query_string, "aws/spans", start_time_ms, end_time_ms)
        if not results or not results[0]:
            return None

        for field in results[0]:
            if field.get("field") == "attributes.session.id":
                return field.get("value")
        return None

    def _execute_query(
        self,
        query_string: str,
        log_group_name: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> List[dict]:
        """Execute a CloudWatch Logs Insights query and wait for results.

        Args:
            query_string: The query string
            log_group_name: Log group to query
            start_time_ms: Start time in milliseconds since epoch
            end_time_ms: End time in milliseconds since epoch

        Returns:
            List of result row dictionaries
        """
        response = self.logs_client.start_query(
            logGroupName=log_group_name,
            startTime=start_time_ms // 1000,
            endTime=end_time_ms // 1000,
            queryString=query_string,
        )
        query_id = response["queryId"]

        timeout = 60
        poll_interval = 2
        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise TimeoutError(f"Query {query_id} timed out after {timeout} seconds")
            result = self.logs_client.get_query_results(queryId=query_id)
            status = result["status"]
            if status == "Complete":
                return result.get("results", [])
            elif status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Query {query_id} failed with status: {status}")
            time.sleep(poll_interval)

    def _query_runtime_logs_individually(
        self,
        trace_ids: List[str],
        log_group: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> List[dict]:
        """Fallback to query runtime logs one trace at a time."""
        results = []
        for trace_id in trace_ids:
            query = f"""fields @timestamp, @message, spanId, traceId, @logStream
            | filter traceId = '{trace_id}'
            | sort @timestamp asc"""
            try:
                results.extend(self._execute_query(query, log_group, start_time_ms, end_time_ms))
            except Exception as e:
                logger.warning("Failed to query runtime logs for trace %s: %s", trace_id, e)
        return results


def fetch_spans_from_cloudwatch(
    session_id: str,
    event_log_group: str,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    region: str = DEFAULT_REGION,
) -> List[dict]:
    """Fetch ADOT spans from CloudWatch with configurable event log group.

    Convenience function that creates a CloudWatchSpanFetcher and fetches spans.

    ADOT spans are always fetched from aws/spans. Event logs can be fetched from
    any configurable log group.

    Args:
        session_id: Session ID from agent execution
        event_log_group: CloudWatch log group name for event logs
            - For Runtime agents: "/aws/bedrock-agentcore/runtimes/{agent_id}-{endpoint}"
            - For custom agents: Any log group you configured (e.g., "/my-app/agent-events")
        start_time: Start time for log query
        end_time: End time for log query
        region: AWS region (default: from DEFAULT_REGION constant)

    Returns:
        List of ADOT span and log record dictionaries

    Example (Runtime agent):
        >>> from datetime import datetime, timedelta, timezone
        >>> start_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        >>> end_time = datetime.now(timezone.utc)
        >>> spans = fetch_spans_from_cloudwatch(
        ...     session_id="abc-123",
        ...     event_log_group="/aws/bedrock-agentcore/runtimes/my-agent-ABC-DEFAULT",
        ...     start_time=start_time,
        ...     end_time=end_time,
        ... )

    Example (Custom agent):
        >>> spans = fetch_spans_from_cloudwatch(
        ...     session_id="abc-123",
        ...     event_log_group="/my-app/agent-events",
        ...     start_time=start_time,
        ...     end_time=end_time,
        ... )
    """
    helper = CloudWatchSpanHelper(region=region)
    return helper.fetch_spans(session_id, event_log_group, start_time, end_time)
