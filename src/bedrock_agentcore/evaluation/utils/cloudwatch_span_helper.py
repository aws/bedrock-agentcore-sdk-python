"""Fetch ADOT spans from CloudWatch for evaluation."""

import json
import logging
import time
from datetime import datetime
from typing import Any, List

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
    ) -> List[dict]:
        """Query a single CloudWatch log group for session data.

        Args:
            log_group_name: Name of the log group to query
            session_id: Session ID to filter by
            start_time: Query start time
            end_time: Query end time

        Returns:
            List of parsed JSON log messages
        """
        query_string = f"""fields @timestamp, @message
        | filter @message like "{session_id}"
        | sort @timestamp asc"""

        max_attempts = 30
        initial_backoff = 0.5
        max_backoff = 5.0

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

        Returns:
            List of ADOT span and log record dictionaries

        Example (Runtime agent):
            >>> from datetime import datetime, timedelta, timezone
            >>> helper = CloudWatchSpanHelper(region="us-west-2")
            >>> start_time = datetime.now(timezone.utc) - timedelta(minutes=10)
            >>> spans = fetcher.fetch_spans(
            ...     session_id="abc-123",
            ...     event_log_group="/aws/bedrock-agentcore/runtimes/my-agent-ABC-DEFAULT",
            ...     start_time=start_time
            ... )

        Example (Custom agent):
            >>> spans = fetcher.fetch_spans(
            ...     session_id="abc-123",
            ...     event_log_group="/my-app/agent-events",
            ...     start_time=start_time
            ... )
        """
        end_time = datetime.now()

        # Query both log groups
        aws_spans = self.query_log_group("aws/spans", session_id, start_time, end_time)
        event_logs = self.query_log_group(event_log_group, session_id, start_time, end_time)

        # Combine and validate
        all_data = aws_spans + event_logs
        valid_items = [item for item in all_data if _is_valid_adot_document(item)]

        logger.info("Fetched %d valid ADOT items from CloudWatch", len(valid_items))
        return valid_items


def fetch_spans_from_cloudwatch(
    session_id: str,
    event_log_group: str,
    start_time: datetime,
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
        region: AWS region (default: from DEFAULT_REGION constant)

    Returns:
        List of ADOT span and log record dictionaries

    Example (Runtime agent):
        >>> from datetime import datetime, timedelta, timezone
        >>> start_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        >>> spans = fetch_spans_from_cloudwatch(
        ...     session_id="abc-123",
        ...     event_log_group="/aws/bedrock-agentcore/runtimes/my-agent-ABC-DEFAULT",
        ...     start_time=start_time
        ... )

    Example (Custom agent):
        >>> spans = fetch_spans_from_cloudwatch(
        ...     session_id="abc-123",
        ...     event_log_group="/my-app/agent-events",
        ...     start_time=start_time
        ... )
    """
    helper = CloudWatchSpanHelper(region=region)
    return helper.fetch_spans(session_id, event_log_group, start_time)
