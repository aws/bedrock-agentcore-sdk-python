"""Span collector abstraction for the evaluation runner."""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List

from bedrock_agentcore._utils.endpoints import DEFAULT_REGION
from bedrock_agentcore.evaluation.utils.cloudwatch_span_helper import CloudWatchSpanHelper

logger = logging.getLogger(__name__)

AWS_SPANS_LOG_GROUP = "aws/spans"


class AgentSpanCollector(ABC):
    """Abstract base class for collecting spans after agent invocation."""

    @abstractmethod
    def collect(self, session_id: str, start_time: datetime, end_time: datetime) -> List[dict]:
        """Collect spans for a given session.

        Args:
            session_id: The session ID to collect spans for.
            start_time: The start time of the session invocation.
            end_time: The end time of the session invocation.

        Returns:
            List of span dictionaries.
        """


class CloudWatchAgentSpanCollector(AgentSpanCollector):
    """Collects spans from CloudWatch using precise attributes.session.id filtering."""

    def __init__(
        self,
        log_group_name: str,
        region: str = DEFAULT_REGION,
        max_wait_seconds: int = 300,
        poll_interval_seconds: int = 30,
    ):
        """Initialize the CloudWatch span collector.

        Args:
            log_group_name: CloudWatch log group name for event logs.
            region: AWS region for CloudWatch client.
            max_wait_seconds: Maximum time to poll for spans before giving up (default 300s).
            poll_interval_seconds: Time between poll attempts (default 30s).
        """
        self.log_group_name = log_group_name
        self.region = region
        self.max_wait_seconds = max_wait_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._helper = CloudWatchSpanHelper(region=region)

    def collect(self, session_id: str, start_time: datetime, end_time: datetime) -> List[dict]:
        """Collect spans from CloudWatch, polling until spans appear or timeout.

        Args:
            session_id: The session ID to collect spans for.
            start_time: The start time of the session invocation.
            end_time: The end time of the session invocation.

        Returns:
            List of ADOT span dictionaries.
        """
        # Widen the query window so spans ingested shortly after the
        # invocation ended are not excluded.  CloudWatch Logs Insights
        # treats endTime as exclusive and ingestion can lag by seconds,
        # so a 60-second buffer avoids missing spans on every retry.
        query_end_time = end_time + timedelta(seconds=60)
        logger.debug(
            "Collecting spans for session_id=%s, log_group=%s, time_range=[%s, %s]",
            session_id,
            self.log_group_name,
            start_time,
            query_end_time,
        )
        deadline = time.monotonic() + self.max_wait_seconds

        while True:
            spans = self._fetch_spans(session_id, start_time, query_end_time)
            logger.debug("fetch_spans returned %d span(s)", len(spans))

            if spans:
                logger.info("Collected %d span(s) for session %s", len(spans), session_id)
                return spans

            if time.monotonic() + self.poll_interval_seconds > deadline:
                logger.warning(
                    "Span collection timed out after %ds for session %s (0 spans found)",
                    self.max_wait_seconds,
                    session_id,
                )
                return spans

            logger.info("No spans found yet, retrying in %ds...", self.poll_interval_seconds)
            time.sleep(self.poll_interval_seconds)

    def _fetch_spans(self, session_id: str, start_time: datetime, end_time: datetime) -> List[dict]:
        """Fetch spans from both aws/spans and the configured log group.

        Queries both log groups with a precise attributes.session.id filter,
        combines results, and returns only valid ADOT span documents.
        """
        query_string = (
            f"fields @timestamp, @message"
            f'\n| filter attributes.session.id = "{session_id}"'
            f"\n| filter ispresent(scope.name)"
            f"\n| filter ispresent(traceId)"
            f"\n| filter ispresent(spanId)"
            f"\n| sort @timestamp asc"
        )

        aws_spans = self._helper.query_log_group(
            AWS_SPANS_LOG_GROUP, session_id, start_time, end_time, query_string=query_string
        )
        event_spans = self._helper.query_log_group(
            self.log_group_name, session_id, start_time, end_time, query_string=query_string
        )

        all_data = aws_spans + event_spans

        logger.info("Fetched %d span items from CloudWatch", len(all_data))
        return all_data
