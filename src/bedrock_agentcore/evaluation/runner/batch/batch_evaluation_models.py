"""Data models for batch evaluation: session source configs, evaluator config, and results."""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, alias_generators, model_validator

from bedrock_agentcore.evaluation.runner.dataset_types import SimulationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data source configs
# ---------------------------------------------------------------------------


class DataSourceConfig(ABC):
    """Abstract base for session span sources passed to the evaluation API.

    .. warning::
        This feature is in preview and may change in future releases.

    Subclass this to support any DataSourceConfig union member
    (cloudWatchLogs or future additions).
    """

    def pre_evaluation_run_hook(self) -> None:
        """Called by the runner after agent invocation, before the evaluation API call.

        Override to add source-specific pre-run behavior such as waiting for
        span ingestion or validating that spans are available.

        Note:
            Implementations may block the calling thread (e.g. to wait for
            CloudWatch ingestion). The runner invokes this synchronously, so
            long-running hooks will delay the evaluation API call by the full
            duration of the hook.
        """
        return None

    @abstractmethod
    def to_data_source_config(
        self,
        session_ids: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Return the dataSourceConfig dict for the evaluation API call.

        The runner always provides all three arguments after agent invocation.
        Implementations use what they need and ignore the rest.

        Args:
            session_ids: Session IDs generated during agent invocation.
            start_time: Earliest session start time across all invocations.
            end_time: Latest session end time across all invocations.

        Returns:
            Dict matching one member of the DataSourceConfig union.
        """


class CloudWatchDataSourceConfig(BaseModel, DataSourceConfig):
    """CloudWatch data source — pulls spans from CloudWatch log groups.

    .. warning::
        This feature is in preview and may change in future releases.

    Attributes:
        service_names: Service names for span filtering. The API accepts exactly one (list of length 1).
        log_group_names: CloudWatch log group names to search (1–5).
        ingestion_delay_seconds: Seconds to wait for spans to appear in
            CloudWatch before submitting the evaluation run. Defaults to 180.
            This sleep blocks the calling thread for the full duration; set
            to 0 to skip the wait.
    """

    service_names: List[str] = Field(min_length=1, max_length=1)
    log_group_names: List[str] = Field(min_length=1, max_length=5)
    ingestion_delay_seconds: int = Field(default=180, ge=0)

    def pre_evaluation_run_hook(self) -> None:
        """Wait for CloudWatch span ingestion before submitting the evaluation run."""
        if self.ingestion_delay_seconds > 0:
            logger.info("Waiting %ds for CloudWatch span ingestion...", self.ingestion_delay_seconds)
            time.sleep(self.ingestion_delay_seconds)

    def to_data_source_config(
        self,
        session_ids: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Return a cloudWatchLogs dataSourceConfig dict for the evaluation API."""
        return {
            "cloudWatchLogs": {
                "serviceNames": self.service_names,
                "logGroupNames": self.log_group_names,
                "filterConfig": {
                    "sessionIds": session_ids,
                    "timeRange": {
                        "startTime": start_time,
                        "endTime": end_time,
                    },
                },
            }
        }


# ---------------------------------------------------------------------------
# Batch eval result models
# ---------------------------------------------------------------------------


class FailedScenario(BaseModel):
    """Information about a scenario that failed during invocation.

    Attributes:
        scenario_id: Scenario identifier.
        error_message: Error description.
    """

    scenario_id: str
    error_message: str


class EvaluatorStatistics(BaseModel):
    """Statistics for an evaluator.

    .. warning::
        This feature is in preview and may change in future releases.

    Attributes:
        average_score: Average evaluation score across all evaluations.
    """

    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)

    average_score: Optional[float] = None


class EvaluatorSummary(BaseModel):
    """Summary statistics for a single evaluator.

    .. warning::
        This feature is in preview and may change in future releases.

    Attributes:
        evaluator_id: Evaluator identifier.
        statistics: Aggregated statistics (average score).
        total_evaluated: Number of items evaluated.
        total_failed: Number of evaluation failures.
    """

    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)

    evaluator_id: Optional[str] = None
    statistics: Optional[EvaluatorStatistics] = None
    total_evaluated: Optional[int] = None
    total_failed: Optional[int] = None


class BatchEvaluationSummary(BaseModel):
    """Aggregated results from a completed batch evaluation.

    .. warning::
        This feature is in preview and may change in future releases.

    Attributes:
        number_of_sessions_completed: Number of sessions that were successfully evaluated.
        number_of_sessions_in_progress: Number of sessions still being evaluated (non-zero
            only in intermediate states).
        number_of_sessions_failed: Number of sessions that failed evaluation.
        total_number_of_sessions: Total number of sessions submitted for evaluation.
        number_of_sessions_ignored: Number of sessions that were ignored.
        evaluator_summaries: Per-evaluator statistics including average score.
    """

    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)

    number_of_sessions_completed: Optional[int] = None
    number_of_sessions_in_progress: Optional[int] = None
    number_of_sessions_failed: Optional[int] = None
    total_number_of_sessions: Optional[int] = None
    number_of_sessions_ignored: Optional[int] = None
    evaluator_summaries: Optional[List[EvaluatorSummary]] = None


class CloudWatchOutputDataConfig(BaseModel):
    """CloudWatch destination for batch evaluation output data.

    .. warning::
        This feature is in preview and may change in future releases.

    Attributes:
        log_group_name: CloudWatch log group where evaluation results are written.
        log_stream_name: CloudWatch log stream for this batch evaluation's results.
    """

    log_group_name: str
    log_stream_name: str


class BatchEvaluationResult(BaseModel):
    """Result returned by :py:meth:`BatchEvaluationRunner.run`.

    .. warning::
        This feature is in preview and may change in future releases.

    Attributes:
        batch_evaluation_id: Unique identifier for the batch evaluation job,
            returned by StartBatchEvaluation.
        batch_evaluation_arn: ARN of the batch evaluation resource.
        batch_evaluation_name: Human-readable name for the batch evaluation job.
        status: Terminal status of the job (e.g. ``"COMPLETED"``).
        created_at: Timestamp when the batch evaluation job was created.
        evaluation_results: Aggregated per-evaluator statistics. Present when
            the job completed successfully; ``None`` otherwise.
        error_details: Service-reported error messages when the job failed.
        agent_invocation_failures: Scenarios that failed during the agent
            invocation phase (before the evaluation job was started). A
            non-empty list does not prevent the job from running — the service
            evaluates only the successfully invoked sessions.
        output_data_config: CloudWatch destination where the service writes
            per-session evaluation result events. Pass to
            :py:meth:`BatchEvaluationRunner.fetch_evaluation_events`
            to read the raw OTel evaluation records.
    """

    batch_evaluation_id: str
    batch_evaluation_arn: str
    batch_evaluation_name: str
    status: str
    created_at: datetime
    evaluation_results: Optional[BatchEvaluationSummary] = None
    error_details: Optional[List[str]] = None
    agent_invocation_failures: List[FailedScenario] = Field(default_factory=list)
    output_data_config: Optional[CloudWatchOutputDataConfig] = None


# ---------------------------------------------------------------------------
# Batch eval config
# ---------------------------------------------------------------------------


class BatchEvaluatorConfig(BaseModel):
    """Configuration for evaluators.

    .. warning::
        This feature is in preview and may change in future releases.

    Attributes:
        evaluator_ids: List of evaluator IDs (built-in names or custom ARNs).
    """

    evaluator_ids: List[str] = Field(min_length=1)


class BatchEvaluationRunConfig(BaseModel):
    """Configuration for a single batch evaluation run.

    .. warning::
        This feature is in preview and may change in future releases.

    Attributes:
        batch_evaluation_name: Human-readable name for the batch evaluation job.
        evaluator_config: Evaluators to run (built-in IDs or custom ARNs).
        data_source: Source from which the service reads agent session spans.
            Use ``CloudWatchDataSourceConfig`` for agents running on AgentCore Runtime.
        max_concurrent_scenarios: Maximum number of scenarios to invoke in
            parallel during the agent invocation phase. Defaults to 5.
        polling_timeout_seconds: Maximum time to wait for the evaluation job
            to reach a terminal state. Defaults to 1800 (30 minutes).
        polling_interval_seconds: Time between GetBatchEvaluation polls.
            Defaults to 30 seconds. Must be less than ``polling_timeout_seconds``.
        simulation_config: Actor simulation settings. Required when the dataset
            contains SimulatedScenario entries.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_evaluation_name: str
    description: Optional[str] = None
    evaluator_config: BatchEvaluatorConfig
    data_source: DataSourceConfig
    max_concurrent_scenarios: int = 5
    polling_timeout_seconds: int = 1800
    polling_interval_seconds: int = 30
    simulation_config: Optional[SimulationConfig] = None

    @model_validator(mode="after")
    def validate_polling(self):
        """Validate that polling_timeout_seconds > polling_interval_seconds and max_concurrent_scenarios > 0."""
        if self.polling_timeout_seconds <= self.polling_interval_seconds:
            raise ValueError(
                f"polling_timeout_seconds ({self.polling_timeout_seconds}) must be greater than "
                f"polling_interval_seconds ({self.polling_interval_seconds})"
            )
        if self.max_concurrent_scenarios <= 0:
            raise ValueError("max_concurrent_scenarios must be > 0")
        return self
