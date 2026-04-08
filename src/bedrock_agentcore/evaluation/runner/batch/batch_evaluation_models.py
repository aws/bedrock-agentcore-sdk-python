"""Data models for batch evaluation: session source configs, evaluator config, and results."""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, alias_generators, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session source configs
# ---------------------------------------------------------------------------


class SessionSourceConfig(ABC):
    """Abstract base for session span sources passed to the evaluation API.

    Subclass this to support any EvaluationSessionSource union member
    (cloudWatchSource or future additions).
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
    def to_api_source(
        self,
        session_ids: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Return the sessionSource dict for the evaluation API call.

        The runner always provides all three arguments after agent invocation.
        Implementations use what they need and ignore the rest.

        Args:
            session_ids: Session IDs generated during agent invocation.
            start_time: Earliest session start time across all invocations.
            end_time: Latest session end time across all invocations.

        Returns:
            Dict matching one member of the EvaluationSessionSource union.
        """


class CloudWatchSessionSourceConfig(BaseModel, SessionSourceConfig):
    """CloudWatch session source — pulls spans from CloudWatch log groups.

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

    def to_api_source(
        self,
        session_ids: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Return a cloudWatchSource session source dict for the evaluation API."""
        return {
            "cloudWatchSource": {
                "serviceNames": self.service_names,
                "logGroupNames": self.log_group_names,
                "sessionInput": {"sessionIds": session_ids},
            }
        }


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class FailedScenario(BaseModel):
    """Information about a scenario that failed during invocation.

    Attributes:
        scenario_id: Scenario identifier.
        error_message: Error description.
    """

    scenario_id: str
    error_message: str


class TokenUsageSummary(BaseModel):
    """Token usage statistics.

    Attributes:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        total_tokens: Total token count.
    """

    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)

    input_tokens: int
    output_tokens: int
    total_tokens: int


class EvaluatorStatistics(BaseModel):
    """Statistics for an evaluator.

    Attributes:
        average_score: Average evaluation score across all evaluations.
        average_token_usage: Average token usage statistics.
    """

    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)

    average_score: Optional[float] = None
    average_token_usage: Optional[TokenUsageSummary] = None


class EvaluatorSummary(BaseModel):
    """Summary statistics for a single evaluator.

    Attributes:
        evaluator_arn: ARN of the evaluator.
        evaluator_id: Evaluator identifier.
        evaluator_name: Human-readable evaluator name.
        statistics: Aggregated statistics (average score and token usage).
        total_evaluated: Number of items evaluated.
        total_failed: Number of evaluation failures.
    """

    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)

    evaluator_arn: Optional[str] = None
    evaluator_id: Optional[str] = None
    evaluator_name: Optional[str] = None
    statistics: Optional[EvaluatorStatistics] = None
    total_evaluated: Optional[int] = None
    total_failed: Optional[int] = None


class BatchEvaluationSummary(BaseModel):
    """Aggregated results from a completed batch evaluation.

    Attributes:
        sessions_completed: Number of sessions that were successfully evaluated.
        sessions_in_progress: Number of sessions still being evaluated (non-zero
            only in intermediate states).
        sessions_failed: Number of sessions that failed evaluation.
        total_sessions: Total number of sessions submitted for evaluation.
        evaluator_summaries: Per-evaluator statistics including average score
            and token usage.
    """

    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)

    sessions_completed: Optional[int] = None
    sessions_in_progress: Optional[int] = None
    sessions_failed: Optional[int] = None
    total_sessions: Optional[int] = None
    evaluator_summaries: Optional[List[EvaluatorSummary]] = None


class CloudWatchOutputDataConfig(BaseModel):
    """CloudWatch destination for batch evaluation output data.

    Attributes:
        log_group_name: CloudWatch log group where evaluation results are written.
        log_stream_name: CloudWatch log stream for this batch evaluation's results.
    """

    log_group_name: str
    log_stream_name: str


class BatchEvaluationResult(BaseModel):
    """Result returned by :py:meth:`BatchEvaluationRunner.run`.

    Attributes:
        batch_evaluate_id: Unique identifier for the batch evaluation job,
            returned by StartBatchEvaluation.
        batch_evaluate_arn: ARN of the batch evaluation resource.
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

    batch_evaluate_id: str
    batch_evaluate_arn: str
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

    Attributes:
        evaluator_ids: List of evaluator IDs (built-in names or custom ARNs).
    """

    evaluator_ids: List[str] = Field(min_length=1)


class BatchEvaluationRunConfig(BaseModel):
    """Configuration for a single batch evaluation run.

    Attributes:
        name: Human-readable name for the batch evaluation job.
        evaluator_config: Evaluators to run (built-in IDs or custom ARNs).
        session_source: Source from which the service reads agent session spans.
            Use ``CloudWatchSessionSourceConfig`` for agents running on AgentCore Runtime.
        max_concurrent_scenarios: Maximum number of scenarios to invoke in
            parallel during the agent invocation phase. Defaults to 5.
        polling_timeout_seconds: Maximum time to wait for the evaluation job
            to reach a terminal state. Defaults to 1800 (30 minutes).
        polling_interval_seconds: Time between GetBatchEvaluation polls.
            Defaults to 30 seconds. Must be less than ``polling_timeout_seconds``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    evaluator_config: BatchEvaluatorConfig
    session_source: SessionSourceConfig
    max_concurrent_scenarios: int = 5
    polling_timeout_seconds: int = 1800
    polling_interval_seconds: int = 30

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
