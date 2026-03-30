"""Scenario executor abstractions for the evaluation framework.

Each ScenarioExecutor subclass owns the invocation logic for a specific scenario type,
keeping the OnDemandEvaluationDatasetRunner agnostic to how turns are produced.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict

from .dataset_types import Scenario
from .invoker_types import AgentInvokerFn, AgentInvokerInput

logger = logging.getLogger(__name__)


class ScenarioExecutionResult(BaseModel):
    """Return value from a scenario execution."""

    scenario_id: str
    session_id: str
    start_time: datetime
    end_time: datetime
    status: str  # "COMPLETED" or "FAILED"
    error: Optional[str] = None


class ScenarioExecutor(BaseModel, ABC):
    """Invokes the test subject for a single scenario."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_invoker: AgentInvokerFn

    @abstractmethod
    def run_scenario(self, scenario: Scenario) -> ScenarioExecutionResult:
        """Execute the scenario and return the result."""


class PredefinedScenarioExecutor(ScenarioExecutor):
    """Runs a PredefinedScenario by iterating its explicit turns."""

    def run_scenario(self, scenario: Scenario) -> ScenarioExecutionResult:
        """Execute a predefined scenario by invoking the agent for each turn."""
        logger.debug("Running scenario %s (%d turn(s))", scenario.scenario_id, len(scenario.turns))
        start_time = datetime.now(timezone.utc)
        session_id = f"{scenario.scenario_id}-{uuid.uuid4()}"
        logger.debug("Generated session_id %s for scenario %s", session_id, scenario.scenario_id)
        try:
            for turn_idx, turn in enumerate(scenario.turns, 1):
                logger.debug(
                    "Invoking turn %d/%d for scenario %s (session_id=%s)",
                    turn_idx,
                    len(scenario.turns),
                    scenario.scenario_id,
                    session_id,
                )
                self.agent_invoker(
                    AgentInvokerInput(
                        payload=turn.input,
                        session_id=session_id,
                    )
                )
            status = "COMPLETED"
            error = None
        except Exception as e:
            logger.exception("Scenario %s failed at invocation: %s", scenario.scenario_id, e)
            status = "FAILED"
            error = str(e)
        end_time = datetime.now(timezone.utc)
        elapsed = (end_time - start_time).total_seconds()
        if status == "COMPLETED":
            logger.info(
                "Scenario %s completed (%d turn(s) in %.1fs), time_range=[%s, %s]",
                scenario.scenario_id,
                len(scenario.turns),
                elapsed,
                start_time,
                end_time,
            )
        return ScenarioExecutionResult(
            scenario_id=scenario.scenario_id,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            status=status,
            error=error,
        )
