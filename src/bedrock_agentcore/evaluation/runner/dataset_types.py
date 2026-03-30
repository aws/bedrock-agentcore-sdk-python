"""Dataset type definitions for the AgentCore Experiment Framework.

Defines how evaluation datasets, scenarios, and turns are structured.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, model_validator

Input = Union[str, Dict[str, Any]]


class Turn(BaseModel):
    """A single conversational turn in an evaluation scenario."""

    input: Input
    expected_response: Optional[str] = None


class Scenario(BaseModel):
    """Base class for evaluation scenarios."""

    scenario_id: str
    assertions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class PredefinedScenario(Scenario):
    """A scenario with a predefined conversation flow."""

    turns: List[Turn]
    expected_trajectory: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_turns_non_empty(self):
        """Validate that turns list is not empty."""
        if not self.turns:
            raise ValueError("turns must not be empty")
        return self


class Dataset(BaseModel):
    """A collection of evaluation scenarios."""

    scenarios: List[Scenario]

    @model_validator(mode="after")
    def validate_scenarios(self):
        """Validate that scenarios list is not empty and has unique IDs."""
        if not self.scenarios:
            raise ValueError("scenarios must not be empty")
        ids = [s.scenario_id for s in self.scenarios]
        duplicates = set(sid for sid in ids if ids.count(sid) > 1)
        if duplicates:
            raise ValueError(f"Duplicate scenario_ids: {duplicates}")
        return self
