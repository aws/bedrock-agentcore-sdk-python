"""Dataset type definitions for the AgentCore Experiment Framework.

Defines how evaluation datasets, scenarios, and turns are structured.
"""

from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, model_validator

Input = Union[str, Dict[str, Any]]


class ActorProfile(BaseModel):
    """Profile describing the simulated actor's identity and objective.

    .. warning::
        This feature is in preview and may change in future releases.

    Attributes:
        traits: Characteristics of the actor (e.g. expertise level, communication style).
        context: Background information about the actor.
        goal: What the actor wants to achieve in the interaction.
    """

    traits: Dict[str, Any] = {}
    context: str
    goal: str


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


class SimulatedScenario(Scenario):
    """A scenario driven by a simulated actor in a multi-turn conversation loop.

    .. warning::
        This feature is in preview and may change in future releases.

    Attributes:
        scenario_description: Human-readable description of what this scenario tests.
        actor_profile: Profile defining the simulated actor's traits, context, and goal.
        input: The initial payload sent to the agent to start the conversation.
            Accepts a plain string, a structured dict, or a ``pydantic.BaseModel``
            instance (e.g. an instance of ``SimulationConfig.input_type``).
        max_turns: Maximum number of conversation turns before the simulation stops.
            Defaults to 10.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_description: str = ""
    actor_profile: ActorProfile
    input: Union[str, Dict[str, Any], BaseModel]
    max_turns: int = 10

    @model_validator(mode="after")
    def validate_max_turns(self):
        """Validate that max_turns is at least 1."""
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        return self


class Dataset(BaseModel):
    """A collection of evaluation scenarios."""

    scenarios: List[Scenario]

    @model_validator(mode="after")
    def validate_scenarios(self):
        """Validate that scenarios list is not empty and has unique IDs."""
        if not self.scenarios:
            raise ValueError("scenarios must not be empty")
        seen: set = set()
        duplicates: set = set()
        for s in self.scenarios:
            (duplicates if s.scenario_id in seen else seen).add(s.scenario_id)
        if duplicates:
            raise ValueError(f"Duplicate scenario_ids: {duplicates}")
        return self


class SimulationConfig(BaseModel):
    """Configuration for actor simulation in SimulatedScenario execution.

    .. warning::
        This feature is in preview and may change in future releases.

    Attributes:
        model_id: Bedrock model ID for the actor agent. Uses the Strands
            default model when None.
        system_prompt_template: Jinja2 system prompt template for the actor.
            Must contain an ``{{ actor_profile }}`` placeholder. When None, the built-in
            ``structured_user_simulator.j2`` template is used.
        input_type: Pydantic model class describing the agent's expected input.
            When set, ``input`` values in SimulatedScenario are validated into
            this type for the first agent call. For subsequent turns the actor is
            schema-constrained via tool-use to produce instances of this type directly.
        output_type: Pydantic model class describing the agent's output schema.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: Optional[str] = None
    system_prompt_template: Optional[str] = None
    input_type: Optional[Type[BaseModel]] = None
    output_type: Optional[Type[BaseModel]] = None
