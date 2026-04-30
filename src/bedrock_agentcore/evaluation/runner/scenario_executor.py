"""Scenario executor abstractions for the evaluation framework.

Each ScenarioExecutor subclass owns the invocation logic for a specific scenario type,
keeping the runners agnostic to how turns are produced.
"""

import json
import logging
import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .dataset_types import Scenario, SimulatedScenario, SimulationConfig
from .invoker_types import AgentInvokerFn, AgentInvokerInput
from .prompts import render_template_file, render_template_string

logger = logging.getLogger(__name__)

_INITIAL_GREETINGS: List[str] = [
    "hi! how can I help you today?",
    "hello! what can I assist you with?",
    "hi there! how may I help you?",
    "good day! what can I do for you?",
    "hello! what would you like to know?",
]


@dataclass
class SimulatorResult:
    """Output from a single actor turn in a simulated conversation.

    .. warning::
        This feature is in preview and may change in future releases.

    Attributes:
        message: The actor's next message. An ``input_type`` instance when
            ``input_type`` is configured; a plain ``str`` or ``None`` otherwise.
            ``None`` when ``stop=True`` regardless of whether ``input_type`` is set.
        reasoning: The actor's internal reasoning for this response.
        stop: ``True`` when the actor signals the conversation should end.
        stop_reason: Why the conversation ended: ``"goal_completed"``,
            ``"max_turns"``, or ``None`` when the conversation is still ongoing.
    """

    message: Any
    reasoning: str
    stop: bool
    stop_reason: Optional[str]


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


class SimulatedScenarioExecutor(ScenarioExecutor):
    """Runs a SimulatedScenario using AgentCoreActorSimulator.

    .. warning::
        This feature is in preview and may change in future releases.

    Uses a dynamically-typed structured output model so the LLM is schema-constrained
    via tool-use enforcement to produce correctly-typed messages, eliminating the need
    for JSON parsing heuristics.
    """

    simulation_config: Optional[SimulationConfig] = None

    def run_scenario(self, scenario: Scenario) -> ScenarioExecutionResult:
        """Execute a simulated scenario using an actor-driven conversation loop."""
        if not isinstance(scenario, SimulatedScenario):
            raise TypeError(f"Expected SimulatedScenario, got {type(scenario).__name__}")
        sim_config = self.simulation_config

        start_time = datetime.now(timezone.utc)
        session_id = f"{scenario.scenario_id}-{uuid.uuid4()}"
        turn_count = 0

        try:
            try:
                from strands_evals.simulation.tools.goal_completion import get_conversation_goal_completion
                from strands_evals.types.simulation import ActorProfile as StrandsActorProfile
            except ImportError as e:
                raise ImportError(
                    "strands-agents-evals is required for SimulatedScenario execution. "
                    "Install it with: pip install 'bedrock-agentcore[simulation]'"
                ) from e

            strands_profile = StrandsActorProfile(
                traits=scenario.actor_profile.traits,
                context=scenario.actor_profile.context,
                actor_goal=scenario.actor_profile.goal,
            )

            system_prompt = _render_system_prompt(sim_config, strands_profile, scenario.scenario_description)

            input_type = sim_config.input_type if sim_config else None
            output_type = sim_config.output_type if sim_config else None
            simulator = AgentCoreActorSimulator(
                actor_profile=strands_profile,
                initial_query=_to_string(scenario.input),
                system_prompt=system_prompt,
                input_type=input_type,
                model=sim_config.model_id if sim_config else None,
                max_turns=scenario.max_turns,
                tools=[get_conversation_goal_completion],
            )

            next_payload = _build_payload(scenario.input, sim_config)

            while True:
                turn_count += 1
                logger.debug(
                    "Turn %d for scenario %s (session_id=%s)",
                    turn_count,
                    scenario.scenario_id,
                    session_id,
                )
                output = self.agent_invoker(AgentInvokerInput(payload=next_payload, session_id=session_id))
                sim_result = simulator.act(_extract_agent_output(output.agent_output, output_type))
                logger.debug(
                    "Turn %d actor result: stop=%s, stop_reason=%s",
                    turn_count,
                    sim_result.stop,
                    sim_result.stop_reason,
                )

                if sim_result.stop:
                    logger.info(
                        "Scenario %s: actor ended conversation (reason=%s)",
                        scenario.scenario_id,
                        sim_result.stop_reason,
                    )
                    break

                if turn_count >= scenario.max_turns:
                    logger.warning(
                        "Scenario %s: executor hit max_turns backstop (%d); simulator did not signal stop",
                        scenario.scenario_id,
                        turn_count,
                    )
                    break

                next_payload = sim_result.message

            status = "COMPLETED"
            error = None

        except Exception as e:
            logger.exception("Scenario %s failed: %s", scenario.scenario_id, e)
            status = "FAILED"
            error = str(e)

        end_time = datetime.now(timezone.utc)
        elapsed = (end_time - start_time).total_seconds()
        if status == "COMPLETED":
            logger.info(
                "Scenario %s completed (%d turn(s) in %.1fs), time_range=[%s, %s]",
                scenario.scenario_id,
                turn_count,
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


class AgentCoreActorSimulator:
    """Actor simulator with dynamically-typed structured output.

    .. warning::
        This feature is in preview and may change in future releases.

    Uses a strands ``Agent`` with a per-scenario Pydantic response model whose
    ``message`` field is typed as ``Optional[input_type]`` when ``input_type``
    is provided. The LLM tool-use schema then enforces the correct message
    structure rather than relying on prompt instructions.

    Response model when ``input_type`` is set::

        SimulatorActorResponse(reasoning: str, stop: bool, message: Optional[input_type])

    Response model when ``input_type`` is ``None``::

        SimulatorActorResponse(reasoning: str, stop: bool, message: Optional[str])

    **Conversation history bootstrap**: the actor's strands ``Agent`` is seeded
    with a two-message history before the first real ``act()`` call:

    - ``user`` (agent's synthetic opener): a random greeting from
      ``_INITIAL_GREETINGS``, standing in for the agent saying hello.
    - ``assistant`` (actor's first turn): the ``initial_query`` string derived
      from ``input``.

    This gives the actor the context that it has already sent its opening
    question, so the first real ``act()`` call — which delivers the agent's
    actual response to that question — arrives with a coherent conversation
    history. The greeting is never sent to the real agent; it exists only to
    orient the actor.
    """

    def __init__(
        self,
        actor_profile: Any,
        initial_query: str,
        system_prompt: str,
        input_type: Optional[Type[BaseModel]] = None,
        model: Optional[str] = None,
        max_turns: int = 10,
        tools: Optional[list] = None,
    ):
        """Initialize the simulator, building the response model and seeding conversation history."""
        from strands import Agent

        self._input_type = input_type
        self._max_turns = max_turns
        self._turn_count = 0
        self._response_model = _make_response_model(input_type)

        conversation_history = [
            {"role": "user", "content": [{"text": random.choice(_INITIAL_GREETINGS)}]},
            {"role": "assistant", "content": [{"text": initial_query.strip()}]},
        ]

        self._agent = Agent(
            system_prompt=system_prompt,
            messages=conversation_history,
            tools=tools or [],
            model=model,
            callback_handler=None,
        )

    def act(self, agent_message: str) -> SimulatorResult:
        """Send the agent's response to the actor and return a SimulatorResult."""
        response = self._agent(agent_message.strip(), structured_output_model=self._response_model)
        self._turn_count += 1

        actor_response = response.structured_output
        stop = bool(actor_response.stop) or self._turn_count >= self._max_turns
        stop_reason: Optional[str] = None
        if stop:
            if actor_response.stop:
                stop_reason = "goal_completed"
            else:
                stop_reason = "max_turns"

        message = actor_response.message

        # Guard: actor signalled continue but produced no message — treat as implicit stop.
        if not stop and message is None:
            input_type_suffix = f" (input_type={self._input_type.__name__})" if self._input_type else ""
            logger.warning(
                "Actor produced null message when stop=False; treating as goal_completed%s", input_type_suffix
            )
            stop = True
            stop_reason = "goal_completed"

        return SimulatorResult(
            message=message,
            reasoning=actor_response.reasoning,
            stop=stop,
            stop_reason=stop_reason,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response_model(input_type: Optional[Type[BaseModel]]) -> Type[BaseModel]:
    """Build a dynamic Pydantic model for actor structured output.

    When ``input_type`` is provided, ``message`` is typed as
    ``Optional[input_type]`` so the LLM tool-use schema enforces the correct
    structure on non-stop turns.
    """
    if input_type is None:
        msg_annotation = Optional[str]
        msg_field = Field(
            None,
            description="The actor's next message to send to the agent. Provide when stop=false. Null when stop=true.",
        )
    else:
        msg_annotation = Optional[input_type]
        msg_field = Field(
            None,
            description=(
                f"Structured message matching the agent's input schema ({input_type.__name__}). "
                "Provide when stop=false. Set to null when stop=true."
            ),
        )

    return create_model(
        "SimulatorActorResponse",
        __base__=BaseModel,
        reasoning=(str, Field(..., description="Internal reasoning for this response.")),
        stop=(
            bool,
            Field(
                False,
                description="Set to true when the conversation goal is met or the conversation should end.",
            ),
        ),
        message=(msg_annotation, msg_field),
    )


def _render_system_prompt(
    sim_config: Optional[SimulationConfig],
    strands_profile: Any,
    scenario_description: str = "",
) -> str:
    """Render the actor system prompt from a Jinja2 template.

    Resolution order:
    1. User-supplied ``system_prompt_template`` (rendered as a Jinja2 string).
    2. Built-in ``structured_user_simulator.j2`` (always used when no custom template).

    The ``actor_profile`` dict and optional ``output_schema`` (when ``output_type`` is
    set) are injected as template variables. Structured input typing is enforced via
    the response model's tool-use schema rather than the system prompt.
    """
    actor_profile_data = strands_profile.model_dump()

    if sim_config and sim_config.system_prompt_template:
        kwargs: dict = {"actor_profile": actor_profile_data, "scenario_description": scenario_description}
        if sim_config.output_type:
            kwargs["output_schema"] = json.dumps(sim_config.output_type.model_json_schema(), indent=2)
        return render_template_string(sim_config.system_prompt_template, **kwargs)

    output_schema = (
        json.dumps(sim_config.output_type.model_json_schema(), indent=2)
        if sim_config and sim_config.output_type
        else None
    )
    return render_template_file(
        "structured_user_simulator.j2",
        actor_profile=actor_profile_data,
        scenario_description=scenario_description,
        output_schema=output_schema,
    )


def _to_string(value: Any) -> str:
    """Serialize a value to a plain string."""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _extract_agent_output(agent_output: Any, output_type: Optional[Type[BaseModel]]) -> str:
    """Serialize the agent's output into a string suitable for the actor.

    When ``output_type`` is provided the agent output is validated against that
    schema and re-serialized as canonical JSON.  If parsing fails the output is
    serialized with ``_to_string`` and returned as-is.

    When ``output_type`` is ``None`` the output is serialized directly with
    ``_to_string``.
    """
    if output_type is None:
        return _to_string(agent_output)

    raw = _to_string(agent_output)
    try:
        parsed = output_type.model_validate_json(raw)
    except ValidationError:
        logger.warning("Agent output could not be parsed as %s; passing through as-is", output_type.__name__)
        return raw

    return parsed.model_dump_json()


def _build_payload(input: Any, sim_config: Optional[SimulationConfig]) -> Any:
    """Return the first agent payload, parsing input into input_type when configured."""
    if sim_config and sim_config.input_type:
        if isinstance(input, sim_config.input_type):
            return input
        if isinstance(input, BaseModel):
            # Wrong BaseModel subtype — coerce via its dict representation so
            # Pydantic validates compatibility and raises ValidationError on mismatch
            # rather than silently passing the wrong type to the agent.
            return sim_config.input_type.model_validate(input.model_dump())
        if isinstance(input, dict):
            return sim_config.input_type.model_validate(input)
        if isinstance(input, str):
            return sim_config.input_type.model_validate_json(input)
    return input
