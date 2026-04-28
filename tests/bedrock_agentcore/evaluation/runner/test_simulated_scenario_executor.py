"""Unit tests for SimulatedScenario types, SimulationConfig, and SimulatedScenarioExecutor."""

import builtins
import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from bedrock_agentcore.evaluation.runner.dataset_types import ActorProfile, SimulatedScenario, SimulationConfig
from bedrock_agentcore.evaluation.runner.invoker_types import AgentInvokerInput, AgentInvokerOutput
from bedrock_agentcore.evaluation.runner.prompts import render_template_file, render_template_string
from bedrock_agentcore.evaluation.runner.scenario_executor import (
    AgentCoreActorSimulator,
    SimulatedScenarioExecutor,
    SimulatorResult,
    _build_payload,
    _extract_agent_output,
    _make_response_model,
    _render_system_prompt,
    _to_string,
)

# ---------------------------------------------------------------------------
# Shared Pydantic models used across tests
# ---------------------------------------------------------------------------


class OrderRequest(BaseModel):
    item: str
    quantity: int


class OrderConfirmation(BaseModel):
    order_id: str
    status: str


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def actor_profile():
    return ActorProfile(
        traits={"expertise": "novice", "tone": "friendly"},
        context="A customer ordering food online",
        goal="Place a pizza order successfully",
    )


@pytest.fixture
def simulated_scenario(actor_profile):
    return SimulatedScenario(
        scenario_id="sim-1",
        scenario_description="Customer orders pizza",
        actor_profile=actor_profile,
        input={"item": "pizza", "quantity": 2},
        max_turns=5,
    )


@pytest.fixture
def mock_strands():
    """Patch strands imports used by AgentCoreActorSimulator and SimulatedScenarioExecutor."""
    mock_agent_instance = MagicMock()
    mock_agent_cls = MagicMock(return_value=mock_agent_instance)
    mock_profile_cls = MagicMock()
    mock_goal_tool = MagicMock()

    modules = {
        "strands": MagicMock(Agent=mock_agent_cls),
        "strands_evals": MagicMock(),
        "strands_evals.simulation": MagicMock(),
        "strands_evals.simulation.tools": MagicMock(),
        "strands_evals.simulation.tools.goal_completion": MagicMock(
            get_conversation_goal_completion=mock_goal_tool
        ),
        "strands_evals.types": MagicMock(),
        "strands_evals.types.simulation": MagicMock(ActorProfile=mock_profile_cls),
    }
    with patch.dict(sys.modules, modules):
        yield mock_agent_cls, mock_agent_instance, mock_profile_cls


def _make_actor_side_effect(messages, stops=None):
    """Build a side_effect for mock_agent_instance that cycles through messages/stops.

    Each call to the mock agent (i.e. each simulator.act() call) returns a
    MagicMock whose structured_output mimics SimulatorActorResponse fields.
    """
    if stops is None:
        # Default: all non-stop until the last message, then stop
        stops = [False] * (len(messages) - 1) + [True] if messages else [True]

    call_count = {"n": 0}

    def side_effect(agent_msg, structured_output_model=None, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        result = MagicMock()
        result.structured_output.message = messages[idx] if idx < len(messages) else None
        result.structured_output.stop = stops[idx] if idx < len(stops) else True
        result.structured_output.reasoning = "test reasoning"
        return result

    return side_effect


# ---------------------------------------------------------------------------
# TestActorProfile
# ---------------------------------------------------------------------------


class TestActorProfile:
    def test_construction_with_all_fields(self):
        profile = ActorProfile(
            traits={"expertise": "expert"},
            context="A senior engineer",
            goal="Debug a production issue",
        )
        assert profile.traits == {"expertise": "expert"}
        assert profile.context == "A senior engineer"
        assert profile.goal == "Debug a production issue"

    def test_traits_defaults_to_empty_dict(self):
        profile = ActorProfile(context="Some context", goal="Some goal")
        assert profile.traits == {}

    def test_context_required(self):
        with pytest.raises(ValidationError):
            ActorProfile(goal="Some goal")

    def test_goal_required(self):
        with pytest.raises(ValidationError):
            ActorProfile(context="Some context")


# ---------------------------------------------------------------------------
# TestSimulatedScenario
# ---------------------------------------------------------------------------


class TestSimulatedScenario:
    def test_construction(self, actor_profile):
        scenario = SimulatedScenario(
            scenario_id="s1",
            scenario_description="desc",
            actor_profile=actor_profile,
            input="hello",
        )
        assert scenario.scenario_id == "s1"
        assert scenario.max_turns == 10  # default

    def test_actor_profile_must_be_actor_profile_type(self):
        with pytest.raises(ValidationError):
            SimulatedScenario(
                scenario_id="s1",
                scenario_description="desc",
                actor_profile="plain string",
                input="hello",
            )

    def test_max_turns_must_be_at_least_one(self, actor_profile):
        with pytest.raises(ValidationError):
            SimulatedScenario(
                scenario_id="s1",
                scenario_description="desc",
                actor_profile=actor_profile,
                input="hello",
                max_turns=0,
            )

    def test_input_accepts_dict(self, actor_profile):
        scenario = SimulatedScenario(
            scenario_id="s1",
            scenario_description="desc",
            actor_profile=actor_profile,
            input={"key": "value"},
        )
        assert scenario.input == {"key": "value"}


# ---------------------------------------------------------------------------
# TestSimulationConfig
# ---------------------------------------------------------------------------


class TestSimulationConfig:
    def test_all_fields_optional(self):
        config = SimulationConfig()
        assert config.model_id is None
        assert config.system_prompt_template is None
        assert config.input_type is None
        assert config.output_type is None

    def test_accepts_pydantic_model_classes(self):
        config = SimulationConfig(input_type=OrderRequest, output_type=OrderConfirmation)
        assert config.input_type is OrderRequest
        assert config.output_type is OrderConfirmation

    def test_accepts_model_id_and_template(self):
        config = SimulationConfig(
            model_id="us.amazon.nova-lite-v1:0",
            system_prompt_template="You are: {{ actor_profile }}",
        )
        assert config.model_id == "us.amazon.nova-lite-v1:0"


# ---------------------------------------------------------------------------
# TestToString
# ---------------------------------------------------------------------------


class TestToString:
    def test_str_passthrough(self):
        assert _to_string("hello") == "hello"

    def test_dict_becomes_json(self):
        result = _to_string({"a": 1})
        assert json.loads(result) == {"a": 1}

    def test_pydantic_model_becomes_json(self):
        model = OrderRequest(item="pizza", quantity=2)
        result = _to_string(model)
        parsed = json.loads(result)
        assert parsed["item"] == "pizza"
        assert parsed["quantity"] == 2

    def test_other_types_use_str(self):
        assert _to_string(42) == "42"


# ---------------------------------------------------------------------------
# TestBuildFirstPayload
# ---------------------------------------------------------------------------


class TestBuildFirstPayload:
    def test_dict_parsed_into_input_type(self):
        config = SimulationConfig(input_type=OrderRequest)
        result = _build_payload({"item": "pizza", "quantity": 2}, config)
        assert isinstance(result, OrderRequest)
        assert result.item == "pizza"

    def test_json_string_parsed_into_input_type(self):
        config = SimulationConfig(input_type=OrderRequest)
        result = _build_payload('{"item": "burger", "quantity": 1}', config)
        assert isinstance(result, OrderRequest)
        assert result.item == "burger"

    def test_invalid_json_string_raises(self):
        config = SimulationConfig(input_type=OrderRequest)
        with pytest.raises(ValidationError):
            _build_payload("not json", config)

    def test_no_input_type_returns_dict_unchanged(self):
        result = _build_payload({"item": "pizza"}, None)
        assert result == {"item": "pizza"}

    def test_no_input_type_returns_string_unchanged(self):
        result = _build_payload("hello", None)
        assert result == "hello"

    def test_correct_basemodel_instance_returned_unchanged(self):
        config = SimulationConfig(input_type=OrderRequest)
        instance = OrderRequest(item="pizza", quantity=2)
        result = _build_payload(instance, config)
        assert result is instance

    def test_wrong_basemodel_compatible_fields_coerced(self):
        # A different BaseModel with the same fields — should be coerced to input_type.
        class AnotherRequest(BaseModel):
            item: str
            quantity: int

        config = SimulationConfig(input_type=OrderRequest)
        result = _build_payload(AnotherRequest(item="sushi", quantity=3), config)
        assert isinstance(result, OrderRequest)
        assert result.item == "sushi"

    def test_wrong_basemodel_incompatible_fields_raises(self):
        # OrderConfirmation has different fields — coercion should raise ValidationError.
        config = SimulationConfig(input_type=OrderRequest)
        with pytest.raises(ValidationError):
            _build_payload(OrderConfirmation(order_id="1", status="ok"), config)


# ---------------------------------------------------------------------------
# TestExtractAgentOutput
# ---------------------------------------------------------------------------


class TestExtractAgentOutput:
    def test_no_output_type_returns_string_unchanged(self):
        assert _extract_agent_output("plain text", None) == "plain text"

    def test_no_output_type_serializes_dict(self):
        result = _extract_agent_output({"answer": "42"}, None)
        assert json.loads(result) == {"answer": "42"}

    def test_valid_output_returns_canonical_json(self):
        class AgentAnswer(BaseModel):
            answer: str

        result = _extract_agent_output('{"answer":  "Hello world"}', AgentAnswer)
        assert json.loads(result) == {"answer": "Hello world"}

    def test_valid_pydantic_instance_returns_canonical_json(self):
        class AgentAnswer(BaseModel):
            answer: str

        result = _extract_agent_output(AgentAnswer(answer="42"), AgentAnswer)
        assert json.loads(result) == {"answer": "42"}

    def test_plain_text_falls_back_when_parse_fails(self):
        class OrderConfirmation(BaseModel):
            order_id: str
            status: str

        plain = "Here is your order summary: item confirmed."
        result = _extract_agent_output(plain, OrderConfirmation)
        assert result == plain


# ---------------------------------------------------------------------------
# TestMakeResponseModel
# ---------------------------------------------------------------------------


class TestMakeResponseModel:
    def test_no_input_type_message_is_str(self):
        model_cls = _make_response_model(None)
        instance = model_cls(reasoning="r", stop=False, message="hello")
        assert instance.message == "hello"

    def test_with_input_type_message_is_optional_input_type(self):
        model_cls = _make_response_model(OrderRequest)
        order = OrderRequest(item="pizza", quantity=1)
        instance = model_cls(reasoning="r", stop=False, message=order)
        assert isinstance(instance.message, OrderRequest)
        assert instance.message.item == "pizza"

    def test_with_input_type_message_can_be_none_on_stop(self):
        model_cls = _make_response_model(OrderRequest)
        instance = model_cls(reasoning="r", stop=True, message=None)
        assert instance.message is None
        assert instance.stop is True

    def test_stop_defaults_to_false(self):
        model_cls = _make_response_model(None)
        instance = model_cls(reasoning="r", message="hi")
        assert instance.stop is False

    def test_reasoning_required(self):
        model_cls = _make_response_model(None)
        with pytest.raises(Exception):
            model_cls(stop=False, message="hi")


# ---------------------------------------------------------------------------
# TestPromptTemplates
# ---------------------------------------------------------------------------


class TestPromptTemplates:
    def test_structured_template_renders_actor_profile(self):
        result = render_template_file(
            "structured_user_simulator.j2",
            actor_profile={"context": "a tester", "goal": "find bugs", "traits": {}},
        )
        assert "a tester" in result
        assert "find bugs" in result

    def test_structured_template_renders_scenario_description(self):
        result = render_template_file(
            "structured_user_simulator.j2",
            actor_profile={"context": "ctx", "goal": "goal", "traits": {}},
            scenario_description="Customer orders pizza",
        )
        assert "Customer orders pizza" in result

    def test_structured_template_renders_output_schema(self):
        schema = json.dumps(OrderConfirmation.model_json_schema(), indent=2)
        result = render_template_file(
            "structured_user_simulator.j2",
            actor_profile={"context": "ctx", "goal": "goal", "traits": {}},
            output_schema=schema,
        )
        assert "order_id" in result or "OrderConfirmation" in result

    def test_structured_template_no_input_schema_section(self):
        """Template must not contain JSON output instructions (handled by response model)."""
        result = render_template_file(
            "structured_user_simulator.j2",
            actor_profile={"context": "ctx", "goal": "goal", "traits": {}},
        )
        assert "MUST be a valid JSON object" not in result
        assert "MANDATORY OUTPUT FORMAT" not in result

    def test_structured_template_contains_stop_instruction(self):
        result = render_template_file(
            "structured_user_simulator.j2",
            actor_profile={"context": "ctx", "goal": "goal", "traits": {}},
        )
        assert "stop=true" in result

    def test_render_template_string_substitutes_variables(self):
        template = "Hello {{ name }}, your goal is {{ goal }}."
        result = render_template_string(template, name="Alice", goal="test the agent")
        assert result == "Hello Alice, your goal is test the agent."

    def test_render_template_string_handles_json_curly_braces(self):
        """Jinja2 must not misinterpret JSON { } as template syntax."""
        template = "Schema: {{ schema }}"
        schema = '{"type": "object", "properties": {"x": {"type": "string"}}}'
        result = render_template_string(template, schema=schema)
        assert schema in result

    def test_render_system_prompt_output_type_injected(self, mock_strands):
        """output_type schema must appear in the rendered prompt."""
        _, _, mock_profile_cls = mock_strands
        mock_profile = MagicMock()
        mock_profile.model_dump.return_value = {"traits": {}, "context": "ctx", "actor_goal": "goal"}

        config = SimulationConfig(output_type=OrderConfirmation)
        rendered = _render_system_prompt(config, mock_profile)

        assert "order_id" in rendered or "OrderConfirmation" in rendered
        assert "None" not in rendered

    def test_render_system_prompt_scenario_description_injected(self, mock_strands):
        mock_profile = MagicMock()
        mock_profile.model_dump.return_value = {"traits": {}, "context": "ctx", "actor_goal": "goal"}

        rendered = _render_system_prompt(None, mock_profile, scenario_description="Customer orders pizza")

        assert "Customer orders pizza" in rendered

    def test_render_system_prompt_always_uses_builtin_template(self, mock_strands):
        """Built-in template is always used (no external default fallback)."""
        mock_profile = MagicMock()
        mock_profile.model_dump.return_value = {"traits": {}, "context": "ctx", "actor_goal": "goal"}

        rendered = _render_system_prompt(None, mock_profile)

        # Built-in template always contains the stop instruction
        assert "stop=true" in rendered

    def test_render_system_prompt_no_input_schema_in_output(self, mock_strands):
        """input_type must NOT inject a JSON output schema into the system prompt."""
        mock_profile = MagicMock()
        mock_profile.model_dump.return_value = {"traits": {}, "context": "ctx", "actor_goal": "goal"}

        config = SimulationConfig(input_type=OrderRequest)
        rendered = _render_system_prompt(config, mock_profile)

        assert "MUST be a valid JSON object" not in rendered
        assert "item" not in rendered  # OrderRequest field must not appear in prompt


# ---------------------------------------------------------------------------
# TestSimulatedScenarioExecutor
# ---------------------------------------------------------------------------


class TestSimulatedScenarioExecutor:
    def test_successful_single_turn(self, simulated_scenario, mock_strands):
        """Actor stops after first turn — agent invoked exactly once."""
        mock_agent_cls, mock_agent_instance, _ = mock_strands
        mock_agent_instance.side_effect = _make_actor_side_effect(
            messages=[None], stops=[True]
        )

        invoker_calls = []

        def invoker(inp):
            invoker_calls.append(inp)
            return AgentInvokerOutput(agent_output="agent response")

        executor = SimulatedScenarioExecutor(agent_invoker=invoker)
        result = executor.run_scenario(simulated_scenario)

        assert result.status == "COMPLETED"
        assert result.scenario_id == "sim-1"
        assert result.session_id.startswith("sim-1-")
        assert result.error is None
        assert result.start_time < result.end_time
        assert len(invoker_calls) == 1

    def test_successful_multi_turn(self, simulated_scenario, mock_strands):
        """Actor continues for 2 turns then stops — 3 agent calls total."""
        mock_agent_cls, mock_agent_instance, _ = mock_strands
        mock_agent_instance.side_effect = _make_actor_side_effect(
            messages=["second msg", "third msg", None],
            stops=[False, False, True],
        )

        invoker_calls = []

        def invoker(inp):
            invoker_calls.append(inp)
            return AgentInvokerOutput(agent_output="agent response")

        executor = SimulatedScenarioExecutor(agent_invoker=invoker)
        result = executor.run_scenario(simulated_scenario)

        assert result.status == "COMPLETED"
        assert len(invoker_calls) == 3

    def test_same_session_id_across_all_turns(self, simulated_scenario, mock_strands):
        mock_agent_cls, mock_agent_instance, _ = mock_strands
        mock_agent_instance.side_effect = _make_actor_side_effect(
            messages=["msg 1", None], stops=[False, True]
        )

        session_ids = []

        def invoker(inp):
            session_ids.append(inp.session_id)
            return AgentInvokerOutput(agent_output="response")

        executor = SimulatedScenarioExecutor(agent_invoker=invoker)
        executor.run_scenario(simulated_scenario)

        assert len(set(session_ids)) == 1, "session_id must be identical across all turns"

    def test_agent_invoker_failure_marks_scenario_failed(self, simulated_scenario, mock_strands):
        mock_agent_cls, mock_agent_instance, _ = mock_strands
        mock_agent_instance.side_effect = _make_actor_side_effect(
            messages=["msg"], stops=[True]
        )

        def failing_invoker(inp):
            raise RuntimeError("agent exploded")

        executor = SimulatedScenarioExecutor(agent_invoker=failing_invoker)
        result = executor.run_scenario(simulated_scenario)

        assert result.status == "FAILED"
        assert "agent exploded" in result.error

    def test_actor_stops_at_max_turns(self, actor_profile, mock_strands):
        """Actor never signals stop — max_turns enforces the limit."""
        mock_agent_cls, mock_agent_instance, _ = mock_strands
        # Actor always returns stop=False; turn count check ends the loop.
        mock_agent_instance.side_effect = _make_actor_side_effect(
            messages=["keep going"] * 10,
            stops=[False] * 10,
        )

        scenario = SimulatedScenario(
            scenario_id="s-maxturn",
            scenario_description="desc",
            actor_profile=actor_profile,
            input="hello",
            max_turns=3,
        )

        invoker_calls = []

        def invoker(inp):
            invoker_calls.append(inp)
            return AgentInvokerOutput(agent_output="response")

        executor = SimulatedScenarioExecutor(agent_invoker=invoker)
        result = executor.run_scenario(scenario)

        assert result.status == "COMPLETED"
        assert mock_agent_instance.call_count == 3

    def test_strands_not_installed_returns_failed(self, simulated_scenario):
        _real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name.startswith("strands"):
                raise ImportError(f"No module named {name!r}")
            return _real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_blocking_import):
            executor = SimulatedScenarioExecutor(agent_invoker=lambda inp: AgentInvokerOutput(agent_output=""))
            result = executor.run_scenario(simulated_scenario)

        assert result.status == "FAILED"
        assert "strands-agents-evals" in result.error

    def test_input_type_parses_input_dict(self, actor_profile, mock_strands):
        """input dict is validated into input_type for the first agent call."""
        mock_agent_cls, mock_agent_instance, _ = mock_strands
        mock_agent_instance.side_effect = _make_actor_side_effect(
            messages=[None], stops=[True]
        )

        scenario = SimulatedScenario(
            scenario_id="s-typed",
            scenario_description="desc",
            actor_profile=actor_profile,
            input={"item": "pizza", "quantity": 2},
        )

        received = []

        def invoker(inp):
            received.append(inp.payload)
            return AgentInvokerOutput(agent_output="ok")

        config = SimulationConfig(input_type=OrderRequest)
        executor = SimulatedScenarioExecutor(agent_invoker=invoker, simulation_config=config)
        executor.run_scenario(scenario)

        assert len(received) == 1
        assert isinstance(received[0], OrderRequest)
        assert received[0].item == "pizza"

    def test_input_type_message_passed_to_agent_as_typed_instance(self, actor_profile, mock_strands):
        """Actor's input_type message is passed directly to the invoker as a typed instance."""
        mock_agent_cls, mock_agent_instance, _ = mock_strands
        burger_order = OrderRequest(item="burger", quantity=1)
        mock_agent_instance.side_effect = _make_actor_side_effect(
            messages=[burger_order, None],
            stops=[False, True],
        )

        scenario = SimulatedScenario(
            scenario_id="s-typed-msg",
            scenario_description="desc",
            actor_profile=actor_profile,
            input={"item": "pizza", "quantity": 2},
        )

        received = []

        def invoker(inp):
            received.append(inp.payload)
            return AgentInvokerOutput(agent_output="ok")

        config = SimulationConfig(input_type=OrderRequest)
        executor = SimulatedScenarioExecutor(agent_invoker=invoker, simulation_config=config)
        executor.run_scenario(scenario)

        # Turn 1: input → OrderRequest
        # Turn 2: actor's typed message → OrderRequest (same type, consistent)
        assert len(received) == 2
        assert isinstance(received[0], OrderRequest)
        assert isinstance(received[1], OrderRequest)
        assert received[1].item == "burger"

    def test_null_message_on_continue_treated_as_stop(self, actor_profile, mock_strands):
        """If actor returns stop=False but message=None with input_type set, treat as goal_completed."""
        mock_agent_cls, mock_agent_instance, _ = mock_strands
        mock_agent_instance.side_effect = _make_actor_side_effect(
            messages=[None], stops=[False]  # stop=False but message=None
        )

        scenario = SimulatedScenario(
            scenario_id="s-null-msg",
            scenario_description="desc",
            actor_profile=actor_profile,
            input={"item": "pizza", "quantity": 1},
        )

        config = SimulationConfig(input_type=OrderRequest)
        executor = SimulatedScenarioExecutor(
            agent_invoker=lambda inp: AgentInvokerOutput(agent_output="ok"),
            simulation_config=config,
        )
        result = executor.run_scenario(scenario)

        assert result.status == "COMPLETED"

    def test_agent_output_serialized_for_actor(self, actor_profile, mock_strands):
        """Agent output is serialized with _to_string and passed as-is to the actor."""
        mock_agent_cls, mock_agent_instance, _ = mock_strands
        mock_agent_instance.side_effect = _make_actor_side_effect(
            messages=[None], stops=[True]
        )

        scenario = SimulatedScenario(
            scenario_id="s-output-single",
            scenario_description="desc",
            actor_profile=actor_profile,
            input="hello",
        )

        class AgentAnswer(BaseModel):
            answer: str

        agent_output = AgentAnswer(answer="Here is the answer.")

        def invoker(inp):
            return AgentInvokerOutput(agent_output=agent_output)

        config = SimulationConfig(output_type=AgentAnswer)
        executor = SimulatedScenarioExecutor(agent_invoker=invoker, simulation_config=config)
        executor.run_scenario(scenario)

        mock_agent_instance.assert_called_once()
        actual_arg = mock_agent_instance.call_args[0][0]
        # Full serialized output passed to actor unchanged
        assert json.loads(actual_arg) == {"answer": "Here is the answer."}

    def test_agent_output_multi_field_schema_passes_json_to_actor(self, actor_profile, mock_strands):
        """When output_type has multiple fields, actor receives the full JSON string."""
        mock_agent_cls, mock_agent_instance, _ = mock_strands
        mock_agent_instance.side_effect = _make_actor_side_effect(
            messages=[None], stops=[True]
        )

        scenario = SimulatedScenario(
            scenario_id="s-output-multi",
            scenario_description="desc",
            actor_profile=actor_profile,
            input="hello",
        )

        agent_output = OrderConfirmation(order_id="ORD-99", status="confirmed")

        def invoker(inp):
            return AgentInvokerOutput(agent_output=agent_output)

        config = SimulationConfig(output_type=OrderConfirmation)
        executor = SimulatedScenarioExecutor(agent_invoker=invoker, simulation_config=config)
        executor.run_scenario(scenario)

        mock_agent_instance.assert_called_once()
        actual_arg = mock_agent_instance.call_args[0][0]
        # Multi-field schema — full JSON passed so actor can use schema context
        assert actual_arg == agent_output.model_dump_json()

    def test_agent_plain_text_passed_through_when_output_type_parse_fails(self, actor_profile, mock_strands):
        """When agent returns plain text that can't be parsed as output_type, pass it through."""
        mock_agent_cls, mock_agent_instance, _ = mock_strands
        mock_agent_instance.side_effect = _make_actor_side_effect(
            messages=[None], stops=[True]
        )

        scenario = SimulatedScenario(
            scenario_id="s-output-fallback",
            scenario_description="desc",
            actor_profile=actor_profile,
            input="hello",
        )

        def invoker(inp):
            return AgentInvokerOutput(agent_output="Here is a plain text response.")

        config = SimulationConfig(output_type=OrderConfirmation)
        executor = SimulatedScenarioExecutor(agent_invoker=invoker, simulation_config=config)
        executor.run_scenario(scenario)

        mock_agent_instance.assert_called_once()
        actual_arg = mock_agent_instance.call_args[0][0]
        assert actual_arg == "Here is a plain text response."

    def test_actor_profile_mapped_to_strands_profile(self, simulated_scenario, mock_strands):
        mock_agent_cls, mock_agent_instance, mock_profile_cls = mock_strands
        mock_agent_instance.side_effect = _make_actor_side_effect(
            messages=[None], stops=[True]
        )

        executor = SimulatedScenarioExecutor(agent_invoker=lambda inp: AgentInvokerOutput(agent_output="ok"))
        executor.run_scenario(simulated_scenario)

        mock_profile_cls.assert_called_once_with(
            traits=simulated_scenario.actor_profile.traits,
            context=simulated_scenario.actor_profile.context,
            actor_goal=simulated_scenario.actor_profile.goal,
        )
