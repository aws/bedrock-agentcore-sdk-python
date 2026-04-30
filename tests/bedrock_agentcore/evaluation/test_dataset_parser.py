"""Tests for FileDatasetProvider."""

import json

import pytest

from bedrock_agentcore.evaluation.runner.dataset_providers import FileDatasetProvider
from bedrock_agentcore.evaluation.runner.dataset_types import (
    ActorProfile,
    Dataset,
    PredefinedScenario,
    SimulatedScenario,
)


class TestFileDatasetProvider:
    def _write_and_load(self, tmp_path, data):
        file_path = tmp_path / "dataset.json"
        file_path.write_text(json.dumps(data))
        return FileDatasetProvider(str(file_path)).get_dataset()

    def test_parse_predefined(self, tmp_path):
        data = {
            "scenarios": [
                {
                    "scenario_id": "s1",
                    "turns": [
                        {"input": "Hello"},
                        {"input": "What is 2+2?"},
                    ],
                }
            ]
        }
        dataset = self._write_and_load(tmp_path, data)
        assert isinstance(dataset, Dataset)
        assert len(dataset.scenarios) == 1
        scenario = dataset.scenarios[0]
        assert isinstance(scenario, PredefinedScenario)
        assert scenario.scenario_id == "s1"
        assert len(scenario.turns) == 2
        assert scenario.turns[0].input == "Hello"
        assert scenario.turns[1].input == "What is 2+2?"

    def test_parse_predefined_full(self, tmp_path):
        data = {
            "scenarios": [
                {
                    "scenario_id": "s1",
                    "turns": [
                        {
                            "input": "Book a flight",
                            "expected_response": "Sure, where to?",
                        }
                    ],
                    "expected_trajectory": ["search_flights"],
                    "assertions": ["Book a flight", "Must confirm before booking"],
                    "metadata": {"category": "travel"},
                }
            ]
        }
        dataset = self._write_and_load(tmp_path, data)
        scenario = dataset.scenarios[0]
        assert scenario.expected_trajectory == ["search_flights"]
        assert scenario.assertions == ["Book a flight", "Must confirm before booking"]
        assert scenario.metadata == {"category": "travel"}
        turn = scenario.turns[0]
        assert turn.expected_response == "Sure, where to?"

    def test_parse_missing_scenarios_key(self, tmp_path):
        with pytest.raises(KeyError):
            self._write_and_load(tmp_path, {})

    def test_parse_empty_scenarios(self, tmp_path):
        with pytest.raises(ValueError, match="scenarios must not be empty"):
            self._write_and_load(tmp_path, {"scenarios": []})

    def test_parse_missing_required_field(self, tmp_path):
        data = {"scenarios": [{"scenario_id": "s1"}]}
        with pytest.raises(ValueError):
            self._write_and_load(tmp_path, data)

    def test_file_not_found(self):
        provider = FileDatasetProvider("/nonexistent/path.json")
        with pytest.raises(FileNotFoundError):
            provider.get_dataset()

    def test_invalid_json(self, tmp_path):
        file_path = tmp_path / "bad.json"
        file_path.write_text("not json")

        provider = FileDatasetProvider(str(file_path))
        with pytest.raises(json.JSONDecodeError):
            provider.get_dataset()

    def test_parse_simulated(self, tmp_path):
        data = {
            "scenarios": [
                {
                    "scenario_id": "sim-1",
                    "scenario_description": "Customer orders pizza",
                    "actor_profile": {
                        "traits": {"expertise": "novice"},
                        "context": "A hungry customer",
                        "goal": "Order a pizza successfully",
                    },
                    "input": "I'd like to order a pizza",
                    "max_turns": 5,
                }
            ]
        }
        dataset = self._write_and_load(tmp_path, data)
        assert len(dataset.scenarios) == 1
        scenario = dataset.scenarios[0]
        assert isinstance(scenario, SimulatedScenario)
        assert scenario.scenario_id == "sim-1"
        assert scenario.scenario_description == "Customer orders pizza"
        assert isinstance(scenario.actor_profile, ActorProfile)
        assert scenario.actor_profile.goal == "Order a pizza successfully"
        assert scenario.input == "I'd like to order a pizza"
        assert scenario.max_turns == 5

    def test_parse_simulated_default_max_turns(self, tmp_path):
        data = {
            "scenarios": [
                {
                    "scenario_id": "sim-2",
                    "scenario_description": "desc",
                    "actor_profile": {"context": "ctx", "goal": "goal"},
                    "input": "hello",
                }
            ]
        }
        dataset = self._write_and_load(tmp_path, data)
        assert dataset.scenarios[0].max_turns == 10

    def test_parse_simulated_with_assertions(self, tmp_path):
        data = {
            "scenarios": [
                {
                    "scenario_id": "sim-3",
                    "scenario_description": "desc",
                    "actor_profile": {"context": "ctx", "goal": "goal"},
                    "input": "hello",
                    "assertions": ["Must greet user", "Must confirm order"],
                }
            ]
        }
        dataset = self._write_and_load(tmp_path, data)
        assert dataset.scenarios[0].assertions == ["Must greet user", "Must confirm order"]

    def test_parse_mixed_predefined_and_simulated(self, tmp_path):
        data = {
            "scenarios": [
                {
                    "scenario_id": "pre-1",
                    "turns": [{"input": "hello"}],
                },
                {
                    "scenario_id": "sim-1",
                    "scenario_description": "desc",
                    "actor_profile": {"context": "ctx", "goal": "goal"},
                    "input": "hello",
                },
            ]
        }
        dataset = self._write_and_load(tmp_path, data)
        assert len(dataset.scenarios) == 2
        assert isinstance(dataset.scenarios[0], PredefinedScenario)
        assert isinstance(dataset.scenarios[1], SimulatedScenario)
