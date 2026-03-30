"""Tests for FileDatasetProvider."""

import json

import pytest

from bedrock_agentcore.evaluation.runner.dataset_providers import FileDatasetProvider
from bedrock_agentcore.evaluation.runner.dataset_types import (
    Dataset,
    PredefinedScenario,
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
        with pytest.raises(KeyError):
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
