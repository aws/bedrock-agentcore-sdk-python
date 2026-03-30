"""Tests for dataset type definitions."""

import pytest

from bedrock_agentcore.evaluation.runner.dataset_types import (
    Dataset,
    PredefinedScenario,
    Scenario,
    Turn,
)


class TestTurn:
    def test_minimal(self):
        t = Turn(input="hello")
        assert t.input == "hello"
        assert t.expected_response is None

    def test_with_expected_response(self):
        t = Turn(input="hello", expected_response="hi")
        assert t.expected_response == "hi"


class TestPredefinedScenario:
    def test_minimal(self):
        s = PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])
        assert s.scenario_id == "s1"
        assert len(s.turns) == 1
        assert s.expected_trajectory is None
        assert s.assertions is None
        assert s.metadata is None

    def test_full(self):
        s = PredefinedScenario(
            scenario_id="s1",
            turns=[Turn(input="hi")],
            expected_trajectory=["step1"],
            assertions=["assert1"],
            metadata={"key": "val"},
        )
        assert s.expected_trajectory == ["step1"]
        assert s.assertions == ["assert1"]
        assert s.metadata == {"key": "val"}

    def test_is_scenario(self):
        s = PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])
        assert isinstance(s, Scenario)

    def test_empty_turns_raises(self):
        with pytest.raises(ValueError, match="turns must not be empty"):
            PredefinedScenario(scenario_id="s1", turns=[])


class TestDataset:
    def _make_scenario(self):
        return PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])

    def test_construction(self):
        ds = Dataset(scenarios=[self._make_scenario()])
        assert len(ds.scenarios) == 1

    def test_empty_scenarios_raises(self):
        with pytest.raises(ValueError, match="scenarios must not be empty"):
            Dataset(scenarios=[])

    def test_duplicate_scenario_ids_raises(self):
        s1 = PredefinedScenario(scenario_id="s1", turns=[Turn(input="hi")])
        s2 = PredefinedScenario(scenario_id="s1", turns=[Turn(input="hello")])
        with pytest.raises(ValueError, match="Duplicate scenario_ids"):
            Dataset(scenarios=[s1, s2])
