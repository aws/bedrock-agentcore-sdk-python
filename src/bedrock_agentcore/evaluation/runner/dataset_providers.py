"""Dataset provider implementations for loading evaluation datasets."""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .dataset_types import ActorProfile, Dataset, PredefinedScenario, Scenario, SimulatedScenario, Turn


class DatasetProvider(ABC):
    """Abstract provider for loading datasets."""

    @abstractmethod
    def get_dataset(self) -> Dataset:
        """Load and return the dataset."""


class FileDatasetProvider(DatasetProvider):
    """A dataset provider that loads a Dataset from a JSON file."""

    def __init__(self, file_path: str):
        """Initialize with a path to a JSON dataset file."""
        self._file_path = file_path

    def get_dataset(self) -> Dataset:
        """Load and return the dataset from the JSON file."""
        with open(self._file_path) as f:
            data = json.load(f)
        scenarios: List[Scenario] = [self._parse_scenario(s) for s in data["scenarios"]]
        return Dataset(scenarios=scenarios)

    @staticmethod
    def _parse_scenario(raw: Dict[str, Any]) -> PredefinedScenario | SimulatedScenario:
        if "turns" in raw:
            return PredefinedScenario(
                scenario_id=raw["scenario_id"],
                turns=[FileDatasetProvider._parse_turn(t) for t in raw["turns"]],
                expected_trajectory=raw.get("expected_trajectory"),
                assertions=raw.get("assertions"),
                metadata=raw.get("metadata"),
            )
        else:
            missing = [k for k in ("scenario_id", "actor_profile", "input") if k not in raw]
            if missing:
                raise ValueError(
                    f"Scenario '{raw.get('scenario_id', '?')}' is missing required fields "
                    f"for SimulatedScenario: {missing}"
                )
            return SimulatedScenario(
                scenario_id=raw["scenario_id"],
                scenario_description=raw.get("scenario_description", ""),
                actor_profile=ActorProfile(**raw["actor_profile"]),
                input=raw["input"],
                max_turns=raw.get("max_turns", 10),
                assertions=raw.get("assertions"),
                metadata=raw.get("metadata"),
            )

    @staticmethod
    def _parse_turn(raw: Dict[str, Any]) -> Turn:
        return Turn(
            input=raw["input"],
            expected_response=raw.get("expected_response"),
        )


class ServiceDatasetProvider(DatasetProvider):
    """A dataset provider that loads a Dataset from the Dataset Management service."""

    def __init__(self, dataset_id: str, version_id: Optional[str] = None, region_name: Optional[str] = None):
        """Initialize with a dataset ID and optional version.

        Args:
            dataset_id: The dataset ID to fetch.
            version_id: Optional version ID. If omitted, fetches DRAFT.
            region_name: AWS region. Falls back to boto3 session region or us-west-2.
        """
        self._dataset_id = dataset_id
        self._version_id = version_id
        self._region_name = region_name

    def get_dataset(self) -> Dataset:
        """Load and return the dataset from the Dataset Management service."""
        from bedrock_agentcore.evaluation.dataset_client import DatasetClient

        client = DatasetClient(region_name=self._region_name)

        # Fetch all examples via pagination
        all_examples: List[Dict[str, Any]] = []
        kwargs: Dict[str, Any] = {"datasetId": self._dataset_id}
        if self._version_id:
            kwargs["datasetVersion"] = self._version_id

        while True:
            response = client.list_dataset_examples(**kwargs)
            all_examples.extend(response.get("examples", []))
            next_token = response.get("nextToken")
            if not next_token:
                break
            kwargs["nextToken"] = next_token

        scenarios: List[Scenario] = [FileDatasetProvider._parse_scenario(example) for example in all_examples]
        return Dataset(scenarios=scenarios)
