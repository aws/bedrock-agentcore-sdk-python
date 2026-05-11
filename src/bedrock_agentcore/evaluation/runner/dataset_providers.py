"""Dataset provider implementations for loading evaluation datasets."""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

from bedrock_agentcore.evaluation.dataset_client import DatasetClient

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


def _get_supported_schema_types() -> set:
    """Collect supported schema types from all ScenarioExecutor subclasses."""
    from .scenario_executor import ScenarioExecutor

    return {cls.model_fields["schema_type"].default for cls in ScenarioExecutor.__subclasses__() if cls.model_fields["schema_type"].default}


class ServiceDatasetProvider(DatasetProvider):
    """A dataset provider that loads a Dataset from the Dataset Management service."""

    def __init__(self, dataset_id: str, version_id: Optional[str] = None, client: Optional[DatasetClient] = None):
        """Initialize with a dataset ID and optional version.

        Args:
            dataset_id: The dataset ID to fetch.
            version_id: Optional version ID. If omitted, fetches DRAFT.
            client: Optional DatasetClient instance. If not provided, a default is created.
        """
        self._dataset_id = dataset_id
        self._version_id = version_id
        self._client = client if client is not None else DatasetClient()

    def get_dataset(self) -> Dataset:
        """Load and return the dataset from the Dataset Management service.

        Fetches the dataset via the presigned download URL returned by GetDataset.
        The URL points to a JSONL file where each line is one example.

        Raises:
            ValueError: If the dataset has no downloadUrl or has an unsupported schemaType.
            RuntimeError: If the dataset content cannot be downloaded.
        """
        kwargs: Dict[str, Any] = {"datasetId": self._dataset_id}
        if self._version_id:
            kwargs["datasetVersion"] = self._version_id

        response = self._client.get_dataset(**kwargs)

        schema_type = response.get("schemaType")
        supported = _get_supported_schema_types()
        if schema_type and schema_type not in supported:
            raise ValueError(
                f"Dataset schema type '{schema_type}' is not supported by the evaluation runners. "
                f"Supported types: {sorted(supported)}"
            )

        download_url = response.get("downloadUrl")
        if not download_url:
            raise ValueError(f"Dataset {self._dataset_id} has no downloadUrl. Status: {response.get('status')}")

        try:
            r = requests.get(download_url)
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Couldn't download dataset from S3 bucket: {e}"
            ) from e

        all_examples: List[Dict[str, Any]] = []
        for line in r.text.strip().split("\n"):
            if line:
                all_examples.append(json.loads(line))

        scenarios: List[Scenario] = [FileDatasetProvider._parse_scenario(example) for example in all_examples]
        return Dataset(scenarios=scenarios)
