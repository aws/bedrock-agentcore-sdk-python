"""Dataset provider implementations for loading evaluation datasets."""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from bedrock_agentcore.evaluation.dataset_client import DatasetClient

from .dataset_types import (
    ActorProfile,
    Dataset,
    PredefinedScenario,
    Scenario,
    SimulatedScenario,
    Turn,
)


def _parse_scenario(raw: Dict[str, Any]) -> PredefinedScenario | SimulatedScenario:
    """Parse a raw dict into a PredefinedScenario or SimulatedScenario.

    The type is inferred from the fields present, matching the service: ``actor_profile`` makes a
    scenario simulated, ``turns`` makes it predefined, and a bare ``input`` is a single-turn predefined
    scenario. Fields the SDK does not know are ignored.
    """
    scenario_id = raw.get("scenario_id", "?")
    # Datasets no longer declare a type, so this is the only guard against a mislabeled example
    # being run as the wrong kind of scenario.
    if "actor_profile" in raw and "turns" in raw:
        raise ValueError(f"Scenario '{scenario_id}' has both 'turns' and 'actor_profile'; a scenario must have one")
    if not any(k in raw for k in ("turns", "actor_profile", "input")):
        raise ValueError(f"Scenario '{scenario_id}' has none of 'turns', 'actor_profile' or 'input'")
    if "actor_profile" not in raw:
        turns = raw.get("turns", [{"input": raw.get("input")}])
        return PredefinedScenario(
            scenario_id=raw["scenario_id"],
            turns=[Turn(input=t["input"], expected_response=t.get("expected_response")) for t in turns],
            expected_trajectory=raw.get("expected_trajectory"),
            assertions=raw.get("assertions"),
            metadata=raw.get("metadata"),
        )
    else:
        missing = [k for k in ("scenario_id", "actor_profile") if k not in raw]
        if missing:
            raise ValueError(
                f"Scenario '{raw.get('scenario_id', '?')}' is missing required fields for SimulatedScenario: {missing}"
            )
        return SimulatedScenario(
            scenario_id=raw["scenario_id"],
            scenario_description=raw.get("scenario_description", ""),
            actor_profile=ActorProfile(**raw["actor_profile"]),
            input=raw.get("input"),
            max_turns=raw.get("max_turns", 10),
            assertions=raw.get("assertions"),
            metadata=raw.get("metadata"),
        )


class DatasetProvider(ABC):
    """Abstract provider for loading datasets."""

    @abstractmethod
    def get_dataset(self) -> Dataset:
        """Load and return the dataset."""


class FileDatasetProvider(DatasetProvider):
    """A dataset provider that loads a Dataset from a JSON or JSONL file.

    JSON format:  {"scenarios": [{...}, {...}]}
    JSONL format: one scenario JSON object per line.
    Format is selected by file extension (".jsonl" → JSONL, otherwise JSON).
    """

    def __init__(self, file_path: str):
        """Initialize with a path to a JSON or JSONL dataset file."""
        self._file_path = file_path

    def get_dataset(self) -> Dataset:
        """Load and return the dataset from the file."""
        if self._file_path.endswith(".jsonl"):
            with open(self._file_path) as f:
                raw_examples = [json.loads(line) for line in f if line.strip()]
        else:
            with open(self._file_path) as f:
                raw_examples = json.load(f)["scenarios"]
        scenarios: List[Scenario] = [_parse_scenario(s) for s in raw_examples]
        return Dataset(scenarios=scenarios)


class DatasetManagementServiceProvider(DatasetProvider):
    """A dataset provider that loads a Dataset from the Dataset Management service."""

    def __init__(
        self,
        dataset_id: str,
        version_id: Optional[str] = None,
        client: Optional[DatasetClient] = None,
    ):
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
            ValueError: If the dataset has no downloadUrl.
            RuntimeError: If the dataset content cannot be downloaded.
        """
        kwargs: Dict[str, Any] = {"datasetId": self._dataset_id}
        if self._version_id:
            kwargs["datasetVersion"] = self._version_id

        response = self._client.get_dataset(**kwargs)

        download_url = response.get("downloadUrl")
        if not download_url:
            raise ValueError(f"Dataset {self._dataset_id} has no downloadUrl. Status: {response.get('status')}")

        try:
            import requests
        except ImportError as e:
            raise ImportError(
                "DatasetManagementServiceProvider requires the 'datasets' extra. "
                "Install it with: pip install 'bedrock-agentcore[datasets]'"
            ) from e

        try:
            r = requests.get(download_url, timeout=60, stream=True)
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Couldn't download dataset from S3 bucket: {e}") from e

        all_examples: List[Dict[str, Any]] = []
        for line in r.iter_lines(decode_unicode=False):
            if line:
                all_examples.append(json.loads(line.decode("utf-8")))

        scenarios: List[Scenario] = [_parse_scenario(example) for example in all_examples]
        return Dataset(scenarios=scenarios)
