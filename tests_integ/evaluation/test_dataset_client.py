"""Integration tests for DatasetClient.

Run with:
    uv run pytest tests_integ/evaluation/test_dataset_client.py -xvs --log-cli-level=INFO
"""

import os
import time

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.evaluation.dataset_client import DatasetClient


@pytest.mark.integration
class TestDatasetClientPassthrough:
    """Read-only passthrough tests. No resources needed."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.client = DatasetClient(region_name=cls.region)

    @pytest.mark.order(1)
    def test_list_datasets_passthrough(self):
        response = self.client.list_datasets()
        assert "datasets" in response

    @pytest.mark.order(2)
    def test_list_datasets_snake_case(self):
        response = self.client.list_datasets(max_results=5)
        assert "datasets" in response

    @pytest.mark.order(3)
    def test_get_nonexistent_dataset_raises(self):
        with pytest.raises(ClientError) as exc_info:
            self.client.get_dataset(datasetId="nonexistent-dataset-id")
        assert exc_info.value.response["Error"]["Code"] == "ResourceNotFoundException"

    @pytest.mark.order(4)
    def test_non_allowlisted_method_raises(self):
        with pytest.raises(AttributeError):
            self.client.not_a_real_method()


@pytest.mark.integration
class TestDatasetCrud:
    """Full CRUD lifecycle tests for datasets."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.client = DatasetClient(region_name=cls.region)
        cls.test_prefix = f"sdk_integ_{int(time.time())}"
        cls.dataset_ids = []

    @classmethod
    def teardown_class(cls):
        for did in cls.dataset_ids:
            try:
                cls.client.delete_dataset(datasetId=did)
            except Exception as e:
                print(f"Failed to delete dataset {did}: {e}")

    @pytest.mark.order(5)
    def test_create_dataset_and_wait_inline(self):
        """Create a dataset with inline examples and wait for ACTIVE."""
        dataset = self.client.create_dataset_and_wait(
            datasetName=f"{self.test_prefix}_predefined",
            schemaType="AGENTCORE_EVALUATION_PREDEFINED_V1",
            source={
                "inlineExamples": {
                    "examples": [
                        {
                            "scenario_id": "test-scenario-1",
                            "turns": [{"input": "Hello", "expected_response": "Hi there!"}],
                        }
                    ]
                }
            },
        )
        self.__class__.dataset_ids.append(dataset["datasetId"])
        assert dataset["status"] == "ACTIVE"

    @pytest.mark.order(6)
    def test_get_dataset(self):
        """Get dataset metadata."""
        if not self.dataset_ids:
            pytest.skip("prerequisite test did not create dataset")
        dataset = self.client.get_dataset(datasetId=self.dataset_ids[0])
        assert dataset["datasetId"] == self.dataset_ids[0]
        assert dataset["status"] == "ACTIVE"

    @pytest.mark.order(7)
    def test_list_datasets_contains_created(self):
        """List datasets includes the one we created."""
        if not self.dataset_ids:
            pytest.skip("prerequisite test did not create dataset")
        response = self.client.list_datasets()
        ids = [d["datasetId"] for d in response["datasets"]]
        assert self.dataset_ids[0] in ids

    @pytest.mark.order(8)
    def test_update_dataset_metadata(self):
        """Update dataset description (sync operation)."""
        if not self.dataset_ids:
            pytest.skip("prerequisite test did not create dataset")
        response = self.client.update_dataset(
            datasetId=self.dataset_ids[0],
            description="updated by integ test",
        )
        assert response["datasetId"] == self.dataset_ids[0]
        # Verify via get
        dataset = self.client.get_dataset(datasetId=self.dataset_ids[0])
        assert dataset.get("description") == "updated by integ test"

    @pytest.mark.order(9)
    def test_add_examples_and_wait(self):
        """Add examples to dataset and wait for ACTIVE."""
        if not self.dataset_ids:
            pytest.skip("prerequisite test did not create dataset")
        dataset = self.client.add_examples_and_wait(
            datasetId=self.dataset_ids[0],
            source={
                "inlineExamples": {
                    "examples": [
                        {
                            "scenario_id": "test-scenario-2",
                            "turns": [{"input": "What is 2+2?", "expected_response": "4"}],
                        }
                    ]
                }
            },
        )
        assert dataset["status"] == "ACTIVE"

    @pytest.mark.order(10)
    def test_list_examples(self):
        """List examples in dataset."""
        if not self.dataset_ids:
            pytest.skip("prerequisite test did not create dataset")
        response = self.client.list_dataset_examples(datasetId=self.dataset_ids[0])
        assert "examples" in response
        assert len(response["examples"]) >= 2

    @pytest.mark.order(11)
    def test_create_dataset_version_and_wait(self):
        """Create a version from DRAFT and wait for ACTIVE."""
        if not self.dataset_ids:
            pytest.skip("prerequisite test did not create dataset")
        dataset = self.client.create_dataset_version_and_wait(
            datasetId=self.dataset_ids[0],
        )
        assert dataset["status"] == "ACTIVE"

    @pytest.mark.order(12)
    def test_list_dataset_versions(self):
        """List versions of the dataset."""
        if not self.dataset_ids:
            pytest.skip("prerequisite test did not create dataset")
        response = self.client.list_dataset_versions(datasetId=self.dataset_ids[0])
        assert "versions" in response
        assert len(response["versions"]) >= 1

    @pytest.mark.order(13)
    def test_delete_dataset_and_wait(self):
        """Delete dataset and wait for removal."""
        if not self.dataset_ids:
            pytest.skip("prerequisite test did not create dataset")
        did = self.dataset_ids.pop(0)
        self.client.delete_dataset_and_wait(datasetId=did)
        with pytest.raises(ClientError) as exc_info:
            self.client.get_dataset(datasetId=did)
        assert exc_info.value.response["Error"]["Code"] == "ResourceNotFoundException"
