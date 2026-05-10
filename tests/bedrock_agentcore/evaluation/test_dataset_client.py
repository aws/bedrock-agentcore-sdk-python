"""Tests for DatasetClient."""

from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore._utils.config import WaitConfig
from bedrock_agentcore.evaluation.dataset_client import DatasetClient


class TestDatasetClientInit:
    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_default_region(self, mock_boto3):
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_boto3.Session.return_value = mock_session

        client = DatasetClient()
        assert client.region_name == "us-east-1"
        mock_session.client.assert_called_once()

    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_explicit_region(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session

        client = DatasetClient(region_name="eu-west-1")
        assert client.region_name == "eu-west-1"

    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_custom_session(self, mock_boto3):
        custom_session = MagicMock()
        custom_session.region_name = "ap-southeast-1"

        client = DatasetClient(boto3_session=custom_session)
        assert client.region_name == "ap-southeast-1"
        custom_session.client.assert_called_once()


class TestDatasetClientPassthrough:
    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_allowed_method_forwarded(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_cp = mock_session.client.return_value
        mock_cp.list_datasets.return_value = {"datasets": []}

        client = DatasetClient()
        result = client.list_datasets(maxResults=10)
        mock_cp.list_datasets.assert_called_once_with(maxResults=10)

    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_snake_case_converted(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_cp = mock_session.client.return_value
        mock_cp.list_datasets.return_value = {"datasets": []}

        client = DatasetClient()
        client.list_datasets(max_results=5)
        mock_cp.list_datasets.assert_called_once_with(maxResults=5)

    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_non_allowed_method_raises(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session

        client = DatasetClient()
        with pytest.raises(AttributeError, match="not_a_real_method"):
            client.not_a_real_method()


class TestDatasetClientCreateAndWait:
    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_create_dataset_and_wait_success(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_cp = mock_session.client.return_value

        mock_cp.create_dataset.return_value = {"datasetId": "ds-123"}
        mock_cp.get_dataset.return_value = {"datasetId": "ds-123", "status": "ACTIVE"}

        client = DatasetClient()
        result = client.create_dataset_and_wait(
            wait_config=WaitConfig(max_wait=10, poll_interval=1),
            datasetName="test",
            schemaType="AGENTCORE_EVALUATION_PREDEFINED_V1",
        )

        assert result["status"] == "ACTIVE"
        mock_cp.create_dataset.assert_called_once()
        mock_cp.get_dataset.assert_called_with(datasetId="ds-123")

    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_create_dataset_and_wait_failure(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_cp = mock_session.client.return_value

        mock_cp.create_dataset.return_value = {"datasetId": "ds-123"}
        mock_cp.get_dataset.return_value = {
            "datasetId": "ds-123",
            "status": "CREATE_FAILED",
            "statusReasons": "Invalid source",
        }

        client = DatasetClient()
        with pytest.raises(RuntimeError, match="CREATE_FAILED"):
            client.create_dataset_and_wait(
                wait_config=WaitConfig(max_wait=5, poll_interval=1),
                datasetName="test",
            )

    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_create_dataset_and_wait_timeout(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_cp = mock_session.client.return_value

        mock_cp.create_dataset.return_value = {"datasetId": "ds-123"}
        mock_cp.get_dataset.return_value = {"datasetId": "ds-123", "status": "CREATING"}

        client = DatasetClient()
        with pytest.raises(TimeoutError):
            client.create_dataset_and_wait(
                wait_config=WaitConfig(max_wait=1, poll_interval=1),
                datasetName="test",
            )


class TestDatasetClientDeleteAndWait:
    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_delete_dataset_and_wait_success(self, mock_boto3):
        from botocore.exceptions import ClientError

        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_cp = mock_session.client.return_value

        mock_cp.delete_dataset.return_value = {"datasetId": "ds-123"}
        mock_cp.get_dataset.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
            "GetDataset",
        )

        client = DatasetClient()
        client.delete_dataset_and_wait(
            wait_config=WaitConfig(max_wait=10, poll_interval=1),
            datasetId="ds-123",
        )

        mock_cp.delete_dataset.assert_called_once_with(datasetId="ds-123")


class TestDatasetClientVersionAndWait:
    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_create_version_and_wait_success(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_cp = mock_session.client.return_value

        mock_cp.create_dataset_version.return_value = {"datasetId": "ds-123"}
        mock_cp.get_dataset.return_value = {"datasetId": "ds-123", "status": "ACTIVE"}

        client = DatasetClient()
        result = client.create_dataset_version_and_wait(
            wait_config=WaitConfig(max_wait=10, poll_interval=1),
            datasetId="ds-123",
        )

        assert result["status"] == "ACTIVE"


class TestDatasetClientExamplesAndWait:
    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_add_examples_and_wait_success(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_cp = mock_session.client.return_value

        mock_cp.add_dataset_examples.return_value = {"datasetId": "ds-123"}
        mock_cp.get_dataset.return_value = {"datasetId": "ds-123", "status": "ACTIVE"}

        client = DatasetClient()
        result = client.add_examples_and_wait(
            wait_config=WaitConfig(max_wait=10, poll_interval=1),
            datasetId="ds-123",
            examples=[{"scenario_id": "s1", "turns": [{"input": "hi"}]}],
        )

        assert result["status"] == "ACTIVE"

    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_update_examples_and_wait_success(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_cp = mock_session.client.return_value

        mock_cp.update_dataset_examples.return_value = {"datasetId": "ds-123"}
        mock_cp.get_dataset.return_value = {"datasetId": "ds-123", "status": "ACTIVE"}

        client = DatasetClient()
        result = client.update_examples_and_wait(
            wait_config=WaitConfig(max_wait=10, poll_interval=1),
            datasetId="ds-123",
            examples=[{"exampleId": "e1", "scenario_id": "s1", "turns": [{"input": "hi"}]}],
        )

        assert result["status"] == "ACTIVE"

    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_delete_examples_and_wait_success(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_cp = mock_session.client.return_value

        mock_cp.delete_dataset_examples.return_value = {"datasetId": "ds-123"}
        mock_cp.get_dataset.return_value = {"datasetId": "ds-123", "status": "ACTIVE"}

        client = DatasetClient()
        result = client.delete_examples_and_wait(
            wait_config=WaitConfig(max_wait=10, poll_interval=1),
            datasetId="ds-123",
            exampleIds=["e1"],
        )

        assert result["status"] == "ACTIVE"

    @patch("bedrock_agentcore.evaluation.dataset_client.boto3")
    def test_add_examples_and_wait_failure(self, mock_boto3):
        mock_session = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_cp = mock_session.client.return_value

        mock_cp.add_dataset_examples.return_value = {"datasetId": "ds-123"}
        mock_cp.get_dataset.return_value = {
            "datasetId": "ds-123",
            "status": "UPDATE_FAILED",
            "statusReasons": "Schema validation failed",
        }

        client = DatasetClient()
        with pytest.raises(RuntimeError, match="UPDATE_FAILED"):
            client.add_examples_and_wait(
                wait_config=WaitConfig(max_wait=5, poll_interval=1),
                datasetId="ds-123",
                examples=[],
            )
