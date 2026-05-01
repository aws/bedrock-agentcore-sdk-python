"""Unit tests for DatasetClient - no external connections."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.dataset import DatasetClient


def test_client_initialization():
    """Test client initialization creates a single bedrock-agentcore-control client."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        mock_session_cls.return_value = mock_session

        client = DatasetClient(region_name="us-west-2")

        assert client.region_name == "us-west-2"
        assert client.integration_source is None
        # Only one client: bedrock-agentcore-control
        assert mock_session.client.call_count == 1

        # Verify config was passed
        for call in mock_session.client.call_args_list:
            assert "config" in call.kwargs

        # Verify the correct service was used
        service_name = mock_session.client.call_args_list[0][0][0]
        assert service_name == "bedrock-agentcore-control"


def test_client_initialization_with_integration_source():
    """Test client initialization with integration_source sets user-agent telemetry."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        mock_session_cls.return_value = mock_session

        client = DatasetClient(region_name="us-west-2", integration_source="langchain")

        assert client.region_name == "us-west-2"
        assert client.integration_source == "langchain"
        assert mock_session.client.call_count == 1


def test_client_initialization_with_boto3_session():
    """Test client initialization with a custom boto3 session."""
    mock_session = MagicMock()
    mock_session.region_name = "eu-west-1"

    client = DatasetClient(boto3_session=mock_session)

    assert client.region_name == "eu-west-1"
    assert mock_session.client.call_count == 1

    # Verify the correct service was used
    call_args = [call[0][0] for call in mock_session.client.call_args_list]
    assert "bedrock-agentcore-control" in call_args


def test_client_initialization_region_fallback_to_default():
    """Test region falls back to us-west-2 when no region is provided."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.region_name = None
        mock_session_cls.return_value = mock_session

        client = DatasetClient()

        assert client.region_name == "us-west-2"


def test_getattr_allowed_methods():
    """Verify each allowed method dispatches correctly via __getattr__."""
    mock_session = MagicMock()
    mock_boto_client = MagicMock()
    mock_session.client.return_value = mock_boto_client

    client = DatasetClient(region_name="us-east-1", boto3_session=mock_session)

    allowed_methods = [
        "create_dataset",
        "get_dataset",
        "list_datasets",
        "update_dataset",
        "delete_dataset",
    ]

    for method_name in allowed_methods:
        method = getattr(client, method_name)
        assert callable(method), f"{method_name} should be callable"


def test_getattr_unknown_method_raises():
    """AttributeError is raised for unknown/disallowed methods."""
    mock_session = MagicMock()
    mock_session.client.return_value = MagicMock()

    client = DatasetClient(region_name="us-east-1", boto3_session=mock_session)

    with pytest.raises(AttributeError, match="has no attribute 'create_evaluator'"):
        _ = client.create_evaluator


def test_create_dataset():
    """Test create_dataset delegates to cp_client."""
    mock_session = MagicMock()
    mock_boto_client = MagicMock()
    mock_session.client.return_value = mock_boto_client
    mock_boto_client.create_dataset.return_value = {"datasetId": "ds-123", "status": "CREATING"}

    client = DatasetClient(region_name="us-west-2", boto3_session=mock_session)
    result = client.create_dataset(name="my-dataset")

    mock_boto_client.create_dataset.assert_called_once_with(name="my-dataset")
    assert result["datasetId"] == "ds-123"


def test_list_datasets():
    """Test list_datasets delegates to cp_client."""
    mock_session = MagicMock()
    mock_boto_client = MagicMock()
    mock_session.client.return_value = mock_boto_client
    mock_boto_client.list_datasets.return_value = {
        "items": [{"datasetId": "ds-1"}, {"datasetId": "ds-2"}]
    }

    client = DatasetClient(region_name="us-west-2", boto3_session=mock_session)
    result = client.list_datasets()

    mock_boto_client.list_datasets.assert_called_once_with()
    assert len(result["items"]) == 2


def test_delete_dataset():
    """Test delete_dataset delegates to cp_client."""
    mock_session = MagicMock()
    mock_boto_client = MagicMock()
    mock_session.client.return_value = mock_boto_client
    mock_boto_client.delete_dataset.return_value = {"datasetId": "ds-123"}

    client = DatasetClient(region_name="us-west-2", boto3_session=mock_session)
    result = client.delete_dataset(datasetIdentifier="ds-123")

    mock_boto_client.delete_dataset.assert_called_once_with(datasetIdentifier="ds-123")
    assert result["datasetId"] == "ds-123"


def test_delete_dataset_and_wait():
    """Test delete_dataset_and_wait polls until the dataset is deleted."""
    from botocore.exceptions import ClientError

    mock_session = MagicMock()
    mock_boto_client = MagicMock()
    mock_session.client.return_value = mock_boto_client

    # delete_dataset returns the datasetId
    mock_boto_client.delete_dataset.return_value = {"datasetId": "ds-123"}

    # get_dataset raises ResourceNotFoundException after deletion
    not_found_error = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
        "GetDataset",
    )
    mock_boto_client.get_dataset.side_effect = not_found_error

    client = DatasetClient(region_name="us-west-2", boto3_session=mock_session)
    # Should not raise
    client.delete_dataset_and_wait(datasetIdentifier="ds-123")

    mock_boto_client.delete_dataset.assert_called_once()
    mock_boto_client.get_dataset.assert_called_once_with(datasetIdentifier="ds-123")


def test_create_dataset_and_wait():
    """Test create_dataset_and_wait polls until READY status."""
    mock_session = MagicMock()
    mock_boto_client = MagicMock()
    mock_session.client.return_value = mock_boto_client

    mock_boto_client.create_dataset.return_value = {"datasetId": "ds-456", "status": "CREATING"}
    mock_boto_client.get_dataset.return_value = {"datasetId": "ds-456", "status": "READY"}

    client = DatasetClient(region_name="us-west-2", boto3_session=mock_session)
    result = client.create_dataset_and_wait(name="my-dataset")

    assert result["status"] == "READY"
    assert result["datasetId"] == "ds-456"
    mock_boto_client.create_dataset.assert_called_once_with(name="my-dataset")
    mock_boto_client.get_dataset.assert_called_once_with(datasetIdentifier="ds-456")


def test_update_dataset_and_wait():
    """Test update_dataset_and_wait polls until READY status."""
    mock_session = MagicMock()
    mock_boto_client = MagicMock()
    mock_session.client.return_value = mock_boto_client

    mock_boto_client.update_dataset.return_value = {"datasetId": "ds-789", "status": "UPDATING"}
    mock_boto_client.get_dataset.return_value = {"datasetId": "ds-789", "status": "READY"}

    client = DatasetClient(region_name="us-west-2", boto3_session=mock_session)
    result = client.update_dataset_and_wait(datasetIdentifier="ds-789", name="updated-name")

    assert result["status"] == "READY"
    mock_boto_client.update_dataset.assert_called_once_with(datasetIdentifier="ds-789", name="updated-name")
    mock_boto_client.get_dataset.assert_called_once_with(datasetIdentifier="ds-789")


def test_create_dataset_and_wait_failed_status():
    """Test create_dataset_and_wait raises RuntimeError on FAILED status."""
    mock_session = MagicMock()
    mock_boto_client = MagicMock()
    mock_session.client.return_value = mock_boto_client

    mock_boto_client.create_dataset.return_value = {"datasetId": "ds-bad", "status": "CREATING"}
    mock_boto_client.get_dataset.return_value = {"datasetId": "ds-bad", "status": "FAILED", "statusReasons": "Bad config"}

    client = DatasetClient(region_name="us-west-2", boto3_session=mock_session)

    with pytest.raises(RuntimeError, match="FAILED"):
        client.create_dataset_and_wait(name="bad-dataset")


def test_getattr_error_message_contains_service():
    """AttributeError message mentions the service name."""
    mock_session = MagicMock()
    mock_session.client.return_value = MagicMock()

    client = DatasetClient(region_name="us-east-1", boto3_session=mock_session)

    with pytest.raises(AttributeError, match="bedrock-agentcore-control"):
        _ = client.unknown_operation
