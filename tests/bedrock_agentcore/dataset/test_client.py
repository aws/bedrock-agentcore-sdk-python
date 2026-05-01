"""Unit tests for DatasetClient - no external connections."""

from unittest.mock import MagicMock, call, patch

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.dataset import DatasetClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(region="us-west-2", **kwargs):
    """Return a DatasetClient backed by a single MagicMock boto3 client."""
    mock_session = MagicMock()
    mock_boto_client = MagicMock()
    mock_session.region_name = region
    mock_session.client.return_value = mock_boto_client
    client = DatasetClient(region_name=region, boto3_session=mock_session, **kwargs)
    return client, mock_boto_client


def _not_found_error():
    return ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
        "GetDataset",
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

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
        for call_item in mock_session.client.call_args_list:
            assert "config" in call_item.kwargs

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
    call_args = [c[0][0] for c in mock_session.client.call_args_list]
    assert "bedrock-agentcore-control" in call_args


def test_client_initialization_region_fallback_to_default():
    """Test region falls back to us-west-2 when no region is provided."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.region_name = None
        mock_session_cls.return_value = mock_session

        client = DatasetClient()

        assert client.region_name == "us-west-2"


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------

def test_dataset_client_exported_from_top_level():
    """DatasetClient must be importable directly from bedrock_agentcore."""
    from bedrock_agentcore import DatasetClient as DC  # noqa: F401
    assert DC is DatasetClient


# ---------------------------------------------------------------------------
# __getattr__ dispatch
# ---------------------------------------------------------------------------

def test_getattr_allowed_methods():
    """Verify each allowed method dispatches correctly via __getattr__."""
    client, _ = _make_client()

    allowed_methods = [
        "create_dataset",
        "get_dataset",
        "list_datasets",
        "update_dataset",
        "delete_dataset",
        "get_paginator",
        "get_waiter",
    ]

    for method_name in allowed_methods:
        method = getattr(client, method_name)
        assert callable(method), f"{method_name} should be callable"


def test_getattr_unknown_method_raises():
    """AttributeError is raised for unknown/disallowed methods."""
    client, _ = _make_client()

    with pytest.raises(AttributeError, match="has no attribute 'create_evaluator'"):
        _ = client.create_evaluator


def test_getattr_error_message_contains_service():
    """AttributeError message mentions the service name."""
    client, _ = _make_client()

    with pytest.raises(AttributeError, match="bedrock-agentcore-control"):
        _ = client.unknown_operation


# ---------------------------------------------------------------------------
# Pass-through CRUD
# ---------------------------------------------------------------------------

def test_create_dataset():
    """Test create_dataset delegates to cp_client."""
    client, mock_boto = _make_client()
    mock_boto.create_dataset.return_value = {"datasetId": "ds-123", "status": "CREATING"}

    result = client.create_dataset(name="my-dataset")

    mock_boto.create_dataset.assert_called_once_with(name="my-dataset")
    assert result["datasetId"] == "ds-123"


def test_list_datasets():
    """Test list_datasets delegates to cp_client."""
    client, mock_boto = _make_client()
    mock_boto.list_datasets.return_value = {
        "items": [{"datasetId": "ds-1"}, {"datasetId": "ds-2"}]
    }

    result = client.list_datasets()

    mock_boto.list_datasets.assert_called_once_with()
    assert len(result["items"]) == 2


def test_delete_dataset():
    """Test delete_dataset delegates to cp_client."""
    client, mock_boto = _make_client()
    mock_boto.delete_dataset.return_value = {"datasetId": "ds-123"}

    result = client.delete_dataset(datasetIdentifier="ds-123")

    mock_boto.delete_dataset.assert_called_once_with(datasetIdentifier="ds-123")
    assert result["datasetId"] == "ds-123"


# ---------------------------------------------------------------------------
# create_dataset_and_wait
# ---------------------------------------------------------------------------

def test_create_dataset_and_wait():
    """Test create_dataset_and_wait polls until READY status."""
    client, mock_boto = _make_client()

    mock_boto.create_dataset.return_value = {"datasetId": "ds-456", "status": "CREATING"}
    mock_boto.get_dataset.return_value = {"dataset": {"datasetId": "ds-456", "status": "READY"}}

    result = client.create_dataset_and_wait(name="my-dataset")

    assert result["status"] == "READY"
    assert result["datasetId"] == "ds-456"
    mock_boto.create_dataset.assert_called_once()
    mock_boto.get_dataset.assert_called_once_with(datasetIdentifier="ds-456")


def test_create_dataset_and_wait_adds_client_token():
    """create_dataset_and_wait injects a clientToken for idempotency."""
    client, mock_boto = _make_client()

    mock_boto.create_dataset.return_value = {"datasetId": "ds-tok", "status": "CREATING"}
    mock_boto.get_dataset.return_value = {"dataset": {"datasetId": "ds-tok", "status": "READY"}}

    client.create_dataset_and_wait(name="tok-dataset")

    create_call_kwargs = mock_boto.create_dataset.call_args[1]
    assert "clientToken" in create_call_kwargs
    # Should be a non-empty string (UUID)
    assert len(create_call_kwargs["clientToken"]) > 0


def test_create_dataset_and_wait_respects_explicit_client_token():
    """Caller-supplied clientToken must not be overwritten."""
    client, mock_boto = _make_client()

    mock_boto.create_dataset.return_value = {"datasetId": "ds-ct", "status": "CREATING"}
    mock_boto.get_dataset.return_value = {"dataset": {"datasetId": "ds-ct", "status": "READY"}}

    client.create_dataset_and_wait(name="ct-dataset", clientToken="my-token-123")

    create_call_kwargs = mock_boto.create_dataset.call_args[1]
    assert create_call_kwargs["clientToken"] == "my-token-123"


def test_create_dataset_and_wait_failed_status():
    """Test create_dataset_and_wait raises RuntimeError on FAILED status."""
    client, mock_boto = _make_client()

    mock_boto.create_dataset.return_value = {"datasetId": "ds-bad", "status": "CREATING"}
    mock_boto.get_dataset.return_value = {
        "dataset": {"datasetId": "ds-bad", "status": "FAILED", "statusReasons": "Bad config"}
    }

    with pytest.raises(RuntimeError, match="FAILED"):
        client.create_dataset_and_wait(name="bad-dataset")


def test_create_dataset_and_wait_update_unsuccessful():
    """create_dataset_and_wait raises RuntimeError on UPDATE_UNSUCCESSFUL status."""
    client, mock_boto = _make_client()

    mock_boto.create_dataset.return_value = {"datasetId": "ds-upd", "status": "CREATING"}
    mock_boto.get_dataset.return_value = {
        "dataset": {"datasetId": "ds-upd", "status": "UPDATE_UNSUCCESSFUL"}
    }

    with pytest.raises(RuntimeError, match="UPDATE_UNSUCCESSFUL"):
        client.create_dataset_and_wait(name="upd-dataset")


def test_create_dataset_and_wait_polls_nested_dataset_key():
    """The poll lambda must unwrap response['dataset'] before reading status."""
    client, mock_boto = _make_client()

    mock_boto.create_dataset.return_value = {"datasetId": "ds-nested", "status": "CREATING"}
    # First call returns CREATING, second returns READY — both wrapped under 'dataset'
    mock_boto.get_dataset.side_effect = [
        {"dataset": {"datasetId": "ds-nested", "status": "CREATING"}},
        {"dataset": {"datasetId": "ds-nested", "status": "READY"}},
    ]

    result = client.create_dataset_and_wait(name="nested-dataset")

    assert result["status"] == "READY"
    assert mock_boto.get_dataset.call_count == 2


# ---------------------------------------------------------------------------
# update_dataset_and_wait
# ---------------------------------------------------------------------------

def test_update_dataset_and_wait():
    """Test update_dataset_and_wait polls until READY status."""
    client, mock_boto = _make_client()

    mock_boto.update_dataset.return_value = {"datasetId": "ds-789", "status": "UPDATING"}
    mock_boto.get_dataset.return_value = {"dataset": {"datasetId": "ds-789", "status": "READY"}}

    result = client.update_dataset_and_wait(datasetIdentifier="ds-789", name="updated-name")

    assert result["status"] == "READY"
    mock_boto.update_dataset.assert_called_once_with(datasetIdentifier="ds-789", name="updated-name")
    mock_boto.get_dataset.assert_called_once_with(datasetIdentifier="ds-789")


def test_update_dataset_and_wait_unwraps_nested_key():
    """update_dataset_and_wait must unwrap response['dataset'] for status check."""
    client, mock_boto = _make_client()

    mock_boto.update_dataset.return_value = {"datasetId": "ds-upd2", "status": "UPDATING"}
    mock_boto.get_dataset.side_effect = [
        {"dataset": {"datasetId": "ds-upd2", "status": "UPDATING"}},
        {"dataset": {"datasetId": "ds-upd2", "status": "READY"}},
    ]

    result = client.update_dataset_and_wait(datasetIdentifier="ds-upd2", name="v2")

    assert result["status"] == "READY"
    assert mock_boto.get_dataset.call_count == 2


# ---------------------------------------------------------------------------
# delete_dataset_and_wait
# ---------------------------------------------------------------------------

def test_delete_dataset_and_wait():
    """Test delete_dataset_and_wait polls until the dataset is deleted."""
    client, mock_boto = _make_client()

    mock_boto.delete_dataset.return_value = {"datasetId": "ds-123"}
    mock_boto.get_dataset.side_effect = _not_found_error()

    client.delete_dataset_and_wait(datasetIdentifier="ds-123")

    mock_boto.delete_dataset.assert_called_once()
    mock_boto.get_dataset.assert_called_once_with(datasetIdentifier="ds-123")


def test_delete_dataset_and_wait_missing_dataset_id_raises():
    """delete_dataset_and_wait raises ValueError when response has no datasetId."""
    client, mock_boto = _make_client()

    mock_boto.delete_dataset.return_value = {}  # missing datasetId

    with pytest.raises(ValueError, match="datasetId"):
        client.delete_dataset_and_wait(datasetIdentifier="ds-missing")


def test_delete_dataset_and_wait_non_404_error_propagates():
    """Non-ResourceNotFoundException errors bubble up during polling."""
    client, mock_boto = _make_client()

    mock_boto.delete_dataset.return_value = {"datasetId": "ds-err"}
    mock_boto.get_dataset.side_effect = ClientError(
        {"Error": {"Code": "InternalServerError", "Message": "Oops"}},
        "GetDataset",
    )

    with pytest.raises(ClientError, match="InternalServerError"):
        client.delete_dataset_and_wait(datasetIdentifier="ds-err")


# ---------------------------------------------------------------------------
# Pagination helpers
# ---------------------------------------------------------------------------

def test_get_paginator_is_accessible():
    """get_paginator must be reachable via __getattr__."""
    client, mock_boto = _make_client()
    mock_boto.get_paginator.return_value = MagicMock()

    paginator = client.get_paginator("list_datasets")

    mock_boto.get_paginator.assert_called_once_with("list_datasets")


def test_get_waiter_is_accessible():
    """get_waiter must be reachable via __getattr__."""
    client, mock_boto = _make_client()
    mock_boto.get_waiter.return_value = MagicMock()

    waiter = client.get_waiter("dataset_ready")

    mock_boto.get_waiter.assert_called_once_with("dataset_ready")
