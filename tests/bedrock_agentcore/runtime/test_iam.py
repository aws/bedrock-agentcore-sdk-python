"""Tests for IAM operations."""

from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from bedrock_agentcore.runtime.iam import (
    delete_role,
    get_iam_client,
    get_or_create_codebuild_execution_role,
    get_or_create_runtime_execution_role,
)


class TestGetIamClient:
    """Tests for get_iam_client."""

    @patch("bedrock_agentcore.runtime.iam.boto3")
    def test_creates_client_with_region(self, mock_boto3: MagicMock) -> None:
        """Test that client is created with specified region."""
        get_iam_client("us-west-2")
        mock_boto3.client.assert_called_once()
        call_kwargs = mock_boto3.client.call_args
        assert call_kwargs[0][0] == "iam"
        assert call_kwargs[1]["region_name"] == "us-west-2"


class TestGetOrCreateRuntimeExecutionRole:
    """Tests for get_or_create_runtime_execution_role."""

    @patch("bedrock_agentcore.runtime.iam.get_iam_client")
    def test_role_exists(self, mock_get_client: MagicMock) -> None:
        """Test when role already exists."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_role.return_value = {
            "Role": {
                "RoleName": "bedrock-agentcore-runtime-abc123",
                "Arn": "arn:aws:iam::123456789012:role/bedrock-agentcore-runtime-abc123",
            }
        }

        result = get_or_create_runtime_execution_role("test-agent", "us-west-2")

        assert "roleArn" in result
        assert result["created"] is False
        mock_client.create_role.assert_not_called()

    @patch("bedrock_agentcore.runtime.iam.boto3")
    @patch("bedrock_agentcore.runtime.iam.get_iam_client")
    def test_role_created(self, mock_get_client: MagicMock, mock_boto3: MagicMock) -> None:
        """Test when role needs to be created."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock STS client for account ID
        mock_sts = MagicMock()
        mock_boto3.client.return_value = mock_sts
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto3.Session.return_value.region_name = "us-west-2"

        # First call raises NoSuchEntityException
        error_response = {"Error": {"Code": "NoSuchEntity"}}
        mock_client.get_role.side_effect = ClientError(error_response, "GetRole")

        mock_client.create_role.return_value = {
            "Role": {
                "RoleName": "bedrock-agentcore-runtime-abc123",
                "Arn": "arn:aws:iam::123456789012:role/bedrock-agentcore-runtime-abc123",
            }
        }

        result = get_or_create_runtime_execution_role("test-agent", "us-west-2")

        assert "roleArn" in result
        assert result["created"] is True
        mock_client.create_role.assert_called_once()
        # Uses put_role_policy for inline policy, not attach_role_policy
        mock_client.put_role_policy.assert_called_once()


class TestGetOrCreateCodeBuildExecutionRole:
    """Tests for get_or_create_codebuild_execution_role."""

    @patch("bedrock_agentcore.runtime.iam.get_iam_client")
    def test_role_exists(self, mock_get_client: MagicMock) -> None:
        """Test when role already exists."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_role.return_value = {
            "Role": {
                "RoleName": "bedrock-agentcore-codebuild-abc123",
                "Arn": "arn:aws:iam::123456789012:role/bedrock-agentcore-codebuild-abc123",
            }
        }

        result = get_or_create_codebuild_execution_role(
            "test-agent",
            "arn:aws:ecr:us-west-2:123456789012:repository/test-repo",
            "us-west-2",
            "test-bucket",
        )

        assert "roleArn" in result
        assert result["created"] is False
        mock_client.create_role.assert_not_called()

    @patch("bedrock_agentcore.runtime.iam.get_iam_client")
    def test_role_created(self, mock_get_client: MagicMock) -> None:
        """Test when role needs to be created."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First call raises NoSuchEntityException
        error_response = {"Error": {"Code": "NoSuchEntity"}}
        mock_client.get_role.side_effect = ClientError(error_response, "GetRole")

        mock_client.create_role.return_value = {
            "Role": {
                "RoleName": "bedrock-agentcore-codebuild-abc123",
                "Arn": "arn:aws:iam::123456789012:role/bedrock-agentcore-codebuild-abc123",
            }
        }

        result = get_or_create_codebuild_execution_role(
            "test-agent",
            "arn:aws:ecr:us-west-2:123456789012:repository/test-repo",
            "us-west-2",
            "test-bucket",
        )

        assert "roleArn" in result
        assert result["created"] is True
        mock_client.create_role.assert_called_once()
        mock_client.put_role_policy.assert_called_once()


class TestDeleteRole:
    """Tests for delete_role."""

    @patch("bedrock_agentcore.runtime.iam.get_iam_client")
    def test_delete_success(self, mock_get_client: MagicMock) -> None:
        """Test successful role deletion."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock list_role_policies (inline policies)
        mock_client.list_role_policies.return_value = {"PolicyNames": ["InlinePolicy"]}

        result = delete_role("test-role", "us-west-2")

        assert result["status"] == "DELETED"
        assert result["roleName"] == "test-role"
        mock_client.list_role_policies.assert_called_once()
        mock_client.delete_role_policy.assert_called_once()
        mock_client.delete_role.assert_called_once()

    @patch("bedrock_agentcore.runtime.iam.get_iam_client")
    def test_delete_no_inline_policies(self, mock_get_client: MagicMock) -> None:
        """Test deletion when role has no inline policies."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # No inline policies
        mock_client.list_role_policies.return_value = {"PolicyNames": []}

        result = delete_role("test-role", "us-west-2")

        assert result["status"] == "DELETED"
        mock_client.delete_role_policy.assert_not_called()
        mock_client.delete_role.assert_called_once()

    @patch("bedrock_agentcore.runtime.iam.get_iam_client")
    def test_delete_not_found(self, mock_get_client: MagicMock) -> None:
        """Test deletion when role not found."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        error_response = {"Error": {"Code": "NoSuchEntity"}}
        mock_client.list_role_policies.side_effect = ClientError(error_response, "ListRolePolicies")

        result = delete_role("test-role", "us-west-2")

        assert result["status"] == "NOT_FOUND"
