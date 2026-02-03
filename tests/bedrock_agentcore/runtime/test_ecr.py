"""Tests for ECR operations."""

from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from bedrock_agentcore.runtime.ecr import (
    build_ecr_uri,
    delete_ecr_repository,
    ensure_ecr_repository,
    get_account_id,
    get_ecr_client,
    get_ecr_login_credentials,
)


class TestGetEcrClient:
    """Tests for get_ecr_client."""

    @patch("bedrock_agentcore.runtime.ecr.boto3")
    def test_creates_client_with_region(self, mock_boto3: MagicMock) -> None:
        """Test that client is created with specified region."""
        get_ecr_client("us-west-2")
        mock_boto3.client.assert_called_once()
        call_kwargs = mock_boto3.client.call_args
        assert call_kwargs[0][0] == "ecr"
        assert call_kwargs[1]["region_name"] == "us-west-2"


class TestEnsureEcrRepository:
    """Tests for ensure_ecr_repository."""

    @patch("bedrock_agentcore.runtime.ecr.get_ecr_client")
    def test_repository_exists(self, mock_get_client: MagicMock) -> None:
        """Test when repository already exists."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.describe_repositories.return_value = {
            "repositories": [
                {
                    "repositoryName": "test-repo",
                    "repositoryUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
                    "repositoryArn": "arn:aws:ecr:us-west-2:123456789012:repository/test-repo",
                }
            ]
        }

        result = ensure_ecr_repository("test-repo", "us-west-2")

        assert result["repositoryName"] == "test-repo"
        assert result["created"] is False
        mock_client.create_repository.assert_not_called()

    @patch("bedrock_agentcore.runtime.ecr.get_ecr_client")
    def test_repository_created(self, mock_get_client: MagicMock) -> None:
        """Test when repository needs to be created."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First call raises RepositoryNotFoundException
        error_response = {"Error": {"Code": "RepositoryNotFoundException"}}
        mock_client.describe_repositories.side_effect = ClientError(error_response, "DescribeRepositories")

        mock_client.create_repository.return_value = {
            "repository": {
                "repositoryName": "test-repo",
                "repositoryUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
                "repositoryArn": "arn:aws:ecr:us-west-2:123456789012:repository/test-repo",
            }
        }

        result = ensure_ecr_repository("test-repo", "us-west-2")

        assert result["repositoryName"] == "test-repo"
        assert result["created"] is True
        mock_client.create_repository.assert_called_once()


class TestDeleteEcrRepository:
    """Tests for delete_ecr_repository."""

    @patch("bedrock_agentcore.runtime.ecr.get_ecr_client")
    def test_delete_success(self, mock_get_client: MagicMock) -> None:
        """Test successful repository deletion."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = delete_ecr_repository("test-repo", "us-west-2")

        assert result["status"] == "DELETED"
        assert result["repositoryName"] == "test-repo"
        mock_client.delete_repository.assert_called_once_with(repositoryName="test-repo", force=False)

    @patch("bedrock_agentcore.runtime.ecr.get_ecr_client")
    def test_delete_not_found(self, mock_get_client: MagicMock) -> None:
        """Test deletion when repository not found."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        error_response = {"Error": {"Code": "RepositoryNotFoundException"}}
        mock_client.delete_repository.side_effect = ClientError(error_response, "DeleteRepository")

        result = delete_ecr_repository("test-repo", "us-west-2")

        assert result["status"] == "NOT_FOUND"


class TestGetAccountId:
    """Tests for get_account_id."""

    @patch("bedrock_agentcore.runtime.ecr.boto3")
    def test_returns_account_id(self, mock_boto3: MagicMock) -> None:
        """Test that account ID is returned."""
        mock_sts = MagicMock()
        mock_boto3.client.return_value = mock_sts
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        result = get_account_id()

        assert result == "123456789012"


class TestBuildEcrUri:
    """Tests for build_ecr_uri."""

    @patch("bedrock_agentcore.runtime.ecr.get_account_id")
    @patch("bedrock_agentcore.runtime.ecr.boto3")
    def test_build_uri(self, mock_boto3: MagicMock, mock_get_account: MagicMock) -> None:
        """Test building ECR URI."""
        mock_get_account.return_value = "123456789012"
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        mock_boto3.Session.return_value = mock_session

        result = build_ecr_uri("my-repo", "us-west-2", "v1.0")

        assert result == "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-repo:v1.0"

    @patch("bedrock_agentcore.runtime.ecr.get_account_id")
    @patch("bedrock_agentcore.runtime.ecr.boto3")
    def test_build_uri_default_tag(self, mock_boto3: MagicMock, mock_get_account: MagicMock) -> None:
        """Test building ECR URI with default tag."""
        mock_get_account.return_value = "123456789012"
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_boto3.Session.return_value = mock_session

        result = build_ecr_uri("my-repo", "us-east-1")

        assert result == "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:latest"


class TestGetEcrLoginCredentials:
    """Tests for get_ecr_login_credentials."""

    @patch("bedrock_agentcore.runtime.ecr.get_ecr_client")
    def test_returns_credentials(self, mock_get_client: MagicMock) -> None:
        """Test that login credentials are returned."""
        import base64

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Base64 encode "AWS:secret-password"
        token = base64.b64encode(b"AWS:secret-password").decode("utf-8")
        mock_client.get_authorization_token.return_value = {
            "authorizationData": [
                {
                    "authorizationToken": token,
                    "proxyEndpoint": "https://123456789012.dkr.ecr.us-west-2.amazonaws.com",
                }
            ]
        }

        username, password, registry = get_ecr_login_credentials("us-west-2")

        assert username == "AWS"
        assert password == "secret-password"
        assert registry == "https://123456789012.dkr.ecr.us-west-2.amazonaws.com"
