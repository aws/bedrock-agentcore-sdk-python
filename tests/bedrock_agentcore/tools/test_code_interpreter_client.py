import datetime
from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter, code_session


class TestCodeInterpreterClient:
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_init(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_control_client = MagicMock()
        mock_data_client = MagicMock()
        mock_session.client.side_effect = [mock_control_client, mock_data_client]
        mock_boto3.Session.return_value = mock_session
        mock_get_control_endpoint.return_value = "https://mock-control-endpoint.com"
        mock_get_data_endpoint.return_value = "https://mock-data-endpoint.com"
        region = "us-west-2"

        # Act
        client = CodeInterpreter(region)

        # Assert
        mock_boto3.Session.assert_called_once()
        assert mock_session.client.call_count == 2
        mock_session.client.assert_any_call(
            "bedrock-agentcore-control",
            region_name=region,
            endpoint_url="https://mock-control-endpoint.com",
        )
        mock_session.client.assert_any_call(
            "bedrock-agentcore", region_name=region, endpoint_url="https://mock-data-endpoint.com"
        )
        assert client.control_plane_client == mock_control_client
        assert client.data_plane_client == mock_data_client
        assert client.identifier is None
        assert client.session_id is None

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    def test_init_with_custom_session(self, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_control_client = MagicMock()
        mock_data_client = MagicMock()
        mock_session.client.side_effect = [mock_control_client, mock_data_client]
        mock_get_control_endpoint.return_value = "https://mock-control-endpoint.com"
        mock_get_data_endpoint.return_value = "https://mock-data-endpoint.com"
        region = "us-west-2"

        # Act
        client = CodeInterpreter(region, session=mock_session)

        # Assert
        assert mock_session.client.call_count == 2
        assert client.control_plane_client == mock_control_client
        assert client.data_plane_client == mock_data_client
        assert client.identifier is None
        assert client.session_id is None

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_property_getters_setters(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session

        client = CodeInterpreter("us-west-2")
        test_identifier = "test.identifier"
        test_session_id = "test-session-id"

        # Act & Assert - identifier
        client.identifier = test_identifier
        assert client.identifier == test_identifier

        # Act & Assert - session_id
        client.session_id = test_session_id
        assert client.session_id == test_session_id

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_create_code_interpreter_minimal(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_control_client = MagicMock()
        mock_data_client = MagicMock()
        mock_session.client.side_effect = [mock_control_client, mock_data_client]
        mock_boto3.Session.return_value = mock_session
        client = CodeInterpreter("us-west-2")

        mock_response = {
            "codeInterpreterArn": "arn:aws:bedrock-agentcore:us-west-2:123456789012:code-interpreter/test-interp",
            "codeInterpreterId": "test-interp-123",
            "createdAt": datetime.datetime.now(),
            "status": "CREATING",
        }
        client.control_plane_client.create_code_interpreter.return_value = mock_response

        # Act
        result = client.create_code_interpreter(
            name="test_interpreter",
            execution_role_arn="arn:aws:iam::123456789012:role/InterpreterRole",
        )

        # Assert
        client.control_plane_client.create_code_interpreter.assert_called_once_with(
            name="test_interpreter",
            executionRoleArn="arn:aws:iam::123456789012:role/InterpreterRole",
            networkConfiguration={"networkMode": "PUBLIC"},
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_create_code_interpreter_with_all_options(
        self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint
    ):
        # Arrange
        mock_session = MagicMock()
        mock_control_client = MagicMock()
        mock_data_client = MagicMock()
        mock_session.client.side_effect = [mock_control_client, mock_data_client]
        mock_boto3.Session.return_value = mock_session
        client = CodeInterpreter("us-west-2")

        mock_response = {
            "codeInterpreterArn": "arn:aws:bedrock-agentcore:us-west-2:123456789012:code-interpreter/test-interp",
            "codeInterpreterId": "test-interp-123",
            "createdAt": datetime.datetime.now(),
            "status": "CREATING",
        }
        client.control_plane_client.create_code_interpreter.return_value = mock_response

        network_config = {
            "networkMode": "VPC",
            "vpcConfig": {"securityGroups": ["sg-123"], "subnets": ["subnet-123"]},
        }
        tags = {"Environment": "Test"}

        # Act
        result = client.create_code_interpreter(
            name="test_interpreter",
            execution_role_arn="arn:aws:iam::123456789012:role/InterpreterRole",
            network_configuration=network_config,
            description="Test interpreter",
            tags=tags,
            client_token="test-token",
        )

        # Assert
        client.control_plane_client.create_code_interpreter.assert_called_once_with(
            name="test_interpreter",
            executionRoleArn="arn:aws:iam::123456789012:role/InterpreterRole",
            networkConfiguration=network_config,
            description="Test interpreter",
            tags=tags,
            clientToken="test-token",
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_delete_code_interpreter(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session
        client = CodeInterpreter("us-west-2")

        mock_response = {
            "codeInterpreterId": "test-interp-123",
            "lastUpdatedAt": datetime.datetime.now(),
            "status": "DELETING",
        }
        client.control_plane_client.delete_code_interpreter.return_value = mock_response

        # Act
        result = client.delete_code_interpreter("test-interp-123")

        # Assert
        client.control_plane_client.delete_code_interpreter.assert_called_once_with(codeInterpreterId="test-interp-123")
        assert result == mock_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_delete_code_interpreter_with_client_token(
        self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint
    ):
        # Arrange
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session
        client = CodeInterpreter("us-west-2")

        mock_response = {
            "codeInterpreterId": "test-interp-123",
            "lastUpdatedAt": datetime.datetime.now(),
            "status": "DELETING",
        }
        client.control_plane_client.delete_code_interpreter.return_value = mock_response

        # Act
        result = client.delete_code_interpreter("test-interp-123", client_token="test-token")

        # Assert
        client.control_plane_client.delete_code_interpreter.assert_called_once_with(
            codeInterpreterId="test-interp-123", clientToken="test-token"
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_get_code_interpreter(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session
        client = CodeInterpreter("us-west-2")

        mock_response = {
            "codeInterpreterArn": "arn:aws:bedrock-agentcore:us-west-2:123456789012:code-interpreter/test-interp",
            "codeInterpreterId": "test-interp-123",
            "name": "test_interpreter",
            "status": "READY",
        }
        client.control_plane_client.get_code_interpreter.return_value = mock_response

        # Act
        result = client.get_code_interpreter("test-interp-123")

        # Assert
        client.control_plane_client.get_code_interpreter.assert_called_once_with(codeInterpreterId="test-interp-123")
        assert result == mock_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_list_code_interpreters_default(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session
        client = CodeInterpreter("us-west-2")

        mock_response = {
            "codeInterpreterSummaries": [
                {"codeInterpreterId": "interp-1", "name": "interpreter_1", "status": "READY"},
                {"codeInterpreterId": "interp-2", "name": "interpreter_2", "status": "CREATING"},
            ]
        }
        client.control_plane_client.list_code_interpreters.return_value = mock_response

        # Act
        result = client.list_code_interpreters()

        # Assert
        client.control_plane_client.list_code_interpreters.assert_called_once_with(maxResults=10)
        assert result == mock_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_list_code_interpreters_with_filters(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session
        client = CodeInterpreter("us-west-2")

        mock_response = {
            "codeInterpreterSummaries": [{"codeInterpreterId": "interp-1", "name": "interpreter_1", "status": "READY"}],
            "nextToken": "next-page-token",
        }
        client.control_plane_client.list_code_interpreters.return_value = mock_response

        # Act
        result = client.list_code_interpreters(interpreter_type="CUSTOM", max_results=50, next_token="token-123")

        # Assert
        client.control_plane_client.list_code_interpreters.assert_called_once_with(
            maxResults=50, type="CUSTOM", nextToken="token-123"
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    @patch("bedrock_agentcore.tools.code_interpreter_client.uuid.uuid4")
    def test_start_with_defaults(self, mock_uuid4, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_data_client = MagicMock()
        mock_session.client.return_value = mock_data_client
        mock_boto3.Session.return_value = mock_session
        mock_uuid4.return_value.hex = "12345678abcdef"

        client = CodeInterpreter("us-west-2")
        mock_response = {"codeInterpreterIdentifier": "aws.codeinterpreter.v1", "sessionId": "session-123"}
        client.data_plane_client.start_code_interpreter_session.return_value = mock_response

        # Act
        session_id = client.start()

        # Assert
        client.data_plane_client.start_code_interpreter_session.assert_called_once_with(
            codeInterpreterIdentifier="aws.codeinterpreter.v1",
            name="code-session-12345678",
            sessionTimeoutSeconds=900,
        )
        assert session_id == "session-123"
        assert client.identifier == "aws.codeinterpreter.v1"
        assert client.session_id == "session-123"

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_start_with_custom_params(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_data_client = MagicMock()
        mock_session.client.return_value = mock_data_client
        mock_boto3.Session.return_value = mock_session

        client = CodeInterpreter("us-west-2")
        mock_response = {"codeInterpreterIdentifier": "custom.interpreter", "sessionId": "custom-session-123"}
        client.data_plane_client.start_code_interpreter_session.return_value = mock_response

        # Act
        session_id = client.start(
            identifier="custom.interpreter",
            name="custom-session",
            session_timeout_seconds=600,
        )

        # Assert
        client.data_plane_client.start_code_interpreter_session.assert_called_once_with(
            codeInterpreterIdentifier="custom.interpreter",
            name="custom-session",
            sessionTimeoutSeconds=600,
        )
        assert session_id == "custom-session-123"
        assert client.identifier == "custom.interpreter"
        assert client.session_id == "custom-session-123"

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_stop_when_session_exists(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_data_client = MagicMock()
        mock_session.client.return_value = mock_data_client
        mock_boto3.Session.return_value = mock_session

        client = CodeInterpreter("us-west-2")
        client.identifier = "test.identifier"
        client.session_id = "test-session-id"

        # Act
        client.stop()

        # Assert
        client.data_plane_client.stop_code_interpreter_session.assert_called_once_with(
            codeInterpreterIdentifier="test.identifier", sessionId="test-session-id"
        )
        assert client.identifier is None
        assert client.session_id is None

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_stop_when_no_session(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_data_client = MagicMock()
        mock_session.client.return_value = mock_data_client
        mock_boto3.Session.return_value = mock_session

        client = CodeInterpreter("us-west-2")
        client.identifier = None
        client.session_id = None

        # Act
        result = client.stop()

        # Assert
        client.data_plane_client.stop_code_interpreter_session.assert_not_called()
        assert result is True

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_get_session(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session
        client = CodeInterpreter("us-west-2")
        client.identifier = "test-interpreter-id"
        client.session_id = "test-session-id"

        mock_response = {
            "sessionId": "test-session-id",
            "codeInterpreterIdentifier": "test-interpreter-id",
            "status": "READY",
        }
        client.data_plane_client.get_code_interpreter_session.return_value = mock_response

        # Act
        result = client.get_session()

        # Assert
        client.data_plane_client.get_code_interpreter_session.assert_called_once_with(
            codeInterpreterIdentifier="test-interpreter-id", sessionId="test-session-id"
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_get_session_with_params(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session
        client = CodeInterpreter("us-west-2")

        mock_response = {
            "sessionId": "other-session-id",
            "codeInterpreterIdentifier": "other-interpreter-id",
            "status": "READY",
        }
        client.data_plane_client.get_code_interpreter_session.return_value = mock_response

        # Act
        result = client.get_session(interpreter_id="other-interpreter-id", session_id="other-session-id")

        # Assert
        client.data_plane_client.get_code_interpreter_session.assert_called_once_with(
            codeInterpreterIdentifier="other-interpreter-id", sessionId="other-session-id"
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_get_session_missing_ids(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session
        client = CodeInterpreter("us-west-2")

        # Act & Assert
        with pytest.raises(ValueError, match="Interpreter ID and Session ID must be provided"):
            client.get_session()

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_list_sessions(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session
        client = CodeInterpreter("us-west-2")
        client.identifier = "test-interpreter-id"

        mock_response = {
            "items": [
                {"sessionId": "session-1", "status": "READY"},
                {"sessionId": "session-2", "status": "TERMINATED"},
            ]
        }
        client.data_plane_client.list_code_interpreter_sessions.return_value = mock_response

        # Act
        result = client.list_sessions()

        # Assert
        client.data_plane_client.list_code_interpreter_sessions.assert_called_once_with(
            codeInterpreterIdentifier="test-interpreter-id", maxResults=10
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_list_sessions_with_filters(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session
        client = CodeInterpreter("us-west-2")

        mock_response = {
            "items": [{"sessionId": "session-1", "status": "READY"}],
            "nextToken": "next-token",
        }
        client.data_plane_client.list_code_interpreter_sessions.return_value = mock_response

        # Act
        result = client.list_sessions(
            interpreter_id="custom-interpreter",
            status="READY",
            max_results=50,
            next_token="token-123",
        )

        # Assert
        client.data_plane_client.list_code_interpreter_sessions.assert_called_once_with(
            codeInterpreterIdentifier="custom-interpreter",
            maxResults=50,
            status="READY",
            nextToken="token-123",
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_list_sessions_missing_interpreter_id(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session
        client = CodeInterpreter("us-west-2")

        # Act & Assert
        with pytest.raises(ValueError, match="Interpreter ID must be provided"):
            client.list_sessions()

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    @patch("bedrock_agentcore.tools.code_interpreter_client.uuid.uuid4")
    def test_invoke_with_existing_session(
        self, mock_uuid4, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint
    ):
        # Arrange
        mock_session = MagicMock()
        mock_data_client = MagicMock()
        mock_session.client.return_value = mock_data_client
        mock_boto3.Session.return_value = mock_session
        mock_uuid4.return_value.hex = "12345678abcdef"

        client = CodeInterpreter("us-west-2")
        client.identifier = "test.identifier"
        client.session_id = "test-session-id"

        mock_response = {"result": "success"}
        client.data_plane_client.invoke_code_interpreter.return_value = mock_response

        # Act
        result = client.invoke(method="testMethod", params={"param1": "value1"})

        # Assert
        client.data_plane_client.invoke_code_interpreter.assert_called_once_with(
            codeInterpreterIdentifier="test.identifier",
            sessionId="test-session-id",
            name="testMethod",
            arguments={"param1": "value1"},
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_control_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_invoke_with_no_session(self, mock_boto3, mock_get_data_endpoint, mock_get_control_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_data_client = MagicMock()
        mock_session.client.return_value = mock_data_client
        mock_boto3.Session.return_value = mock_session

        client = CodeInterpreter("us-west-2")
        client.identifier = None
        client.session_id = None

        mock_start_response = {"codeInterpreterIdentifier": "aws.codeinterpreter.v1", "sessionId": "session-123"}
        client.data_plane_client.start_code_interpreter_session.return_value = mock_start_response

        mock_invoke_response = {"result": "success"}
        client.data_plane_client.invoke_code_interpreter.return_value = mock_invoke_response

        # Act
        result = client.invoke(method="testMethod", params=None)

        # Assert
        client.data_plane_client.start_code_interpreter_session.assert_called_once()
        client.data_plane_client.invoke_code_interpreter.assert_called_once_with(
            codeInterpreterIdentifier="aws.codeinterpreter.v1",
            sessionId="session-123",
            name="testMethod",
            arguments={},
        )
        assert result == mock_invoke_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.CodeInterpreter")
    def test_code_session_context_manager(self, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Act
        with code_session("us-west-2"):
            pass

        # Assert
        mock_client_class.assert_called_once_with("us-west-2", session=None)
        mock_client.start.assert_called_once_with()
        mock_client.stop.assert_called_once()

    @patch("bedrock_agentcore.tools.code_interpreter_client.CodeInterpreter")
    def test_code_session_context_manager_with_session(self, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_session = MagicMock()

        # Act
        with code_session("us-west-2", session=mock_session):
            pass

        # Assert
        mock_client_class.assert_called_once_with("us-west-2", session=mock_session)
        mock_client.start.assert_called_once_with()
        mock_client.stop.assert_called_once()

    @patch("bedrock_agentcore.tools.code_interpreter_client.CodeInterpreter")
    def test_code_session_context_manager_with_identifier(self, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Act
        with code_session("us-west-2", identifier="custom-interpreter-123"):
            pass

        # Assert
        mock_client_class.assert_called_once_with("us-west-2", session=None)
        mock_client.start.assert_called_once_with(identifier="custom-interpreter-123")
        mock_client.stop.assert_called_once()
