"""Tests for GatewayClient Knowledge Base target helper methods."""

from unittest.mock import MagicMock, Mock

from bedrock_agentcore.gateway.client import GatewayClient


class TestCreateKnowledgeBaseTarget:
    """Tests for create_knowledge_base_target."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = GatewayClient(boto3_session=mock_session)
        client.create_gateway_target_and_wait = Mock(return_value={"status": "READY", "targetId": "t-123"})
        return client

    def test_minimal(self):
        client = self._make_client()

        result = client.create_knowledge_base_target(
            gateway_identifier="gw-123",
            knowledge_base_id="KB456",
        )

        assert result["status"] == "READY"
        client.create_gateway_target_and_wait.assert_called_once_with(
            wait_config=None,
            gatewayIdentifier="gw-123",
            name="kb-KB456",
            targetConfiguration={
                "mcp": {
                    "connector": {
                        "source": {"connectorId": "bedrock-knowledge-bases"},
                        "enabled": ["Retrieve"],
                        "configurations": [
                            {
                                "name": "Retrieve",
                                "parameterValues": {"knowledgeBaseId": "KB456"},
                            }
                        ],
                    },
                },
            },
            credentialProviderConfigurations=[
                {"credentialProviderType": "GATEWAY_IAM_ROLE"},
            ],
        )

    def test_with_all_options(self):
        client = self._make_client()

        result = client.create_knowledge_base_target(
            gateway_identifier="gw-123",
            knowledge_base_id="KB456",
            name="custom-target",
            description="Search product docs",
            retrieval_configuration={"vectorSearchConfiguration": {"numberOfResults": 5}},
            parameter_overrides=[{"path": "$.retrievalQuery.text", "visible": True}],
        )

        assert result["status"] == "READY"
        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        assert call_kwargs["name"] == "custom-target"
        connector = call_kwargs["targetConfiguration"]["mcp"]["connector"]
        config = connector["configurations"][0]
        assert config["description"] == "Search product docs"
        assert config["parameterValues"]["knowledgeBaseId"] == "KB456"
        expected_retrieval = {"vectorSearchConfiguration": {"numberOfResults": 5}}
        assert config["parameterValues"]["retrievalConfiguration"] == expected_retrieval
        assert config["parameterOverrides"] == [{"path": "$.retrievalQuery.text", "visible": True}]

    def test_kwargs_override_target_configuration(self):
        client = self._make_client()

        custom_target_config = {"mcp": {"lambda": {"lambdaArn": "arn:..."}}}
        client.create_knowledge_base_target(
            gateway_identifier="gw-123",
            knowledge_base_id="KB456",
            targetConfiguration=custom_target_config,
        )

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        assert call_kwargs["targetConfiguration"] == custom_target_config

    def test_kwargs_override_credential_provider(self):
        client = self._make_client()

        custom_creds = [{"credentialProviderType": "CUSTOM"}]
        client.create_knowledge_base_target(
            gateway_identifier="gw-123",
            knowledge_base_id="KB456",
            credentialProviderConfigurations=custom_creds,
        )

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        assert call_kwargs["credentialProviderConfigurations"] == custom_creds

    def test_default_name(self):
        client = self._make_client()

        client.create_knowledge_base_target(
            gateway_identifier="gw-123",
            knowledge_base_id="ABCDEF123",
        )

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        assert call_kwargs["name"] == "kb-ABCDEF123"

    def test_wait_config_passed_through(self):
        from bedrock_agentcore._utils.config import WaitConfig

        client = self._make_client()
        wc = WaitConfig(max_wait=60, poll_interval=5)

        client.create_knowledge_base_target(
            gateway_identifier="gw-123",
            knowledge_base_id="KB456",
            wait_config=wc,
        )

        assert client.create_gateway_target_and_wait.call_args[1]["wait_config"] == wc


class TestCreateAgenticRetrieveTarget:
    """Tests for create_agentic_retrieve_target."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = GatewayClient(boto3_session=mock_session)
        client.create_gateway_target_and_wait = Mock(return_value={"status": "READY", "targetId": "t-456"})
        return client

    def test_minimal(self):
        client = self._make_client()

        result = client.create_agentic_retrieve_target(
            gateway_identifier="gw-123",
            retrievers=[{"knowledgeBaseId": "KB1"}],
            model_arn="arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6-v1",
        )

        assert result["status"] == "READY"
        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        connector = call_kwargs["targetConfiguration"]["mcp"]["connector"]
        assert connector["source"]["connectorId"] == "bedrock-agentic-retrieve"
        assert connector["enabled"] == ["AgenticRetrieveStream"]
        config = connector["configurations"][0]
        assert config["name"] == "AgenticRetrieveStream"
        assert config["parameterValues"]["retrievers"] == [{"knowledgeBaseId": "KB1"}]
        agentic_config = config["parameterValues"]["agenticRetrieveConfiguration"]
        model_arn = agentic_config["foundationModelConfiguration"]["bedrock"]["modelArn"]
        assert model_arn == "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6-v1"

    def test_with_all_options(self):
        client = self._make_client()

        retrievers = [
            {"knowledgeBaseId": "KB1", "description": "Product docs"},
            {"knowledgeBaseId": "KB2", "description": "FAQ"},
        ]

        result = client.create_agentic_retrieve_target(
            gateway_identifier="gw-123",
            retrievers=retrievers,
            model_arn="arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6-v1",
            name="multi-kb-target",
            description="Search across all KBs",
            max_agent_iteration=5,
            parameter_overrides=[{"path": "$.messages", "visible": True}],
        )

        assert result["status"] == "READY"
        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        assert call_kwargs["name"] == "multi-kb-target"
        connector = call_kwargs["targetConfiguration"]["mcp"]["connector"]
        config = connector["configurations"][0]
        assert config["description"] == "Search across all KBs"
        assert config["parameterValues"]["retrievers"] == retrievers
        assert config["parameterValues"]["agenticRetrieveConfiguration"]["maxAgentIteration"] == 5
        assert config["parameterOverrides"] == [{"path": "$.messages", "visible": True}]

    def test_kwargs_override_target_configuration(self):
        client = self._make_client()

        custom_target_config = {"mcp": {"lambda": {"lambdaArn": "arn:..."}}}
        client.create_agentic_retrieve_target(
            gateway_identifier="gw-123",
            retrievers=[{"knowledgeBaseId": "KB1"}],
            model_arn="arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6-v1",
            targetConfiguration=custom_target_config,
        )

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        assert call_kwargs["targetConfiguration"] == custom_target_config

    def test_default_name_contains_timestamp(self):
        client = self._make_client()

        client.create_agentic_retrieve_target(
            gateway_identifier="gw-123",
            retrievers=[{"knowledgeBaseId": "KB1"}],
            model_arn="arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6-v1",
        )

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        assert call_kwargs["name"].startswith("agentic-retrieve-")

    def test_default_credential_provider(self):
        client = self._make_client()

        client.create_agentic_retrieve_target(
            gateway_identifier="gw-123",
            retrievers=[{"knowledgeBaseId": "KB1"}],
            model_arn="arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6-v1",
        )

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        assert call_kwargs["credentialProviderConfigurations"] == [
            {"credentialProviderType": "GATEWAY_IAM_ROLE"},
        ]
