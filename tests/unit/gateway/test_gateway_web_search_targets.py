"""Tests for GatewayClient Web Search target helper methods."""

from unittest.mock import MagicMock, Mock

from bedrock_agentcore.gateway.client import GatewayClient


class TestCreateWebSearchTarget:
    """Tests for create_web_search_target."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = GatewayClient(boto3_session=mock_session)
        client.create_gateway_target_and_wait = Mock(return_value={"status": "READY", "targetId": "t-789"})
        return client

    def test_minimal(self):
        client = self._make_client()

        result = client.create_web_search_target(gateway_identifier="gw-123")

        assert result["status"] == "READY"
        client.create_gateway_target_and_wait.assert_called_once_with(
            wait_config=None,
            gatewayIdentifier="gw-123",
            name="amazon-web-search",
            targetConfiguration={
                "mcp": {
                    "connector": {
                        "source": {"connectorId": "web-search"},
                        "enabled": ["WebSearch"],
                        "configurations": [{"name": "WebSearch", "parameterValues": {}}],
                    },
                },
            },
            credentialProviderConfigurations=[
                {"credentialProviderType": "GATEWAY_IAM_ROLE"},
            ],
        )

    def test_with_all_options(self):
        client = self._make_client()

        result = client.create_web_search_target(
            gateway_identifier="gw-123",
            name="custom-search",
            description="Search the public web",
            exclude_domains=["example.com", "spam.example"],
            include_domains=["allowed.example"],
            connector_version="1.2.0",
            parameter_overrides=[{"path": "$.maxResults", "visible": True}],
        )

        assert result["status"] == "READY"
        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        assert call_kwargs["name"] == "custom-search"
        connector = call_kwargs["targetConfiguration"]["mcp"]["connector"]
        assert connector["source"] == {"connectorId": "web-search", "version": "1.2.0"}
        assert connector["enabled"] == ["WebSearch"]
        config = connector["configurations"][0]
        assert config["name"] == "WebSearch"
        assert config["description"] == "Search the public web"
        assert config["parameterValues"] == {
            "domainFilter": {
                "include": ["allowed.example"],
                "exclude": ["example.com", "spam.example"],
            }
        }
        assert config["parameterOverrides"] == [{"path": "$.maxResults", "visible": True}]

    def test_include_domains_only(self):
        client = self._make_client()

        client.create_web_search_target(
            gateway_identifier="gw-123",
            include_domains=["docs.aws.amazon.com"],
        )

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        config = call_kwargs["targetConfiguration"]["mcp"]["connector"]["configurations"][0]
        assert config["parameterValues"] == {"domainFilter": {"include": ["docs.aws.amazon.com"]}}

    def test_connector_version_omitted_by_default(self):
        client = self._make_client()

        client.create_web_search_target(gateway_identifier="gw-123")

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        source = call_kwargs["targetConfiguration"]["mcp"]["connector"]["source"]
        assert source == {"connectorId": "web-search"}

    def test_no_domain_filter_when_no_exclude_domains(self):
        client = self._make_client()

        client.create_web_search_target(gateway_identifier="gw-123")

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        config = call_kwargs["targetConfiguration"]["mcp"]["connector"]["configurations"][0]
        assert config["parameterValues"] == {}

    def test_empty_exclude_domains_is_omitted(self):
        client = self._make_client()

        client.create_web_search_target(gateway_identifier="gw-123", exclude_domains=[])

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        config = call_kwargs["targetConfiguration"]["mcp"]["connector"]["configurations"][0]
        assert config["parameterValues"] == {}

    def test_parameter_values_always_present(self):
        """The service drops configurations without parameterValues, then rejects the
        request as empty, so the key is sent even when there is nothing to configure."""
        client = self._make_client()

        client.create_web_search_target(gateway_identifier="gw-123", description="Search the web")

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        config = call_kwargs["targetConfiguration"]["mcp"]["connector"]["configurations"][0]
        assert "parameterValues" in config

    def test_kwargs_override_target_configuration(self):
        client = self._make_client()

        custom_target_config = {"mcp": {"lambda": {"lambdaArn": "arn:..."}}}
        client.create_web_search_target(
            gateway_identifier="gw-123",
            targetConfiguration=custom_target_config,
        )

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        assert call_kwargs["targetConfiguration"] == custom_target_config

    def test_kwargs_override_credential_provider(self):
        client = self._make_client()

        custom_creds = [{"credentialProviderType": "CUSTOM"}]
        client.create_web_search_target(
            gateway_identifier="gw-123",
            credentialProviderConfigurations=custom_creds,
        )

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        assert call_kwargs["credentialProviderConfigurations"] == custom_creds

    def test_default_credential_provider(self):
        client = self._make_client()

        client.create_web_search_target(gateway_identifier="gw-123")

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        assert call_kwargs["credentialProviderConfigurations"] == [
            {"credentialProviderType": "GATEWAY_IAM_ROLE"},
        ]

    def test_wait_config_passed_through(self):
        from bedrock_agentcore._utils.config import WaitConfig

        client = self._make_client()
        wc = WaitConfig(max_wait=60, poll_interval=5)

        client.create_web_search_target(gateway_identifier="gw-123", wait_config=wc)

        assert client.create_gateway_target_and_wait.call_args[1]["wait_config"] == wc
