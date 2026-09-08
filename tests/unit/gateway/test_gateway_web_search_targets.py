"""Tests for GatewayClient Web Search target helper methods."""

from unittest.mock import MagicMock, Mock

import pytest

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
        """parameterValues is sent even when empty.

        The service drops every configuration whose parameterValues is absent and then
        rejects the request as empty, so the key always goes on the wire.
        """
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

    def test_include_domains_pins_the_version_that_supports_it(self):
        """An include list is rejected server-side on the connector's default version."""
        client = self._make_client()

        client.create_web_search_target(
            gateway_identifier="gw-123",
            include_domains=["docs.aws.amazon.com"],
        )

        connector = client.create_gateway_target_and_wait.call_args[1]["targetConfiguration"]["mcp"]["connector"]
        assert connector["source"] == {"connectorId": "web-search", "version": "1.2.0"}
        assert connector["configurations"][0]["parameterValues"] == {
            "domainFilter": {"include": ["docs.aws.amazon.com"]}
        }

    def test_a_newer_pinned_version_is_kept(self):
        client = self._make_client()

        client.create_web_search_target(
            gateway_identifier="gw-123",
            include_domains=["docs.aws.amazon.com"],
            connector_version="1.3.0",
        )

        source = client.create_gateway_target_and_wait.call_args[1]["targetConfiguration"]["mcp"]["connector"]["source"]
        assert source["version"] == "1.3.0"

    def test_include_domains_with_an_older_pinned_version_raises(self):
        client = self._make_client()

        with pytest.raises(ValueError, match="requires connector version 1.2.0 or later, got 1.1.0"):
            client.create_web_search_target(
                gateway_identifier="gw-123",
                include_domains=["docs.aws.amazon.com"],
                connector_version="1.1.0",
            )

        client.create_gateway_target_and_wait.assert_not_called()

    def test_a_two_component_version_is_not_read_as_older(self):
        client = self._make_client()

        client.create_web_search_target(
            gateway_identifier="gw-123",
            include_domains=["docs.aws.amazon.com"],
            connector_version="1.2",
        )

        source = client.create_gateway_target_and_wait.call_args[1]["targetConfiguration"]["mcp"]["connector"]["source"]
        assert source["version"] == "1.2"

    def test_an_unrecognized_pinned_version_is_passed_through(self):
        """A version this SDK cannot read is the service's to accept or reject."""
        client = self._make_client()

        client.create_web_search_target(
            gateway_identifier="gw-123",
            include_domains=["docs.aws.amazon.com"],
            connector_version="LATEST",
        )

        source = client.create_gateway_target_and_wait.call_args[1]["targetConfiguration"]["mcp"]["connector"]["source"]
        assert source["version"] == "LATEST"

    def test_exclude_domains_alone_pins_no_version(self):
        client = self._make_client()

        client.create_web_search_target(gateway_identifier="gw-123", exclude_domains=["spam.example"])

        source = client.create_gateway_target_and_wait.call_args[1]["targetConfiguration"]["mcp"]["connector"]["source"]
        assert source == {"connectorId": "web-search"}

    @pytest.mark.parametrize("argument", ["include_domains", "exclude_domains"])
    def test_domain_list_maximum(self, argument):
        client = self._make_client()
        domains = [f"d{index}.example" for index in range(101)]

        with pytest.raises(ValueError, match="at most 100 domains, got 101"):
            client.create_web_search_target(gateway_identifier="gw-123", **{argument: domains})

        client.create_gateway_target_and_wait.assert_not_called()

    def test_empty_exclude_domains_is_omitted(self):
        client = self._make_client()

        client.create_web_search_target(gateway_identifier="gw-123", exclude_domains=[])

        call_kwargs = client.create_gateway_target_and_wait.call_args[1]
        config = call_kwargs["targetConfiguration"]["mcp"]["connector"]["configurations"][0]
        assert config["parameterValues"] == {}

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

    def test_wait_config_passed_through(self):
        from bedrock_agentcore._utils.config import WaitConfig

        client = self._make_client()
        wc = WaitConfig(max_wait=60, poll_interval=5)

        client.create_web_search_target(gateway_identifier="gw-123", wait_config=wc)

        assert client.create_gateway_target_and_wait.call_args[1]["wait_config"] == wc
