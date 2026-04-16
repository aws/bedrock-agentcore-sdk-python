"""Tests for PolicyEngineClient."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.policy.client import PolicyEngineClient


class TestPolicyEngineClientInit:
    """Tests for PolicyEngineClient initialization."""

    def test_init_with_region(self):
        mock_session = MagicMock()
        mock_session.region_name = "eu-west-1"
        client = PolicyEngineClient(region_name="us-west-2", boto3_session=mock_session)
        assert client.region_name == "us-west-2"

    def test_init_default_region_from_session(self):
        mock_session = MagicMock()
        mock_session.region_name = "eu-west-1"
        client = PolicyEngineClient(boto3_session=mock_session)
        assert client.region_name == "eu-west-1"

    def test_init_default_region_fallback(self):
        mock_session = MagicMock()
        mock_session.region_name = None
        client = PolicyEngineClient(boto3_session=mock_session)
        assert client.region_name == "us-west-2"

    def test_init_creates_cp_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        PolicyEngineClient(boto3_session=mock_session)

        mock_session.client.assert_called_once()
        call_args = mock_session.client.call_args
        assert call_args[0][0] == "bedrock-agentcore-control"

    def test_init_with_integration_source(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = PolicyEngineClient(boto3_session=mock_session, integration_source="langchain")
        assert client.integration_source == "langchain"


class TestPolicyEngineClientPassthrough:
    """Tests for __getattr__ passthrough."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = PolicyEngineClient(boto3_session=mock_session)
        client.cp_client = Mock()
        return client

    def test_policy_engine_crud_forwarded(self):
        client = self._make_client()
        client.cp_client.create_policy_engine.return_value = {"policyEngineId": "pe-123"}

        result = client.create_policy_engine(name="test-engine")

        client.cp_client.create_policy_engine.assert_called_once_with(name="test-engine")
        assert result["policyEngineId"] == "pe-123"

    def test_policy_crud_forwarded(self):
        client = self._make_client()
        client.cp_client.create_policy.return_value = {"policyId": "pol-123"}

        result = client.create_policy(
            policyEngineId="pe-123",
            name="test-policy",
            definition={"cedar": {"statement": "permit(principal, action, resource);"}},
        )

        client.cp_client.create_policy.assert_called_once()
        assert result["policyId"] == "pol-123"

    def test_snake_case_kwargs_converted(self):
        client = self._make_client()
        client.cp_client.get_policy_engine.return_value = {"policyEngineId": "pe-123"}

        client.get_policy_engine(policy_engine_id="pe-123")

        client.cp_client.get_policy_engine.assert_called_once_with(policyEngineId="pe-123")

    def test_non_allowlisted_method_raises_attribute_error(self):
        client = self._make_client()

        with pytest.raises(AttributeError, match="has no attribute 'not_a_real_method'"):
            client.not_a_real_method()

    def test_all_cp_methods_in_allowlist(self):
        expected = {
            "create_policy_engine",
            "get_policy_engine",
            "list_policy_engines",
            "update_policy_engine",
            "delete_policy_engine",
            "create_policy",
            "get_policy",
            "list_policies",
            "update_policy",
            "delete_policy",
            "start_policy_generation",
            "get_policy_generation",
            "list_policy_generations",
            "list_policy_generation_assets",
        }
        assert expected == PolicyEngineClient._ALLOWED_CP_METHODS


class TestGeneratePolicy:
    """Tests for generate_policy orchestration method."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = PolicyEngineClient(boto3_session=mock_session)
        client.cp_client = Mock()
        return client

    def test_generate_policy_immediate_success(self):
        client = self._make_client()
        client.cp_client.start_policy_generation.return_value = {"policyGenerationId": "gen-123"}
        client.cp_client.get_policy_generation.return_value = {"status": "GENERATED", "policyGenerationId": "gen-123"}

        result = client.generate_policy(
            policy_engine_id="pe-123",
            name="test-gen",
            resource={"arn": "arn:aws:test"},
            content={"rawText": "allow refunds"},
        )

        assert result["status"] == "GENERATED"
        client.cp_client.start_policy_generation.assert_called_once()

    def test_generate_policy_with_fetch_assets(self):
        client = self._make_client()
        client.cp_client.start_policy_generation.return_value = {"policyGenerationId": "gen-123"}
        client.cp_client.get_policy_generation.return_value = {"status": "GENERATED", "policyGenerationId": "gen-123"}
        client.cp_client.list_policy_generation_assets.return_value = {
            "policyGenerationAssets": [{"assetId": "asset-1"}]
        }

        result = client.generate_policy(
            policy_engine_id="pe-123",
            name="test-gen",
            resource={"arn": "arn:aws:test"},
            content={"rawText": "allow refunds"},
            fetch_assets=True,
        )

        assert result["generatedPolicies"] == [{"assetId": "asset-1"}]
        client.cp_client.list_policy_generation_assets.assert_called_once()

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch("bedrock_agentcore._utils.polling.time.time", side_effect=[0, 0, 1, 1])
    def test_generate_policy_polls_through_generating(self, _mock_time, _mock_sleep):
        client = self._make_client()
        client.cp_client.start_policy_generation.return_value = {"policyGenerationId": "gen-123"}
        client.cp_client.get_policy_generation.side_effect = [
            {"status": "GENERATING"},
            {"status": "GENERATED", "policyGenerationId": "gen-123"},
        ]

        result = client.generate_policy(
            policy_engine_id="pe-123",
            name="test-gen",
            resource={"arn": "arn:aws:test"},
            content={"rawText": "allow refunds"},
        )

        assert result["status"] == "GENERATED"
        assert client.cp_client.get_policy_generation.call_count == 2

    def test_generate_policy_failed(self):
        client = self._make_client()
        client.cp_client.start_policy_generation.return_value = {"policyGenerationId": "gen-123"}
        client.cp_client.get_policy_generation.return_value = {
            "status": "GENERATE_FAILED",
            "statusReasons": ["Invalid input"],
        }

        with pytest.raises(RuntimeError, match="GENERATE_FAILED"):
            client.generate_policy(
                policy_engine_id="pe-123",
                name="test-gen",
                resource={"arn": "arn:aws:test"},
                content={"rawText": "bad input"},
            )

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch("bedrock_agentcore._utils.polling.time.time", side_effect=[0, 0, 0, 301])
    def test_generate_policy_timeout(self, _mock_time, _mock_sleep):
        client = self._make_client()
        client.cp_client.start_policy_generation.return_value = {"policyGenerationId": "gen-123"}
        client.cp_client.get_policy_generation.return_value = {"status": "GENERATING"}

        with pytest.raises(TimeoutError):
            client.generate_policy(
                policy_engine_id="pe-123",
                name="test-gen",
                resource={"arn": "arn:aws:test"},
                content={"rawText": "allow refunds"},
            )


class TestCreatePolicyFromGenerationAsset:
    """Tests for create_policy_from_generation_asset."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = PolicyEngineClient(boto3_session=mock_session)
        client.cp_client = Mock()
        return client

    def test_creates_policy_with_generation_definition(self):
        client = self._make_client()
        client.cp_client.create_policy.return_value = {"policyId": "pol-123", "policyArn": "arn:test"}

        result = client.create_policy_from_generation_asset(
            policy_engine_id="pe-123",
            name="generated-policy",
            policy_generation_id="gen-456",
            policy_generation_asset_id="asset-789",
        )

        assert result["policyId"] == "pol-123"
        call_kwargs = client.cp_client.create_policy.call_args[1]
        assert call_kwargs["definition"] == {
            "policyGeneration": {
                "policyGenerationId": "gen-456",
                "policyGenerationAssetId": "asset-789",
            }
        }

    def test_passes_optional_params(self):
        client = self._make_client()
        client.cp_client.create_policy.return_value = {"policyId": "pol-123"}

        client.create_policy_from_generation_asset(
            policy_engine_id="pe-123",
            name="generated-policy",
            policy_generation_id="gen-456",
            policy_generation_asset_id="asset-789",
            description="From NL generation",
            validation_mode="IGNORE_ALL_FINDINGS",
            client_token="token-abc",
        )

        call_kwargs = client.cp_client.create_policy.call_args[1]
        assert call_kwargs["description"] == "From NL generation"
        assert call_kwargs["validationMode"] == "IGNORE_ALL_FINDINGS"
        assert call_kwargs["clientToken"] == "token-abc"


class TestWaitForActive:
    """Tests for *_and_wait methods using shared wait_until."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = PolicyEngineClient(boto3_session=mock_session)
        client.cp_client = Mock()
        return client

    def test_create_engine_and_wait_immediate(self):
        client = self._make_client()
        client.cp_client.create_policy_engine.return_value = {"policyEngineId": "pe-123"}
        client.cp_client.get_policy_engine.return_value = {"status": "ACTIVE", "policyEngineId": "pe-123"}

        result = client.create_policy_engine_and_wait(name="test")

        assert result["status"] == "ACTIVE"

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch("bedrock_agentcore._utils.polling.time.time", side_effect=[0, 0, 1, 1])
    def test_create_engine_and_wait_polls(self, _mock_time, _mock_sleep):
        client = self._make_client()
        client.cp_client.create_policy_engine.return_value = {"policyEngineId": "pe-123"}
        client.cp_client.get_policy_engine.side_effect = [
            {"status": "CREATING"},
            {"status": "ACTIVE", "policyEngineId": "pe-123"},
        ]

        result = client.create_policy_engine_and_wait(name="test")

        assert result["status"] == "ACTIVE"
        assert client.cp_client.get_policy_engine.call_count == 2

    def test_create_engine_and_wait_failed_status(self):
        client = self._make_client()
        client.cp_client.create_policy_engine.return_value = {"policyEngineId": "pe-123"}
        client.cp_client.get_policy_engine.return_value = {
            "status": "CREATE_FAILED",
            "statusReasons": ["something broke"],
        }

        with pytest.raises(RuntimeError, match="CREATE_FAILED"):
            client.create_policy_engine_and_wait(name="test")

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch("bedrock_agentcore._utils.polling.time.time", side_effect=[0, 0, 0, 301])
    def test_create_engine_and_wait_timeout(self, _mock_time, _mock_sleep):
        client = self._make_client()
        client.cp_client.create_policy_engine.return_value = {"policyEngineId": "pe-123"}
        client.cp_client.get_policy_engine.return_value = {"status": "CREATING"}

        with pytest.raises(TimeoutError):
            client.create_policy_engine_and_wait(name="test")

    def test_create_policy_and_wait_immediate(self):
        client = self._make_client()
        client.cp_client.create_policy.return_value = {
            "policyEngineId": "pe-123",
            "policyId": "pol-123",
        }
        client.cp_client.get_policy.return_value = {"status": "ACTIVE", "policyId": "pol-123"}

        result = client.create_policy_and_wait(
            policyEngineId="pe-123",
            name="test",
            definition={"cedar": {"statement": "permit(principal, action, resource);"}},
        )

        assert result["status"] == "ACTIVE"

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch("bedrock_agentcore._utils.polling.time.time", side_effect=[0, 0, 1, 1])
    def test_create_policy_and_wait_polls(self, _mock_time, _mock_sleep):
        client = self._make_client()
        client.cp_client.create_policy.return_value = {
            "policyEngineId": "pe-123",
            "policyId": "pol-123",
        }
        client.cp_client.get_policy.side_effect = [
            {"status": "CREATING"},
            {"status": "ACTIVE", "policyId": "pol-123"},
        ]

        result = client.create_policy_and_wait(
            policyEngineId="pe-123",
            name="test",
            definition={"cedar": {"statement": "permit(principal, action, resource);"}},
        )

        assert result["status"] == "ACTIVE"
        assert client.cp_client.get_policy.call_count == 2

    def test_create_policy_and_wait_failed_status(self):
        client = self._make_client()
        client.cp_client.create_policy.return_value = {
            "policyEngineId": "pe-123",
            "policyId": "pol-123",
        }
        client.cp_client.get_policy.return_value = {
            "status": "CREATE_FAILED",
            "statusReasons": ["something broke"],
        }

        with pytest.raises(RuntimeError, match="CREATE_FAILED"):
            client.create_policy_and_wait(
                policyEngineId="pe-123",
                name="test",
                definition={"cedar": {"statement": "permit(principal, action, resource);"}},
            )


class TestCreateOrGet:
    """Tests for create_or_get_policy_engine and create_or_get_policy."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = PolicyEngineClient(boto3_session=mock_session)
        client.cp_client = Mock()
        return client

    def _conflict_error(self, operation="CreatePolicyEngine"):
        return ClientError(
            {"Error": {"Code": "ConflictException", "Message": "already exists"}},
            operation,
        )

    def test_create_or_get_engine_creates_new(self):
        client = self._make_client()
        client.cp_client.create_policy_engine.return_value = {"policyEngineId": "pe-123"}
        client.cp_client.get_policy_engine.return_value = {"status": "ACTIVE", "policyEngineId": "pe-123"}

        result = client.create_or_get_policy_engine(name="test-engine")

        assert result["status"] == "ACTIVE"
        client.cp_client.create_policy_engine.assert_called_once()

    def test_create_or_get_engine_finds_existing(self):
        client = self._make_client()
        client.cp_client.create_policy_engine.side_effect = self._conflict_error()
        client.cp_client.list_policy_engines.return_value = {
            "policyEngines": [{"name": "test-engine", "policyEngineId": "pe-existing"}],
        }
        client.cp_client.get_policy_engine.return_value = {"status": "ACTIVE", "policyEngineId": "pe-existing"}

        result = client.create_or_get_policy_engine(name="test-engine")

        assert result["policyEngineId"] == "pe-existing"

    def test_create_or_get_engine_not_found_reraises(self):
        client = self._make_client()
        client.cp_client.create_policy_engine.side_effect = self._conflict_error()
        client.cp_client.list_policy_engines.return_value = {"policyEngines": []}

        with pytest.raises(ClientError):
            client.create_or_get_policy_engine(name="ghost-engine")

    def test_create_or_get_policy_creates_new(self):
        client = self._make_client()
        client.cp_client.create_policy.return_value = {"policyId": "pol-123"}
        client.cp_client.get_policy.return_value = {"status": "ACTIVE", "policyId": "pol-123"}

        result = client.create_or_get_policy(
            policy_engine_id="pe-123",
            name="test-policy",
            definition={"cedar": {"statement": "permit(principal, action, resource);"}},
        )

        assert result["status"] == "ACTIVE"
        client.cp_client.create_policy.assert_called_once()

    def test_create_or_get_policy_finds_existing(self):
        client = self._make_client()
        client.cp_client.create_policy.side_effect = self._conflict_error("CreatePolicy")
        client.cp_client.list_policies.return_value = {
            "policies": [{"name": "test-policy", "policyId": "pol-existing"}],
        }
        client.cp_client.get_policy.return_value = {"status": "ACTIVE", "policyId": "pol-existing"}

        result = client.create_or_get_policy(
            policy_engine_id="pe-123",
            name="test-policy",
            definition={"cedar": {"statement": "permit(principal, action, resource);"}},
        )

        assert result["policyId"] == "pol-existing"

    def test_non_conflict_error_reraises(self):
        client = self._make_client()
        client.cp_client.create_policy_engine.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "bad input"}},
            "CreatePolicyEngine",
        )

        with pytest.raises(ClientError, match="ValidationException"):
            client.create_or_get_policy_engine(name="bad-engine")
