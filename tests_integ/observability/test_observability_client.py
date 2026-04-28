"""Integration tests for ObservabilityClient.

Requires:
    OBSERVABILITY_TEST_MEMORY_ID: ID of a persistent memory resource in the test account.
    OBSERVABILITY_TEST_MEMORY_ARN: Full ARN of that memory resource.
"""

import os

import pytest

from bedrock_agentcore.observability.client import ObservabilityClient


@pytest.mark.integration
class TestObservabilityDelivery:
    """Tests enable/disable/status lifecycle using a real memory resource."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.memory_id = os.environ.get("OBSERVABILITY_TEST_MEMORY_ID")
        cls.memory_arn = os.environ.get("OBSERVABILITY_TEST_MEMORY_ARN")
        if not cls.memory_id or not cls.memory_arn:
            pytest.fail("OBSERVABILITY_TEST_MEMORY_ID and OBSERVABILITY_TEST_MEMORY_ARN must be set")
        cls.client = ObservabilityClient(region_name=cls.region)

    @classmethod
    def teardown_class(cls):
        try:
            cls.client.disable_observability_for_resource(
                resource_id=cls.memory_id,
                delete_log_group=True,
            )
        except Exception as e:
            print(f"Teardown: {e}")

    @pytest.mark.order(1)
    def test_enable_observability_logs_and_traces(self):
        result = self.client.enable_observability_for_resource(
            resource_arn=self.memory_arn,
            resource_id=self.memory_id,
            resource_type="memory",
            enable_logs=True,
            enable_traces=True,
        )
        assert result["status"] == "success"
        assert result["logs_enabled"] is True
        assert result["traces_enabled"] is True
        assert "APPLICATION_LOGS" in result["log_group"]

    @pytest.mark.order(2)
    def test_get_status_shows_configured(self):
        status = self.client.get_observability_status(resource_id=self.memory_id)
        assert status["logs"]["configured"] is True
        assert status["traces"]["configured"] is True

    @pytest.mark.order(3)
    def test_enable_is_idempotent(self):
        result = self.client.enable_observability_for_resource(
            resource_arn=self.memory_arn,
            resource_id=self.memory_id,
            resource_type="memory",
        )
        assert result["status"] == "success"

    @pytest.mark.order(4)
    def test_disable_observability(self):
        result = self.client.disable_observability_for_resource(
            resource_id=self.memory_id,
            delete_log_group=True,
        )
        assert result["status"] == "success"
        assert len(result["deleted"]) > 0

    @pytest.mark.order(5)
    def test_get_status_shows_not_configured_after_disable(self):
        status = self.client.get_observability_status(resource_id=self.memory_id)
        assert status["logs"]["configured"] is False
        assert status["traces"]["configured"] is False

    @pytest.mark.order(6)
    def test_disable_is_idempotent(self):
        result = self.client.disable_observability_for_resource(
            resource_id=self.memory_id,
        )
        assert result["status"] == "success"


@pytest.mark.integration
class TestObservabilityArnParsing:
    """Tests that ARN parsing works with real API calls."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.memory_id = os.environ.get("OBSERVABILITY_TEST_MEMORY_ID")
        cls.memory_arn = os.environ.get("OBSERVABILITY_TEST_MEMORY_ARN")
        if not cls.memory_id or not cls.memory_arn:
            pytest.fail("OBSERVABILITY_TEST_MEMORY_ID and OBSERVABILITY_TEST_MEMORY_ARN must be set")
        cls.client = ObservabilityClient(region_name=cls.region)

    @classmethod
    def teardown_class(cls):
        try:
            cls.client.disable_observability_for_resource(
                resource_id=cls.memory_id,
                delete_log_group=True,
            )
        except Exception as e:
            print(f"Teardown: {e}")

    @pytest.mark.order(7)
    def test_enable_with_arn_only(self):
        """Test that resource_type and resource_id are inferred from ARN."""
        result = self.client.enable_observability_for_resource(
            resource_arn=self.memory_arn,
        )
        assert result["status"] == "success"
        assert result["resource_type"] == "memory"
        assert result["resource_id"] == self.memory_id


@pytest.mark.integration
class TestTransactionSearch:
    """Tests enable_transaction_search with real X-Ray and CloudWatch APIs."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.memory_id = os.environ.get("OBSERVABILITY_TEST_MEMORY_ID")
        if not cls.memory_id:
            pytest.fail("OBSERVABILITY_TEST_MEMORY_ID must be set")
        cls.client = ObservabilityClient(region_name=cls.region)

    @pytest.mark.order(8)
    def test_enable_transaction_search(self):
        result = self.client.enable_transaction_search()
        assert result is True

    @pytest.mark.order(9)
    def test_enable_transaction_search_is_idempotent(self):
        result = self.client.enable_transaction_search()
        assert result is True
