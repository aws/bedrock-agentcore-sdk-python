"""Tests for AgentCorePaymentsPluginConfig."""

import pytest

from bedrock_agentcore.payments.integrations.config import AgentCorePaymentsPluginConfig


class TestAgentCorePaymentsPluginConfigValidation:
    """Test validation of AgentCorePaymentsPluginConfig required fields."""

    def test_empty_payment_manager_arn_raises_error(self):
        """Test that empty payment_manager_arn raises ValueError."""
        with pytest.raises(ValueError, match="payment_manager_arn is required"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="",
                user_id="test-user",
                payment_instrument_id="test-instrument",
                payment_session_id="test-session",
            )

    def test_invalid_arn_format_raises_error(self):
        """Test that invalid ARN format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ARN format"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="not-an-arn",
                user_id="test-user",
                payment_instrument_id="test-instrument",
                payment_session_id="test-session",
            )

    def test_empty_user_id_raises_error(self):
        """Test that empty user_id raises ValueError."""
        with pytest.raises(ValueError, match="user_id is required"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="",
                payment_instrument_id="test-instrument",
                payment_session_id="test-session",
            )

    def test_empty_payment_instrument_id_raises_error(self):
        """Test that empty payment_instrument_id raises ValueError via update method."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
        )
        with pytest.raises(ValueError, match="payment_instrument_id cannot be empty"):
            config.update_payment_instrument_id("")

    def test_empty_payment_session_id_raises_error(self):
        """Test that empty payment_session_id raises ValueError via update method."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
        )
        with pytest.raises(ValueError, match="payment_session_id cannot be empty"):
            config.update_payment_session_id("")

    def test_whitespace_only_payment_manager_arn(self):
        """Test that whitespace-only payment_manager_arn raises Invalid ARN format error."""
        with pytest.raises(ValueError, match="Invalid ARN format"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="   ",
                user_id="test-user",
                payment_instrument_id="test-instrument",
                payment_session_id="test-session",
            )

    def test_valid_config(self):
        """Test that valid configuration is accepted."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        assert config.payment_manager_arn == "arn:aws:payment:us-east-1:123456789012:payment-manager/test"
        assert config.user_id == "test-user"
        assert config.payment_instrument_id == "test-instrument"
        assert config.payment_session_id == "test-session"

    def test_config_with_network_preferences(self):
        """Test configuration with network preferences."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            network_preferences_config=["eip155:1", "solana:mainnet"],
        )
        assert config.network_preferences_config == ["eip155:1", "solana:mainnet"]

    def test_config_without_region(self):
        """Test configuration without explicit region."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        assert config.region is None


class TestAgentCorePaymentsPluginConfigAutoPayment:
    """Test auto_payment configuration field."""

    def test_auto_payment_defaults_to_true(self):
        """Test that auto_payment defaults to True."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        assert config.auto_payment is True

    def test_auto_payment_can_be_set_to_false(self):
        """Test that auto_payment can be explicitly set to False."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            auto_payment=False,
        )
        assert config.auto_payment is False

    def test_auto_payment_can_be_set_to_true(self):
        """Test that auto_payment can be explicitly set to True."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            auto_payment=True,
        )
        assert config.auto_payment is True

    def test_auto_payment_rejects_non_boolean_string(self):
        """Test that auto_payment validation rejects string values."""
        with pytest.raises(ValueError, match="auto_payment must be a boolean"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="test-user",
                payment_instrument_id="test-instrument",
                payment_session_id="test-session",
                auto_payment="true",  # type: ignore
            )

    def test_auto_payment_rejects_non_boolean_int(self):
        """Test that auto_payment validation rejects integer values."""
        with pytest.raises(ValueError, match="auto_payment must be a boolean"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="test-user",
                payment_instrument_id="test-instrument",
                payment_session_id="test-session",
                auto_payment=1,  # type: ignore
            )

    def test_auto_payment_rejects_non_boolean_none(self):
        """Test that auto_payment validation rejects None values."""
        with pytest.raises(ValueError, match="auto_payment must be a boolean"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="test-user",
                payment_instrument_id="test-instrument",
                payment_session_id="test-session",
                auto_payment=None,  # type: ignore
            )

    def test_auto_payment_rejects_non_boolean_dict(self):
        """Test that auto_payment validation rejects dict values."""
        with pytest.raises(ValueError, match="auto_payment must be a boolean"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="test-user",
                payment_instrument_id="test-instrument",
                payment_session_id="test-session",
                auto_payment={},  # type: ignore
            )

    def test_auto_payment_rejects_non_boolean_list(self):
        """Test that auto_payment validation rejects list values."""
        with pytest.raises(ValueError, match="auto_payment must be a boolean"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="test-user",
                payment_instrument_id="test-instrument",
                payment_session_id="test-session",
                auto_payment=[],  # type: ignore
            )

    def test_auto_payment_validation_error_message(self):
        """Test that validation error message is descriptive."""
        with pytest.raises(ValueError) as exc_info:
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="test-user",
                payment_instrument_id="test-instrument",
                payment_session_id="test-session",
                auto_payment="invalid",  # type: ignore
            )
        assert "auto_payment must be a boolean" in str(exc_info.value)
        assert "str" in str(exc_info.value)

    def test_backward_compatibility_without_auto_payment(self):
        """Test backward compatibility - config works without auto_payment field."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        # Should not raise any exception
        assert config is not None
        assert config.auto_payment is True

    def test_config_with_all_fields(self):
        """Test configuration with all fields including auto_payment."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            region="us-east-1",
            auto_payment=False,
        )
        assert config.payment_manager_arn == "arn:aws:payment:us-east-1:123456789012:payment-manager/test"
        assert config.user_id == "test-user"
        assert config.payment_instrument_id == "test-instrument"
        assert config.payment_session_id == "test-session"
        assert config.region == "us-east-1"
        assert config.auto_payment is False

    def test_auto_payment_field_is_accessible(self):
        """Test that auto_payment field is accessible after initialization."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            auto_payment=False,
        )
        # Should be able to access the field
        assert hasattr(config, "auto_payment")
        assert config.auto_payment is False


class TestAgentCorePaymentsPluginConfigOptionalFields:
    """Test optional payment_instrument_id and payment_session_id fields."""

    def test_config_without_instrument_and_session(self):
        """Test config creation without payment_instrument_id and payment_session_id."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
        )
        assert config.payment_instrument_id is None
        assert config.payment_session_id is None

    def test_config_with_instrument_and_session(self):
        """Test config creation with explicit payment_instrument_id and payment_session_id."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="instrument-123",
            payment_session_id="session-456",
        )
        assert config.payment_instrument_id == "instrument-123"
        assert config.payment_session_id == "session-456"

    def test_update_payment_instrument_id(self):
        """Test updating payment_instrument_id after creation."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
        )
        assert config.payment_instrument_id is None
        config.update_payment_instrument_id("new-instrument")
        assert config.payment_instrument_id == "new-instrument"

    def test_update_payment_session_id(self):
        """Test updating payment_session_id after creation."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
        )
        assert config.payment_session_id is None
        config.update_payment_session_id("new-session")
        assert config.payment_session_id == "new-session"

    def test_update_payment_instrument_id_empty_raises(self):
        """Test that updating with empty string raises ValueError."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
        )
        with pytest.raises(ValueError, match="payment_instrument_id cannot be empty"):
            config.update_payment_instrument_id("")

    def test_update_payment_session_id_empty_raises(self):
        """Test that updating with empty string raises ValueError."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
        )
        with pytest.raises(ValueError, match="payment_session_id cannot be empty"):
            config.update_payment_session_id("")


class TestAgentCorePaymentsPluginConfigMaxInterruptRetries:
    """Test max_interrupt_retries configuration field."""

    def test_max_interrupt_retries_defaults_to_5(self):
        """Test that max_interrupt_retries defaults to 5."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
        )
        assert config.max_interrupt_retries == 5

    def test_max_interrupt_retries_custom_value(self):
        """Test setting a custom max_interrupt_retries value."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            max_interrupt_retries=10,
        )
        assert config.max_interrupt_retries == 10

    def test_max_interrupt_retries_zero_disables(self):
        """Test setting max_interrupt_retries to 0."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            max_interrupt_retries=0,
        )
        assert config.max_interrupt_retries == 0


class TestAgentCorePaymentsPluginConfigAgentName:
    """Test agent_name configuration field."""

    def test_agent_name_defaults_to_none(self):
        """Test that agent_name defaults to None."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
        )
        assert config.agent_name is None

    def test_agent_name_can_be_set(self):
        """Test that agent_name can be explicitly set."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            agent_name="my-agent",
        )
        assert config.agent_name == "my-agent"

    def test_agent_name_included_in_full_config(self):
        """Test configuration with all fields including agent_name."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            region="us-east-1",
            auto_payment=True,
            max_interrupt_retries=3,
            agent_name="my-payment-agent",
        )
        assert config.agent_name == "my-payment-agent"
        assert config.payment_manager_arn == "arn:aws:payment:us-east-1:123456789012:payment-manager/test"
        assert config.user_id == "test-user"

    def test_backward_compatibility_without_agent_name(self):
        """Test backward compatibility - config works without agent_name field."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        assert config is not None
        assert config.agent_name is None


class TestAgentCorePaymentsPluginConfigBearerAuth:
    """Test bearer_token and token_provider configuration fields."""

    def test_bearer_token_defaults_to_none(self):
        """Test that bearer_token defaults to None."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
        )
        assert config.bearer_token is None

    def test_token_provider_defaults_to_none(self):
        """Test that token_provider defaults to None."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
        )
        assert config.token_provider is None

    def test_bearer_token_can_be_set(self):
        """Test that bearer_token can be explicitly set."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            bearer_token="eyJhbGciOiJSUzI1NiJ9.test",
        )
        assert config.bearer_token == "eyJhbGciOiJSUzI1NiJ9.test"

    def test_token_provider_can_be_set(self):
        """Test that token_provider can be set to a callable."""

        def provider():
            return "fresh-token"

        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            token_provider=provider,
        )
        assert config.token_provider is provider
        assert config.token_provider() == "fresh-token"

    def test_mutual_exclusivity_raises_value_error(self):
        """Test that setting both bearer_token and token_provider raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="test-user",
                bearer_token="token",
                token_provider=lambda: "token",
            )

    def test_bearer_token_with_all_fields(self):
        """Test configuration with all fields including bearer_token."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
            region="us-east-1",
            auto_payment=True,
            max_interrupt_retries=3,
            agent_name="my-agent",
            bearer_token="my-jwt",
        )
        assert config.bearer_token == "my-jwt"
        assert config.agent_name == "my-agent"
        assert config.token_provider is None

    def test_backward_compatibility_without_bearer_fields(self):
        """Test backward compatibility - config works without bearer fields."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_instrument_id="test-instrument",
            payment_session_id="test-session",
        )
        assert config.bearer_token is None
        assert config.token_provider is None


class TestAgentCorePaymentsPluginConfigUserId:
    """Test user_id behavior with SigV4 vs bearer auth."""

    def test_user_id_required_for_sigv4(self):
        """Test that user_id is required when no bearer auth is configured."""
        with pytest.raises(ValueError, match="user_id is required for SigV4"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            )

    def test_user_id_optional_with_bearer_token(self):
        """Test that user_id is optional when bearer_token is set."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            bearer_token="my-jwt",
        )
        assert config.user_id is None
        assert config.bearer_token == "my-jwt"

    def test_user_id_optional_with_token_provider(self):
        """Test that user_id is optional when token_provider is set."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            token_provider=lambda: "fresh",
        )
        assert config.user_id is None

    def test_user_id_can_be_set_with_bearer_token(self):
        """Test that user_id can still be provided with bearer auth."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="explicit-user",
            bearer_token="my-jwt",
        )
        assert config.user_id == "explicit-user"
        assert config.bearer_token == "my-jwt"


class TestAgentCorePaymentsPluginConfigBearerTokenValidation:
    """Test bearer_token and token_provider type validation."""

    def test_bearer_token_non_string_raises_error(self):
        """Test that non-string bearer_token raises ValueError."""
        with pytest.raises(ValueError, match="bearer_token must be a string"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="test-user",
                bearer_token=12345,  # type: ignore
            )

    def test_token_provider_non_callable_raises_error(self):
        """Test that non-callable token_provider raises ValueError."""
        with pytest.raises(ValueError, match="token_provider must be callable"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="test-user",
                token_provider="not-callable",  # type: ignore
            )


class TestAgentCorePaymentsPluginConfigPaymentToolAllowlist:
    """Test payment_tool_allowlist configuration field."""

    def test_payment_tool_allowlist_accepts_valid_list(self):
        """Test that payment_tool_allowlist accepts a valid list of strings."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
            payment_tool_allowlist=["http_request", "api_call", "fetch_data"],
        )
        assert config.payment_tool_allowlist == ["http_request", "api_call", "fetch_data"]

    def test_payment_tool_allowlist_rejects_non_list(self):
        """Test that payment_tool_allowlist rejects non-list values."""
        with pytest.raises(ValueError, match="payment_tool_allowlist must be a list"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="test-user",
                payment_tool_allowlist="http_request",  # type: ignore
            )

    def test_payment_tool_allowlist_rejects_list_with_non_strings(self):
        """Test that payment_tool_allowlist rejects list with non-string entries."""
        with pytest.raises(ValueError, match="All entries in payment_tool_allowlist must be strings"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="test-user",
                payment_tool_allowlist=["http_request", 123, "api_call"],  # type: ignore
            )

    def test_payment_tool_allowlist_defaults_to_none(self):
        """Test that payment_tool_allowlist defaults to None."""
        config = AgentCorePaymentsPluginConfig(
            payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
            user_id="test-user",
        )
        assert config.payment_tool_allowlist is None


class TestAgentCorePaymentsPluginConfigWhitespaceUserId:
    """Test whitespace-only user_id validation."""

    def test_whitespace_only_user_id_raises_value_error(self):
        """Test that whitespace-only user_id raises ValueError."""
        with pytest.raises(ValueError, match="user_id cannot be whitespace-only"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="   ",
            )

    def test_whitespace_tabs_user_id_raises_value_error(self):
        """Test that tabs-only user_id raises ValueError."""
        with pytest.raises(ValueError, match="user_id cannot be whitespace-only"):
            AgentCorePaymentsPluginConfig(
                payment_manager_arn="arn:aws:payment:us-east-1:123456789012:payment-manager/test",
                user_id="\t\t",
            )
