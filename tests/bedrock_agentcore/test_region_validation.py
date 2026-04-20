"""Tests for region validation and SSRF prevention.

Covers the fix for V2177374595 — unvalidated region parameter in endpoint
construction allows request redirection to non-AWS hosts.
"""

import pytest

from bedrock_agentcore._utils.endpoints import (
    InvalidRegionError,
    _validate_endpoint_url,
    get_control_plane_endpoint,
    get_data_plane_endpoint,
    validate_region,
)

# ---------------------------------------------------------------------------
# Layer 1: validate_region() — regex input validation
# ---------------------------------------------------------------------------


class TestValidateRegion:
    """Tests for the validate_region() function."""

    @pytest.mark.parametrize(
        "region",
        [
            "us-east-1",
            "us-west-2",
            "eu-west-1",
            "eu-central-1",
            "ap-southeast-1",
            "ap-northeast-1",
            "sa-east-1",
            "ca-central-1",
            "me-south-1",
            "af-south-1",
            # GovCloud
            "us-gov-west-1",
            "us-gov-east-1",
        ],
    )
    def test_valid_regions_accepted(self, region):
        assert validate_region(region) == region

    @pytest.mark.parametrize(
        "region",
        [
            # SSRF payloads (CVE-2026-22611 class)
            "x@attacker.com:443/#",
            "evil.com/#",
            "us-east-1.attacker.com",
            "attacker.com:443",
            # URL control characters
            "us-east-1/../../",
            "us-east-1?foo=bar",
            "us-east-1#fragment",
            # Newline injection (the \Z vs $ distinction)
            "us-east-1\n",
            "us-east-1\r\n",
            # Whitespace
            " us-east-1",
            "us-east-1 ",
            # Uppercase
            "US-EAST-1",
            "Us-East-1",
            # Empty / None-like
            "",
            # Very long string
            "a" * 200,
            # Unicode homoglyphs
            "us-e\u0430st-1",  # Cyrillic 'a'
            # Command injection attempts
            "us-east-1$(whoami)",
            "us-east-1`id`",
            # Null byte
            "us-east-1\x00",
        ],
    )
    def test_malicious_regions_rejected(self, region):
        with pytest.raises(InvalidRegionError):
            validate_region(region)

    def test_non_string_rejected(self):
        with pytest.raises(InvalidRegionError):
            validate_region(123)  # type: ignore[arg-type]

    def test_none_rejected(self):
        with pytest.raises(InvalidRegionError):
            validate_region(None)  # type: ignore[arg-type]

    def test_error_is_valueerror_subclass(self):
        """Ensure existing `except ValueError` handlers still catch it."""
        with pytest.raises(ValueError):
            validate_region("evil.com")


# ---------------------------------------------------------------------------
# Layer 2: _validate_endpoint_url() — output hostname validation
# ---------------------------------------------------------------------------


class TestValidateEndpointUrl:
    """Tests for defense-in-depth URL hostname check."""

    def test_valid_aws_url(self):
        url = "https://bedrock-agentcore.us-east-1.amazonaws.com"
        assert _validate_endpoint_url(url) == url

    def test_valid_china_url(self):
        url = "https://bedrock-agentcore.cn-north-1.amazonaws.com.cn"
        assert _validate_endpoint_url(url) == url

    def test_valid_api_aws_url(self):
        url = "https://bedrock-agentcore.us-east-1.api.aws"
        assert _validate_endpoint_url(url) == url

    def test_attacker_url_rejected(self):
        url = "https://bedrock-agentcore.x@attacker.com:443/#.amazonaws.com"
        with pytest.raises(InvalidRegionError, match="non-AWS host"):
            _validate_endpoint_url(url)

    def test_non_aws_host_rejected(self):
        url = "https://not-amazonaws.com/something"
        with pytest.raises(InvalidRegionError, match="non-AWS host"):
            _validate_endpoint_url(url)


# ---------------------------------------------------------------------------
# Endpoint functions (both layers combined)
# ---------------------------------------------------------------------------


class TestEndpointFunctions:
    """Tests for get_data_plane_endpoint / get_control_plane_endpoint."""

    def test_valid_data_plane_endpoint(self):
        url = get_data_plane_endpoint("us-west-2")
        assert url == "https://bedrock-agentcore.us-west-2.amazonaws.com"

    def test_valid_control_plane_endpoint(self):
        url = get_control_plane_endpoint("eu-west-1")
        assert url == "https://bedrock-agentcore-control.eu-west-1.amazonaws.com"

    def test_malicious_region_rejected_dp(self):
        with pytest.raises(InvalidRegionError):
            get_data_plane_endpoint("x@attacker.com:443/#")

    def test_malicious_region_rejected_cp(self):
        with pytest.raises(InvalidRegionError):
            get_control_plane_endpoint("evil.com/#")

    def test_env_override_skips_region_validation(self):
        """Environment overrides skip region validation but are still URL-validated."""
        import bedrock_agentcore._utils.endpoints as ep

        original_override = ep.DP_ENDPOINT_OVERRIDE
        try:
            # Valid AWS override works
            ep.DP_ENDPOINT_OVERRIDE = "https://bedrock-agentcore.us-east-1.amazonaws.com"
            result = ep.get_data_plane_endpoint("not-a-region")
            assert result == "https://bedrock-agentcore.us-east-1.amazonaws.com"

            # Malicious override is rejected
            ep.DP_ENDPOINT_OVERRIDE = "https://attacker.com"
            with pytest.raises(InvalidRegionError, match="non-AWS host"):
                ep.get_data_plane_endpoint("not-a-region")
        finally:
            ep.DP_ENDPOINT_OVERRIDE = original_override

    def test_govcloud_regions(self):
        url = get_data_plane_endpoint("us-gov-west-1")
        assert "us-gov-west-1" in url


# ---------------------------------------------------------------------------
# build_runtime_url (ARN extraction path)
# ---------------------------------------------------------------------------


class TestBuildRuntimeUrlValidation:
    """Tests for SSRF prevention in build_runtime_url."""

    def test_valid_arn_builds_url(self):
        from bedrock_agentcore.runtime.a2a import build_runtime_url

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-agent"
        url = build_runtime_url(arn)
        assert "bedrock-agentcore.us-east-1.amazonaws.com" in url

    def test_malicious_arn_region_rejected(self):
        from bedrock_agentcore.runtime.a2a import build_runtime_url

        malicious_arn = "arn:aws:bedrock-agentcore:evil.com/#:123456789012:runtime/agent1"
        with pytest.raises(InvalidRegionError):
            build_runtime_url(malicious_arn)

    def test_explicit_malicious_region_override_rejected(self):
        from bedrock_agentcore.runtime.a2a import build_runtime_url

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-agent"
        with pytest.raises(InvalidRegionError):
            build_runtime_url(arn, region="x@attacker.com:443/#")

    def test_valid_region_override(self):
        from bedrock_agentcore.runtime.a2a import build_runtime_url

        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-agent"
        url = build_runtime_url(arn, region="eu-west-1")
        assert "bedrock-agentcore.eu-west-1.amazonaws.com" in url


# ---------------------------------------------------------------------------
# Client constructor fail-fast validation
# ---------------------------------------------------------------------------


class TestClientConstructorValidation:
    """Tests that all client constructors reject malicious regions early."""

    def test_agent_core_runtime_client_rejects_bad_region(self):
        from botocore.exceptions import InvalidRegionError as BotocoreInvalidRegionError

        with pytest.raises(BotocoreInvalidRegionError):
            from bedrock_agentcore.runtime.agent_core_runtime_client import AgentCoreRuntimeClient

            AgentCoreRuntimeClient("x@attacker.com:443/#")

    def test_browser_client_rejects_bad_region(self):
        """BrowserClient still uses validate_region() for WebSocket URL construction."""
        with pytest.raises(InvalidRegionError):
            from bedrock_agentcore.tools.browser_client import BrowserClient

            BrowserClient("evil.com/#")

    def test_code_interpreter_rejects_bad_region(self):
        """CodeInterpreter delegates to boto3 which has its own region validation."""
        from botocore.exceptions import InvalidRegionError as BotocoreInvalidRegionError

        with pytest.raises(BotocoreInvalidRegionError):
            from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

            CodeInterpreter("evil.com/#")

    def test_identity_client_rejects_bad_region(self):
        """IdentityClient delegates to boto3 which has its own region validation."""
        from botocore.exceptions import InvalidRegionError as BotocoreInvalidRegionError

        with pytest.raises(BotocoreInvalidRegionError):
            from bedrock_agentcore.services.identity import IdentityClient

            IdentityClient("x@attacker.com:443/#")

    def test_resource_policy_client_rejects_bad_region(self):
        """ResourcePolicyClient delegates to boto3 which has its own region validation."""
        from botocore.exceptions import InvalidRegionError as BotocoreInvalidRegionError

        with pytest.raises(BotocoreInvalidRegionError):
            from bedrock_agentcore.services.resource_policy import ResourcePolicyClient

            ResourcePolicyClient("evil.com/#")

    def test_memory_controlplane_rejects_bad_region(self):
        """MemoryControlPlaneClient delegates to boto3 which has its own region validation."""
        from botocore.exceptions import InvalidRegionError as BotocoreInvalidRegionError

        with pytest.raises(BotocoreInvalidRegionError):
            from bedrock_agentcore.memory.controlplane import MemoryControlPlaneClient

            MemoryControlPlaneClient(region_name="x@attacker.com:443/#")
