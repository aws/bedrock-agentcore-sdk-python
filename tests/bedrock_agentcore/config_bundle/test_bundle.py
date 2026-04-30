"""Tests for ConfigBundleRef."""

import pytest

from bedrock_agentcore.config_bundle.bundle import ConfigBundleRef


class TestConfigBundleRef:
    def test_bundle_id_extracted_from_arn(self):
        ref = ConfigBundleRef(
            bundle_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:configuration-bundle/my-agent-ab12cd34ef",
            bundle_version="2",
        )
        assert ref.bundle_id == "my-agent-ab12cd34ef"

    def test_bundle_id_extracted_from_short_arn(self):
        ref = ConfigBundleRef(
            bundle_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:bundle/my-agent",
            bundle_version="1",
        )
        assert ref.bundle_id == "my-agent"

    def test_bundle_version_preserved(self):
        ref = ConfigBundleRef(
            bundle_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:bundle/my-agent",
            bundle_version="42",
        )
        assert ref.bundle_version == "42"

    def test_frozen(self):
        ref = ConfigBundleRef(
            bundle_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:bundle/my-agent",
            bundle_version="1",
        )
        with pytest.raises((AttributeError, TypeError)):
            ref.bundle_arn = "other"

    def test_empty_arn_raises(self):
        with pytest.raises(ValueError, match="bundle_arn must not be empty"):
            ConfigBundleRef(bundle_arn="", bundle_version="1")

    def test_empty_version_raises(self):
        with pytest.raises(ValueError, match="bundle_version must not be empty"):
            ConfigBundleRef(bundle_arn="arn:aws:...:bundle/my-agent", bundle_version="")

    def test_arn_without_slash_raises(self):
        with pytest.raises(ValueError, match="does not contain a valid bundle ID segment"):
            ConfigBundleRef(bundle_arn="not-an-arn", bundle_version="1")

    def test_arn_with_trailing_slash_raises(self):
        with pytest.raises(ValueError, match="does not contain a valid bundle ID segment"):
            ConfigBundleRef(bundle_arn="arn:aws:...:bundle/", bundle_version="1")

    def test_equality(self):
        ref1 = ConfigBundleRef(bundle_arn="arn:aws:...:bundle/my-agent", bundle_version="1")
        ref2 = ConfigBundleRef(bundle_arn="arn:aws:...:bundle/my-agent", bundle_version="1")
        assert ref1 == ref2

    def test_inequality_different_version(self):
        ref1 = ConfigBundleRef(bundle_arn="arn:aws:...:bundle/my-agent", bundle_version="1")
        ref2 = ConfigBundleRef(bundle_arn="arn:aws:...:bundle/my-agent", bundle_version="2")
        assert ref1 != ref2
