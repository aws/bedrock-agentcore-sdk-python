"""Tests for baggage parsing utilities."""

from bedrock_agentcore.config_bundle.baggage import _extract_baggage, _parse_config_bundle_baggage
from bedrock_agentcore.config_bundle.bundle import ConfigBundleRef

ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:bundle/my-agent"
ARN2 = "arn:aws:bedrock-agentcore:us-west-2:123456789012:bundle/other-agent"
ARN_KEY = "aws.agentcore.configbundle_arn"
VERSION_KEY = "aws.agentcore.configbundle_version"


class TestExtractBaggage:
    def test_single_baggage_header(self):
        headers = [("baggage", f"{ARN_KEY}={ARN},{VERSION_KEY}=2")]
        result = _extract_baggage(headers)
        assert result == {ARN_KEY: [ARN], VERSION_KEY: ["2"]}

    def test_multiple_baggage_headers_same_name(self):
        headers = [
            ("baggage", f"{ARN_KEY}={ARN},{VERSION_KEY}=2"),
            ("baggage", f"{ARN_KEY}={ARN2},{VERSION_KEY}=5"),
        ]
        result = _extract_baggage(headers)
        assert result[ARN_KEY] == [ARN, ARN2]
        assert result[VERSION_KEY] == ["2", "5"]

    def test_duplicate_key_in_single_header(self):
        headers = [("baggage", f"{ARN_KEY}={ARN},{ARN_KEY}={ARN2}")]
        result = _extract_baggage(headers)
        assert result[ARN_KEY] == [ARN, ARN2]

    def test_ignores_non_baggage_headers(self):
        headers = [
            ("content-type", "application/json"),
            ("baggage", f"{ARN_KEY}={ARN},{VERSION_KEY}=1"),
            ("authorization", "Bearer token"),
        ]
        result = _extract_baggage(headers)
        assert set(result.keys()) == {ARN_KEY, VERSION_KEY}

    def test_case_insensitive_header_name(self):
        headers = [("Baggage", f"{ARN_KEY}={ARN},{VERSION_KEY}=3")]
        result = _extract_baggage(headers)
        assert result[ARN_KEY] == [ARN]

    def test_strips_properties_after_semicolon(self):
        headers = [("baggage", f"{ARN_KEY}={ARN};meta=x,{VERSION_KEY}=1;ttl=60")]
        result = _extract_baggage(headers)
        assert result[ARN_KEY] == [ARN]
        assert result[VERSION_KEY] == ["1"]

    def test_empty_baggage_header(self):
        result = _extract_baggage([])
        assert result == {}

    def test_malformed_entries_skipped(self):
        headers = [("baggage", f"no-equals,{ARN_KEY}={ARN},{VERSION_KEY}=2")]
        result = _extract_baggage(headers)
        assert result[ARN_KEY] == [ARN]
        assert result[VERSION_KEY] == ["2"]

    def test_empty_key_skipped(self):
        # "=value" has no key — must not insert "" into result
        headers = [("baggage", f"=orphan,{ARN_KEY}={ARN}")]
        result = _extract_baggage(headers)
        assert "" not in result
        assert result[ARN_KEY] == [ARN]

    def test_empty_value_skipped(self):
        # "key=" has an empty value — must not append "" to the list
        headers = [("baggage", f"{ARN_KEY}=,{VERSION_KEY}=2")]
        result = _extract_baggage(headers)
        assert ARN_KEY not in result
        assert result[VERSION_KEY] == ["2"]

    def test_extra_whitespace_stripped(self):
        headers = [("baggage", f"  {ARN_KEY} = {ARN} , {VERSION_KEY} = 7 ")]
        result = _extract_baggage(headers)
        assert result[ARN_KEY] == [ARN]
        assert result[VERSION_KEY] == ["7"]


class TestParseConfigBundleBaggage:
    def test_single_bundle(self):
        ref = _parse_config_bundle_baggage({ARN_KEY: [ARN], VERSION_KEY: ["2"]})
        assert ref == ConfigBundleRef(bundle_arn=ARN, bundle_version="2")

    def test_multiple_arns_uses_first(self):
        ref = _parse_config_bundle_baggage({ARN_KEY: [ARN, ARN2], VERSION_KEY: ["2", "5"]})
        assert ref == ConfigBundleRef(bundle_arn=ARN, bundle_version="2")

    def test_empty_baggage(self):
        assert _parse_config_bundle_baggage({}) is None

    def test_missing_version(self):
        assert _parse_config_bundle_baggage({ARN_KEY: [ARN]}) is None

    def test_missing_arn(self):
        assert _parse_config_bundle_baggage({VERSION_KEY: ["2"]}) is None

    def test_unrelated_keys_ignored(self):
        ref = _parse_config_bundle_baggage({ARN_KEY: [ARN], VERSION_KEY: ["2"], "other_key": ["other_value"]})
        assert ref == ConfigBundleRef(bundle_arn=ARN, bundle_version="2")

    def test_returns_single_ref(self):
        ref = _parse_config_bundle_baggage({ARN_KEY: [ARN], VERSION_KEY: ["1"]})
        assert isinstance(ref, ConfigBundleRef)
