"""Tests for build_user_agent_suffix."""

from bedrock_agentcore._utils import user_agent
from bedrock_agentcore._utils.user_agent import build_user_agent_suffix


def _base() -> str:
    return f"bedrock-agentcore/{user_agent.SDK_VERSION}"


def test_no_arguments_returns_base():
    assert build_user_agent_suffix() == _base()


def test_integration_source_only():
    assert build_user_agent_suffix("strands") == f"{_base()} (integration_source=strands)"


def test_integration_source_and_feature():
    assert build_user_agent_suffix("strands", feature="payments") == (
        f"{_base()} (integration_source=strands; feature=payments)"
    )


def test_feature_only():
    assert build_user_agent_suffix(feature="payments") == f"{_base()} (feature=payments)"


def test_tokens_are_sanitized_and_lowercased():
    # Injection characters and spaces are stripped; value is lowercased.
    assert build_user_agent_suffix("Ra w-SDK) evil", feature="Pay;ments") == (
        f"{_base()} (integration_source=raw-sdkevil; feature=payments)"
    )
