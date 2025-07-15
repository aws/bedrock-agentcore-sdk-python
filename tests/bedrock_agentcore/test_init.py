"""Tests for bedrock_agentcore.__init__ module."""

import pytest


def test_getattr_raises_for_unknown_attribute():
    """Test that __getattr__ raises AttributeError for unknown attributes."""
    import bedrock_agentcore

    with pytest.raises(AttributeError, match="module 'bedrock_agentcore' has no attribute 'UnknownAttribute'"):
        _ = bedrock_agentcore.UnknownAttribute


def test_all_exports():
    """Test that all expected exports are available."""
    import bedrock_agentcore

    # Test direct imports
    assert hasattr(bedrock_agentcore.runtime, "BedrockAgentCoreApp")
    assert hasattr(bedrock_agentcore.runtime, "RequestContext")
    assert hasattr(bedrock_agentcore.runtime, "BedrockAgentCoreContext")

    # Test __all__ contains expected items
    expected_all = [
        "BedrockAgentCoreApp",
        "RequestContext",
        "BedrockAgentCoreContext",
        "PingStatus",
    ]
    assert sorted(bedrock_agentcore.__all__) == sorted(expected_all)
