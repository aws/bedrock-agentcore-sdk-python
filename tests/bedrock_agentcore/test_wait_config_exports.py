"""Public imports for the configuration accepted by polling clients."""

from importlib import import_module

import pytest

from bedrock_agentcore._utils.config import WaitConfig


@pytest.mark.parametrize("package", ["runtime", "evaluation", "gateway", "knowledge_base", "policy"])
def test_wait_config_public_export(package):
    module = import_module(f"bedrock_agentcore.{package}")
    assert "WaitConfig" in module.__all__
    assert module.WaitConfig is WaitConfig
    config = module.WaitConfig(max_wait=900, poll_interval=5)
    assert config.max_wait == 900
    assert config.poll_interval == 5
