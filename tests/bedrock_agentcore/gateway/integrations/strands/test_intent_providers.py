"""Tests for IntentProvider and StrandsIntentProvider."""

from unittest.mock import Mock, patch

import pytest

from bedrock_agentcore.gateway.integrations.strands.plugins.agentcore_tool_search.intent_providers import (
    StrandsIntentProvider,
    IntentProvider,
)


class TestIntentProviderInterface:
    """Test IntentProvider abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that IntentProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IntentProvider()

    def test_subclass_must_implement_derive_intent(self):
        """Test that subclass without derive_intent raises TypeError."""

        class IncompleteProvider(IntentProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_subclass_with_derive_intent_works(self):
        """Test that a proper subclass can be instantiated."""

        class ValidProvider(IntentProvider):
            def derive_intent(self, messages: list[dict], model=None) -> str:
                return "test"

        provider = ValidProvider()
        assert provider.derive_intent([]) == "test"


class TestStrandsIntentProvider:
    """Test StrandsIntentProvider class."""

    def test_init_default_message_window(self):
        """Test default message window is 5."""
        provider = StrandsIntentProvider()
        assert provider._message_window == 5

    def test_init_custom_message_window(self):
        """Test custom message window."""
        provider = StrandsIntentProvider(message_window=3)
        assert provider._message_window == 3

    def test_init_with_explicit_model(self):
        """Test initialization with explicit model."""
        model = Mock()
        provider = StrandsIntentProvider(model=model)
        assert provider._explicit_model is model

    def test_empty_messages_returns_empty_string(self):
        """Test empty messages returns empty string without calling LLM."""
        provider = StrandsIntentProvider()
        result = provider.derive_intent([])
        assert result == ""

    @patch(
        "bedrock_agentcore.gateway.integrations.strands.plugins.agentcore_tool_search.intent_providers.strands_intent_provider.Agent"
    )
    def test_derive_intent_calls_agent(self, mock_agent_class):
        """Test derive_intent creates an Agent and calls it."""
        mock_agent = Mock()
        mock_agent.return_value = "user wants weather info"
        mock_agent_class.return_value = mock_agent

        provider = StrandsIntentProvider(message_window=2)
        messages = [
            {"role": "user", "content": [{"text": "What is the weather?"}]},
        ]

        result = provider.derive_intent(messages)

        assert result == "user wants weather info"
        mock_agent_class.assert_called_once()

    @patch(
        "bedrock_agentcore.gateway.integrations.strands.plugins.agentcore_tool_search.intent_providers.strands_intent_provider.Agent"
    )
    def test_derive_intent_uses_explicit_model(self, mock_agent_class):
        """Test derive_intent uses explicit model over agent model."""
        mock_agent = Mock()
        mock_agent.return_value = "intent"
        mock_agent_class.return_value = mock_agent

        explicit_model = Mock(name="explicit-model")
        provider = StrandsIntentProvider(model=explicit_model)
        messages = [{"role": "user", "content": [{"text": "hello"}]}]

        provider.derive_intent(messages, model=Mock(name="agent-model"))

        # Explicit model takes priority
        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["model"] is explicit_model

    @patch(
        "bedrock_agentcore.gateway.integrations.strands.plugins.agentcore_tool_search.intent_providers.strands_intent_provider.Agent"
    )
    def test_derive_intent_uses_agent_model_when_no_explicit(self, mock_agent_class):
        """Test derive_intent falls back to agent model when no explicit model."""
        mock_agent = Mock()
        mock_agent.return_value = "intent"
        mock_agent_class.return_value = mock_agent

        agent_model = Mock(name="agent-model")
        provider = StrandsIntentProvider()
        messages = [{"role": "user", "content": [{"text": "hello"}]}]

        provider.derive_intent(messages, model=agent_model)

        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["model"] is agent_model

    @patch(
        "bedrock_agentcore.gateway.integrations.strands.plugins.agentcore_tool_search.intent_providers.strands_intent_provider.Agent"
    )
    def test_derive_intent_no_model_kwarg_when_none(self, mock_agent_class):
        """Test derive_intent omits model kwarg when no model available."""
        mock_agent = Mock()
        mock_agent.return_value = "intent"
        mock_agent_class.return_value = mock_agent

        provider = StrandsIntentProvider()
        messages = [{"role": "user", "content": [{"text": "hello"}]}]

        provider.derive_intent(messages, model=None)

        call_kwargs = mock_agent_class.call_args[1]
        assert "model" not in call_kwargs

    @patch(
        "bedrock_agentcore.gateway.integrations.strands.plugins.agentcore_tool_search.intent_providers.strands_intent_provider.Agent"
    )
    def test_derive_intent_respects_message_window(self, mock_agent_class):
        """Test only last N messages are used."""
        mock_agent = Mock()
        mock_agent.return_value = "intent"
        mock_agent_class.return_value = mock_agent

        provider = StrandsIntentProvider(message_window=3)
        messages = [
            {"role": "user", "content": [{"text": "first"}]},
            {"role": "user", "content": [{"text": "second"}]},
            {"role": "user", "content": [{"text": "third"}]},
            {"role": "user", "content": [{"text": "fourth"}]},
            {"role": "user", "content": [{"text": "fifth"}]},
        ]

        provider.derive_intent(messages)

        # Window=3 takes last 3 messages; only user messages are formatted
        call_args = mock_agent.call_args[0][0]
        assert "first" not in call_args
        assert "second" not in call_args
        assert "third" in call_args
        assert "fourth" in call_args
        assert "fifth" in call_args

    @patch(
        "bedrock_agentcore.gateway.integrations.strands.plugins.agentcore_tool_search.intent_providers.strands_intent_provider.Agent"
    )
    def test_derive_intent_handles_exception(self, mock_agent_class):
        """Test derive_intent returns empty string on exception."""
        mock_agent_class.side_effect = RuntimeError("LLM unavailable")

        provider = StrandsIntentProvider()
        messages = [{"role": "user", "content": [{"text": "hello"}]}]

        result = provider.derive_intent(messages)

        assert result == ""

    def test_format_messages_for_prompt(self):
        """Test message formatting only includes user messages."""
        provider = StrandsIntentProvider()
        messages = [
            {"role": "user", "content": [{"text": "Hello"}, {"text": "world"}]},
            {"role": "assistant", "content": [{"text": "Hi there"}]},
            {"role": "user", "content": [{"text": "What is the weather?"}]},
        ]

        result = provider._format_messages_for_prompt(messages)

        assert "Hello world" in result
        assert "What is the weather?" in result
        assert "Hi there" not in result

    def test_format_messages_handles_missing_role(self):
        """Test formatting skips messages without user role."""
        provider = StrandsIntentProvider()
        messages = [{"content": [{"text": "no role"}]}]

        result = provider._format_messages_for_prompt(messages)

        assert result == ""

    def test_format_messages_handles_non_text_blocks(self):
        """Test formatting skips non-text content blocks."""
        provider = StrandsIntentProvider()
        messages = [
            {"role": "user", "content": [{"image": "data"}, {"text": "only this"}]},
        ]

        result = provider._format_messages_for_prompt(messages)

        assert "only this" in result
        assert "data" not in result

    def test_format_messages_excludes_tool_results(self):
        """Test formatting excludes assistant tool results to avoid PII leakage."""
        provider = StrandsIntentProvider()
        messages = [
            {"role": "user", "content": [{"text": "Check my account"}]},
            {"role": "assistant", "content": [{"toolUse": {"name": "get_account", "input": {}}}]},
            {"role": "user", "content": [{"toolResult": {"content": [{"text": "SSN: 123-45-6789"}]}}]},
            {"role": "user", "content": [{"text": "Now send an email"}]},
        ]

        result = provider._format_messages_for_prompt(messages)

        assert "Check my account" in result
        assert "Now send an email" in result
        assert "SSN" not in result
        assert "get_account" not in result
