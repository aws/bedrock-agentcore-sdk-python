import logging

from strands import Agent

from .intent_provider import IntentProvider

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = (
    "You are an intent classifier. Given the recent conversation messages, "
    "produce a concise one-sentence description of what the user is trying to accomplish. "
    "Focus on the type of task, not the specific details. "
    "Reply with ONLY the intent description, nothing else."
)


class DefaultIntentProvider(IntentProvider):
    """LLM-based intent provider that classifies the last N messages."""

    def __init__(self, message_window: int = 5, model=None):
        self._message_window = message_window
        self._explicit_model = model

    def derive_intent(self, messages: list[dict], model=None) -> str:
        """Derive intent using an LLM. Falls back to agent's model if no explicit model set."""
        try:
            recent_messages = messages[-self._message_window:] if messages else []
            if not recent_messages:
                return ""

            kwargs = {"system_prompt": INTENT_SYSTEM_PROMPT, "tools": []}
            # Priority: explicit model > agent's model > Strands default
            resolved_model = self._explicit_model or model
            if resolved_model:
                kwargs["model"] = resolved_model

            intent_agent = Agent(**kwargs)
            response = intent_agent(self._format_messages_for_prompt(recent_messages))
            return str(response).strip()
        except Exception as e:
            logger.error("Failed to derive intent: %s", e)
            return ""

    def _format_messages_for_prompt(self, messages: list[dict]) -> str:
        """Format messages into a text prompt for the intent LLM."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", [])
            text = ""
            if isinstance(content, list):
                text = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and "text" in block
                )
            parts.append(f"{role}: {text}")
        return "\n".join(parts)
