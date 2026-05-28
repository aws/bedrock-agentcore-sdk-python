"""Intent provider abstract interface."""

from abc import ABC, abstractmethod


class IntentProvider(ABC):
    """Abstract interface for deriving user intent from conversation messages.

    Subclasses must implement the `derive_intent` method to analyze conversation
    messages and return a concise intent string.
    """

    @abstractmethod
    def derive_intent(self, messages: list[dict], model=None) -> str:
        """Analyze conversation messages and return a concise intent string.

        Args:
            messages: List of conversation message dicts in Strands format.
            model: Optional model instance from the parent agent. Implementations
                can use this for LLM-based intent derivation.

        Returns:
            A plain text string describing the user's intent.
            Returns empty string if intent cannot be determined.
        """
        ...
