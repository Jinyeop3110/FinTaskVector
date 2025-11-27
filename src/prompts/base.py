"""Base prompt class for Financial QA."""

from abc import ABC, abstractmethod
from typing import Optional


class BasePrompt(ABC):
    """Abstract base class for prompt templates."""

    SYSTEM_PROMPT = (
        "You are a financial expert. Given the context and question, "
        "provide the numerical answer. Be precise and give only the final number."
    )

    def __init__(self, include_system: bool = True):
        """
        Initialize prompt template.

        Args:
            include_system: Whether to include system prompt
        """
        self.include_system = include_system

    @abstractmethod
    def format(
        self,
        question: str,
        context: str,
        **kwargs,
    ) -> str | list[dict]:
        """
        Format the prompt.

        Args:
            question: The question to answer
            context: Financial document context
            **kwargs: Additional arguments

        Returns:
            Formatted prompt (string or chat format)
        """
        pass

    def _to_chat_format(self, user_content: str) -> list[dict]:
        """Convert to chat message format."""
        messages = []
        if self.include_system:
            messages.append({"role": "system", "content": self.SYSTEM_PROMPT})
        messages.append({"role": "user", "content": user_content})
        return messages
