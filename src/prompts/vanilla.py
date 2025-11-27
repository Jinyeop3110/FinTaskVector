"""Vanilla (zero-shot) prompt template for FinQA."""

from typing import Optional
from .base import BasePrompt


class VanillaPrompt(BasePrompt):
    """Zero-shot prompt - direct answer without reasoning."""

    SYSTEM_PROMPT = (
        "You are a financial expert. Given the context and question, "
        "provide the numerical answer. Be precise and give only the final number. "
        "For yes/no questions, answer with 'yes' or 'no'."
    )

    TEMPLATE = """Context:
{context}

Question: {question}

Answer (provide only the numerical value, or yes/no for comparison questions):"""

    def __init__(self, include_system: bool = True):
        """
        Initialize vanilla prompt.

        Args:
            include_system: Whether to include system prompt
        """
        self.include_system = include_system

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _to_chat_format(self, user_content: str) -> list[dict]:
        """Convert to chat message format."""
        messages = []
        if self.include_system:
            messages.append({"role": "system", "content": self._get_system_prompt()})
        messages.append({"role": "user", "content": user_content})
        return messages

    def format(
        self,
        question: str,
        context: str,
        **kwargs,
    ) -> list[dict]:
        """
        Format zero-shot prompt.

        Args:
            question: The question to answer
            context: Financial document context

        Returns:
            Formatted prompt in chat format
        """
        user_content = self.TEMPLATE.format(
            context=context,
            question=question,
        )
        return self._to_chat_format(user_content)
