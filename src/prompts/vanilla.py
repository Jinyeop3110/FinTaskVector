"""Vanilla (zero-shot) prompt template for Financial QA."""

from typing import Optional
from .base import BasePrompt


class VanillaPrompt(BasePrompt):
    """Zero-shot prompt without examples."""

    TEMPLATE = """Context:
{context}

Question: {question}

Answer (provide only the numerical value):"""

    def format(
        self,
        question: str,
        context: str,
        **kwargs,
    ) -> str | list[dict]:
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
