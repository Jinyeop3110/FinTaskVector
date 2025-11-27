"""Vanilla (zero-shot) prompt template for FinQA."""

from typing import Optional
from .base import BasePrompt, DSL_DESCRIPTION


class VanillaPrompt(BasePrompt):
    """Zero-shot prompt without examples."""

    # Template for direct answer mode
    TEMPLATE_ANSWER = """Context:
{context}

Question: {question}

Answer (provide only the numerical value):"""

    # Template for program synthesis mode
    TEMPLATE_PROGRAM = """{dsl_description}

Context:
{context}

Question: {question}

Output only the program (e.g., "divide(100, 50), multiply(#0, 2)"), nothing else.

Program:"""

    def __init__(
        self,
        include_system: bool = True,
        output_program: bool = False,
    ):
        """
        Initialize vanilla prompt.

        Args:
            include_system: Whether to include system prompt
            output_program: If True, prompt for program output; else direct answer
        """
        super().__init__(include_system, output_program)

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
        if self.output_program:
            user_content = self.TEMPLATE_PROGRAM.format(
                dsl_description=DSL_DESCRIPTION,
                context=context,
                question=question,
            )
        else:
            user_content = self.TEMPLATE_ANSWER.format(
                context=context,
                question=question,
            )
        return self._to_chat_format(user_content)
