"""Few-shot prompt template for Financial QA."""

from typing import Optional
from .base import BasePrompt


class FewShotPrompt(BasePrompt):
    """Few-shot prompt with in-context learning examples."""

    EXAMPLE_TEMPLATE = """Context:
{context}

Question: {question}

Answer: {answer}

---
"""

    QUERY_TEMPLATE = """Context:
{context}

Question: {question}

Answer (provide only the numerical value):"""

    def __init__(
        self,
        n_shots: int = 3,
        include_system: bool = True,
    ):
        """
        Initialize few-shot prompt.

        Args:
            n_shots: Number of examples to include
            include_system: Whether to include system prompt
        """
        super().__init__(include_system)
        self.n_shots = n_shots

    def format_example(self, example: dict) -> str:
        """Format a single ICL example."""
        return self.EXAMPLE_TEMPLATE.format(
            context=example.get("context", "")[:1500],  # Truncate for length
            question=example["question"],
            answer=example["answer"],
        )

    def format(
        self,
        question: str,
        context: str,
        icl_examples: Optional[list[dict]] = None,
        **kwargs,
    ) -> str | list[dict]:
        """
        Format few-shot prompt with examples.

        Args:
            question: The question to answer
            context: Financial document context
            icl_examples: List of ICL examples with 'context', 'question', 'answer'

        Returns:
            Formatted prompt in chat format
        """
        parts = []

        # Add ICL examples
        if icl_examples:
            parts.append("Here are some examples:\n")
            for ex in icl_examples[: self.n_shots]:
                parts.append(self.format_example(ex))
            parts.append("Now answer the following:\n")

        # Add the query
        parts.append(
            self.QUERY_TEMPLATE.format(
                context=context,
                question=question,
            )
        )

        user_content = "\n".join(parts)
        return self._to_chat_format(user_content)
