"""Chain-of-thought prompt template for Financial QA."""

from typing import Optional
from .base import BasePrompt


class ChainOfThoughtPrompt(BasePrompt):
    """Chain-of-thought prompt that encourages step-by-step reasoning."""

    SYSTEM_PROMPT = (
        "You are a financial expert. Given the context and question, "
        "think step by step to solve the problem. Show your reasoning, "
        "then provide the final numerical answer."
    )

    TEMPLATE = """Context:
{context}

Question: {question}

Let's solve this step by step:
1. First, identify the relevant numbers from the context.
2. Determine what calculation is needed.
3. Perform the calculation.
4. State the final answer.

Solution:"""

    EXAMPLE_TEMPLATE = """Context:
{context}

Question: {question}

Let's solve this step by step:
{reasoning}

Final Answer: {answer}

---
"""

    def __init__(
        self,
        n_shots: int = 2,
        include_system: bool = True,
    ):
        """
        Initialize chain-of-thought prompt.

        Args:
            n_shots: Number of examples to include
            include_system: Whether to include system prompt
        """
        super().__init__(include_system)
        self.n_shots = n_shots

    def format_example(self, example: dict) -> str:
        """Format a single CoT example."""
        # Use program as reasoning if available, otherwise generic
        program = example.get("program", "")
        if program:
            reasoning = f"Using the formula: {program}"
        else:
            reasoning = "Extracting values and computing..."

        return self.EXAMPLE_TEMPLATE.format(
            context=example.get("context", "")[:1000],
            question=example["question"],
            reasoning=reasoning,
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
        Format chain-of-thought prompt.

        Args:
            question: The question to answer
            context: Financial document context
            icl_examples: Optional ICL examples

        Returns:
            Formatted prompt in chat format
        """
        parts = []

        # Add ICL examples if provided
        if icl_examples:
            parts.append("Here are some examples of step-by-step solutions:\n")
            for ex in icl_examples[: self.n_shots]:
                parts.append(self.format_example(ex))
            parts.append("Now solve the following:\n")

        # Add the query
        parts.append(
            self.TEMPLATE.format(
                context=context,
                question=question,
            )
        )

        user_content = "\n".join(parts)
        return self._to_chat_format(user_content)
