"""Base prompt class for FinQA."""

from abc import ABC, abstractmethod
from typing import Optional


# DSL operations and description for program synthesis
DSL_DESCRIPTION = """
Write a program to compute the answer using the following DSL operations.

Operations:
- add(a, b): Add two numbers. Example: add(10, 5) = 15
- subtract(a, b): Subtract b from a. Example: subtract(10, 5) = 5
- multiply(a, b): Multiply two numbers. Example: multiply(10, 5) = 50
- divide(a, b): Divide a by b. Example: divide(10, 5) = 2
- exp(a, b): Raise a to the power of b. Example: exp(2, 3) = 8
- greater(a, b): Return "yes" if a > b, else "no". Example: greater(10, 5) = "yes"
- table_sum(row): Sum all values in a table row
- table_average(row): Average of all values in a table row
- table_max(row): Maximum value in a table row
- table_min(row): Minimum value in a table row

Rules:
1. Use #N to refer to the result of step N (0-indexed). Example: #0 refers to the first operation's result.
2. Use const_X for constants: const_100 = 100, const_1000 = 1000, const_1000000 = 1000000.
3. Percentages like "23.6%" should be written as 23.6% in the program.
4. Chain operations sequentially. Each operation can reference previous results with #N.

Example programs:
- "What percentage of total revenue (7018) is segment revenue (1947)?" → divide(1947, 7018)
- "What is the change in net assets from 2303 to 2309.9?" → subtract(2309.9, 2303)
- "What is the growth rate from 991.1 to 959.2?" → subtract(959.2, 991.1), divide(#0, 991.1)
- "What is the total of 15553 and 7376?" → add(15553, 7376)
- "What is the percentage decrease from 34.8 to 1.2?" → subtract(34.8, 1.2), divide(#0, 34.8), multiply(#1, const_100)
- "Did value A (11000) exceed value B (3300000)?" → multiply(607, 18.13), multiply(#0, const_1000), multiply(3.3, const_1000000), greater(#1, #2)
- "What is total operating expenses if segment expense is 9896 and it represents 23.6%?" → divide(9896, 23.6%)
- "Convert 1327657 shares at $42.61 to millions." → multiply(1327657, 42.61), divide(#0, const_1000000)
- "What is the average of values in the revenue row?" → table_average(revenue)
""".strip()


class BasePrompt(ABC):
    """Abstract base class for prompt templates."""

    # System prompts for different modes
    SYSTEM_PROMPT_ANSWER = (
        "You are a financial expert. Given the context and question, "
        "provide the numerical answer. Be precise and give only the final number."
    )

    SYSTEM_PROMPT_PROGRAM = (
        "You are a financial expert. Given the context and question, "
        "write a program using the provided DSL to compute the answer. "
        "Output only the program, nothing else."
    )

    def __init__(
        self,
        include_system: bool = True,
        output_program: bool = False,
    ):
        """
        Initialize prompt template.

        Args:
            include_system: Whether to include system prompt
            output_program: If True, prompt for program output; else direct answer
        """
        self.include_system = include_system
        self.output_program = output_program

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

    def _get_system_prompt(self) -> str:
        """Get appropriate system prompt based on output mode."""
        if self.output_program:
            return self.SYSTEM_PROMPT_PROGRAM
        return self.SYSTEM_PROMPT_ANSWER

    def _to_chat_format(self, user_content: str) -> list[dict]:
        """Convert to chat message format."""
        messages = []
        if self.include_system:
            messages.append({"role": "system", "content": self._get_system_prompt()})
        messages.append({"role": "user", "content": user_content})
        return messages
