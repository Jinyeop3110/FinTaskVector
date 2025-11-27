"""Few-shot prompt template for FinQA."""

from typing import Optional
from .base import BasePrompt, DSL_DESCRIPTION


class FewShotPrompt(BasePrompt):
    """Few-shot prompt with in-context learning examples."""

    # Example templates for answer mode
    EXAMPLE_TEMPLATE_ANSWER = """Context:
{context}

Question: {question}

Answer: {answer}

---
"""

    # Example templates for program mode
    EXAMPLE_TEMPLATE_PROGRAM = """Context:
{context}

Question: {question}

Program: {program}

---
"""

    # Query templates
    QUERY_TEMPLATE_ANSWER = """Context:
{context}

Question: {question}

Answer:"""

    QUERY_TEMPLATE_PROGRAM = """Context:
{context}

Question: {question}

Output only the program, nothing else.

Program:"""

    def __init__(
        self,
        n_shots: int = 3,
        include_system: bool = True,
        output_program: bool = False,
        max_context_len: int = 1500,
    ):
        """
        Initialize few-shot prompt.

        Args:
            n_shots: Number of examples to include
            include_system: Whether to include system prompt
            output_program: If True, prompt for program output; else direct answer
            max_context_len: Maximum context length per example (truncation)
        """
        super().__init__(include_system, output_program)
        self.n_shots = n_shots
        self.max_context_len = max_context_len

    def _truncate_context(self, context: str) -> str:
        """Truncate context to max length."""
        if len(context) <= self.max_context_len:
            return context
        return context[:self.max_context_len] + "..."

    def format_example(self, example: dict) -> str:
        """Format a single ICL example."""
        context = self._truncate_context(example.get("context", ""))

        if self.output_program:
            return self.EXAMPLE_TEMPLATE_PROGRAM.format(
                context=context,
                question=example["question"],
                program=example.get("program", ""),
            )
        else:
            return self.EXAMPLE_TEMPLATE_ANSWER.format(
                context=context,
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
            icl_examples: List of ICL examples with 'context', 'question', 'answer', 'program'

        Returns:
            Formatted prompt in chat format
        """
        parts = []

        # Add DSL description for program mode
        if self.output_program:
            parts.append(DSL_DESCRIPTION)
            parts.append("")

        # Add ICL examples
        if icl_examples:
            parts.append("Here are some examples:\n")
            for ex in icl_examples[:self.n_shots]:
                parts.append(self.format_example(ex))
            parts.append("Now answer the following:\n")

        # Add the query
        if self.output_program:
            parts.append(
                self.QUERY_TEMPLATE_PROGRAM.format(
                    context=context,
                    question=question,
                )
            )
        else:
            parts.append(
                self.QUERY_TEMPLATE_ANSWER.format(
                    context=context,
                    question=question,
                )
            )

        user_content = "\n".join(parts)
        return self._to_chat_format(user_content)
