"""Chain-of-thought prompt template for FinQA."""

from typing import Optional
from .base import BasePrompt, DSL_DESCRIPTION


class ChainOfThoughtPrompt(BasePrompt):
    """Chain-of-thought prompt that encourages step-by-step reasoning."""

    SYSTEM_PROMPT_ANSWER = (
        "You are a financial expert. Given the context and question, "
        "think step by step to solve the problem. Show your reasoning, "
        "then provide the final numerical answer on the last line as 'Answer: <number>'."
    )

    SYSTEM_PROMPT_PROGRAM = (
        "You are a financial expert. Given the context and question, "
        "think step by step about what calculations are needed, "
        "then write the program using the DSL. Output the program on the last line."
    )

    TEMPLATE_ANSWER = """Context:
{context}

Question: {question}

Let's solve this step by step:
1. First, identify the relevant numbers from the context.
2. Determine what calculation is needed.
3. Perform the calculation.
4. State the final answer.

Solution:"""

    TEMPLATE_PROGRAM = """{dsl_description}

Context:
{context}

Question: {question}

Let's think about what operations we need:
1. Identify the relevant values from the context.
2. Determine the sequence of operations.
3. Write the program.

Program:"""

    EXAMPLE_TEMPLATE_ANSWER = """Context:
{context}

Question: {question}

Solution:
{reasoning}

Answer: {answer}

---
"""

    EXAMPLE_TEMPLATE_PROGRAM = """Context:
{context}

Question: {question}

Reasoning: {reasoning}

Program: {program}

---
"""

    def __init__(
        self,
        n_shots: int = 2,
        include_system: bool = True,
        output_program: bool = False,
        max_context_len: int = 1000,
    ):
        """
        Initialize chain-of-thought prompt.

        Args:
            n_shots: Number of examples to include
            include_system: Whether to include system prompt
            output_program: If True, prompt for program output; else direct answer
            max_context_len: Maximum context length per example
        """
        super().__init__(include_system, output_program)
        self.n_shots = n_shots
        self.max_context_len = max_context_len

    def _get_system_prompt(self) -> str:
        """Get appropriate system prompt for CoT."""
        if self.output_program:
            return self.SYSTEM_PROMPT_PROGRAM
        return self.SYSTEM_PROMPT_ANSWER

    def _truncate_context(self, context: str) -> str:
        """Truncate context to max length."""
        if len(context) <= self.max_context_len:
            return context
        return context[:self.max_context_len] + "..."

    def format_example(self, example: dict) -> str:
        """Format a single CoT example."""
        context = self._truncate_context(example.get("context", ""))

        # Generate reasoning from program if available
        program = example.get("program", "")
        if program:
            reasoning = f"The calculation requires: {program}"
        else:
            reasoning = "Extracting values and computing the result."

        if self.output_program:
            return self.EXAMPLE_TEMPLATE_PROGRAM.format(
                context=context,
                question=example["question"],
                reasoning=reasoning,
                program=program,
            )
        else:
            return self.EXAMPLE_TEMPLATE_ANSWER.format(
                context=context,
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
            if self.output_program:
                parts.append(DSL_DESCRIPTION)
                parts.append("")
            parts.append("Here are some examples of step-by-step solutions:\n")
            for ex in icl_examples[:self.n_shots]:
                parts.append(self.format_example(ex))
            parts.append("Now solve the following:\n")

        # Add the query
        if self.output_program:
            if not icl_examples:
                parts.append(
                    self.TEMPLATE_PROGRAM.format(
                        dsl_description=DSL_DESCRIPTION,
                        context=context,
                        question=question,
                    )
                )
            else:
                parts.append(f"""Context:
{context}

Question: {question}

Program:""")
        else:
            if not icl_examples:
                parts.append(
                    self.TEMPLATE_ANSWER.format(
                        context=context,
                        question=question,
                    )
                )
            else:
                parts.append(f"""Context:
{context}

Question: {question}

Solution:""")

        user_content = "\n".join(parts)
        return self._to_chat_format(user_content)
