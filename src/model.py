"""vLLM inference wrapper for Financial QA."""

import time
from dataclasses import dataclass
from typing import Optional

from vllm import LLM, SamplingParams


@dataclass
class GenerationStats:
    """Statistics from generation."""
    responses: list[str]
    total_input_tokens: int
    total_output_tokens: int
    latency_seconds: float

    @property
    def avg_input_tokens(self) -> float:
        return self.total_input_tokens / len(self.responses) if self.responses else 0

    @property
    def avg_output_tokens(self) -> float:
        return self.total_output_tokens / len(self.responses) if self.responses else 0

    @property
    def tokens_per_second(self) -> float:
        return self.total_output_tokens / self.latency_seconds if self.latency_seconds > 0 else 0


class VLLMInference:
    """Wrapper for vLLM inference."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
    ):
        """
        Initialize vLLM model.

        Args:
            model_name: HuggingFace model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            gpu_memory_utilization: Fraction of GPU memory to use
            dtype: Data type for model weights
        """
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        prompts: list[str] | list[list[dict]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        return_stats: bool = False,
    ) -> list[str] | GenerationStats:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of prompts (strings or chat format)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            return_stats: If True, return GenerationStats with token counts and latency

        Returns:
            List of generated responses, or GenerationStats if return_stats=True
        """
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

        # Handle chat format
        if prompts and isinstance(prompts[0], list):
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
                for p in prompts
            ]
        else:
            formatted_prompts = prompts

        start_time = time.time()
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        latency = time.time() - start_time

        responses = [output.outputs[0].text.strip() for output in outputs]

        if return_stats:
            total_input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
            total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            return GenerationStats(
                responses=responses,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                latency_seconds=latency,
            )

        return responses

    def generate_single(
        self,
        prompt: str | list[dict],
        **kwargs,
    ) -> str:
        """Generate response for a single prompt."""
        results = self.generate([prompt], **kwargs)
        return results[0]
