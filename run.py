"""Main entry point for FinQA evaluation."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from src.datasets import FinQADataset
from src.evaluate import (
    evaluate_predictions,
    evaluate_by_operation,
    execution_accuracy,
)
from src.model import VLLMInference
from src.prompts import VanillaPrompt, FewShotPrompt, ChainOfThoughtPrompt

logger = logging.getLogger(__name__)

PROMPT_TYPES = {
    "vanilla": VanillaPrompt,
    "few_shot": FewShotPrompt,
    "cot": ChainOfThoughtPrompt,
}


def setup_logging(output_dir: Path) -> None:
    """Setup logging to both console and file."""
    log_file = output_dir / "run.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers
    root_logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_session_name(model: str, prompt_type: str, n_shots: int, tag: str | None = None) -> str:
    """Generate session folder name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.split("/")[-1]

    if prompt_type == "few_shot":
        mode = f"{n_shots}shot"
    else:
        mode = prompt_type

    parts = [model_short, mode, timestamp]
    if tag:
        parts.insert(0, tag)
    return "_".join(parts)


def run_evaluation(
    model: VLLMInference,
    dataset: FinQADataset,
    prompt_template,
    icl_pool: FinQADataset | None = None,
    n_shots: int = 0,
    max_samples: int | None = None,
    batch_size: int = 8,
    max_tokens: int = 256,
) -> tuple[list[dict], dict]:
    """
    Run evaluation on the dataset.

    Args:
        model: vLLM model wrapper
        dataset: FinQA dataset (test set)
        prompt_template: Prompt template instance
        icl_pool: Dataset to sample ICL examples from (train set)
        n_shots: Number of ICL examples (for few-shot)
        max_samples: Maximum samples to evaluate (None for all)
        batch_size: Batch size for inference
        max_tokens: Maximum tokens to generate

    Returns:
        Tuple of (predictions list, metrics dict)
    """
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    logger.info(f"Running evaluation on {n_samples} samples")

    all_predictions = []
    all_prompts = []
    all_ground_truths = []
    all_programs = []

    # Prepare prompts
    for idx in tqdm(range(n_samples), desc="Preparing prompts"):
        example = dataset[idx]
        question = example["question"]
        context = example["context"]
        ground_truth = example["answer"]
        program = example.get("program", "")

        # Get ICL examples if needed
        icl_examples = None
        if n_shots > 0 and icl_pool is not None:
            icl_examples = dataset.get_icl_examples(n_shots, idx, icl_pool=icl_pool)

        # Format prompt
        prompt = prompt_template.format(
            question=question,
            context=context,
            icl_examples=icl_examples,
        )

        all_prompts.append(prompt)
        all_ground_truths.append(ground_truth)
        all_programs.append(program)
        all_predictions.append({
            "id": example.get("id", f"test_{idx}"),
            "idx": idx,
            "question": question,
            "context": context[:500] + "..." if len(context) > 500 else context,
            "ground_truth": ground_truth,
            "program": program,
        })

    # Batch inference
    logger.info("Running inference...")
    all_responses = []
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating"):
        batch = all_prompts[i : i + batch_size]
        responses = model.generate(batch, max_tokens=max_tokens)
        all_responses.extend(responses)

    # Add predictions and per-example metrics
    for pred, response in zip(all_predictions, all_responses):
        pred["prediction"] = response
        pred["execution_accuracy"] = execution_accuracy(response, pred["ground_truth"])

    # Compute aggregate metrics
    predictions_list = [p["prediction"] for p in all_predictions]
    metrics = evaluate_predictions(predictions_list, all_ground_truths)

    # Add per-operation breakdown
    op_metrics = evaluate_by_operation(predictions_list, all_ground_truths, all_programs)
    metrics["by_operation"] = op_metrics

    return all_predictions, metrics


def save_predictions(predictions: list[dict], output_path: str) -> None:
    """Save predictions to JSON file."""
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FinQA Evaluation")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="vanilla",
        choices=["vanilla", "few_shot", "cot"],
        help="Prompt template type",
    )
    parser.add_argument("--n_shots", type=int, default=3, help="Number of ICL examples (for few_shot)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None, help="Optional tag for session name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Numerical tolerance for evaluation")
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    # Create session directory
    session_name = get_session_name(args.model, args.prompt_type, args.n_shots, args.tag)
    output_dir = Path(args.output_dir) / session_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("FinQA Evaluation (czyssrs/FinQA)")
    logger.info("=" * 60)
    logger.info(f"Session: {session_name}")
    logger.info(f"Configuration: {vars(args)}")

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    # Initialize prompt template
    prompt_class = PROMPT_TYPES[args.prompt_type]
    if args.prompt_type == "few_shot":
        prompt_template = prompt_class(n_shots=args.n_shots)
    elif args.prompt_type == "cot":
        prompt_template = prompt_class(n_shots=args.n_shots if args.n_shots > 0 else 2)
    else:
        prompt_template = prompt_class()

    logger.info(f"Using prompt type: {args.prompt_type}")

    # Initialize datasets
    test_dataset = FinQADataset(split=args.split, seed=args.seed)
    test_dataset.load()
    logger.info(f"Loaded {args.split} dataset with {len(test_dataset)} examples")

    # Load train set for ICL examples if needed
    icl_pool = None
    if args.prompt_type in ["few_shot", "cot"] and args.n_shots > 0:
        icl_pool = FinQADataset(split="train", seed=args.seed)
        icl_pool.load()
        logger.info(f"Loaded train dataset with {len(icl_pool)} examples for ICL")

    # Initialize model
    logger.info(f"Loading model: {args.model}")
    model = VLLMInference(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel,
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    predictions, metrics = run_evaluation(
        model=model,
        dataset=test_dataset,
        prompt_template=prompt_template,
        icl_pool=icl_pool,
        n_shots=args.n_shots,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

    # Save results
    logger.info("Saving results...")
    save_predictions(predictions, str(output_dir / "predictions.json"))

    # Metrics JSON
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Summary
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Samples evaluated: {metrics['n_samples']}")
    logger.info(f"Execution Accuracy: {metrics['execution_accuracy']:.4f} ({metrics['n_correct']}/{metrics['n_samples']})")
    logger.info(f"Exact Match: {metrics['exact_match']:.4f}")
    logger.info("-" * 40)
    logger.info("Accuracy by Operation:")
    for op, op_metrics in metrics["by_operation"].items():
        if op_metrics["total"] > 0:
            logger.info(f"  {op}: {op_metrics['accuracy']:.4f} ({op_metrics['total']} samples)")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")

    return metrics


if __name__ == "__main__":
    main()
