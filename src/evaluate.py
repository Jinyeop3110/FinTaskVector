"""Evaluation metrics for FinQA (Financial Question Answering)."""

import re
from typing import Optional


def extract_number(text: str) -> Optional[float]:
    """
    Extract numerical value from text.

    Handles:
    - Plain numbers: 123, 123.45
    - Percentages: 12.5%, 12.5 %
    - Currency: $1,234.56
    - Negative: -123, (123)
    - Scientific: 1.23e-4

    Args:
        text: Text containing a number

    Returns:
        Extracted float or None if no number found
    """
    if not text:
        return None

    text = text.strip()

    # Try to find the last number in the text (usually the answer)
    # Pattern matches various number formats
    patterns = [
        r"[-+]?\d+\.?\d*%",  # Percentage
        r"[-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?",  # Currency with commas
        r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?",  # Scientific notation
        r"\([\d,.]+\)",  # Negative in parentheses
    ]

    # Combine patterns
    combined = "|".join(f"({p})" for p in patterns)
    matches = re.findall(combined, text)

    if not matches:
        return None

    # Get the last match (flatten tuple and filter empty)
    last_match = None
    for match_tuple in matches:
        for m in match_tuple:
            if m:
                last_match = m

    if not last_match:
        return None

    # Clean and convert
    num_str = last_match.strip()

    # Handle percentage
    is_percent = "%" in num_str
    num_str = num_str.replace("%", "")

    # Handle currency and commas
    num_str = num_str.replace("$", "").replace(",", "")

    # Handle parentheses for negative
    if num_str.startswith("(") and num_str.endswith(")"):
        num_str = "-" + num_str[1:-1]

    try:
        value = float(num_str)
        if is_percent:
            value = value / 100
        return value
    except ValueError:
        return None


def numerical_match(
    prediction: str,
    ground_truth: str,
    tolerance: float = 1e-3,
) -> float:
    """
    Check if predicted number matches ground truth within tolerance.

    Args:
        prediction: Model prediction (may contain text)
        ground_truth: Ground truth answer
        tolerance: Relative tolerance for comparison

    Returns:
        1.0 if match, 0.0 otherwise
    """
    pred_num = extract_number(prediction)
    gt_num = extract_number(ground_truth)

    if pred_num is None or gt_num is None:
        return 0.0

    # Handle zero case
    if gt_num == 0:
        return float(abs(pred_num) < tolerance)

    # Relative error
    rel_error = abs(pred_num - gt_num) / abs(gt_num)
    return float(rel_error <= tolerance)


def execution_accuracy(
    prediction: str,
    ground_truth: str,
    tolerance: float = 1e-3,
) -> float:
    """
    FinQA execution accuracy metric.

    Compares extracted numerical values with tolerance.
    This is the primary metric used in FinQA evaluation.

    Args:
        prediction: Model prediction
        ground_truth: Ground truth (exe_ans from dataset)
        tolerance: Relative tolerance (default 0.1%)

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    return numerical_match(prediction, ground_truth, tolerance)


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Exact string match after normalization.

    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_clean = prediction.strip().lower()
    gt_clean = ground_truth.strip().lower()
    return float(pred_clean == gt_clean)


def evaluate_predictions(
    predictions: list[str],
    ground_truths: list[str],
    tolerance: float = 1e-3,
) -> dict[str, float]:
    """
    Evaluate predictions using FinQA metrics.

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers
        tolerance: Tolerance for numerical comparison

    Returns:
        Dictionary with metric scores
    """
    assert len(predictions) == len(ground_truths), "Length mismatch"

    exec_acc_scores = []
    exact_match_scores = []

    for pred, gt in zip(predictions, ground_truths):
        exec_acc_scores.append(execution_accuracy(pred, gt, tolerance))
        exact_match_scores.append(exact_match(pred, gt))

    return {
        "execution_accuracy": sum(exec_acc_scores) / len(exec_acc_scores),
        "exact_match": sum(exact_match_scores) / len(exact_match_scores),
        "n_samples": len(predictions),
        "n_correct": int(sum(exec_acc_scores)),
    }


def evaluate_by_operation(
    predictions: list[str],
    ground_truths: list[str],
    programs: list[str],
    tolerance: float = 1e-3,
) -> dict[str, dict]:
    """
    Evaluate predictions grouped by operation type.

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers
        programs: List of program strings (to extract operation)
        tolerance: Tolerance for numerical comparison

    Returns:
        Dictionary with per-operation metrics
    """
    operations = [
        "add", "subtract", "multiply", "divide",
        "exp", "greater", "table_sum", "table_average",
    ]

    results = {op: {"correct": 0, "total": 0} for op in operations}
    results["other"] = {"correct": 0, "total": 0}

    for pred, gt, prog in zip(predictions, ground_truths, programs):
        score = execution_accuracy(pred, gt, tolerance)

        # Find which operation this belongs to
        found_op = False
        for op in operations:
            if op in prog:
                results[op]["correct"] += score
                results[op]["total"] += 1
                found_op = True
                break

        if not found_op:
            results["other"]["correct"] += score
            results["other"]["total"] += 1

    # Compute accuracy per operation
    for op in results:
        total = results[op]["total"]
        if total > 0:
            results[op]["accuracy"] = results[op]["correct"] / total
        else:
            results[op]["accuracy"] = 0.0

    return results
