"""
FinQA Dataset (czyssrs/FinQA)

A dataset for numerical reasoning over financial data.
Source: https://github.com/czyssrs/FinQA
Paper: https://arxiv.org/abs/2109.00122

Fields per example:
- pre_text: Text before the table
- post_text: Text after the table
- table: Financial table (list of rows)
- qa: Contains question, program, exe_ans (execution answer)

Evaluation:
- Execution Accuracy: Compare predicted answer to gold exe_ans
"""

import json
import logging
import random
import re
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def count_program_steps(program: str) -> int:
    """
    Count the number of operation steps in a FinQA program.

    Args:
        program: Program string like "subtract(5829, 5735), divide(#0, 5735)"

    Returns:
        Number of operation steps
    """
    pattern = r'(\w+)\([^)]+\)'
    matches = re.findall(pattern, program)
    return len(matches)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "finqa"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset"


def table_to_text(table: list[list[str]]) -> str:
    """
    Convert table to linearized text format.

    Args:
        table: List of rows, first row is header

    Returns:
        Linearized table string
    """
    if not table:
        return ""

    header = table[0]
    rows_text = []

    for row in table[1:]:
        row_parts = []
        for h, v in zip(header, row):
            if v and v.strip():
                row_parts.append(f"{h}: {v}")
        if row_parts:
            rows_text.append(" | ".join(row_parts))

    return " ; ".join(rows_text)


def download_file(url: str, output_path: Path) -> None:
    """Download a file from URL."""
    logger.info(f"Downloading {url}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    logger.info(f"Saved to {output_path}")


def download_finqa(
    output_dir: Optional[str] = None,
    n_steps: Optional[int] = None,
    merge_to_qa: bool = False,
) -> dict:
    """
    Download the FinQA dataset from GitHub (czyssrs/FinQA).

    Args:
        output_dir: Directory to save data
        n_steps: If specified, only keep examples with exactly this many steps.
                 Steps are counted as the number of operations in the program.
        merge_to_qa: If True, merge all splits into a single qa.json file

    Returns:
        Dictionary with dataset statistics
    """
    output_dir = Path(output_dir) if output_dir else DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    # Download raw files
    files = ["train.json", "dev.json", "test.json"]
    for fname in files:
        url = f"{GITHUB_RAW_BASE}/{fname}"
        raw_path = raw_dir / fname
        if not raw_path.exists():
            download_file(url, raw_path)
        else:
            logger.info(f"Using cached {raw_path}")

    # Process each split
    stats = {"splits": {}, "n_steps_filter": n_steps}
    all_processed = []  # For merge_to_qa

    for fname in files:
        split_name = fname.replace(".json", "")
        with open(raw_dir / fname) as f:
            raw_data = json.load(f)

        processed = []
        filtered_count = 0

        for entry in raw_data:
            qa = entry.get("qa", {})
            program = qa.get("program", "")

            # Apply n_steps filter if specified
            if n_steps is not None:
                steps = count_program_steps(program)
                if steps != n_steps:
                    filtered_count += 1
                    continue

            # Build context from pre_text + table + post_text
            pre_text = " ".join(entry.get("pre_text", []))
            post_text = " ".join(entry.get("post_text", []))
            table = entry.get("table", [])
            table_text = table_to_text(table)

            context = f"{pre_text}\n\nTable:\n{table_text}\n\n{post_text}".strip()

            # exe_ans is the gold execution answer (numerical)
            exe_ans = qa.get("exe_ans")
            if exe_ans is not None:
                # Format answer consistently
                if isinstance(exe_ans, float):
                    # Round to avoid floating point issues
                    if exe_ans == int(exe_ans):
                        answer = str(int(exe_ans))
                    else:
                        answer = str(round(exe_ans, 6))
                else:
                    answer = str(exe_ans)
            else:
                answer = ""

            item = {
                "id": entry.get("id", ""),
                "context": context,
                "question": qa.get("question", ""),
                "answer": answer,
                "program": program,
                "program_re": qa.get("program_re", ""),
                "gold_inds": qa.get("gold_inds", {}),
                "split": split_name,  # Track which split this came from
                "metadata": {
                    "pre_text": entry.get("pre_text", []),
                    "post_text": entry.get("post_text", []),
                    "table": table,
                    "table_ori": entry.get("table_ori", []),
                },
            }
            processed.append(item)

        # Save processed split
        with open(output_dir / f"{split_name}.json", "w") as f:
            json.dump(processed, f, indent=2)

        all_processed.extend(processed)
        stats["splits"][split_name] = len(processed)

        if n_steps is not None:
            logger.info(f"Processed {split_name}: {len(processed)} examples ({filtered_count} filtered out, kept {n_steps}-step only)")
        else:
            logger.info(f"Processed {split_name}: {len(processed)} examples")

    # If merge_to_qa, create single qa.json with all data
    if merge_to_qa:
        qa_path = output_dir / "qa.json"
        with open(qa_path, "w") as f:
            json.dump(all_processed, f, indent=2)
        logger.info(f"Merged all splits into {qa_path}: {len(all_processed)} examples")
        stats["qa_json_count"] = len(all_processed)

    # Compute overall statistics
    all_data = all_processed
    stats["total_samples"] = len(all_data)

    if all_data:
        stats["avg_context_length"] = sum(len(ex["context"].split()) for ex in all_data) / len(all_data)
        stats["avg_question_length"] = sum(len(ex["question"].split()) for ex in all_data) / len(all_data)

        # Count program operations
        ops = {}
        step_counts = {}
        for ex in all_data:
            prog = ex.get("program", "")
            steps = count_program_steps(prog)
            step_counts[steps] = step_counts.get(steps, 0) + 1

            for op in ["add", "subtract", "multiply", "divide", "exp", "greater",
                       "table_sum", "table_average", "table_max", "table_min"]:
                if op in prog:
                    ops[op] = ops.get(op, 0) + 1
        stats["operation_counts"] = ops
        stats["step_distribution"] = step_counts
    else:
        stats["avg_context_length"] = 0
        stats["avg_question_length"] = 0
        stats["operation_counts"] = {}
        stats["step_distribution"] = {}

    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 50)
    logger.info("FinQA Dataset Statistics (czyssrs/FinQA):")
    if n_steps is not None:
        logger.info(f"  Filter: {n_steps}-step examples only")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Train: {stats['splits'].get('train', 0)}")
    logger.info(f"  Dev: {stats['splits'].get('dev', 0)}")
    logger.info(f"  Test: {stats['splits'].get('test', 0)}")
    if stats.get("avg_context_length"):
        logger.info(f"  Avg context length: {stats['avg_context_length']:.1f} words")
    logger.info(f"  Operations: {list(stats.get('operation_counts', {}).keys())}")
    if merge_to_qa:
        logger.info(f"  Merged to qa.json: {stats.get('qa_json_count', 0)} examples")
    logger.info("=" * 50)

    return stats


class FinQADataset:
    """Dataset loader for FinQA (czyssrs/FinQA)."""

    # Supported operations for program execution
    OPERATIONS = [
        "add", "subtract", "multiply", "divide", "exp", "greater",
        "table_sum", "table_average", "table_max", "table_min"
    ]

    def __init__(
        self,
        split: str = "test",
        data_dir: Optional[str] = None,
        seed: int = 42,
        use_qa_json: bool = False,
    ):
        """
        Initialize the dataset loader.

        Args:
            split: Dataset split ('train', 'dev', 'test') or 'all' when use_qa_json=True
            data_dir: Directory containing processed data
            seed: Random seed for sampling
            use_qa_json: If True, load from qa.json and filter by split field
        """
        self.split = split
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.seed = seed
        self.rng = random.Random(seed)
        self.use_qa_json = use_qa_json
        self._data: list[dict] = []

    def load(self) -> list[dict]:
        """Load the dataset from processed JSON."""
        if self.use_qa_json:
            # Load from merged qa.json
            json_path = self.data_dir / "qa.json"
            if not json_path.exists():
                raise FileNotFoundError(
                    f"qa.json not found at {json_path}. "
                    "Run download_finqa(merge_to_qa=True) first."
                )
            with open(json_path) as f:
                all_data = json.load(f)

            # Filter by split if not 'all'
            if self.split == "all":
                self._data = all_data
            else:
                self._data = [ex for ex in all_data if ex.get("split") == self.split]

            logger.info(f"Loaded {len(self._data)} examples from qa.json (split={self.split})")
        else:
            # Original behavior: load from split-specific file
            json_path = self.data_dir / f"{self.split}.json"
            if not json_path.exists():
                raise FileNotFoundError(
                    f"Processed data not found at {json_path}. "
                    "Run download_finqa() first."
                )
            with open(json_path) as f:
                self._data = json.load(f)
        return self._data

    @property
    def data(self) -> list[dict]:
        """Get the loaded data, loading if necessary."""
        if not self._data:
            self.load()
        return self._data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def sample(self, n: int, exclude_idx: Optional[int] = None, rng: Optional[random.Random] = None) -> list[dict]:
        """Sample n examples from the dataset."""
        rng = rng if rng is not None else self.rng
        indices = list(range(len(self.data)))
        if exclude_idx is not None:
            indices.remove(exclude_idx)
        sampled_indices = rng.sample(indices, min(n, len(indices)))
        return [self.data[i] for i in sampled_indices]

    def get_icl_examples(
        self,
        n_shots: int,
        target_idx: int,
        icl_pool: Optional["FinQADataset"] = None,
        rng: Optional[random.Random] = None,
    ) -> list[dict]:
        """Get ICL examples for a target question."""
        pool = icl_pool if icl_pool is not None else self
        exclude = target_idx if icl_pool is None else None
        return pool.sample(n_shots, exclude_idx=exclude, rng=rng)

    @staticmethod
    def str_to_num(text: str) -> Optional[float]:
        """Convert string to number, handling percentages and special formats."""
        text = text.strip()
        if not text:
            return None

        # Remove common formatting
        text = text.replace(",", "").replace("$", "").replace(" ", "")

        # Handle percentages
        if text.endswith("%"):
            try:
                return float(text[:-1]) / 100
            except ValueError:
                return None

        # Handle parentheses for negative numbers
        if text.startswith("(") and text.endswith(")"):
            text = "-" + text[1:-1]

        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def execute_program(program: str, precision: int = 5) -> Optional[float]:
        """
        Execute a FinQA program and return the result.

        Args:
            program: Program string like "subtract(5829, 5735)"
            precision: Decimal precision for result

        Returns:
            Computed result or None if execution fails
        """
        # Parse program tokens
        tokens = program.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
        tokens = [t.strip() for t in tokens if t.strip() and t != "EOF"]

        if not tokens:
            return None

        results = {}  # Store intermediate results
        step = 0
        i = 0

        while i < len(tokens):
            op = tokens[i]
            if op not in FinQADataset.OPERATIONS:
                i += 1
                continue

            # Find arguments
            if i + 1 >= len(tokens) or tokens[i + 1] != "(":
                i += 1
                continue

            # Find closing paren
            paren_count = 1
            j = i + 2
            args = []
            current_arg = ""

            while j < len(tokens) and paren_count > 0:
                if tokens[j] == "(":
                    paren_count += 1
                    current_arg += tokens[j]
                elif tokens[j] == ")":
                    paren_count -= 1
                    if paren_count > 0:
                        current_arg += tokens[j]
                elif tokens[j] == ",":
                    if current_arg:
                        args.append(current_arg.strip())
                    current_arg = ""
                else:
                    current_arg += tokens[j]
                j += 1

            if current_arg:
                args.append(current_arg.strip())

            # Resolve arguments
            resolved_args = []
            for arg in args:
                if arg.startswith("#"):
                    # Reference to previous result
                    ref_idx = int(arg[1:])
                    if ref_idx in results:
                        resolved_args.append(results[ref_idx])
                    else:
                        return None
                else:
                    num = FinQADataset.str_to_num(arg)
                    if num is not None:
                        resolved_args.append(num)
                    else:
                        return None

            # Execute operation
            try:
                if op == "add" and len(resolved_args) == 2:
                    result = resolved_args[0] + resolved_args[1]
                elif op == "subtract" and len(resolved_args) == 2:
                    result = resolved_args[0] - resolved_args[1]
                elif op == "multiply" and len(resolved_args) == 2:
                    result = resolved_args[0] * resolved_args[1]
                elif op == "divide" and len(resolved_args) == 2:
                    if resolved_args[1] == 0:
                        return None
                    result = resolved_args[0] / resolved_args[1]
                elif op == "exp" and len(resolved_args) == 2:
                    result = resolved_args[0] ** resolved_args[1]
                elif op == "greater" and len(resolved_args) == 2:
                    result = 1.0 if resolved_args[0] > resolved_args[1] else 0.0
                else:
                    return None

                results[step] = result
                step += 1
            except Exception:
                return None

            i = j

        if not results:
            return None

        # Return last result, rounded
        final = results[max(results.keys())]
        return round(final, precision)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_finqa()
