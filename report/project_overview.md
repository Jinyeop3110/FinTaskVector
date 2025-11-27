# Financial QA Evaluation Framework

## Project Overview

This is a **Financial QA (Question Answering) Evaluation Framework** that compares different prompting strategies for financial numerical reasoning tasks using the **FinQA dataset** and vLLM for model inference.

### Key Details

| Item | Value |
|------|-------|
| **Dataset** | FinQA (czyssrs/FinQA) |
| **Default Model** | Qwen/Qwen2.5-7B-Instruct |
| **Framework** | vLLM for efficient batch inference |
| **Task** | Numerical reasoning over financial documents |

---

## Architecture

```
Financial_task_vector/
├── src/
│   ├── datasets/        # Data loading (finqa.py)
│   ├── prompts/         # Prompt templates (vanilla, few_shot, cot)
│   ├── model.py         # vLLM inference wrapper
│   ├── evaluate.py      # Evaluation metrics
│   └── __init__.py
├── configs/             # YAML config files
├── data/finqa/          # Downloaded dataset
├── scripts/             # Shell scripts
├── outputs/             # Results directory
├── report/              # Reports and analysis
├── run.py               # Main entry point
└── requirements.txt     # Dependencies
```

---

## Main Components

### 1. Data Pipeline (`src/datasets/finqa.py`)

- **FinQADataset class**: Loads and manages the FinQA dataset
- Downloads from GitHub (train/dev/test splits)
- Linearizes financial tables to text
- Supports sampling for ICL (in-context learning) examples
- Program execution for mathematical operations

### 2. Prompt Templates (`src/prompts/`)

| Strategy | Description | Default Shots |
|----------|-------------|---------------|
| **VanillaPrompt** | Zero-shot, direct Q&A | 0 |
| **FewShotPrompt** | N examples from training data | 3 |
| **ChainOfThoughtPrompt** | Step-by-step reasoning | 2 |

### 3. Model Inference (`src/model.py`)

**VLLMInference class**:
- Wrapper around vLLM for efficient inference
- Tensor parallelism across GPUs
- Configurable batch inference
- Chat format prompt handling

### 4. Evaluation Metrics (`src/evaluate.py`)

- **Execution Accuracy**: Primary metric (numerical with tolerance)
- **Exact Match**: String-level equality
- **Per-Operation Breakdown**: Accuracy by operation type (add, subtract, multiply, etc.)

---

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Qwen/Qwen2.5-7B-Instruct | HuggingFace model ID |
| `--prompt_type` | vanilla | vanilla, few_shot, cot |
| `--n_shots` | 3 | Number of ICL examples |
| `--max_samples` | None | Limit evaluation samples |
| `--batch_size` | 8 | Batch size for inference |
| `--max_tokens` | 256 | Max output tokens |
| `--tensor_parallel` | 1 | Number of GPUs |
| `--tolerance` | 1e-3 | Numerical tolerance |
| `--split` | test | Dataset split |

---

## Usage

### Step 1: Download Dataset

```bash
./scripts/download_finqa.sh
```

### Step 2: Run Evaluation

**Zero-shot:**
```bash
python run.py --n_shots 0 --max_samples 50
```

**Few-shot (3 examples):**
```bash
python run.py --n_shots 3 --max_samples 50
```

**Chain-of-thought:**
```bash
python run.py --prompt_type cot --n_shots 2 --max_samples 50
```

### Step 3: Run Comparison

```bash
./scripts/run_comparison.sh "Qwen/Qwen2.5-7B-Instruct" 100
```

---

## Output Structure

```
outputs/{model}_{mode}_{timestamp}/
├── config.yaml          # Run configuration
├── run.log              # Execution log
├── predictions.json     # Per-sample predictions
└── metrics.json         # Aggregated metrics
```

### Example Metrics Output

```json
{
  "execution_accuracy": 0.67,
  "exact_match": 0.65,
  "n_samples": 100,
  "n_correct": 67,
  "by_operation": {
    "add": {"accuracy": 0.75, "total": 20},
    "subtract": {"accuracy": 0.60, "total": 15},
    "multiply": {"accuracy": 0.70, "total": 10}
  }
}
```

---

## Dependencies

```
vllm>=0.6.0
datasets>=2.14.0
transformers>=4.40.0
pyyaml>=6.0
tqdm>=4.66.0
pandas>=2.0.0
openpyxl>=3.1.0
requests>=2.28.0
```

---

## Workflow Diagram

```
Dataset Download (GitHub)
         ↓
Load FinQA (train/test splits)
         ↓
For each sample:
  - Prepare prompt (vanilla/few-shot/CoT)
  - Sample ICL examples if needed
  - Batch inference via vLLM
         ↓
Post-process predictions:
  - Extract numerical values
  - Compare with tolerance
         ↓
Compute metrics:
  - Execution accuracy
  - Exact match
  - Per-operation breakdown
         ↓
Save results (JSON, config, logs)
```

---

## References

- **FinQA Paper**: https://arxiv.org/abs/2109.00122
- **FinQA Repository**: https://github.com/czyssrs/FinQA
- **vLLM**: https://github.com/vllm-project/vllm
