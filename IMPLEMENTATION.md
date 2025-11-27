# Implementation Guide

## Project Overview
Financial QA evaluation comparing single-question vs in-context learning (ICL) prompting strategies using vLLM.

## Dataset

### FinQA
- **Source**: [AfterQuery/FinanceQA](https://huggingface.co/datasets/AfterQuery/FinanceQA)
- **Paper**: [arXiv:2501.18062](https://arxiv.org/abs/2501.18062)
- **Task**: Direct text-based financial QA
- **Splits**: test (148 examples)
- **Fields**: context, question, answer, chain_of_thought, company, question_type

## Directory Structure
```
Financial_task_vector/
├── configs/                    # YAML configuration files
│   ├── zero_shot.yaml
│   └── few_shot.yaml
├── data/                       # Dataset storage (gitignored)
│   └── finqa/
│       ├── test.json
│       └── statistics.json
├── outputs/                    # Experiment results (gitignored)
│   └── {model}_{mode}_{timestamp}/
│       ├── config.yaml
│       ├── run.log
│       ├── predictions.json
│       ├── results.xlsx
│       └── metrics.json
├── scripts/
│   ├── download_finqa.sh
│   ├── run_zero_shot.sh
│   ├── run_few_shot.sh
│   └── run_comparison.sh
├── src/
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── finqa.py
│   ├── prompts.py
│   ├── model.py
│   └── evaluate.py
├── run.py
├── requirements.txt
└── IMPLEMENTATION.md
```

## Code Style

### Python
- Python 3.10+
- Type hints required for function signatures
- Docstrings: Google style
- Line length: 100 chars max
- Imports: stdlib → third-party → local (separated by blank lines)

### Naming Conventions
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

## Quick Start

```bash
# 1. Download dataset
./scripts/download_finqa.sh

# 2. Run evaluation
# Zero-shot
python run.py --n_shots 0 --max_samples 50

# 3-shot ICL
python run.py --n_shots 3 --max_samples 50
```

## Prompt Formats

### Zero-shot
```
System: You are a financial expert assistant...

Question: {question}

Answer:
```

### Few-shot ICL
```
System: You are a financial expert assistant...

Question: {example_q1}
Answer: {example_a1}

Question: {example_q2}
Answer: {example_a2}

Question: {target_question}
Answer:
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Exact Match | Normalized string equality |
| F1 Score | Token-level precision/recall |
| ROUGE-L | Longest common subsequence F1 |

## Git Workflow
1. Create feature branch: `git checkout -b feature/<name>`
2. Implement changes
3. Update IMPLEMENTATION.md if structure changes
4. Commit with descriptive message
