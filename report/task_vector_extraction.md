# Task Vector Extraction

## Overview

Task vectors are extracted hidden representations from transformer models that capture task-specific information. This module extracts hidden states at specified token positions and layers, enabling analysis of how different prompting strategies affect internal representations.

---

## Architecture

```
Financial_task_vector/
├── task_vector_generation.py    # Main extraction script
├── task_vector_config.yaml      # Extraction configuration
├── task_vector/                 # Averaged task vectors (output)
│   └── {session}/
│       ├── config.yaml
│       ├── metadata.json
│       ├── task_vectors.pt      # All averaged vectors
│       ├── layer_4.pt           # Per-layer files
│       ├── layer_8.pt
│       └── ...
└── raw_representations/         # Per-sample representations (output)
    └── {session}/
        ├── sample_00000.pt
        ├── sample_00001.pt
        └── ...
```

---

## Configuration

The extraction is controlled by `task_vector_config.yaml`:

### Token Positions

Specify which token's hidden state to extract:

```yaml
token_positions:
  - "last"      # Last token of prompt (most common)
  - "first"     # First token
  - -2          # Second to last token
  - -3          # Third to last token
```

| Position | Description | Use Case |
|----------|-------------|----------|
| `"last"` | Last token before generation | Standard task vector extraction |
| `"first"` | First token (often BOS) | Sentence-level representations |
| Negative int | Index from end | Multiple position analysis |
| Positive int | Index from start | Specific token targeting |

### Layers

Specify which transformer layers to extract from:

```yaml
layers:
  - 4
  - 8
  - 12
  - 16
  - 20
  - 24
  - 27  # Last layer (Qwen2.5-1.5B has 28 layers)
```

| Model | Total Layers | Recommended Extraction Layers |
|-------|--------------|------------------------------|
| Qwen2.5-1.5B | 28 | 4, 8, 12, 16, 20, 24, 27 |
| Qwen2.5-7B | 28 | 4, 8, 12, 16, 20, 24, 27 |

### Prompt Configuration

Reference existing prompt configs to ensure consistent prompting:

```yaml
prompt_config: "configs/qwen2.5-1.5b/cot_5shot_answer.yaml"
```

This ensures the same prompts used for evaluation are used for extraction.

---

## Output Format

### Task Vectors (`task_vector/{session}/`)

**`task_vectors.pt`**: Dictionary containing averaged representations

```python
{
    "layer_4_pos_last": tensor([...]),      # (hidden_size,)
    "layer_4_pos_last_std": tensor([...]),  # Standard deviation
    "layer_8_pos_last": tensor([...]),
    ...
}
```

**`layer_{N}.pt`**: Per-layer files for convenience

```python
{
    "layer_12_pos_last": tensor([...]),
    "layer_12_pos_last_std": tensor([...]),
}
```

**`metadata.json`**: Extraction metadata

```json
{
    "session_name": "cot_5shot_train_Qwen2.5-1.5B-Instruct_task_vector_20251127_160000",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "n_samples": 256,
    "layers": [4, 8, 12, 16, 20, 24, 27],
    "token_positions": ["last"],
    "prompt_type": "cot",
    "split": "train",
    "hidden_size": 1536,
    "sample_ids": ["train_0", "train_1", ...]
}
```

### Raw Representations (`raw_representations/{session}/`)

**`sample_{idx:05d}.pt`**: Per-sample hidden states

```python
{
    "layer_4_pos_last": tensor([...]),   # (hidden_size,)
    "layer_8_pos_last": tensor([...]),
    ...
    "_sample_id": "train_0",
    "_sample_idx": 0,
    "_seq_len": 1024,
    "_token_indices": [1023]
}
```

---

## Usage

### Basic Extraction

```bash
python task_vector_generation.py --config task_vector_config.yaml
```

### Custom Configuration

1. Create a custom config:

```yaml
# my_extraction_config.yaml
model: "Qwen/Qwen2.5-1.5B-Instruct"
token_positions:
  - "last"
  - -2
layers: [12, 24, 27]
prompt_config: "configs/qwen2.5-1.5b/vanilla_0shot_answer.yaml"
split: "train"
max_samples: 512
tag: "vanilla_multi_pos"
```

2. Run extraction:

```bash
python task_vector_generation.py --config my_extraction_config.yaml
```

---

## Loading Extracted Vectors

### Load Averaged Task Vectors

```python
import torch

# Load all task vectors
task_vectors = torch.load("task_vector/{session}/task_vectors.pt")

# Access specific layer/position
layer_12_vec = task_vectors["layer_12_pos_last"]  # (hidden_size,)
layer_12_std = task_vectors["layer_12_pos_last_std"]

print(f"Shape: {layer_12_vec.shape}")  # torch.Size([1536])
```

### Load Per-Layer Files

```python
# Load just layer 12
layer_12 = torch.load("task_vector/{session}/layer_12.pt")
```

### Load Raw Representations

```python
# Load single sample
sample = torch.load("raw_representations/{session}/sample_00000.pt")

# Access hidden state
hidden = sample["layer_12_pos_last"]
sample_id = sample["_sample_id"]
```

### Batch Load All Samples

```python
from pathlib import Path
import torch

session_dir = Path("raw_representations/{session}")
all_hiddens = []

for pt_file in sorted(session_dir.glob("sample_*.pt")):
    sample = torch.load(pt_file)
    all_hiddens.append(sample["layer_12_pos_last"])

# Stack into tensor
hiddens = torch.stack(all_hiddens)  # (n_samples, hidden_size)
```

---

## Analysis Examples

### Compare Task Vectors Across Prompting Strategies

```python
import torch

# Load task vectors from different configs
cot_vec = torch.load("task_vector/cot_5shot_.../task_vectors.pt")
vanilla_vec = torch.load("task_vector/vanilla_0shot_.../task_vectors.pt")

# Compute cosine similarity at layer 12
cot_l12 = cot_vec["layer_12_pos_last"]
van_l12 = vanilla_vec["layer_12_pos_last"]

cos_sim = torch.nn.functional.cosine_similarity(
    cot_l12.unsqueeze(0),
    van_l12.unsqueeze(0)
)
print(f"Cosine similarity (layer 12): {cos_sim.item():.4f}")
```

### Layer-wise Analysis

```python
import matplotlib.pyplot as plt

layers = [4, 8, 12, 16, 20, 24, 27]
similarities = []

for layer in layers:
    cot = cot_vec[f"layer_{layer}_pos_last"]
    van = vanilla_vec[f"layer_{layer}_pos_last"]
    sim = torch.nn.functional.cosine_similarity(cot.unsqueeze(0), van.unsqueeze(0))
    similarities.append(sim.item())

plt.plot(layers, similarities, marker='o')
plt.xlabel("Layer")
plt.ylabel("Cosine Similarity (CoT vs Vanilla)")
plt.title("Task Vector Similarity Across Layers")
plt.savefig("layer_similarity.png")
```

---

## Session Naming Convention

Output session names follow this pattern:

```
{tag}_{model_short}_task_vector_{timestamp}
```

Examples:
- `cot_5shot_train_Qwen2.5-1.5B-Instruct_task_vector_20251127_160000`
- `vanilla_0shot_Qwen2.5-7B-Instruct_task_vector_20251127_161500`

---

## Memory Considerations

| Setting | Memory Impact |
|---------|---------------|
| More layers | Linear increase |
| More token positions | Linear increase |
| More samples | Disk space for raw_representations |
| batch_size | Lower = less GPU memory |

**Recommendation**: For Qwen2.5-1.5B with 7 layers and 256 samples:
- Raw representations: ~2.5 GB
- Task vectors: ~50 MB

---

## Integration with Evaluation

The task vector extraction uses the **same prompt templates** as the evaluation pipeline (`run.py`). This ensures:

1. Consistent prompts between evaluation and representation extraction
2. Ability to correlate task vectors with model performance
3. Reproducible experiments across different prompt configurations

To extract task vectors for a specific evaluation config:

```yaml
# task_vector_config.yaml
prompt_config: "configs/qwen2.5-1.5b/cot_5shot_answer.yaml"  # Same as evaluation
split: "train"  # Use train set for task vectors
```
