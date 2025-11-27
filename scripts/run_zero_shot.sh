#!/bin/bash
# Run zero-shot evaluation

set -e

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
MAX_SAMPLES=${2:-100}

echo "Running zero-shot evaluation with $MODEL"

python run.py \
    --model "$MODEL" \
    --n_shots 0 \
    --max_samples "$MAX_SAMPLES" \
    --batch_size 8 \
    --output_dir outputs
