#!/bin/bash
# Run few-shot ICL evaluation

set -e

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
N_SHOTS=${2:-3}
MAX_SAMPLES=${3:-100}

echo "Running ${N_SHOTS}-shot evaluation with $MODEL"

python run.py \
    --model "$MODEL" \
    --n_shots "$N_SHOTS" \
    --max_samples "$MAX_SAMPLES" \
    --batch_size 8 \
    --output_dir outputs
