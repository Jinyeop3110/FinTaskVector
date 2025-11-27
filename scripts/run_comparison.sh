#!/bin/bash
# Run comparison: zero-shot vs few-shot

set -e

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
MAX_SAMPLES=${2:-100}

echo "============================================"
echo "Running comparison experiment"
echo "Model: $MODEL"
echo "Max samples: $MAX_SAMPLES"
echo "============================================"

echo ""
echo ">>> Zero-shot evaluation"
python run.py \
    --model "$MODEL" \
    --n_shots 0 \
    --max_samples "$MAX_SAMPLES" \
    --output_dir outputs

echo ""
echo ">>> 1-shot evaluation"
python run.py \
    --model "$MODEL" \
    --n_shots 1 \
    --max_samples "$MAX_SAMPLES" \
    --output_dir outputs

echo ""
echo ">>> 3-shot evaluation"
python run.py \
    --model "$MODEL" \
    --n_shots 3 \
    --max_samples "$MAX_SAMPLES" \
    --output_dir outputs

echo ""
echo ">>> 5-shot evaluation"
python run.py \
    --model "$MODEL" \
    --n_shots 5 \
    --max_samples "$MAX_SAMPLES" \
    --output_dir outputs

echo ""
echo "============================================"
echo "Comparison complete! Check outputs/ for results"
echo "============================================"
