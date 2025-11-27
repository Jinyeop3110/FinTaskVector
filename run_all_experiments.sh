#!/bin/bash
# Master script to run all FinQA experiments using config files
# Uses 3-step filtered data from qa.json

set -e

cd /home/yeopjin/orcd/pool/workspace/Financial_task_vector

# Configuration
PYTHON="/home/yeopjin/orcd/pool/conda_install/envs/eelma/bin/python"
CONFIG_DIR="configs/qwen2.5-1.5b"

# Output directory for logs
LOG_DIR="logs/experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "FinQA Experiment Suite (3-step filtered)"
echo "=========================================="
echo "Config directory: $CONFIG_DIR"
echo "Log directory: $LOG_DIR"
echo "=========================================="

# Function to run experiment on specific GPU
run_experiment() {
    local gpu=$1
    local config=$2
    local tag=$(basename "$config" .yaml)
    local log_file="$LOG_DIR/${tag}.log"

    echo "[GPU $gpu] Starting: $tag"

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON run.py \
        --config "$config" \
        --tensor_parallel 1 \
        > "$log_file" 2>&1

    local status=$?
    if [ $status -eq 0 ]; then
        echo "[GPU $gpu] Completed: $tag"
    else
        echo "[GPU $gpu] FAILED: $tag (see $log_file)"
    fi
    return $status
}

# Export function and variables for parallel execution
export -f run_experiment
export PYTHON LOG_DIR

echo ""
echo "Running experiments in parallel across 4 GPUs..."
echo ""

# Batch 1: GPU 0-3 (vanilla + cot 0-2 shot)
echo "=== Batch 1/2 ==="
run_experiment 0 "$CONFIG_DIR/vanilla_0shot.yaml" &
PID0=$!
run_experiment 1 "$CONFIG_DIR/cot_0shot.yaml" &
PID1=$!
run_experiment 2 "$CONFIG_DIR/cot_1shot.yaml" &
PID2=$!
run_experiment 3 "$CONFIG_DIR/cot_2shot.yaml" &
PID3=$!

# Wait for batch 1
wait $PID0 $PID1 $PID2 $PID3
echo "Batch 1 completed."
echo ""

# Batch 2: GPU 0-2 (cot 3-5 shot)
echo "=== Batch 2/2 ==="
run_experiment 0 "$CONFIG_DIR/cot_3shot.yaml" &
PID0=$!
run_experiment 1 "$CONFIG_DIR/cot_4shot.yaml" &
PID1=$!
run_experiment 2 "$CONFIG_DIR/cot_5shot.yaml" &
PID2=$!

# Wait for batch 2
wait $PID0 $PID1 $PID2
echo "Batch 2 completed."
echo ""

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Logs saved to: $LOG_DIR"
echo ""

# Generate summary
echo "Generating results summary..."
$PYTHON - << 'PYTHON_SCRIPT'
import json
from pathlib import Path

outputs_dir = Path("outputs")
results = []

# Find all experiment directories
for exp_dir in sorted(outputs_dir.glob("*")):
    if not exp_dir.is_dir():
        continue

    # Check for metrics
    metrics_file = exp_dir / "metrics.json"
    agg_metrics_file = exp_dir / "aggregate_metrics.json"

    if agg_metrics_file.exists():
        with open(agg_metrics_file) as f:
            metrics = json.load(f)
        acc = metrics.get("mean_accuracy", 0)
        std = metrics.get("std_accuracy", 0)
        n_runs = metrics.get("n_runs", 1)
        if "per_run_metrics" in metrics and metrics["per_run_metrics"]:
            inf_stats = metrics["per_run_metrics"][0].get("inference_stats", {})
        else:
            inf_stats = {}
    elif metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        acc = metrics.get("execution_accuracy", 0)
        std = 0
        n_runs = 1
        inf_stats = metrics.get("inference_stats", {})
    else:
        continue

    name = exp_dir.name

    results.append({
        "name": name,
        "accuracy": acc,
        "std": std,
        "n_runs": n_runs,
        "latency": inf_stats.get("total_latency_seconds", 0),
        "input_tokens": inf_stats.get("avg_input_tokens_per_sample", 0),
        "output_tokens": inf_stats.get("avg_output_tokens_per_sample", 0),
    })

# Print summary table
print("\n" + "="*100)
print("EXPERIMENT RESULTS SUMMARY")
print("="*100)
print(f"{'Experiment':<50} {'Accuracy':<15} {'Latency':<10} {'In Tok':<10} {'Out Tok':<10}")
print("-"*100)

for r in results:
    acc_str = f"{r['accuracy']*100:.2f}% ± {r['std']*100:.2f}%" if r['std'] > 0 else f"{r['accuracy']*100:.2f}%"
    print(f"{r['name'][:50]:<50} {acc_str:<15} {r['latency']:.1f}s{'':<5} {r['input_tokens']:.0f}{'':<6} {r['output_tokens']:.0f}")

print("="*100)
PYTHON_SCRIPT

echo ""
echo "Done! Results are in the outputs/ directory."
