#!/bin/bash
# Download FinQA dataset (czyssrs/FinQA)
# Paper: https://arxiv.org/abs/2109.00122
# Source: https://github.com/czyssrs/FinQA
#
# Usage:
#   ./download_finqa.sh                    # Download full dataset
#   ./download_finqa.sh --n_steps 3        # Filter to 3-step examples only
#   ./download_finqa.sh --merge_to_qa      # Merge all splits into qa.json
#   ./download_finqa.sh --n_steps 3 --merge_to_qa  # Both options

set -e
cd "$(dirname "$0")/.."

# Parse arguments
N_STEPS=""
MERGE_TO_QA="False"

while [[ $# -gt 0 ]]; do
    case $1 in
        --n_steps)
            N_STEPS="$2"
            shift 2
            ;;
        --merge_to_qa)
            MERGE_TO_QA="True"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--n_steps N] [--merge_to_qa]"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Downloading FinQA Dataset"
echo "Source: https://github.com/czyssrs/FinQA"
if [ -n "$N_STEPS" ]; then
    echo "Filter: ${N_STEPS}-step examples only"
fi
if [ "$MERGE_TO_QA" = "True" ]; then
    echo "Output: Merging to qa.json"
fi
echo "============================================"

# Use full python path for compatibility
PYTHON="${PYTHON:-/home/yeopjin/orcd/pool/conda_install/envs/eelma/bin/python}"

# Build Python arguments
if [ -n "$N_STEPS" ]; then
    N_STEPS_ARG="n_steps=$N_STEPS"
else
    N_STEPS_ARG="n_steps=None"
fi

$PYTHON -c "
import logging
logging.basicConfig(level=logging.INFO)
from src.datasets.finqa import download_finqa
download_finqa($N_STEPS_ARG, merge_to_qa=$MERGE_TO_QA)
"

echo ""
echo "Done! Data saved to data/finqa/"
ls -la data/finqa/
