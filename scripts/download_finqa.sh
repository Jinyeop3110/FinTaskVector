#!/bin/bash
# Download FinQA dataset (czyssrs/FinQA)
# Paper: https://arxiv.org/abs/2109.00122
# Source: https://github.com/czyssrs/FinQA

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "Downloading FinQA Dataset"
echo "Source: https://github.com/czyssrs/FinQA"
echo "============================================"

# Use full python path for compatibility
PYTHON="${PYTHON:-/home/yeopjin/orcd/pool/conda_install/envs/eelma/bin/python}"

$PYTHON -c "
import logging
logging.basicConfig(level=logging.INFO)
from src.datasets.finqa import download_finqa
download_finqa()
"

echo ""
echo "Done! Data saved to data/finqa/"
ls -la data/finqa/
