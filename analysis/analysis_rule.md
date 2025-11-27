# Analysis Rules

## Folder Structure

```
analysis/
├── code/        # All analysis scripts
├── plot/        # Generated plots and figures
├── raw_data/    # Raw data files for processing
└── analysis_rule.md
```

---

## Workflow

### Step 1: Raw Data Processing

- Place raw data files in `raw_data/`
- Create data processing scripts in `code/`
- Naming convention: `process_<data_name>.py`

### Step 2: Plotting Code

- Create plotting scripts in `code/`
- Naming convention: `plot_<figure_name>.py`
- Scripts should read processed data and generate figures

### Step 3: Generate Plots

- All generated plots should be saved to `plot/`
- Naming convention: `<figure_name>.<format>` (e.g., `accuracy_comparison.png`)

---

## Code Organization Rules

### For each analysis task, create separate files:

1. **Data Processing Code** (`code/`)
   ```
   code/
   ├── process_metrics.py
   ├── process_predictions.py
   └── ...
   ```

2. **Plotting Code** (`code/`)
   ```
   code/
   ├── plot_accuracy.py
   ├── plot_comparison.py
   └── ...
   ```

3. **Output Plots** (`plot/`)
   ```
   plot/
   ├── accuracy_comparison.png
   ├── operation_breakdown.png
   └── ...
   ```

---

## Example Pipeline

```bash
# 1. Process raw data
python code/process_metrics.py --input raw_data/results.json --output raw_data/processed.csv

# 2. Generate plots
python code/plot_accuracy.py --input raw_data/processed.csv --output plot/accuracy.png
```

---

## Naming Conventions

| Type | Location | Pattern | Example |
|------|----------|---------|---------|
| Raw data | `raw_data/` | `<source>_<type>.<ext>` | `finqa_metrics.json` |
| Processing script | `code/` | `process_<name>.py` | `process_metrics.py` |
| Plotting script | `code/` | `plot_<name>.py` | `plot_accuracy.py` |
| Output figure | `plot/` | `<name>.<format>` | `accuracy_comparison.png` |

---

## Best Practices

1. **Modularity**: Keep processing and plotting code separate
2. **Reproducibility**: Scripts should be runnable with clear inputs/outputs
3. **Documentation**: Add docstrings and comments to scripts
4. **Version Control**: Track code changes, not raw data or generated plots
