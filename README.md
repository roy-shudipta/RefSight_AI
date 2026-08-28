# RefSight

This repository contains the analysis code for *RefSight: An AI framework for
explaining spatiotemporal risk in football refereeing*.

The workflow has three main stages:

1. Thirty-two spatiotemporal features are calculated from framewise tracking
   and event data.
2. Dynamic time warping (DTW) retrieves seven tactically similar correct
   decisions for each incorrect decision. The selected events are extracted as
   pre-event clips.
3. LSTM, TCN, and Transformer models rank the incorrect clip within each
   matched set. Calibration, explanation, aggregation, uncertainty, and
   statistical corroboration are then completed.

## Repository structure

```text
Feature Engineering/
    features.py                         Feature definitions
    run_all.py                          Batch feature creation
    README.md                           Input schema and feature dictionary
DTW and Clip Extraction/DTW/
    dtw_retrieval/                      DTW matching method
    run_dtw.py                          Retrieval entry point
    extract_clips.py                    Model-clip extraction
    README_DTW.md                       DTW and extraction instructions
Model Training/
    models/                             LSTM, TCN, and Transformer models
    train.py                            One model run
    run_seeds_explain.py                Three models by twenty seeds
    explanation_and_calibration/        Per-run analyses
    consensus_aggregation/              Cross-run aggregation
    analysis/                           Bootstrap and conditional logit analyses
    README.md                           Full modelling instructions
```

## Data availability

The licensed tracking and event data are not distributed with this repository.
DTW match tables, extracted clips, model checkpoints, logs, and result tables
are also not included. These files may contain restricted information and must
be stored according to the data provider's terms.

Local paths are supplied through command-line arguments or environment
variables. The `.gitignore` file excludes the expected data and result formats.

## Software environment

Python 3.11.7 was used for the study. From the repository root, one environment
can be prepared for the full workflow:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r "Feature Engineering/requirements.txt"
python -m pip install -r "DTW and Clip Extraction/DTW/requirements.txt"
python -m pip install -r "Model Training/requirements.txt"
```

## Workflow

All commands in this section are run from the repository root.

### 1. Create the spatiotemporal features

```bash
python "Feature Engineering/run_all.py" \
  --input-dir /path/to/raw_tracking \
  --output-dir /path/to/featured_tracking
```

One featured Parquet file is written for each input match. The input schema and
the complete feature dictionary are described in
[Feature Engineering/README.md](<Feature Engineering/README.md>).

### 2. Retrieve matched decisions with DTW

```bash
python "DTW and Clip Extraction/DTW/run_dtw.py" \
  --tracking-folder /path/to/featured_tracking \
  --out-matches /path/to/private_outputs/dtw_matches.csv \
  --top-k 7 \
  --band-fraction 0.10
```

### 3. Extract the matched pre-event clips

```bash
python "DTW and Clip Extraction/DTW/extract_clips.py" \
  --tracking-folder /path/to/featured_tracking \
  --matches-csv /path/to/private_outputs/dtw_matches.csv \
  --output-dir /path/to/private_data/time_series_clips \
  --top-k 7
```

The clip directory contains `metadata.csv`, `schema.json`, and a `sequences/`
folder. Further details are provided in
[DTW and Clip Extraction/DTW/README_DTW.md](<DTW and Clip Extraction/DTW/README_DTW.md>).

### 4. Train the models and run the per-model analyses

```bash
export REFSIGHT_TIME_SERIES_DIR=/path/to/private_data/time_series_clips
python "Model Training/run_seeds_explain.py" \
  --runs-root /path/to/private_results/runs
```

This command runs twenty seeds for each of the three model types. It also runs
the per-model permutation, calibration, risk-profile, and operating-range
methods.

### 5. Aggregate the trained models

```bash
python "Model Training/consensus_aggregation/aggregate_consensus_risk.py" \
  --runs-root /path/to/private_results/runs

python "Model Training/consensus_aggregation/aggregate_permutation.py" \
  --runs-root /path/to/private_results/runs

python "Model Training/consensus_aggregation/aggregate_operating_ranges.py" \
  --runs-root /path/to/private_results/runs
```

Bootstrap uncertainty and conditional logistic corroboration are described in
[Model Training/README.md](<Model Training/README.md>).

## Copyright

Copyright © 2026 Shudipta Roy. All rights reserved.

This code is provided for scholarly peer review and inspection. Access to this
repository does not grant permission to copy, modify, distribute, sublicense,
or reuse the code without prior written permission from the copyright holder,
except where permitted by applicable law or GitHub's Terms of Service.
