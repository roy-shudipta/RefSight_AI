# RefSight model training

This directory contains the listwise modelling and analysis workflow used in
RefSight.

Each matched query contains eight candidates:

```text
candidate 0      expert-labelled incorrect decision
candidates 1–7  DTW-matched correct decisions
```

The incorrect clip is trained to receive the highest score within its matched list.

## Input data

The clip directory is supplied through `REFSIGHT_TIME_SERIES_DIR`:

```bash
export REFSIGHT_TIME_SERIES_DIR=/path/to/time_series_clips
```

The directory must contain:

```text
metadata.csv
schema.json
sequences/
```

Each file in `sequences/` is a variable-length NumPy array with 32 columns.
`schema.json` records the exact feature order. `metadata.csv` must contain:

```text
query_id
decision_type
candidate_rank
file_path
n_frames
Referee Name
```

Feature names and public column identifiers are listed in the
[feature-engineering documentation](../Feature%20Engineering/README.md).

The code checks for 1,849 complete query lists. It also checks the reported
referee-level split of 989 training, 374 validation, and 486 test queries.

The licensed input data and generated result tables are not included in this repository.

## Environment

Python 3.11.7 was used in the study.

```bash
cd "Model Training"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training method

All candidates in a query are assigned the shortest valid candidate length,
capped at 375 frames. The event-aligned tail is retained. Shorter inputs are
right-padded. The 32 features are standardised with valid frames from the
training partition only. Missing and padded values are then set to zero in
standardised space.

The fixed architectures are:

| Model | Architecture | Parameters |
|---|---|---:|
| LSTM | Two bidirectional layers, 64 units per direction, dropout 0.1, masked attention pooling | 150,018 |
| TCN | Four causal residual blocks, 64 channels, kernel 5, dilations 1/2/4/8, dropout 0.5, masked attention pooling | 156,482 |
| Transformer | Four layers, width 128, eight heads, feed-forward width 256, sinusoidal positions, dropout 0.1, masked attention pooling | 534,658 |

Listwise softmax cross-entropy is used with candidate 0 as the target and
temperature 0.5. Adam is used with learning rate `3e-4`, weight decay `5e-3`,
batch size 16, and gradient clipping at norm 5. Training lasts 50 epochs. The
checkpoint with the highest validation Top 1 accuracy is retained.

One model can be trained as follows:

```bash
export REFSIGHT_MODEL_NAME=transformer
export REFSIGHT_SEED=1
export REFSIGHT_RESULTS_DIR=/path/to/results/transformer/seed_001
python train.py
```

The model name must be `lstm`, `tcn`, or `transformer`.

## Sixty model runs

The complete reliability analysis uses seeds 1–20 for all three architectures:

```bash
python run_seeds_explain.py --runs-root /path/to/results/runs
```

For each run, the model is trained and the following analyses are completed:

- Full-series permutation is repeated five times for every feature. The mean decrease in held-out Top 1 accuracy is reported in percentage points.
- The same permutation is applied in moving 2-second event-aligned windows for every feature.
- Isotonic calibration is fitted on validation scores and applied to test scores.
- Real-unit feature means are calculated over the final 5 seconds and divided into six quantile bins.
- A directional operating-range threshold is retained when its risk step is at least 0.5 percentage points. Bins with fewer than 30 clips receive zero weight.

Run outputs are written to `results/runs/<model>/seed_XXX/` unless another root is supplied.

## Consensus analyses

The consensus contextual risk is calculated after all 60 runs are complete.
Unless `--out-dir` is supplied, aggregate outputs are written under
`results/aggregate/` in this directory.

```bash
python consensus_aggregation/aggregate_consensus_risk.py \
  --runs-root /path/to/results/runs
```

Calibrated scores are averaged across seeds within each architecture. The three architecture means are then averaged. The script calculates ECE with 10 equal-width bins, Brier score, AUROC, and the retrospective triage curve.

Permutation results are pooled with the same two-stage rule:

```bash
python consensus_aggregation/aggregate_permutation.py \
  --runs-root /path/to/results/runs
```

Operating ranges are pooled across seeds and model types:

```bash
python consensus_aggregation/aggregate_operating_ranges.py \
  --runs-root /path/to/results/runs
```

Within each model type, medians are taken across seeds. A direction is supported by a model type when at least 70% of its seeds select that direction. A final operating range is supported when at least two of the three model types agree.

## Uncertainty and statistical corroboration

Consensus uncertainty is estimated with 5,000 query-level bootstrap resamples. Whole matched lists are resampled together:

```bash
python analysis/bootstrap_uncertainty.py --bootstrap 5000
```

The model-independent corroboration uses all 1,849 matched sets. Each eligible feature is fitted in a separate conditional logistic regression. Referee-level cluster bootstrap uncertainty uses 1,000 resamples. Benjamini–Hochberg correction is applied across features. Team-shape and ball-position matching features are excluded from this analysis.

```bash
python analysis/conditional_logit_corroboration.py --bootstrap 1000
```

## Main files

```text
config.py                         Fixed paths, features, models, and training settings
data_prep.py                      Referee split, matched lists, and training-only scaling
dataset.py                        Shared-length trimming, padding, and tensor loading
engine.py                         Listwise training and ranking metrics
models/                           LSTM, TCN, and Transformer encoders
run_seeds_explain.py              The 3 × 20 run workflow
explanation_and_calibration/      Per-run permutation, calibration, and thresholds
consensus_aggregation/            Cross-seed and cross-model aggregation
analysis/                         Bootstrap and conditional-logit analyses
```

All generated checkpoints and result files should remain outside version control.
