# DTW Retrieval and Clip Extraction

This folder contains the method used to form the matched decision sets in
RefSight. Seven tactically similar correct decisions are retrieved for each
incorrect decision. The selected events are then converted into the
variable-length clips used by the sequence models.

Tracking data, retrieved-match tables, extracted arrays, and model results are
not included. These files must be supplied by an authorised user.

## Code structure

```text
DTW/
|-- dtw_retrieval/     DTW retrieval method
|-- run_dtw.py         DTW retrieval entry point
|-- extract_clips.py   Clip-extraction entry point
|-- requirements.txt
`-- README_DTW.md
```

## Pre-event windows

Only live-play rows are expected from the feature-engineering stage. Each window
ends immediately before the decision event. Up to 15 seconds are retained at
25 Hz. At least 2 seconds are required. A window is stopped at a frame or time
discontinuity.

The featured files must retain the match, frame, event, broad event type, and
referee fields listed in the feature-engineering documentation.

## Matching features

DTW matching is based on team organisation and ball position. Referee movement
is not used. The following variables are included:

- Home and away team centroid coordinates.
- Home and away team length and width.
- Home and away team stretch index.
- Home and away team surface area.
- Ball x and y position.

The team columns use the public names defined in the
[feature-engineering documentation](../../Feature%20Engineering/README.md).

These variables are standardised with statistics fitted on correct-decision
windows.

## Candidate retrieval

Correct candidates are restricted to the same referee and broad event type as
the incorrect query. Physical, technical, and disciplinary events are eligible.
Offside events are excluded.

Exact multivariate DTW is calculated for eligible candidates from the same
match. Cross-match candidates are processed in two phases. Exact DTW is first
calculated for a reproducibly seeded sample. This establishes a distance
threshold. LB_Keogh is then used to remove candidates that cannot enter the
retained set. Exact DTW is calculated for every surviving candidate.

A Sakoe-Chiba band is applied. Its width is
`ceil(0.10 * max(query_length, candidate_length))`. Euclidean distance is used
as the frame-level cost. Within-match and cross-match candidates are merged and
ranked. The seven candidates with the smallest DTW distances are retained.

One Numba implementation is used for DTW calculation.

## Installation

Python 3.11 is recommended.
The following commands are run from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r "DTW and Clip Extraction/DTW/requirements.txt"
```

## Run DTW retrieval

```bash
python "DTW and Clip Extraction/DTW/run_dtw.py" \
  --tracking-folder /path/to/featured_tracking \
  --out-matches /path/to/private_outputs/dtw_matches.csv \
  --top-k 7 \
  --band-fraction 0.10
```

The input and output paths may also be provided through environment variables:

```bash
export REFSIGHT_FEATURED_TRACKING_DIR=/path/to/featured_tracking
export REFSIGHT_DTW_MATCHES_CSV=/path/to/private_outputs/dtw_matches.csv
python "DTW and Clip Extraction/DTW/run_dtw.py"
```

## Extract model clips

```bash
python "DTW and Clip Extraction/DTW/extract_clips.py" \
  --tracking-folder /path/to/featured_tracking \
  --matches-csv /path/to/private_outputs/dtw_matches.csv \
  --output-dir /path/to/private_data/time_series_clips \
  --top-k 7
```

Each complete query list contains one incorrect clip and seven correct clips.
The 32 model variables are written in the order defined by
`MODEL_FEATURE_COLUMNS`. Missing feature values are preserved. Model scaling and
fixed-length padding are applied later during model training.

The generated CSV and NumPy files may contain restricted information. They
should remain outside the public repository.
