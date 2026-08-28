#!/usr/bin/env python3
"""Calibrated scores are combined into the consensus contextual risk."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_TRAINING_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("lstm", "tcn", "transformer")
EXPECTED_SEEDS = set(range(1, 21))
CALIBRATION_BINS = 10
PREDICTION_FILE = "clip_risk_predictions_test.csv"
EXPECTED_TEST_QUERIES = 486
EXPECTED_CANDIDATES = 8


def _resolve(path: str | Path) -> Path:
    """A path is resolved from the Model Training directory."""
    path = Path(path).expanduser()
    return (
        path.resolve() if path.is_absolute() else (MODEL_TRAINING_ROOT / path).resolve()
    )


def load_predictions(runs_root: Path) -> pd.DataFrame:
    """Calibrated test predictions are loaded from all 60 trained models."""
    rows = []
    for model_name in MODEL_NAMES:
        model_files = sorted((runs_root / model_name).glob(f"seed_*/{PREDICTION_FILE}"))
        seeds = {
            int(re.search(r"seed_(\d+)", str(file_path)).group(1))
            for file_path in model_files
        }
        if seeds != EXPECTED_SEEDS or len(model_files) != len(EXPECTED_SEEDS):
            raise ValueError(
                f"{model_name} does not contain calibrated predictions for seeds 1 to 20."
            )

        for file_path in model_files:
            predictions = pd.read_csv(file_path)
            required = {
                "query_id",
                "candidate_rank",
                "decision_type",
                "file_path",
                "n_frames",
                "p_incorrect",
            }
            missing = sorted(required - set(predictions.columns))
            if missing:
                raise ValueError(
                    f"Prediction columns are missing from {file_path}: {missing}"
                )
            predictions = predictions.loc[:, sorted(required)].copy()
            predictions["p_incorrect"] = pd.to_numeric(
                predictions["p_incorrect"],
                errors="coerce",
            )
            predictions["n_frames"] = pd.to_numeric(
                predictions["n_frames"],
                errors="coerce",
            )
            if predictions[["p_incorrect", "n_frames"]].isna().any().any():
                raise ValueError(f"Invalid numeric values were found in {file_path}.")

            predictions["model"] = model_name
            predictions["seed"] = int(re.search(r"seed_(\d+)", str(file_path)).group(1))
            rows.append(predictions)
    return pd.concat(rows, ignore_index=True)


def build_consensus(predictions: pd.DataFrame) -> pd.DataFrame:
    """Scores are averaged across seeds and then across model types."""
    key_columns = ["query_id", "candidate_rank"]
    expected_clips = None
    for (_, _), run in predictions.groupby(["model", "seed"]):
        clip_keys = set(map(tuple, run[key_columns].to_numpy()))
        if expected_clips is None:
            expected_clips = clip_keys
        elif clip_keys != expected_clips:
            raise ValueError("The held-out clip set is not identical across all runs.")

    per_model = (
        predictions.groupby(["model", *key_columns], as_index=False)["p_incorrect"]
        .mean()
        .rename(columns={"p_incorrect": "model_risk"})
    )
    wide = per_model.pivot(
        index=key_columns,
        columns="model",
        values="model_risk",
    )
    if wide.isna().any().any() or set(wide.columns) != set(MODEL_NAMES):
        raise ValueError("All three model types are required for every held-out clip.")

    wide = wide.rename(columns={name: f"risk_{name}" for name in MODEL_NAMES})
    risk_columns = [f"risk_{name}" for name in MODEL_NAMES]
    wide["consensus_risk"] = wide[risk_columns].mean(axis=1)

    metadata_columns = ["decision_type", "file_path", "n_frames"]
    metadata_variation = predictions.groupby(key_columns)[metadata_columns].nunique(
        dropna=False
    )
    if metadata_variation.gt(1).any().any():
        raise ValueError("Held-out clip metadata is not identical across all runs.")
    metadata = predictions.drop_duplicates(key_columns).set_index(key_columns)[
        metadata_columns
    ]
    consensus = wide.join(metadata).reset_index()
    rank_sets = consensus.groupby("query_id")["candidate_rank"].apply(
        lambda ranks: set(pd.to_numeric(ranks, errors="coerce"))
    )
    expected_ranks = set(range(EXPECTED_CANDIDATES))
    if (
        consensus["query_id"].nunique() != EXPECTED_TEST_QUERIES
        or not rank_sets.map(lambda ranks: ranks == expected_ranks).all()
    ):
        raise ValueError(
            "The expected 486 complete held-out query lists were not found."
        )
    consensus["y"] = (
        consensus["decision_type"].astype(str).str.lower().eq("incorrect")
    ).astype(int)
    return consensus


def calibration_summary(
    consensus: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """ECE, Brier score, and AUROC are calculated on the test clips."""
    probabilities = consensus["consensus_risk"].to_numpy(dtype=float)
    labels = consensus["y"].to_numpy(dtype=int)
    edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    bin_ids = np.digitize(probabilities, edges[1:-1], right=True)

    rows = []
    ece = 0.0
    for bin_id in range(CALIBRATION_BINS):
        selected = bin_ids == bin_id
        count = int(selected.sum())
        if count == 0:
            continue
        predicted = float(probabilities[selected].mean())
        observed = float(labels[selected].mean())
        ece += (count / len(labels)) * abs(predicted - observed)
        rows.append(
            {
                "bin": bin_id,
                "lo": float(edges[bin_id]),
                "hi": float(edges[bin_id + 1]),
                "n": count,
                "mean_predicted_risk": predicted,
                "observed_incorrect_rate": observed,
            }
        )

    from sklearn.metrics import roc_auc_score

    metrics = {
        "n_clips": int(len(labels)),
        "n_queries": int(consensus["query_id"].nunique()),
        "base_incorrect_rate": float(labels.mean()),
        "ece": float(ece),
        "brier": float(np.mean(np.square(probabilities - labels))),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "calibration_bins": CALIBRATION_BINS,
        "interpretation": "Contextual risk within the matched comparison design.",
    }
    return pd.DataFrame(rows), metrics


def triage_curve(consensus: pd.DataFrame) -> pd.DataFrame:
    """Incorrect-decision recovery is calculated across review fractions."""
    ordered = consensus.sort_values(
        ["consensus_risk", "query_id", "candidate_rank"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    reviewed = np.arange(1, len(ordered) + 1)
    total_incorrect = int(ordered["y"].sum())
    return pd.DataFrame(
        {
            "clips_reviewed": reviewed,
            "review_fraction": reviewed / len(ordered),
            "incorrect_recovered": ordered["y"].cumsum(),
            "incorrect_recovery_fraction": ordered["y"].cumsum() / total_incorrect,
        }
    )


def main() -> None:
    """Consensus, calibration, and triage tables are written."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--out-dir", default="results/aggregate/consensus_risk")
    arguments = parser.parse_args()

    runs_root = _resolve(arguments.runs_root)
    output_dir = _resolve(arguments.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    consensus = build_consensus(load_predictions(runs_root))
    reliability, metrics = calibration_summary(consensus)
    triage = triage_curve(consensus)

    consensus.to_csv(output_dir / "consensus_clip_risk.csv", index=False)
    reliability.to_csv(output_dir / "consensus_reliability_table.csv", index=False)
    triage.to_csv(output_dir / "consensus_triage_curve.csv", index=False)
    (output_dir / "consensus_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"Consensus outputs were saved in {output_dir}.", flush=True)


if __name__ == "__main__":
    main()
