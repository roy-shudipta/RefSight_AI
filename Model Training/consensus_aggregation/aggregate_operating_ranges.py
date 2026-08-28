#!/usr/bin/env python3
"""Operating ranges and calibrated risk curves are pooled across all runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_TRAINING_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("lstm", "tcn", "transformer")
EXPECTED_SEEDS = set(range(1, 21))
SEED_SUPPORT_THRESHOLD = 0.70
MINIMUM_SUPPORTING_MODELS = 2


def _resolve(path: str | Path) -> Path:
    """A path is resolved from the Model Training directory."""
    path = Path(path).expanduser()
    return (
        path.resolve() if path.is_absolute() else (MODEL_TRAINING_ROOT / path).resolve()
    )


def _seed_from_path(path: Path) -> int:
    """A seed number is read from a run directory name."""
    match = re.search(r"seed_(\d+)", str(path))
    if match is None:
        raise ValueError(f"A seed number was not found in {path}.")
    return int(match.group(1))


def load_run_outputs(runs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thresholds and risk profiles are loaded from all 60 runs."""
    thresholds = []
    profiles = []
    for model_name in MODEL_NAMES:
        run_directories = sorted((runs_root / model_name).glob("seed_*"))
        seeds = {_seed_from_path(path) for path in run_directories if path.is_dir()}
        if seeds != EXPECTED_SEEDS or len(run_directories) != len(EXPECTED_SEEDS):
            raise ValueError(f"{model_name} does not contain seeds 1 to 20.")

        for run_directory in run_directories:
            seed = _seed_from_path(run_directory)
            threshold_path = (
                run_directory
                / "thresholds"
                / "operating_ranges"
                / "operating_range_thresholds.csv"
            )
            profile_path = run_directory / "risk_profiles_all_features.csv"
            if not threshold_path.exists() or not profile_path.exists():
                raise FileNotFoundError(
                    f"Operating-range outputs are missing in {run_directory}."
                )

            threshold_table = pd.read_csv(threshold_path)
            threshold_table["model"] = model_name
            threshold_table["seed"] = seed
            thresholds.append(threshold_table)

            profile_table = pd.read_csv(profile_path)
            profile_table["model"] = model_name
            profile_table["seed"] = seed
            profile_table["midpoint"] = 0.5 * (
                pd.to_numeric(profile_table["lo"], errors="coerce")
                + pd.to_numeric(profile_table["hi"], errors="coerce")
            )
            profiles.append(profile_table)

    return pd.concat(thresholds, ignore_index=True), pd.concat(
        profiles, ignore_index=True
    )


def pool_thresholds(thresholds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Threshold medians and directional support are calculated in two stages."""
    model_rows = []
    for (model_name, feature), group in thresholds.groupby(["model", "feature"]):
        for direction in ("low", "high"):
            selected = group.loc[group["direction"].eq(direction)]
            support = len(selected) / len(EXPECTED_SEEDS)
            model_rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "direction": direction,
                    "seed_support": support,
                    "supporting_seeds": int(len(selected)),
                    "threshold_median": float(selected["threshold"].median()),
                    "delta_risk_pp_median": float(selected["delta_risk_pp"].median()),
                    "model_supports_direction": support >= SEED_SUPPORT_THRESHOLD,
                }
            )
    by_model = pd.DataFrame(model_rows)

    pooled_rows = []
    for (feature, direction), group in by_model.groupby(["feature", "direction"]):
        supported = group.loc[group["model_supports_direction"]]
        supporting_models = int(len(supported))
        pooled_rows.append(
            {
                "feature": feature,
                "direction": direction,
                "threshold": float(supported["threshold_median"].median()),
                "delta_risk_pp": float(supported["delta_risk_pp_median"].median()),
                "supporting_model_types": supporting_models,
                "total_model_types": len(MODEL_NAMES),
                "supported_operating_range": (
                    supporting_models >= MINIMUM_SUPPORTING_MODELS
                ),
            }
        )
    return by_model, pd.DataFrame(pooled_rows)


def pool_risk_curves(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrated risks are averaged across seeds and then across model types."""
    by_model = profiles.groupby(["model", "feature", "bin"], as_index=False).agg(
        midpoint=("midpoint", "median"),
        risk=("risk_mean", "mean"),
        n=("n", "median"),
    )
    pooled = by_model.groupby(["feature", "bin"], as_index=False).agg(
        midpoint=("midpoint", "median"),
        consensus_risk=("risk", "mean"),
        n=("n", "median"),
    )
    return by_model, pooled


def main() -> None:
    """Pooled operating-range and risk-curve tables are written."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--out-dir", default="results/aggregate/operating_ranges")
    arguments = parser.parse_args()

    output_dir = _resolve(arguments.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds, profiles = load_run_outputs(_resolve(arguments.runs_root))
    by_model_thresholds, pooled_thresholds = pool_thresholds(thresholds)
    by_model_curves, pooled_curves = pool_risk_curves(profiles)

    by_model_thresholds.to_csv(
        output_dir / "thresholds_by_model_direction.csv",
        index=False,
    )
    pooled_thresholds.to_csv(
        output_dir / "thresholds_pooled_direction.csv",
        index=False,
    )
    by_model_curves.to_csv(output_dir / "risk_curves_by_model.csv", index=False)
    pooled_curves.to_csv(output_dir / "risk_curves_consensus.csv", index=False)
    print(f"Operating-range outputs were saved in {output_dir}.", flush=True)


if __name__ == "__main__":
    main()
