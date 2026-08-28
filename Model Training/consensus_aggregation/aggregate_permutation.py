#!/usr/bin/env python3
"""Full-series and windowed permutation results are pooled across all runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


MODEL_TRAINING_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("lstm", "tcn", "transformer")
EXPECTED_SEEDS = set(range(1, 21))
TOP_FEATURE_COUNT = 10
SEED_SUPPORT_THRESHOLD = 0.70
FPS = 25
GLOBAL_FILE = "permutation_importance_listwise_top1.csv"
WINDOW_FILE = "windowed_permutation_surface_top1.csv"


def _resolve(path: str | Path) -> Path:
    """A path is resolved from the Model Training directory."""
    path = Path(path).expanduser()
    return (
        path.resolve() if path.is_absolute() else (MODEL_TRAINING_ROOT / path).resolve()
    )


def _seed_from_path(path: Path) -> int:
    """A seed number is read from a run path."""
    match = re.search(r"seed_(\d+)", str(path))
    if match is None:
        raise ValueError(f"A seed number was not found in {path}.")
    return int(match.group(1))


def load_results(runs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permutation tables are loaded from all 60 runs."""
    global_rows = []
    window_rows = []
    for model_name in MODEL_NAMES:
        run_directories = sorted((runs_root / model_name).glob("seed_*"))
        seeds = {_seed_from_path(path) for path in run_directories if path.is_dir()}
        if seeds != EXPECTED_SEEDS or len(run_directories) != len(EXPECTED_SEEDS):
            raise ValueError(f"{model_name} does not contain seeds 1 to 20.")

        for run_directory in run_directories:
            seed = _seed_from_path(run_directory)
            global_path = run_directory / GLOBAL_FILE
            window_path = run_directory / WINDOW_FILE
            if not global_path.exists() or not window_path.exists():
                raise FileNotFoundError(
                    f"Permutation outputs are missing in {run_directory}."
                )

            global_table = pd.read_csv(global_path).loc[:, ["feature", "drop"]]
            global_table["drop"] = pd.to_numeric(global_table["drop"], errors="coerce")
            global_table["model"] = model_name
            global_table["seed"] = seed
            global_rows.append(global_table.dropna(subset=["drop"]))

            window_table = pd.read_csv(window_path).loc[
                :, ["feature", "offset_frames_before_event", "drop"]
            ]
            window_table["drop"] = pd.to_numeric(window_table["drop"], errors="coerce")
            window_table["offset_frames_before_event"] = pd.to_numeric(
                window_table["offset_frames_before_event"],
                errors="coerce",
            )
            window_table["model"] = model_name
            window_table["seed"] = seed
            window_rows.append(
                window_table.dropna(subset=["drop", "offset_frames_before_event"])
            )

    return pd.concat(global_rows, ignore_index=True), pd.concat(
        window_rows, ignore_index=True
    )


def pool_global(global_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Global importance is pooled across seeds and model types."""
    ranked = global_results.sort_values(
        ["model", "seed", "drop"],
        ascending=[True, True, False],
    ).copy()
    ranked["top_feature"] = (
        ranked.groupby(["model", "seed"]).cumcount() < TOP_FEATURE_COUNT
    )
    by_model = ranked.groupby(["model", "feature"], as_index=False).agg(
        median_drop_pp=("drop", "median"),
        top_feature_seed_support=("top_feature", "mean"),
    )
    by_model["model_supports_feature"] = (
        by_model["top_feature_seed_support"] >= SEED_SUPPORT_THRESHOLD
    )
    consensus = (
        by_model.groupby("feature", as_index=False)
        .agg(
            pooled_median_drop_pp=("median_drop_pp", "median"),
            supporting_model_types=("model_supports_feature", "sum"),
        )
        .sort_values("pooled_median_drop_pp", ascending=False)
    )
    consensus["total_model_types"] = len(MODEL_NAMES)
    return by_model, consensus


def pool_windowed(
    window_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Windowed importance is pooled across seeds and model types."""
    by_model = (
        window_results.groupby(
            ["model", "feature", "offset_frames_before_event"],
            as_index=False,
        )["drop"]
        .median()
        .rename(columns={"drop": "median_drop_pp"})
    )
    consensus = (
        by_model.groupby(
            ["feature", "offset_frames_before_event"],
            as_index=False,
        )["median_drop_pp"]
        .median()
        .rename(columns={"median_drop_pp": "pooled_median_drop_pp"})
    )
    consensus["seconds_before_event"] = consensus["offset_frames_before_event"] / FPS
    peak_indices = consensus.groupby("feature")["pooled_median_drop_pp"].idxmax()
    peaks = consensus.loc[peak_indices].sort_values(
        "pooled_median_drop_pp",
        ascending=False,
    )
    return by_model, consensus, peaks


def main() -> None:
    """Pooled permutation tables are written."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--out-dir", default="results/aggregate/permutation")
    arguments = parser.parse_args()

    output_dir = _resolve(arguments.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_results, window_results = load_results(_resolve(arguments.runs_root))
    global_by_model, global_consensus = pool_global(global_results)
    window_by_model, window_consensus, window_peaks = pool_windowed(window_results)

    global_by_model.to_csv(output_dir / "permutation_by_model.csv", index=False)
    global_consensus.to_csv(output_dir / "permutation_consensus.csv", index=False)
    window_by_model.to_csv(
        output_dir / "windowed_permutation_by_model.csv", index=False
    )
    window_consensus.to_csv(
        output_dir / "windowed_permutation_consensus.csv", index=False
    )
    window_peaks.to_csv(output_dir / "windowed_permutation_peaks.csv", index=False)
    print(f"Permutation outputs were saved in {output_dir}.", flush=True)


if __name__ == "__main__":
    main()
