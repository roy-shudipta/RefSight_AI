#!/usr/bin/env python3
"""One operating-range threshold is estimated from each calibrated risk curve."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


MINIMUM_BIN_COUNT = 30
MINIMUM_RISK_STEP_PP = 0.5


def best_threshold(feature_curve: pd.DataFrame) -> tuple[float, str, float]:
    """The largest supported risk step across adjacent bin cuts is returned."""
    curve = feature_curve.sort_values("bin").copy()
    curve["midpoint"] = 0.5 * (
        pd.to_numeric(curve["lo"], errors="coerce")
        + pd.to_numeric(curve["hi"], errors="coerce")
    )
    curve["n"] = pd.to_numeric(curve["n"], errors="coerce")
    curve["risk_mean"] = pd.to_numeric(curve["risk_mean"], errors="coerce")
    curve = curve.dropna(subset=["midpoint", "n", "risk_mean"])
    if len(curve) < 2:
        return float("nan"), "none", float("nan")

    midpoints = curve["midpoint"].to_numpy(dtype=float)
    counts = curve["n"].to_numpy(dtype=float)
    risks = curve["risk_mean"].to_numpy(dtype=float)
    weights = np.where(counts >= MINIMUM_BIN_COUNT, counts, 0.0)

    best = (float("nan"), "none", float("nan"))
    for cut in range(1, len(curve)):
        low_weights = weights[:cut]
        high_weights = weights[cut:]
        if low_weights.sum() == 0 or high_weights.sum() == 0:
            continue

        low_risk = float(np.average(risks[:cut], weights=low_weights))
        high_risk = float(np.average(risks[cut:], weights=high_weights))
        if high_risk >= low_risk:
            direction = "high"
            risk_step = 100.0 * (high_risk - low_risk)
        else:
            direction = "low"
            risk_step = 100.0 * (low_risk - high_risk)

        if risk_step > best[2]:
            threshold = 0.5 * (midpoints[cut - 1] + midpoints[cut])
            best = (float(threshold), direction, float(risk_step))

    if not np.isfinite(best[0]) or best[2] < MINIMUM_RISK_STEP_PP:
        return float("nan"), "none", float("nan")
    return best


def main() -> None:
    """Run-specific thresholds are written for later consensus aggregation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=os.getenv("OVERRIDE_RESULTS_DIR", "."),
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("OVERRIDE_MODEL_NAME", "unknown"),
    )
    arguments = parser.parse_args()

    results_dir = Path(arguments.results_dir).expanduser().resolve()
    risk_path = results_dir / "risk_profiles_all_features.csv"
    if not risk_path.exists():
        raise FileNotFoundError(
            f"The calibrated risk profiles were not found at {risk_path}"
        )

    profiles = pd.read_csv(risk_path)
    required = {"feature", "bin", "lo", "hi", "n", "risk_mean"}
    missing = sorted(required - set(profiles.columns))
    if missing:
        raise ValueError("Risk-profile columns are missing: " + ", ".join(missing))

    rows = []
    for feature, curve in profiles.groupby("feature", sort=True):
        threshold, direction, risk_step = best_threshold(curve)
        rows.append(
            {
                "model": arguments.model_name,
                "feature": feature,
                "threshold": threshold,
                "direction": direction,
                "delta_risk_pp": risk_step,
            }
        )

    output_dir = results_dir / "thresholds" / "operating_ranges"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "operating_range_thresholds.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
