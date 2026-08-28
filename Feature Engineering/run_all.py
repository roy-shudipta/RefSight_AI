#!/usr/bin/env python3
"""Tracking features are created for each match CSV file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import traceback

import numpy as np
import pandas as pd

from features import MODEL_FEATURE_COLUMNS, run_all_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_tracking"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "featured_tracking"

_NUMERIC_COLUMN = re.compile(
    r"^(ball_(x|y|z|speed)|"
    r"Referee_(x|y|speed)|"
    r"AsstRef[12]_(x|y)|"
    r"(home|away)_(\d+)_(x|y|speed)|"
    r"pitchLength|pitchWidth)$"
)


def _configured_path(environment_name: str, default: Path) -> Path:
    """A path is read from the environment or from the project default."""
    value = os.getenv(environment_name)
    return Path(value).expanduser() if value else default


def parse_arguments() -> argparse.Namespace:
    """Command-line arguments are parsed."""
    parser = argparse.ArgumentParser(
        description="Create RefSight features from tracking CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_configured_path("REFSIGHT_RAW_TRACKING_DIR", DEFAULT_INPUT_DIR),
        help="Folder containing one tracking CSV file per match.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_configured_path("REFSIGHT_FEATURED_TRACKING_DIR", DEFAULT_OUTPUT_DIR),
        help="Folder used for the featured Parquet files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Existing Parquet files are replaced.",
    )
    return parser.parse_args()


def coerce_tracking_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Tracking measurements are converted to numeric values."""
    out = frame.copy()
    columns = [column for column in out.columns if _NUMERIC_COLUMN.match(column)]
    out[columns] = out[columns].apply(pd.to_numeric, errors="coerce")
    return out


def normalise_live_column(frame: pd.DataFrame) -> pd.DataFrame:
    """The live-play field is converted to strict Boolean values."""
    if "live" not in frame.columns:
        raise ValueError("The required 'live' column is missing.")

    out = frame.copy()
    values = out["live"]
    if pd.api.types.is_bool_dtype(values.dtype):
        parsed = values.fillna(False).astype(bool)
    else:
        numeric = pd.to_numeric(values, errors="coerce")
        text = values.astype("string").str.strip().str.lower()
        text_values = text.map(
            {
                "true": 1.0,
                "false": 0.0,
                "yes": 1.0,
                "no": 0.0,
                "1": 1.0,
                "0": 0.0,
            }
        )
        combined = numeric.where(numeric.notna(), text_values)
        if combined.isna().any():
            raise ValueError("The 'live' column contains unsupported values.")
        parsed = combined.ne(0)
    out["live"] = parsed
    return out


def prepare_tracking(frame: pd.DataFrame) -> pd.DataFrame:
    """The tracking table is prepared for safe feature creation."""
    out = coerce_tracking_columns(frame)
    out = normalise_live_column(out)

    missing_ball = {"ball_x", "ball_y"} - set(out.columns)
    if missing_ball:
        joined = ", ".join(sorted(missing_ball))
        raise ValueError(f"Required ball columns are missing: {joined}")

    # Ball coordinates outside live play are treated as missing.
    out.loc[~out["live"], ["ball_x", "ball_y"]] = np.nan
    return out


def process_file(csv_path: Path, output_dir: Path) -> Path:
    """One CSV file is processed and one Parquet file is saved."""
    frame = pd.read_csv(csv_path, low_memory=False)
    frame = prepare_tracking(frame)
    featured = run_all_features(frame)

    # Only valid live-play rows are saved.
    featured = featured.loc[featured["live"]].copy()
    missing = sorted(set(MODEL_FEATURE_COLUMNS) - set(featured.columns))
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"The featured output is incomplete: {joined}")

    output_path = output_dir / f"{csv_path.stem}.parquet"
    featured.to_parquet(output_path, index=False)
    return output_path


def run_directory(input_dir: Path, output_dir: Path, overwrite: bool = False) -> int:
    """All CSV files in one folder are processed in name order."""
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"The input folder does not exist: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    processed = 0
    for csv_path in csv_files:
        output_path = output_dir / f"{csv_path.stem}.parquet"
        if output_path.exists() and not overwrite:
            print(f"SKIP  {csv_path.name}")
            continue
        try:
            saved_path = process_file(csv_path, output_dir)
            processed += 1
            print(f"OK    {csv_path.name} -> {saved_path.name}")
        except Exception:
            failures += 1
            print(f"FAIL  {csv_path.name}")
            print(traceback.format_exc())

    print(f"Processed: {processed}; failed: {failures}; found: {len(csv_files)}")
    return failures


def main() -> int:
    """The batch command is run."""
    arguments = parse_arguments()
    return run_directory(
        arguments.input_dir,
        arguments.output_dir,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
