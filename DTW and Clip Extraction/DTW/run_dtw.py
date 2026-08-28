#!/usr/bin/env python3
"""DTW-matched correct decisions are retrieved for incorrect decisions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


DTW_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DTW_ROOT.parents[1]
if str(DTW_ROOT) not in sys.path:
    sys.path.insert(0, str(DTW_ROOT))

from dtw_retrieval.columns import Columns
from dtw_retrieval.config import DtwConfig
from dtw_retrieval.filters import build_event_index
from dtw_retrieval.io import MatchStore, build_manifest, list_parquet_files
from dtw_retrieval.retrieval import run_retrieval
from dtw_retrieval.scaling import fit_scaler_on_correct


def _path_from_environment(name: str, default: Path) -> Path:
    """A path is read from the environment or from the default."""
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


def parse_arguments() -> argparse.Namespace:
    """Command-line arguments are parsed."""
    parser = argparse.ArgumentParser(
        description="Retrieve DTW-matched correct decisions for each incorrect decision."
    )
    parser.add_argument(
        "--tracking-folder",
        type=Path,
        default=_path_from_environment(
            "REFSIGHT_FEATURED_TRACKING_DIR",
            PROJECT_ROOT / "data" / "featured_tracking",
        ),
        help="Folder that contains featured match Parquet files.",
    )
    parser.add_argument(
        "--out-matches",
        type=Path,
        default=_path_from_environment(
            "REFSIGHT_DTW_MATCHES_CSV",
            PROJECT_ROOT / "outputs" / "dtw" / "dtw_matches.csv",
        ),
        help="Output CSV for the ranked DTW matches.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=7,
        help="Number of correct candidates retained for each query.",
    )
    parser.add_argument(
        "--band-fraction",
        type=float,
        default=0.10,
        help="Sakoe-Chiba band as a fraction of the longer sequence.",
    )
    arguments = parser.parse_args()
    if arguments.top_k <= 0:
        parser.error("--top-k must be a positive integer.")
    if not 0.0 < arguments.band_fraction <= 1.0:
        parser.error("--band-fraction must be greater than 0 and at most 1.")
    return arguments


def main() -> int:
    """The DTW retrieval command is run."""
    arguments = parse_arguments()
    tracking_folder = arguments.tracking_folder.expanduser().resolve()
    out_matches = arguments.out_matches.expanduser().resolve()

    files = list_parquet_files(tracking_folder)
    if not files:
        raise FileNotFoundError(
            f"No featured Parquet files were found in: {tracking_folder}"
        )

    columns = Columns()
    config = DtwConfig(
        top_k=arguments.top_k,
        band_frac=arguments.band_fraction,
    )
    out_matches.parent.mkdir(parents=True, exist_ok=True)

    print(f"Tracking folder: {tracking_folder}")
    print(f"Parquet files: {len(files)}")
    manifest = build_manifest(files, columns)
    match_map, event_index = build_event_index(manifest, columns, config)

    print(f"Matches in manifest: {len(manifest)}")
    print(f"Incorrect events: {len(event_index.incorrect)}")
    print(f"Correct events: {len(event_index.correct)}")

    referee_matches: dict[str, set[str]] = {}
    for event in event_index.correct + event_index.incorrect:
        if event.referee_name:
            referee_matches.setdefault(event.referee_name, set()).add(event.match_id)
    print(f"Referees: {len(referee_matches)}")

    store = MatchStore(max_items=config.max_match_cache)
    scaler = fit_scaler_on_correct(
        match_map,
        event_index,
        store,
        columns,
        config,
    )
    run_retrieval(
        match_map=match_map,
        index=event_index,
        store=store,
        cols=columns,
        cfg=config,
        scaler=scaler,
        out_matches_csv=out_matches,
    )

    print(f"Saved matches: {out_matches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
