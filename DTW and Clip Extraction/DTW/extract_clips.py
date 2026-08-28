#!/usr/bin/env python3
"""Variable-length model clips are extracted from DTW matches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FPS = 25
MAX_WINDOW_SECONDS = 15
MIN_WINDOW_SECONDS = 2
MAX_GAP_FRAMES = 12
MAX_TIME_GAP_SECONDS = 0.5

MODEL_FEATURE_COLUMNS = (
    "referee_speed",
    "ball_height",
    "ball_speed",
    "referee_to_ball_distance",
    "interaction_angle",
    "occlusion_count",
    "distance_covered_previous_5s",
    "trajectory_directness_5s",
    "fastest_player_speed",
    "closest_player_speed",
    "turn_angle",
    "trajectory_directness_2s",
    "lateral_view_angle",
    "referee_to_lead_assistant_distance",
    "ball_angle_referee_vs_lead_assistant",
    "complementary_coverage_index",
    "home_team_centroid_x",
    "home_team_centroid_y",
    "home_team_length",
    "home_team_width",
    "home_team_stretch_index",
    "home_team_spread",
    "home_team_surface_area",
    "home_team_centroid_to_ball_distance",
    "away_team_centroid_x",
    "away_team_centroid_y",
    "away_team_length",
    "away_team_width",
    "away_team_stretch_index",
    "away_team_spread",
    "away_team_surface_area",
    "away_team_centroid_to_ball_distance",
)

IDENTIFIER_COLUMNS = (
    "frameIdx",
    "MergeID",
    "matchSeconds",
    "event",
)

METADATA_COLUMNS = ("Referee Name",)

MATCH_COLUMNS = {
    "inc_mergeid",
    "inc_frame",
    "best_mergeid",
    "best_cor_frame",
    "rank",
}


def _path_from_environment(name: str, default: Path) -> Path:
    """A path is read from the environment or from the default."""
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


def parse_arguments() -> argparse.Namespace:
    """Command-line arguments are parsed."""
    parser = argparse.ArgumentParser(
        description="Extract model-ready source clips from DTW matches."
    )
    parser.add_argument(
        "--tracking-folder",
        type=Path,
        default=_path_from_environment(
            "REFSIGHT_FEATURED_TRACKING_DIR",
            PROJECT_ROOT / "data" / "featured_tracking",
        ),
    )
    parser.add_argument(
        "--matches-csv",
        type=Path,
        default=_path_from_environment(
            "REFSIGHT_DTW_MATCHES_CSV",
            PROJECT_ROOT / "outputs" / "dtw" / "dtw_matches.csv",
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_path_from_environment(
            "REFSIGHT_TIME_SERIES_DIR",
            PROJECT_ROOT / "data" / "time_series_clips",
        ),
    )
    parser.add_argument("--top-k", type=int, default=7)
    return parser.parse_args()


def continuous_pre_event_window(
    frame: pd.DataFrame,
    event_frame: int,
) -> pd.DataFrame | None:
    """The longest valid pre-event window between 2 s and 15 s is returned."""
    before_event = frame.loc[frame["frameIdx"] < int(event_frame)].copy()
    if before_event.empty:
        return None
    before_event = before_event.sort_values("frameIdx")
    frames = before_event["frameIdx"].to_numpy(dtype=int)
    times = pd.to_numeric(
        before_event["matchSeconds"],
        errors="coerce",
    ).to_numpy(dtype=float)
    end = len(before_event) - 1
    start = end
    maximum_frames = MAX_WINDOW_SECONDS * FPS
    minimum_frames = MIN_WINDOW_SECONDS * FPS

    while start > 0:
        frame_gap = frames[start] - frames[start - 1]
        time_gap = times[start] - times[start - 1]
        if frame_gap > MAX_GAP_FRAMES:
            break
        if np.isfinite(time_gap) and time_gap > MAX_TIME_GAP_SECONDS:
            break
        start -= 1
        if frames[end] - frames[start] >= maximum_frames:
            break

    window = before_event.iloc[start : end + 1].copy()
    if len(window) > maximum_frames:
        window = window.iloc[-maximum_frames:].copy()
    return window if len(window) >= minimum_frames else None


def feature_matrix(window: pd.DataFrame) -> np.ndarray:
    """The configured feature columns are returned without imputation."""
    numeric = window.loc[:, list(MODEL_FEATURE_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return numeric.to_numpy(dtype=np.float32)


def _dataset_match_id(
    tracking: ds.Dataset,
    match_id: object,
) -> int | str:
    """A match identifier is converted to the dataset field type."""
    match_type = tracking.schema.field("MergeID").type
    if pa.types.is_integer(match_type):
        return int(float(match_id))
    return str(match_id)


def read_event_slice(
    tracking: ds.Dataset,
    match_id: object,
    event_frame: int,
    columns: list[str],
) -> pd.DataFrame:
    """The frames needed for one pre-event clip are read."""
    typed_match_id = _dataset_match_id(tracking, match_id)
    margin = MAX_WINDOW_SECONDS * FPS + MAX_GAP_FRAMES
    first_frame = max(0, int(event_frame) - margin)
    condition = (
        (ds.field("MergeID") == typed_match_id)
        & (ds.field("frameIdx") <= int(event_frame))
        & (ds.field("frameIdx") >= first_frame)
    )
    return tracking.to_table(columns=columns, filter=condition).to_pandas()


def event_metadata(frame: pd.DataFrame, event_frame: int) -> dict[str, object]:
    """Available event metadata is read from the event frame."""
    event_rows = frame.loc[
        frame["frameIdx"].eq(int(event_frame)) & frame["event"].eq(1)
    ]
    if event_rows.empty:
        return {}
    columns = [column for column in METADATA_COLUMNS if column in event_rows.columns]
    return event_rows.iloc[0].reindex(columns).to_dict()


def _safe_file_part(value: object) -> str:
    """A value is made safe for use in a local file name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def validate_inputs(tracking: ds.Dataset, matches: pd.DataFrame) -> list[str]:
    """Required tracking and DTW-match columns are checked."""
    missing_matches = sorted(MATCH_COLUMNS - set(matches.columns))
    if missing_matches:
        raise ValueError("DTW match columns are missing: " + ", ".join(missing_matches))

    required_tracking = (
        set(IDENTIFIER_COLUMNS) | set(METADATA_COLUMNS) | set(MODEL_FEATURE_COLUMNS)
    )
    missing_tracking = sorted(required_tracking - set(tracking.schema.names))
    if missing_tracking:
        raise ValueError(
            "Featured tracking columns are missing: " + ", ".join(missing_tracking)
        )
    return (
        list(IDENTIFIER_COLUMNS) + list(METADATA_COLUMNS) + list(MODEL_FEATURE_COLUMNS)
    )


def extract_clips(
    tracking_folder: Path,
    matches_csv: Path,
    output_dir: Path,
    top_k: int,
) -> None:
    """Matched query lists are written as variable-length NumPy arrays."""
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if not tracking_folder.is_dir():
        raise FileNotFoundError(f"Tracking folder was not found: {tracking_folder}")
    if not matches_csv.is_file():
        raise FileNotFoundError(f"DTW matches file was not found: {matches_csv}")

    tracking = ds.dataset(str(tracking_folder), format="parquet")
    matches = pd.read_csv(matches_csv)
    columns = validate_inputs(tracking, matches)
    matches["rank"] = pd.to_numeric(matches["rank"], errors="raise").astype(int)
    matches = matches.loc[matches["rank"].between(1, top_k)].copy()
    matches = matches.sort_values(["inc_mergeid", "inc_frame", "rank"])

    sequence_dir = output_dir / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict[str, object]] = []
    expected_ranks = set(range(1, top_k + 1))

    grouped = matches.groupby(["inc_mergeid", "inc_frame"], sort=False)
    for (query_match, query_frame), candidates in grouped:
        candidates = candidates.drop_duplicates("rank", keep="first")
        if set(candidates["rank"].tolist()) != expected_ranks:
            continue

        query_frame = int(query_frame)
        query_slice = read_event_slice(
            tracking,
            query_match,
            query_frame,
            columns,
        )
        query_window = continuous_pre_event_window(query_slice, query_frame)
        if query_window is None:
            continue

        query_name = f"{_safe_file_part(query_match)}_{query_frame}"
        query_file = f"{query_name}_incorrect.npy"
        query_array = feature_matrix(query_window)
        pending_arrays: list[tuple[str, np.ndarray]] = [(query_file, query_array)]
        pending_metadata: list[dict[str, object]] = [
            {
                "query_id": query_name,
                "decision_type": "Incorrect",
                "candidate_rank": 0,
                "MergeID": query_match,
                "event_frame": query_frame,
                "n_frames": len(query_array),
                "file_path": query_file,
                **event_metadata(query_slice, query_frame),
            }
        ]

        complete_list = True
        for candidate in candidates.itertuples(index=False):
            candidate_match = candidate.best_mergeid
            candidate_frame = int(candidate.best_cor_frame)
            candidate_rank = int(candidate.rank)
            candidate_slice = read_event_slice(
                tracking,
                candidate_match,
                candidate_frame,
                columns,
            )
            candidate_window = continuous_pre_event_window(
                candidate_slice,
                candidate_frame,
            )
            if candidate_window is None:
                complete_list = False
                break
            candidate_array = feature_matrix(candidate_window)
            candidate_file = f"{query_name}_correct{candidate_rank}.npy"
            pending_arrays.append((candidate_file, candidate_array))
            pending_metadata.append(
                {
                    "query_id": query_name,
                    "decision_type": "Correct",
                    "candidate_rank": candidate_rank,
                    "MergeID": candidate_match,
                    "event_frame": candidate_frame,
                    "n_frames": len(candidate_array),
                    "file_path": candidate_file,
                    **event_metadata(candidate_slice, candidate_frame),
                }
            )

        if not complete_list:
            continue

        # One incorrect clip and all ranked correct clips are saved together.
        for file_name, array in pending_arrays:
            np.save(sequence_dir / file_name, array)
        metadata_rows.extend(pending_metadata)

    if not metadata_rows:
        raise RuntimeError("No complete query lists could be extracted.")

    schema_path = output_dir / "schema.json"
    schema_path.write_text(
        json.dumps({"ordered_columns": list(MODEL_FEATURE_COLUMNS)}, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    print(f"Saved clips: {len(metadata)}")
    print(
        f"Saved query lists: {metadata['query_id'].nunique() if not metadata.empty else 0}"
    )
    print(f"Output folder: {output_dir}")


def main() -> None:
    """The clip-extraction command is run."""
    arguments = parse_arguments()
    extract_clips(
        tracking_folder=arguments.tracking_folder.expanduser().resolve(),
        matches_csv=arguments.matches_csv.expanduser().resolve(),
        output_dir=arguments.output_dir.expanduser().resolve(),
        top_k=arguments.top_k,
    )


if __name__ == "__main__":
    main()
