from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DtwConfig:
    fps: int = 25

    # A pre-event window is defined.
    window_seconds: int = 15
    min_cont_seconds: int = 2

    # The DTW settings are defined.
    band_frac: float = 0.10
    top_k: int = 7

    # Frame and time discontinuities are limited.
    max_gap_frames: int = 12
    max_time_gap_seconds: float = 0.5

    # Eligible event types are defined.
    allowed_types: Tuple[str, ...] = ("Phys", "Tech", "Disc")
    excluded_eventnames: Tuple[str, ...] = ("Offside-Active", "Offside-Position")

    # Team organisation and ball position are used for DTW matching.
    feature_cols: Tuple[str, ...] = (
        "home_team_centroid_x",
        "home_team_centroid_y",
        "away_team_centroid_x",
        "away_team_centroid_y",
        "home_team_length",
        "home_team_width",
        "away_team_length",
        "away_team_width",
        "home_team_surface_area",
        "away_team_surface_area",
        "home_team_stretch_index",
        "away_team_stretch_index",
        "ball_x",
        "ball_y",
    )

    # Memory use is limited by cache sizes.
    max_seq_cache: int = 20000
    max_match_cache: int = 6
