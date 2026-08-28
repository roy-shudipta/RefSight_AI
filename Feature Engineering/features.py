"""Spatiotemporal features are created from framewise tracking data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


FPS = 25
HOME_PLAYER_IDS = tuple(f"home_{i}" for i in range(1, 12))
AWAY_PLAYER_IDS = tuple(f"away_{i}" for i in range(1, 12))
PLAYER_IDS = HOME_PLAYER_IDS + AWAY_PLAYER_IDS

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


def required_input_columns() -> tuple[str, ...]:
    """The required tracking columns are returned."""
    columns = {
        "ball_x",
        "ball_y",
        "ball_z",
        "ball_speed",
        "Referee_x",
        "Referee_y",
        "Referee_speed",
        "AsstRef1_x",
        "AsstRef1_y",
        "AsstRef2_x",
        "AsstRef2_y",
        "pitchLength",
        "lastTouch",
    }
    for player_id in PLAYER_IDS:
        columns.update(
            {
                f"{player_id}_x",
                f"{player_id}_y",
                f"{player_id}_speed",
            }
        )
    return tuple(sorted(columns))


def validate_input_columns(frame: pd.DataFrame) -> None:
    """The input schema is checked before features are created."""
    missing = sorted(set(required_input_columns()) - set(frame.columns))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Required tracking columns are missing: {joined}")

    if "live_bool" not in frame.columns and "live" not in frame.columns:
        raise ValueError("A 'live' or 'live_bool' column is required.")

    if "period" not in frame.columns and "gamePeriod" not in frame.columns:
        raise ValueError("A 'period' or 'gamePeriod' column is required.")


def _live_mask(frame: pd.DataFrame) -> pd.Series:
    """A strict Boolean live-play mask is returned."""
    column = "live_bool" if "live_bool" in frame.columns else "live"
    values = frame[column]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)

    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip().str.lower()
    text_values = text.map(
        {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "1": 1.0, "0": 0.0}
    )
    parsed = numeric.where(numeric.notna(), text_values)
    if parsed.isna().any():
        raise ValueError("The live-play column contains unsupported values.")
    return parsed.ne(0)


def _segments(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Live and non-live blocks are assigned separate segment numbers."""
    live = _live_mask(frame)
    if live.empty:
        return live, pd.Series(dtype="int64", index=frame.index)
    changed = live.ne(live.shift(1, fill_value=bool(live.iloc[0])))
    return live, changed.cumsum()


def _numeric_array(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Numeric columns are returned as a floating-point matrix."""
    return (
        frame.loc[:, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )


def _player_coordinates(
    frame: pd.DataFrame,
    player_ids: Sequence[str] = PLAYER_IDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Player coordinates are returned in stable slot order."""
    x = _numeric_array(frame, [f"{player_id}_x" for player_id in player_ids])
    y = _numeric_array(frame, [f"{player_id}_y" for player_id in player_ids])
    return x, y


def _nearest_player_indices(
    frame: pd.DataFrame,
    player_ids: Sequence[str] = PLAYER_IDS,
) -> tuple[np.ndarray, np.ndarray]:
    """The nearest player slot and distance are found for each frame."""
    player_x, player_y = _player_coordinates(frame, player_ids)
    ball_x = pd.to_numeric(frame["ball_x"], errors="coerce").to_numpy(dtype=float)
    ball_y = pd.to_numeric(frame["ball_y"], errors="coerce").to_numpy(dtype=float)
    distances = np.hypot(player_x - ball_x[:, None], player_y - ball_y[:, None])
    valid = np.isfinite(distances)
    safe = np.where(valid, distances, np.inf)
    indices = np.argmin(safe, axis=1)
    has_player = valid.any(axis=1)
    indices = np.where(has_player, indices, -1)
    nearest = np.where(
        has_player, safe[np.arange(len(frame)), np.maximum(indices, 0)], np.nan
    )
    return indices.astype(int), nearest


# =============================================================================
# Referee speed, ball height, and ball speed
# Direct tracking measurements are copied into the public feature schema.
# =============================================================================
def add_direct_tracking_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Referee speed, ball height, and ball speed are added."""
    out = frame.copy()
    out["referee_speed"] = pd.to_numeric(out["Referee_speed"], errors="coerce")
    out["ball_height"] = pd.to_numeric(out["ball_z"], errors="coerce")
    out["ball_speed"] = pd.to_numeric(out["ball_speed"], errors="coerce")
    return out


# =============================================================================
# Referee to ball distance
# The Euclidean distance from the referee to the ball is calculated.
# =============================================================================
def add_referee_to_ball_distance(frame: pd.DataFrame) -> pd.DataFrame:
    """The referee-to-ball distance is added in metres."""
    if "referee_to_ball_distance" in frame.columns:
        return frame
    out = frame.copy()
    out["referee_to_ball_distance"] = np.hypot(
        pd.to_numeric(out["Referee_x"], errors="coerce")
        - pd.to_numeric(out["ball_x"], errors="coerce"),
        pd.to_numeric(out["Referee_y"], errors="coerce")
        - pd.to_numeric(out["ball_y"], errors="coerce"),
    )
    return out


# =============================================================================
# Interaction angle
# The angle between the nearest home and away players is calculated.
# The angle is measured from the referee's position.
# =============================================================================
def add_interaction_angle(frame: pd.DataFrame) -> pd.DataFrame:
    """The angle between the nearest home and away players is added."""
    if "interaction_angle" in frame.columns:
        return frame
    out = frame.copy()
    home_idx, _ = _nearest_player_indices(out, HOME_PLAYER_IDS)
    away_idx, _ = _nearest_player_indices(out, AWAY_PLAYER_IDS)
    home_x, home_y = _player_coordinates(out, HOME_PLAYER_IDS)
    away_x, away_y = _player_coordinates(out, AWAY_PLAYER_IDS)
    rows = np.arange(len(out))

    home_point_x = np.full(len(out), np.nan)
    home_point_y = np.full(len(out), np.nan)
    away_point_x = np.full(len(out), np.nan)
    away_point_y = np.full(len(out), np.nan)
    valid_home = home_idx >= 0
    valid_away = away_idx >= 0
    home_point_x[valid_home] = home_x[rows[valid_home], home_idx[valid_home]]
    home_point_y[valid_home] = home_y[rows[valid_home], home_idx[valid_home]]
    away_point_x[valid_away] = away_x[rows[valid_away], away_idx[valid_away]]
    away_point_y[valid_away] = away_y[rows[valid_away], away_idx[valid_away]]

    ref_x = pd.to_numeric(out["Referee_x"], errors="coerce").to_numpy(dtype=float)
    ref_y = pd.to_numeric(out["Referee_y"], errors="coerce").to_numpy(dtype=float)
    home_vec_x = home_point_x - ref_x
    home_vec_y = home_point_y - ref_y
    away_vec_x = away_point_x - ref_x
    away_vec_y = away_point_y - ref_y
    dot = home_vec_x * away_vec_x + home_vec_y * away_vec_y
    denominator = np.hypot(home_vec_x, home_vec_y) * np.hypot(away_vec_x, away_vec_y)
    cosine = np.full(len(out), np.nan)
    valid = np.isfinite(denominator) & (denominator > 0)
    cosine[valid] = dot[valid] / denominator[valid]
    out["interaction_angle"] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return out


# =============================================================================
# Occlusion count
# Players inside the referee's sight cone are counted.
# Only players in front of the target player are counted.
# =============================================================================
def add_occlusion_count(
    frame: pd.DataFrame, half_angle_deg: float = 2.5
) -> pd.DataFrame:
    """Players inside the referee's narrow sight cone are counted."""
    if "occlusion_count" in frame.columns:
        return frame
    out = frame.copy()
    nearest_idx, _ = _nearest_player_indices(out)
    player_x, player_y = _player_coordinates(out)
    ref_x = pd.to_numeric(out["Referee_x"], errors="coerce").to_numpy(dtype=float)
    ref_y = pd.to_numeric(out["Referee_y"], errors="coerce").to_numpy(dtype=float)
    rows = np.arange(len(out))

    target_x = np.full(len(out), np.nan)
    target_y = np.full(len(out), np.nan)
    has_target = nearest_idx >= 0
    target_x[has_target] = player_x[rows[has_target], nearest_idx[has_target]]
    target_y[has_target] = player_y[rows[has_target], nearest_idx[has_target]]
    axis_x = target_x - ref_x
    axis_y = target_y - ref_y
    axis_distance = np.hypot(axis_x, axis_y)
    axis_unit_x = np.divide(
        axis_x,
        axis_distance,
        out=np.full(len(out), np.nan),
        where=np.isfinite(axis_distance) & (axis_distance > 0),
    )
    axis_unit_y = np.divide(
        axis_y,
        axis_distance,
        out=np.full(len(out), np.nan),
        where=np.isfinite(axis_distance) & (axis_distance > 0),
    )
    player_vec_x = player_x - ref_x[:, None]
    player_vec_y = player_y - ref_y[:, None]
    player_distance = np.hypot(player_vec_x, player_vec_y)
    dot = player_vec_x * axis_unit_x[:, None] + player_vec_y * axis_unit_y[:, None]
    cosine = np.divide(
        dot,
        player_distance,
        out=np.full_like(dot, np.nan),
        where=np.isfinite(player_distance) & (player_distance > 0),
    )
    target_cells = np.zeros_like(cosine, dtype=bool)
    target_cells[has_target, nearest_idx[has_target]] = True
    inside_angle = cosine >= np.cos(np.deg2rad(half_angle_deg))
    lies_between = player_distance < axis_distance[:, None]
    inside = inside_angle & lies_between & ~target_cells
    counts = inside.sum(axis=1).astype(float)
    invalid = ~np.isfinite(axis_distance) | (axis_distance <= 0)
    counts[invalid] = np.nan
    out["occlusion_count"] = counts
    return out


# =============================================================================
# Distance covered in the previous 5 s
# Referee movement over the previous five seconds is summed.
# =============================================================================
def add_distance_covered_previous_5s(
    frame: pd.DataFrame,
    fps: int = FPS,
) -> pd.DataFrame:
    """The distance covered in the previous five seconds is added."""
    if "distance_covered_previous_5s" in frame.columns:
        return frame
    out = frame.copy()
    live, segment = _segments(out)
    ref_x = pd.to_numeric(out["Referee_x"], errors="coerce")
    ref_y = pd.to_numeric(out["Referee_y"], errors="coerce")
    dx = ref_x.groupby(segment).diff()
    dy = ref_y.groupby(segment).diff()
    step = pd.Series(np.hypot(dx, dy), index=out.index).where(live, 0.0)
    covered = step.groupby(segment).transform(
        lambda values: values.rolling(5 * fps, min_periods=1).sum()
    )
    out["distance_covered_previous_5s"] = covered.where(live)
    return out


# =============================================================================
# Trajectory directness over 5 s
# Net displacement is divided by the full movement path.
# =============================================================================
def add_trajectory_directness_5s(frame: pd.DataFrame, fps: int = FPS) -> pd.DataFrame:
    """Five-second trajectory directness is added."""
    if "trajectory_directness_5s" in frame.columns:
        return frame
    out = frame.copy()
    live, segment = _segments(out)
    window = 5 * fps
    ref_x = pd.to_numeric(out["Referee_x"], errors="coerce")
    ref_y = pd.to_numeric(out["Referee_y"], errors="coerce")

    # A short mean is applied to reduce tracking jitter.
    smooth_x = ref_x.groupby(segment).transform(
        lambda values: values.rolling(3, center=True, min_periods=1).mean()
    )
    smooth_y = ref_y.groupby(segment).transform(
        lambda values: values.rolling(3, center=True, min_periods=1).mean()
    )
    dx = smooth_x.groupby(segment).diff()
    dy = smooth_y.groupby(segment).diff()
    step = pd.Series(np.hypot(dx, dy), index=out.index).where(live, 0.0)
    path_length = step.groupby(segment).transform(
        lambda values: values.rolling(window, min_periods=1).sum()
    )
    lag_x = smooth_x.groupby(segment).shift(window)
    lag_y = smooth_y.groupby(segment).shift(window)
    net_distance = np.hypot(smooth_x - lag_x, smooth_y - lag_y)
    ratio = net_distance / path_length.replace(0, np.nan)
    ratio = ratio.clip(0, 1).mask(path_length < 1.0)
    out["trajectory_directness_5s"] = ratio.where(live)
    return out


# =============================================================================
# Fastest player speed and closest player speed
# The two player-speed summaries are calculated for each frame.
# =============================================================================
def add_player_speed_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    """The fastest and nearest-player speeds are added."""
    outputs = {"fastest_player_speed", "closest_player_speed"}
    if outputs.issubset(frame.columns):
        return frame
    out = frame.copy()
    speed_columns = [f"{player_id}_speed" for player_id in PLAYER_IDS]
    speeds = _numeric_array(out, speed_columns)
    nearest_idx, _ = _nearest_player_indices(out)
    # Fastest player speed
    # The highest valid player speed is selected.
    safe_speeds = np.where(np.isfinite(speeds), speeds, -np.inf)
    fastest = safe_speeds.max(axis=1)
    fastest[fastest == -np.inf] = np.nan
    # Closest player speed
    # The speed of the player nearest to the ball is selected.
    rows = np.arange(len(out))
    closest = np.full(len(out), np.nan)
    valid = nearest_idx >= 0
    closest[valid] = speeds[rows[valid], nearest_idx[valid]]
    out["fastest_player_speed"] = fastest
    out["closest_player_speed"] = closest
    return out


# =============================================================================
# Turn angle and trajectory directness over 2 s
# Referee movement is prepared once for both features.
# =============================================================================
def add_turn_angle_and_trajectory_directness_2s(
    frame: pd.DataFrame,
    fps: int = FPS,
) -> pd.DataFrame:
    """Turn angle and two-second trajectory directness are added."""
    outputs = {"turn_angle", "trajectory_directness_2s"}
    if outputs.issubset(frame.columns):
        return frame
    out = frame.copy()
    _, segment = _segments(out)
    dt = 1.0 / fps
    ref_x = pd.to_numeric(out["Referee_x"], errors="coerce")
    ref_y = pd.to_numeric(out["Referee_y"], errors="coerce")

    def interpolate(values: pd.Series) -> pd.Series:
        return values.groupby(segment).transform(
            lambda part: part.interpolate(limit=1, limit_direction="both")
        )

    ref_x = interpolate(ref_x)
    ref_y = interpolate(ref_y)
    initial_dx = ref_x.groupby(segment).diff()
    initial_dy = ref_y.groupby(segment).diff()
    bad_step = np.hypot(initial_dx, initial_dy) > (12.0 * dt)
    ref_x = interpolate(ref_x.mask(bad_step))
    ref_y = interpolate(ref_y.mask(bad_step))

    # A five-frame mean is applied before direction is estimated.
    smooth_x = ref_x.groupby(segment).transform(
        lambda values: values.rolling(5, center=True, min_periods=1).mean()
    )
    smooth_y = ref_y.groupby(segment).transform(
        lambda values: values.rolling(5, center=True, min_periods=1).mean()
    )
    dx = smooth_x.groupby(segment).diff()
    dy = smooth_y.groupby(segment).diff()
    displacement = pd.Series(np.hypot(dx, dy), index=out.index)
    speed = displacement / dt
    # Turn angle
    # The change between consecutive movement headings is calculated.
    heading = pd.Series(np.degrees(np.arctan2(dy, dx)), index=out.index)
    turn_change = heading.groupby(segment).diff()
    turn = ((turn_change + 180.0) % 360.0) - 180.0
    moving = (speed > 2.0) & (displacement > 0.05)
    out["turn_angle"] = turn.where(moving)
    # Trajectory directness over 2 s
    # Net displacement is divided by the full movement path.
    window = 2 * fps
    lag_x = smooth_x.groupby(segment).shift(window)
    lag_y = smooth_y.groupby(segment).shift(window)
    net_distance = np.hypot(smooth_x - lag_x, smooth_y - lag_y)
    path_length = displacement.groupby(segment).transform(
        lambda values: values.rolling(window, min_periods=window).sum()
    )
    out["trajectory_directness_2s"] = (net_distance / path_length).where(
        path_length > 0
    )
    return out


def _slot_columns(team: str) -> tuple[list[str], list[str]]:
    """The coordinate columns for one team are returned."""
    player_ids = HOME_PLAYER_IDS if team == "home" else AWAY_PLAYER_IDS
    return (
        [f"{player_id}_x" for player_id in player_ids],
        [f"{player_id}_y" for player_id in player_ids],
    )


def _detect_goalkeeper_slot(frame: pd.DataFrame, team: str) -> str:
    """The slot nearest to either goal is treated as the goalkeeper."""
    x_columns, y_columns = _slot_columns(team)
    pitch_length = pd.to_numeric(frame["pitchLength"], errors="coerce").to_numpy(
        dtype=float
    )
    left_goal = -0.5 * pitch_length
    right_goal = 0.5 * pitch_length
    best_slot = 0
    best_distance = np.inf
    for slot, (x_column, y_column) in enumerate(zip(x_columns, y_columns)):
        x = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(pitch_length)
        if not valid.any():
            continue
        distance = np.minimum(
            np.hypot(x[valid] - left_goal[valid], y[valid]),
            np.hypot(x[valid] - right_goal[valid], y[valid]),
        ).mean()
        if distance < best_distance:
            best_slot = slot
            best_distance = float(distance)
    return f"{team}_{best_slot + 1}"


def _convex_hull_area(x: np.ndarray, y: np.ndarray) -> float:
    """The convex-hull area is calculated for one frame."""
    points = np.column_stack([x, y])
    points = np.unique(points[np.isfinite(points).all(axis=1)], axis=0)
    if len(points) < 3:
        return np.nan
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        return float(
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in points:
        while (
            len(lower) >= 2
            and cross(np.asarray(lower[-2]), np.asarray(lower[-1]), point) <= 0
        ):
            lower.pop()
        lower.append((float(point[0]), float(point[1])))
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while (
            len(upper) >= 2
            and cross(np.asarray(upper[-2]), np.asarray(upper[-1]), point) <= 0
        ):
            upper.pop()
        upper.append((float(point[0]), float(point[1])))
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        return np.nan
    hull_x = hull[:, 0]
    hull_y = hull[:, 1]
    return float(
        0.5
        * abs(np.dot(hull_x, np.roll(hull_y, -1)) - np.dot(hull_y, np.roll(hull_x, -1)))
    )


# =============================================================================
# Team organisation features
# The goalkeeper is excluded before each team feature is calculated.
# The same calculations are applied to the home and away teams.
# =============================================================================
def _team_shape(
    frame: pd.DataFrame, team: str, goalkeeper: str
) -> dict[str, np.ndarray]:
    """Team shape is calculated after the goalkeeper is excluded."""
    team_ids = HOME_PLAYER_IDS if team == "home" else AWAY_PLAYER_IDS
    player_ids = [player_id for player_id in team_ids if player_id != goalkeeper]
    x, y = _player_coordinates(frame, player_ids)
    valid = np.isfinite(x) & np.isfinite(y)
    paired_x = np.where(valid, x, np.nan)
    paired_y = np.where(valid, y, np.nan)
    with np.errstate(all="ignore"):
        # Team centroid
        # The mean outfield-player position is calculated.
        centroid_x = np.nanmean(paired_x, axis=1)
        centroid_y = np.nanmean(paired_y, axis=1)

        # Team length
        # The longitudinal span of the outfield players is calculated.
        length = np.nanmax(paired_x, axis=1) - np.nanmin(paired_x, axis=1)

        # Team width
        # The lateral span of the outfield players is calculated.
        width = np.nanmax(paired_y, axis=1) - np.nanmin(paired_y, axis=1)

        # Stretch index
        # The mean player distance from the team centroid is calculated.
        distance = np.hypot(
            paired_x - centroid_x[:, None],
            paired_y - centroid_y[:, None],
        )
        stretch = np.nanmean(distance, axis=1)

    # Spread
    # Overall pairwise spacing between outfield players is calculated.
    count = valid.sum(axis=1).astype(float)
    x_zero = np.where(valid, x, 0.0)
    y_zero = np.where(valid, y, 0.0)
    sum_squared = np.sum(x_zero**2 + y_zero**2, axis=1)
    sum_x = np.sum(x_zero, axis=1)
    sum_y = np.sum(y_zero, axis=1)
    pairwise_squared = 2.0 * count * sum_squared - 2.0 * (sum_x**2 + sum_y**2)
    spread = np.sqrt(np.maximum(pairwise_squared, 0.0))
    spread[count < 2] = np.nan

    # Surface area
    # The convex-hull area of the outfield players is calculated.
    area = np.asarray(
        [_convex_hull_area(paired_x[i], paired_y[i]) for i in range(len(frame))]
    )

    # Centroid to ball distance
    # The distance from the team centroid to the ball is calculated.
    ball_x = pd.to_numeric(frame["ball_x"], errors="coerce").to_numpy(dtype=float)
    ball_y = pd.to_numeric(frame["ball_y"], errors="coerce").to_numpy(dtype=float)
    centroid_to_ball_distance = np.hypot(centroid_x - ball_x, centroid_y - ball_y)
    return {
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "length": length,
        "width": width,
        "stretch_index": stretch,
        "spread": spread,
        "surface_area": area,
        "centroid_to_ball_distance": centroid_to_ball_distance,
    }


def add_team_organisation(frame: pd.DataFrame) -> pd.DataFrame:
    """Team organisation features are added for both teams."""
    expected = {
        f"{team}_team_{metric}"
        for team in ("home", "away")
        for metric in (
            "centroid_x",
            "centroid_y",
            "length",
            "width",
            "stretch_index",
            "spread",
            "surface_area",
            "centroid_to_ball_distance",
        )
    }
    if expected.issubset(frame.columns):
        return frame
    out = frame.copy()
    for team in ("home", "away"):
        goalkeeper = _detect_goalkeeper_slot(out, team)
        values = _team_shape(out, team, goalkeeper)
        for metric, result in values.items():
            out[f"{team}_team_{metric}"] = result
    return out


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    """A unit vector is returned when its direction is defined."""
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9:
        return np.array([np.nan, np.nan])
    return vector / norm


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """The angle between two vectors is returned in degrees."""
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return np.nan
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return np.nan
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _period_column(frame: pd.DataFrame) -> str:
    """The available period column name is returned."""
    return "gamePeriod" if "gamePeriod" in frame.columns else "period"


def _attack_goal_by_period(frame: pd.DataFrame) -> dict[tuple[object, str], float]:
    """The attacked goal is inferred for each team and period."""
    period_column = _period_column(frame)
    pitch = pd.to_numeric(frame["pitchLength"], errors="coerce").dropna()
    half_length = float(pitch.iloc[0] / 2.0) if not pitch.empty else 52.5
    home_goalkeeper = _detect_goalkeeper_slot(frame, "home")
    home_gk_x = pd.to_numeric(frame[f"{home_goalkeeper}_x"], errors="coerce")
    attack_goal: dict[tuple[object, str], float] = {}
    for period in frame[period_column].dropna().unique():
        in_period = frame[period_column].eq(period)
        values = home_gk_x[in_period].dropna()
        if len(values) < 10:
            continue
        left_share = float(values.lt(0).mean())
        right_share = float(values.gt(0).mean())
        if left_share >= 0.60:
            attack_goal[(period, "home")] = half_length
            attack_goal[(period, "away")] = -half_length
        elif right_share >= 0.60:
            attack_goal[(period, "home")] = -half_length
            attack_goal[(period, "away")] = half_length
    return attack_goal


def _centroid_distance_with_all_players(frame: pd.DataFrame, team: str) -> np.ndarray:
    """Centroid-to-ball distance is calculated for possession evidence."""
    player_ids = HOME_PLAYER_IDS if team == "home" else AWAY_PLAYER_IDS
    x, y = _player_coordinates(frame, player_ids)
    with np.errstate(all="ignore"):
        centroid_x = np.nanmean(x, axis=1)
        centroid_y = np.nanmean(y, axis=1)
    ball_x = pd.to_numeric(frame["ball_x"], errors="coerce").to_numpy(dtype=float)
    ball_y = pd.to_numeric(frame["ball_y"], errors="coerce").to_numpy(dtype=float)
    return np.hypot(centroid_x - ball_x, centroid_y - ball_y)


def _possession_state(frame: pd.DataFrame, fps: int) -> tuple[np.ndarray, np.ndarray]:
    """A stable attacking team is estimated from tracking evidence."""
    live, segment = _segments(frame)
    _, home_distance = _nearest_player_indices(frame, HOME_PLAYER_IDS)
    _, away_distance = _nearest_player_indices(frame, AWAY_PLAYER_IDS)
    home_centroid = _centroid_distance_with_all_players(frame, "home")
    away_centroid = _centroid_distance_with_all_players(frame, "away")
    last_touch = frame["lastTouch"].astype("string").str.lower().fillna("").to_numpy()
    votes = np.zeros(len(frame), dtype=float)
    votes[last_touch == "home"] += 1.0
    votes[last_touch == "away"] -= 1.0
    home_closer = np.isfinite(home_distance) & (
        ~np.isfinite(away_distance) | (home_distance <= away_distance)
    )
    away_closer = np.isfinite(away_distance) & (
        ~np.isfinite(home_distance) | (away_distance < home_distance)
    )
    votes[home_closer] += 1.0
    votes[away_closer] -= 1.0
    centroid_valid = np.isfinite(home_centroid) & np.isfinite(away_centroid)
    votes[centroid_valid & (home_centroid < away_centroid)] += 0.5
    votes[centroid_valid & (away_centroid < home_centroid)] -= 0.5
    evidence = (
        pd.Series(votes, index=frame.index)
        .groupby(segment)
        .transform(lambda values: values.rolling(2 * fps, min_periods=1).sum())
    )
    confidence = evidence.abs().to_numpy(dtype=float)
    possession = np.full(len(frame), "unknown", dtype=object)
    segment_values = segment.to_numpy()
    starts = np.ones(len(frame), dtype=bool)
    if len(frame) > 1:
        starts[1:] = segment_values[1:] != segment_values[:-1]
    state = "unknown"
    held = 0
    minimum_hold = max(1, int(round(0.6 * fps)))
    for i, value in enumerate(evidence.to_numpy(dtype=float)):
        if starts[i]:
            state = "unknown"
            held = 0
        if state == "unknown":
            if value >= 10:
                state = "home"
            elif value <= -10:
                state = "away"
            held = 0
        else:
            held += 1
            if held >= minimum_hold and value >= 10 and state != "home":
                state = "home"
                held = 0
            elif held >= minimum_hold and value <= -10 and state != "away":
                state = "away"
                held = 0
        possession[i] = state if live.iloc[i] else "unknown"
    return possession, confidence


# =============================================================================
# Lateral view angle
# The referee's viewing angle is measured against the attacking direction.
# =============================================================================
def add_lateral_view_angle(frame: pd.DataFrame, fps: int = FPS) -> pd.DataFrame:
    """The referee's lateral view angle is added in degrees."""
    if "lateral_view_angle" in frame.columns:
        return frame
    out = frame.copy()
    live, segment = _segments(out)
    period_column = _period_column(out)
    attack_goal = _attack_goal_by_period(out)
    possession, confidence = _possession_state(out, fps)
    home_idx, _ = _nearest_player_indices(out, HOME_PLAYER_IDS)
    away_idx, _ = _nearest_player_indices(out, AWAY_PLAYER_IDS)
    home_x, home_y = _player_coordinates(out, HOME_PLAYER_IDS)
    away_x, away_y = _player_coordinates(out, AWAY_PLAYER_IDS)
    ball_x = pd.to_numeric(out["ball_x"], errors="coerce")
    ball_y = pd.to_numeric(out["ball_y"], errors="coerce")
    ball_vx = ball_x.groupby(segment).diff() * fps
    ball_vy = ball_y.groupby(segment).diff() * fps
    smooth_vx = ball_vx.groupby(segment).transform(
        lambda values: values.rolling(fps, min_periods=1).mean()
    )
    smooth_vy = ball_vy.groupby(segment).transform(
        lambda values: values.rolling(fps, min_periods=1).mean()
    )
    speed = pd.to_numeric(out["ball_speed"], errors="coerce").to_numpy(dtype=float)
    ref_x = pd.to_numeric(out["Referee_x"], errors="coerce").to_numpy(dtype=float)
    ref_y = pd.to_numeric(out["Referee_y"], errors="coerce").to_numpy(dtype=float)
    last_touch = out["lastTouch"].astype("string").str.lower().fillna("").to_numpy()
    periods = out[period_column].to_numpy()
    segment_values = segment.to_numpy()
    starts = np.ones(len(out), dtype=bool)
    if len(out) > 1:
        starts[1:] = segment_values[1:] != segment_values[:-1]
    angles = np.full(len(out), np.nan)
    previous_axis = np.array([np.nan, np.nan])

    for i in range(len(out)):
        if starts[i]:
            previous_axis = np.array([np.nan, np.nan])
        if not live.iloc[i]:
            continue
        team = possession[i]
        if team not in {"home", "away"}:
            team = last_touch[i]
        if team not in {"home", "away"}:
            continue
        player_idx = home_idx[i] if team == "home" else away_idx[i]
        if player_idx < 0:
            continue
        carrier_x = home_x[i, player_idx] if team == "home" else away_x[i, player_idx]
        carrier_y = home_y[i, player_idx] if team == "home" else away_y[i, player_idx]
        if not np.isfinite([carrier_x, carrier_y, ref_x[i], ref_y[i]]).all():
            continue
        referee_vector = np.array([ref_x[i] - carrier_x, ref_y[i] - carrier_y])
        goal_x = attack_goal.get((periods[i], team), np.nan)
        slow_axis = _unit_vector(np.array([goal_x - carrier_x, 0.0]))
        fast_axis = _unit_vector(np.array([smooth_vx.iloc[i], smooth_vy.iloc[i]]))
        frame_speed = speed[i]
        if not np.isfinite(frame_speed):
            frame_speed = float(np.hypot(smooth_vx.iloc[i], smooth_vy.iloc[i]))
        weight = float(np.clip((frame_speed - 1.0) / 3.0, 0.0, 1.0))
        contested = confidence[i] < 6.0 and frame_speed < 1.5
        if contested and np.isfinite(previous_axis).all():
            axis = previous_axis
        elif not np.isfinite(slow_axis).all():
            axis = fast_axis
        elif not np.isfinite(fast_axis).all():
            axis = slow_axis
        else:
            axis = _unit_vector((1.0 - weight) * slow_axis + weight * fast_axis)
        if not np.isfinite(axis).all():
            continue
        previous_axis = axis
        angles[i] = _angle_degrees(axis, referee_vector)
    out["lateral_view_angle"] = angles
    return out


def _lead_assistant_mask(
    frame: pd.DataFrame,
    fps: int,
    corridor_m: float = 8.0,
    minimum_switch_seconds: float = 0.6,
) -> np.ndarray:
    """The lead assistant is selected with temporal stabilisation."""
    ar1_y = pd.to_numeric(frame["AsstRef1_y"], errors="coerce").to_numpy(dtype=float)
    ar2_y = pd.to_numeric(frame["AsstRef2_y"], errors="coerce").to_numpy(dtype=float)
    ball_y = pd.to_numeric(frame["ball_y"], errors="coerce").to_numpy(dtype=float)
    ar1_is_top = float(np.nanmedian(ar1_y)) >= float(np.nanmedian(ar2_y))
    minimum_switch = max(1, int(round(minimum_switch_seconds * fps)))

    def desired(ball_value: float, previous: bool) -> bool:
        if not np.isfinite(ball_value):
            return previous
        if ball_value > corridor_m:
            return ar1_is_top
        if ball_value < -corridor_m:
            return not ar1_is_top
        return previous

    selected = np.zeros(len(frame), dtype=bool)
    if len(frame) == 0:
        return selected
    previous = ar1_is_top
    previous = desired(ball_y[0], previous)
    selected[0] = previous
    pending = 0
    for i in range(1, len(frame)):
        wanted = desired(ball_y[i], previous)
        if wanted == previous:
            pending = 0
        else:
            pending += 1
            if pending >= minimum_switch:
                previous = wanted
                pending = 0
        selected[i] = previous
    return selected


# =============================================================================
# Referee and assistant coverage geometry
# The lead assistant is selected before the three features are calculated.
# =============================================================================
def add_officiating_coverage(frame: pd.DataFrame, fps: int = FPS) -> pd.DataFrame:
    """Referee and assistant coverage features are added."""
    outputs = {
        "referee_to_lead_assistant_distance",
        "ball_angle_referee_vs_lead_assistant",
        "complementary_coverage_index",
    }
    if outputs.issubset(frame.columns):
        return frame
    out = frame.copy()
    ar1_selected = _lead_assistant_mask(out, fps)
    ar1_x = pd.to_numeric(out["AsstRef1_x"], errors="coerce").to_numpy(dtype=float)
    ar1_y = pd.to_numeric(out["AsstRef1_y"], errors="coerce").to_numpy(dtype=float)
    ar2_x = pd.to_numeric(out["AsstRef2_x"], errors="coerce").to_numpy(dtype=float)
    ar2_y = pd.to_numeric(out["AsstRef2_y"], errors="coerce").to_numpy(dtype=float)
    lead_x = np.where(ar1_selected, ar1_x, ar2_x)
    lead_y = np.where(ar1_selected, ar1_y, ar2_y)
    ref_x = pd.to_numeric(out["Referee_x"], errors="coerce").to_numpy(dtype=float)
    ref_y = pd.to_numeric(out["Referee_y"], errors="coerce").to_numpy(dtype=float)
    ball_x = pd.to_numeric(out["ball_x"], errors="coerce").to_numpy(dtype=float)
    ball_y = pd.to_numeric(out["ball_y"], errors="coerce").to_numpy(dtype=float)
    # Referee to lead assistant distance
    # The distance between the referee and lead assistant is calculated.
    distance = np.hypot(ref_x - lead_x, ref_y - lead_y)
    out["referee_to_lead_assistant_distance"] = distance

    # Ball angle between referee and lead assistant
    # The two viewing directions from the ball are compared.
    referee_ball_x = ref_x - ball_x
    referee_ball_y = ref_y - ball_y
    assistant_ball_x = lead_x - ball_x
    assistant_ball_y = lead_y - ball_y
    denominator = np.hypot(referee_ball_x, referee_ball_y) * np.hypot(
        assistant_ball_x, assistant_ball_y
    )
    cosine = np.full(len(out), np.nan)
    valid = np.isfinite(denominator) & (denominator > 1e-9)
    cosine[valid] = (
        referee_ball_x[valid] * assistant_ball_x[valid]
        + referee_ball_y[valid] * assistant_ball_y[valid]
    ) / denominator[valid]
    theta = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    out["ball_angle_referee_vs_lead_assistant"] = theta

    # Complementary coverage index
    # Spatial separation and ball angle are combined.
    separation = np.clip(distance / 40.0, 0.0, 1.0)
    angular = np.clip(np.sin(np.radians(theta)), 0.0, 1.0)
    out["complementary_coverage_index"] = separation * angular
    return out


FEATURE_BUILDERS = (
    add_direct_tracking_features,
    add_referee_to_ball_distance,
    add_interaction_angle,
    add_occlusion_count,
    add_distance_covered_previous_5s,
    add_trajectory_directness_5s,
    add_player_speed_summaries,
    add_turn_angle_and_trajectory_directness_2s,
    add_team_organisation,
    add_lateral_view_angle,
    add_officiating_coverage,
)


def run_all_features(frame: pd.DataFrame) -> pd.DataFrame:
    """All configured features are added in a fixed order."""
    validate_input_columns(frame)
    out = frame.copy()
    for builder in FEATURE_BUILDERS:
        try:
            out = builder(out)
        except Exception as error:
            raise RuntimeError(
                f"Feature creation failed in {builder.__name__}."
            ) from error
    missing = sorted(set(MODEL_FEATURE_COLUMNS) - set(out.columns))
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Required feature columns were not created: {joined}")
    return out
