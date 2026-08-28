from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import Columns
from .config import DtwConfig


def window_matrix(
    df: pd.DataFrame,
    end_frame: int,
    cols: Columns,
    cfg: DtwConfig,
    feature_cols: list[str],
) -> np.ndarray | None:
    """A continuous pre-event feature matrix is returned."""
    if cols.frame not in df.columns:
        return None

    pre = df[df[cols.frame] < int(end_frame)].copy()
    if pre.empty:
        return None

    pre = pre.sort_values(cols.frame)

    frames = pre[cols.frame].to_numpy()
    times = pre[cols.time].to_numpy() if cols.time in pre.columns else None

    end_idx = len(pre) - 1
    start_idx = end_idx

    max_len_frames = cfg.window_seconds * cfg.fps
    min_len_frames = cfg.min_cont_seconds * cfg.fps

    while start_idx > 0:
        # The window is stopped at a frame discontinuity.
        if frames[start_idx] - frames[start_idx - 1] > cfg.max_gap_frames:
            break

        # The window is stopped at a time discontinuity.
        if times is not None:
            t0 = times[start_idx - 1]
            t1 = times[start_idx]
            if not (np.isnan(t0) or np.isnan(t1)):
                if (t1 - t0) > cfg.max_time_gap_seconds:
                    break

        start_idx -= 1
        if (frames[end_idx] - frames[start_idx]) >= max_len_frames:
            break

    seg = pre.iloc[start_idx : end_idx + 1]
    if len(seg) > max_len_frames:
        seg = seg.iloc[-max_len_frames:]
    if len(seg) < min_len_frames:
        return None

    missing_features = [column for column in feature_cols if column not in seg.columns]
    if missing_features:
        raise ValueError(
            "DTW feature columns are missing: " + ", ".join(missing_features)
        )

    X = seg[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X
