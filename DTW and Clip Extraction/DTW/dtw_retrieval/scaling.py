from __future__ import annotations

from sklearn.preprocessing import StandardScaler

from .columns import Columns
from .config import DtwConfig
from .filters import EventIndex
from .io import MatchStore, MatchInfo
from .windows import window_matrix


def fit_scaler_on_correct(
    match_map: dict[str, MatchInfo],
    index: EventIndex,
    store: MatchStore,
    cols: Columns,
    cfg: DtwConfig,
) -> StandardScaler:
    """A standard scaler is fitted with correct-decision windows."""
    feature_cols = list(cfg.feature_cols)
    needed_cols = set(feature_cols + [cols.frame, cols.time])
    scaler = StandardScaler(with_mean=True, with_std=True)
    fitted_windows = 0

    for event in index.correct:
        info = match_map[event.match_id]
        frame = store.get(event.match_id, info.path, columns=list(needed_cols))
        frame = frame.sort_values(cols.frame)

        sequence = window_matrix(
            frame,
            event.frame,
            cols=cols,
            cfg=cfg,
            feature_cols=feature_cols,
        )
        if sequence is not None:
            scaler.partial_fit(sequence)
            fitted_windows += 1

    if fitted_windows == 0:
        raise RuntimeError("No valid correct-decision windows were found for scaling.")
    return scaler
