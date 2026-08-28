from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .columns import Columns
from .config import DtwConfig
from .io import MatchInfo


@dataclass(frozen=True)
class EventRow:
    match_id: str
    frame: int
    time: float
    event_idx: int
    event_name: str
    event_type: str
    referee_name: str


@dataclass
class EventIndex:
    incorrect: List[EventRow]
    correct: List[EventRow]

    correct_by_match: Dict[str, List[EventRow]]
    correct_by_ref_type: Dict[tuple[str, str], List[EventRow]]


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def _norm_referee(v: Any) -> str:
    """A referee identifier is normalised to a stable string."""
    if v is None:
        return ""
    # Missing values are converted to an empty identifier.
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass

    # Numeric identifiers are converted without a trailing decimal.
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        return str(int(fv)) if fv.is_integer() else str(fv)

    # Text identifiers are stripped and normalised when possible.
    s = str(v).strip()
    if s == "":
        return ""
    try:
        fv = float(s)
        return str(int(fv)) if fv.is_integer() else s
    except Exception:
        return s


def filter_event_table(
    df: pd.DataFrame,
    cols: Columns,
    cfg: DtwConfig,
) -> pd.DataFrame:
    """Eligible decision-event rows are returned."""
    out = df.copy()

    # Rows without a coded decision event are excluded.
    if cols.event_flag in out.columns:
        out = out[out[cols.event_flag] == 1]

    # Events with a time difference above two seconds are excluded.
    event_time_col = getattr(cols, "event_time", "Time")
    if (event_time_col in out.columns) and (cols.time in out.columns):
        t_event = pd.to_numeric(out[event_time_col], errors="coerce")
        t_track = pd.to_numeric(out[cols.time], errors="coerce")
        valid = t_event.notna() & t_track.notna()
        diff = (t_event - t_track).abs()

        mask_keep = (~valid) | (diff <= 2.0)
        out = out[mask_keep]

    # Events outside the eligible broad event types are excluded.
    if cols.event_type in out.columns:
        allowed = set(cfg.allowed_types)
        mask_keep = out[cols.event_type].isin(allowed)
        out = out[mask_keep]

    # Offside events are excluded.
    if cols.event_name in out.columns:
        excluded = set(cfg.excluded_eventnames)
        mask_keep = ~out[cols.event_name].isin(excluded)
        out = out[mask_keep]

    return out


def build_event_index(
    manifest: pd.DataFrame, cols: Columns, cfg: DtwConfig
) -> tuple[dict[str, MatchInfo], EventIndex]:
    match_map: dict[str, MatchInfo] = {}

    incorrect: List[EventRow] = []
    correct: List[EventRow] = []

    need = [
        cols.match_id,
        cols.frame,
        cols.time,
        cols.event_time,
        cols.event_flag,
        cols.incorrect,
        cols.event_name,
        cols.event_type,
        cols.referee_name,
    ]
    for _, r in manifest.iterrows():
        path = Path(r["file"])
        mid = str(r[cols.match_id])
        match_map[mid] = MatchInfo(match_id=mid, path=path)

        df = pd.read_parquet(path, columns=[c for c in need if c is not None])
        df = df.sort_values(cols.frame).copy()

        df = filter_event_table(df, cols, cfg)

        if df.empty:
            continue

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            fr = int(row[cols.frame])
            t = (
                float(row[cols.time])
                if (cols.time in df.columns and not pd.isna(row[cols.time]))
                else np.nan
            )

            evn = (
                _safe_str(row[cols.event_name]) if cols.event_name in df.columns else ""
            )
            typ = (
                _safe_str(row[cols.event_type]) if cols.event_type in df.columns else ""
            )
            refn = (
                _norm_referee(row[cols.referee_name])
                if cols.referee_name in df.columns
                else ""
            )

            inc = int(row[cols.incorrect]) == 1

            er = EventRow(
                match_id=mid,
                frame=fr,
                time=t,
                event_idx=int(i),
                event_name=evn,
                event_type=typ,
                referee_name=refn,
            )
            if inc:
                incorrect.append(er)
            else:
                correct.append(er)

    correct_by_match: Dict[str, List[EventRow]] = {}
    correct_by_ref_type: Dict[tuple[str, str], List[EventRow]] = {}

    for er in correct:
        correct_by_match.setdefault(er.match_id, []).append(er)

        # Events without a referee identifier are not added to a shared pool.
        if er.referee_name != "":
            correct_by_ref_type.setdefault((er.referee_name, er.event_type), []).append(
                er
            )

    return match_map, EventIndex(
        incorrect=incorrect,
        correct=correct,
        correct_by_match=correct_by_match,
        correct_by_ref_type=correct_by_ref_type,
    )


def choose_candidate_pool(q: EventRow, index: EventIndex) -> List[EventRow]:
    """Correct events from the same referee and event type are returned."""
    if q.referee_name == "":
        return []
    return index.correct_by_ref_type.get((q.referee_name, q.event_type), [])
