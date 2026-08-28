from __future__ import annotations

from pathlib import Path
from typing import Optional
from collections import OrderedDict

import numpy as np
import pandas as pd

from .columns import Columns
from .config import DtwConfig
from .filters import EventRow, EventIndex, choose_candidate_pool
from .io import MatchInfo, MatchStore
from .windows import window_matrix
from .dtw_core import dtw_distance, two_phase_retrieve


class SeqCache:
    def __init__(self, max_items: int):
        self.max_items = max_items
        self._cache: "OrderedDict[tuple[str, int], np.ndarray]" = OrderedDict()

    def get(self, key: tuple[str, int]) -> Optional[np.ndarray]:
        if key in self._cache:
            v = self._cache.pop(key)
            self._cache[key] = v
            return v
        return None

    def put(self, key: tuple[str, int], value: np.ndarray) -> None:
        self._cache[key] = value
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


def _stable_seed(q: EventRow) -> int:
    s = (
        q.match_id + "|" + q.event_name + "|" + q.event_type + "|" + str(q.frame)
    ).encode("utf-8")
    b = s[:8].ljust(8, b"\0")
    return int(np.frombuffer(b, dtype=np.uint64)[0] % (2**32 - 1))


def run_retrieval(
    match_map: dict[str, MatchInfo],
    index: EventIndex,
    store: MatchStore,
    cols: Columns,
    cfg: DtwConfig,
    scaler,
    out_matches_csv: Path,
) -> None:
    feature_cols = list(cfg.feature_cols)

    needed_cols = set(feature_cols + [cols.frame, cols.time])

    seq_cache = SeqCache(cfg.max_seq_cache)

    def get_seq(match_id: str, frame: int) -> Optional[np.ndarray]:
        key = (match_id, int(frame))
        cached = seq_cache.get(key)
        if cached is not None:
            return cached

        info = match_map[match_id]
        df = store.get(match_id, info.path, columns=list(needed_cols))
        df = df.sort_values(cols.frame)

        X = window_matrix(df, frame, cols=cols, cfg=cfg, feature_cols=feature_cols)
        if X is None:
            return None

        Xs = scaler.transform(X)
        seq_cache.put(key, Xs)
        return Xs

    main_rows: list[dict] = []

    for q in index.incorrect:
        q_seq = get_seq(q.match_id, q.frame)
        if q_seq is None:
            continue

        cand_pool = choose_candidate_pool(q, index)

        # Correct events from the same match, referee, and event type are selected.
        within_all = index.correct_by_match.get(q.match_id, [])
        within = [
            candidate
            for candidate in within_all
            if candidate.event_type == q.event_type
            and candidate.referee_name == q.referee_name
        ]

        best_kth = 1e308
        cand_all: list[tuple[str, int, float, int, float, str, str]] = []

        # Exact DTW is calculated for all eligible within-match candidates.
        for c in within:
            c_seq = get_seq(c.match_id, c.frame)
            if c_seq is None:
                continue

            # A longer candidate is aligned to the query endpoint.
            c_use = c_seq[-len(q_seq) :] if len(c_seq) > len(q_seq) else c_seq

            d = dtw_distance(
                q_seq, c_use, band_frac=cfg.band_frac, best_so_far=best_kth
            )
            cand_all.append(
                (
                    c.match_id,
                    c.frame,
                    float(d),
                    c.event_idx,
                    c.time,
                    c.event_name,
                    c.event_type,
                )
            )

            if len(cand_all) >= cfg.top_k:
                cand_all.sort(key=lambda t: (t[2], t[1]))
                best_kth = cand_all[cfg.top_k - 1][2]

        # Cross-match candidates are restricted to the same referee and event type.
        cross = [c for c in cand_pool if c.match_id != q.match_id]
        if cross:
            cross_seqs: list[np.ndarray] = []
            cross_meta: list[EventRow] = []
            for c in cross:
                c_seq = get_seq(c.match_id, c.frame)
                if c_seq is None:
                    continue
                cross_seqs.append(c_seq)
                cross_meta.append(c)

            if cross_seqs:
                cross_idx_dist = two_phase_retrieve(
                    q_seq=q_seq,
                    cand_seqs=cross_seqs,
                    k=cfg.top_k,
                    band_frac=cfg.band_frac,
                    initial_best=best_kth,
                    seed=_stable_seed(q),
                )
                for idx, dist in cross_idx_dist:
                    c = cross_meta[int(idx)]
                    cand_all.append(
                        (
                            c.match_id,
                            c.frame,
                            float(dist),
                            c.event_idx,
                            c.time,
                            c.event_name,
                            c.event_type,
                        )
                    )

        if not cand_all:
            continue

        cand_all.sort(key=lambda t: (t[2], t[1]))
        chosen = cand_all[: cfg.top_k]

        for rank, (
            best_mid,
            best_fr,
            best_dist,
            best_eidx,
            best_time,
            best_ev,
            best_tp,
        ) in enumerate(chosen, start=1):
            row_out = {
                "inc_mergeid": q.match_id,
                "inc_event_idx": int(q.event_idx),
                "inc_frame": int(q.frame),
                "inc_time": float(q.time) if not np.isnan(q.time) else np.nan,
                "inc_eventname": q.event_name,
                "inc_type": q.event_type,
                "best_mergeid": best_mid,
                "best_cor_event_idx": int(best_eidx),
                "best_cor_frame": int(best_fr),
                "best_cor_time": float(best_time)
                if not (best_time is None or np.isnan(best_time))
                else np.nan,
                "best_eventname": best_ev,
                "best_type": best_tp,
                "dtw": float(best_dist),
                "rank": int(rank),
            }

            main_rows.append(row_out)

    if not main_rows:
        raise RuntimeError("No eligible DTW matches were retrieved.")

    matches = pd.DataFrame(main_rows).sort_values(
        ["inc_mergeid", "inc_event_idx", "rank"]
    )
    matches.to_csv(out_matches_csv, index=False)
