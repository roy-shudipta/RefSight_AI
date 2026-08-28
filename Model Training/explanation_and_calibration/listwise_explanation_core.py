"""Listwise permutation, calibration, and risk profiles are calculated."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from config import (
    OUTPUT_DIR,
    RESULTS_DIR,
    TGT_LEN,
    MODEL_NAME,
    TOP_K_CORRECT,
    BATCH_SIZE,
)
from utils import get_device, set_seed
from data_prep import (
    load_metadata,
    split_dataset,
    fit_scaler,
    filter_valid_queries_listwise,
    build_query_lists,
)
from models import build_model


# Full-series permutation settings are defined here.
PERM_ENABLE = True
PERM_BATCH_SIZE = 32
PERM_METRIC = "top1"
PERM_REPEATS = 5
PERM_SEED = 7

# Event-aligned windowed permutation settings are defined here.
WINDOW_ENABLE = True
WINDOW_BATCH_SIZE = 16
WINDOW_METRIC = "top1"
WINDOW_FRAMES = 50
WINDOW_STEP_FRAMES = 25
WINDOW_SEED = 7

# Calibrated risk-profile settings are defined here.
RISK_ENABLE = True
RISK_LAST_SECONDS = 5
RISK_FPS = 25
RISK_N_BINS = 6
RISK_MIN_COUNT_PER_BIN = 30


def read_schema_columns(schema_path: Path) -> List[str]:
    """The stored feature order is returned."""
    with open(schema_path, "r") as f:
        obj = json.load(f)

    cols = obj.get("ordered_columns", None)
    if not cols or not isinstance(cols, list):
        raise ValueError(f"schema.json missing 'ordered_columns': {schema_path}")
    return [str(c) for c in cols]


def model_input_feature_names(
    schema_path: Path, scaler: Dict[str, np.ndarray]
) -> np.ndarray:
    """Feature names are returned in model input order."""
    all_cols = read_schema_columns(schema_path)
    keep_cols = np.asarray(scaler["keep_cols"]).astype(int)
    return np.array([all_cols[index] for index in keep_cols], dtype=object)


def _scaler_statistics(scaler: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Training means and standard deviations are returned."""
    return (
        np.asarray(scaler["mean"], dtype=np.float32),
        np.asarray(scaler["std"], dtype=np.float32),
    )


@torch.no_grad()
def collect_listwise_to_cpu(loader) -> Dict[str, np.ndarray]:
    """Listwise batches are collected as CPU arrays."""
    sequences, lengths = [], []
    for batch_sequences, batch_lengths, _, _ in loader:
        sequences.append(
            batch_sequences.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        lengths.append(
            batch_lengths.detach().cpu().numpy().astype(np.int32, copy=False)
        )
    return {
        "X": np.concatenate(sequences, axis=0),
        "L": np.concatenate(lengths, axis=0),
    }


@torch.no_grad()
def scores_via_arrays(
    model: nn.Module,
    X: np.ndarray,
    L: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Candidate scores are calculated from listwise arrays."""
    model.eval()
    query_count, candidate_count, time_steps, feature_count = X.shape
    output = np.empty((query_count, candidate_count), dtype=np.float32)

    for start in range(0, query_count, batch_size):
        end = min(query_count, start + batch_size)
        sequences = torch.from_numpy(X[start:end]).to(device)
        lengths = torch.from_numpy(L[start:end]).to(device)
        current_batch_size = sequences.shape[0]
        flat_sequences = sequences.view(
            current_batch_size * candidate_count,
            time_steps,
            feature_count,
        )
        flat_lengths = lengths.view(current_batch_size * candidate_count)
        flat_scores = model(flat_sequences, flat_lengths)
        output[start:end] = (
            flat_scores.view(current_batch_size, candidate_count)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    return output


def _topk_acc(scores: np.ndarray, k: int) -> float:
    """The proportion of queries with candidate zero in the top k is returned."""
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores, axis=1)
    hits = np.any(order[:, :k] == 0, axis=1)
    return float(hits.mean()) if hits.size else float("nan")


def _mrr(scores: np.ndarray) -> float:
    """Mean reciprocal rank is calculated for candidate zero."""
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores, axis=1)
    positions = np.argmax(order == 0, axis=1) + 1
    return float(np.mean(1.0 / positions)) if positions.size else float("nan")


def compute_metric(scores: np.ndarray, metric: str) -> float:
    """The selected listwise metric is calculated."""
    metric = str(metric).lower()
    if metric == "top1":
        return _topk_acc(scores, k=1)
    if metric == "top2":
        return _topk_acc(scores, k=2)
    if metric == "mrr":
        return _mrr(scores)
    raise ValueError("metric must be one of: top1, top2, mrr")


def _permute_feature_listwise_inplace(
    X: np.ndarray,
    feat_idx: int,
    *,
    rng: np.random.Generator,
) -> None:
    """One full feature series is shuffled across clips."""
    N, M, _, _ = X.shape
    j = int(feat_idx)
    clip_indices = [(query, candidate) for query in range(N) for candidate in range(M)]
    original = np.stack(
        [X[query, candidate, :, j] for query, candidate in clip_indices]
    )
    for target, source in enumerate(rng.permutation(len(clip_indices))):
        query, candidate = clip_indices[target]
        X[query, candidate, :, j] = original[source]


def permutation_importance_listwise(
    model: nn.Module,
    data: Dict[str, np.ndarray],
    feature_names: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    metric: str,
    n_repeats: int,
    seed: int,
) -> Dict[str, Any]:
    """Full-series permutation importance is calculated for every feature."""
    X = data["X"]
    L = data["L"]

    names = feature_names
    idxs = np.arange(len(feature_names))

    base_scores = scores_via_arrays(model, X, L, device=device, batch_size=batch_size)
    baseline = compute_metric(base_scores, metric)
    print(f"[Permutation] Baseline {metric} = {baseline:.3f}", flush=True)

    rng = np.random.default_rng(seed)
    drops = np.zeros((len(idxs),), dtype=np.float32)
    X_work = X.copy()

    # One full feature series is shuffled across candidate clips.
    for fi, j in enumerate(idxs):
        rep_vals = []
        for _ in range(n_repeats):
            _permute_feature_listwise_inplace(X_work, int(j), rng=rng)
            sc = scores_via_arrays(
                model, X_work, L, device=device, batch_size=batch_size
            )
            rep_vals.append(float(compute_metric(sc, metric)))
            X_work[:] = X

        score_mean = float(np.mean(rep_vals))
        drops[fi] = 100.0 * float(baseline - score_mean)

        if (fi + 1) % 5 == 0 or (fi + 1) == len(idxs):
            print(f"[Permutation] done {fi+1}/{len(idxs)} features", flush=True)

    return {
        "names": names,
        "idxs": idxs.astype(int),
        "drops": drops.astype(np.float32),
        "baseline": float(baseline),
        "metric": metric,
        "n_repeats": int(n_repeats),
        "units": "percentage points",
    }


def _permute_window_rel_end_inplace(
    X: np.ndarray,
    L: np.ndarray,
    *,
    feat_idx: int,
    offset: int,
    window: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One event-aligned feature window is shuffled across candidate clips."""
    N, M, T, _ = X.shape
    j = int(feat_idx)
    clip_count = N * M
    flat_lengths = L.reshape(clip_count).astype(np.int32)
    ends = np.clip(flat_lengths - int(offset), 0, T)
    starts = np.maximum(ends - int(window), 0)
    segment_lengths = (ends - starts).astype(np.int32)
    buffered = np.zeros((clip_count, window), dtype=np.float32)

    valid = np.flatnonzero(segment_lengths > 0)
    for clip in valid:
        query, candidate = divmod(int(clip), M)
        start = int(starts[clip])
        end = int(ends[clip])
        length = int(segment_lengths[clip])
        buffered[clip, :length] = X[query, candidate, start:end, j]

    for target, source in zip(valid, rng.permutation(valid)):
        target_length = int(segment_lengths[target])
        source_length = int(segment_lengths[source])
        shared_length = min(target_length, source_length)
        if shared_length == 0:
            continue
        query, candidate = divmod(int(target), M)
        target_end = int(ends[target])
        X[query, candidate, target_end - shared_length : target_end, j] = buffered[
            source,
            source_length - shared_length : source_length,
        ]

    return buffered, starts, segment_lengths


def _restore_window_rel_end_inplace(
    X: np.ndarray,
    *,
    feat_idx: int,
    buf_vals: np.ndarray,
    buf_start: np.ndarray,
    buf_len: np.ndarray,
) -> None:
    """Original feature-window values are restored."""
    N, M, _, _ = X.shape
    j = int(feat_idx)
    for clip in np.flatnonzero(buf_len > 0):
        query, candidate = divmod(int(clip), M)
        start = int(buf_start[clip])
        length = int(buf_len[clip])
        X[query, candidate, start : start + length, j] = buf_vals[clip, :length]


def windowed_permutation_importance_listwise(
    model: nn.Module,
    data: Dict[str, np.ndarray],
    feature_names: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    metric: str,
    window: int,
    step: int,
    seed: int,
) -> Dict[str, Any]:
    """Event-aligned windowed permutation is calculated for every feature."""
    X = data["X"]
    L = data["L"]

    _, _, T, _ = X.shape
    base_scores = scores_via_arrays(model, X, L, device=device, batch_size=batch_size)
    baseline = compute_metric(base_scores, metric)
    print(f"[Temporal permutation] Baseline {metric} = {baseline:.3f}", flush=True)
    rng = np.random.default_rng(seed)

    idxs = np.arange(len(feature_names))
    names = np.array([feature_names[j] for j in idxs], dtype=object)

    offsets = list(range(0, max(1, T - window + 1), step))
    W = len(offsets)
    drops = np.zeros((len(idxs), W), dtype=np.float32)

    X_work = X.copy()

    # Each feature window is shuffled across candidate clips.
    for fi, j in enumerate(idxs):
        j = int(j)
        for wi, off in enumerate(offsets):
            buf_vals, buf_start, buf_len = _permute_window_rel_end_inplace(
                X_work,
                L,
                feat_idx=j,
                offset=int(off),
                window=int(window),
                rng=rng,
            )

            sc = scores_via_arrays(
                model, X_work, L, device=device, batch_size=batch_size
            )
            score = compute_metric(sc, metric)
            drops[fi, wi] = 100.0 * float(baseline - score)

            _restore_window_rel_end_inplace(
                X_work,
                feat_idx=j,
                buf_vals=buf_vals,
                buf_start=buf_start,
                buf_len=buf_len,
            )

        if (fi + 1) % 5 == 0 or (fi + 1) == len(idxs):
            print(
                f"[Temporal permutation] done {fi+1}/{len(idxs)} features", flush=True
            )

    return {
        "names": names,
        "idxs": idxs.astype(int),
        "offsets": np.array(offsets, dtype=int),
        "window": int(window),
        "step": int(step),
        "drops": drops,
        "baseline": float(baseline),
        "metric": metric,
        "units": "percentage points",
    }


def build_single_clip_arrays(
    meta: pd.DataFrame,
    scaler: Dict[str, np.ndarray],
    *,
    tgt_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardised single-clip arrays are prepared for calibration."""
    from utils import coerce_numeric, pad_or_trim

    keep_cols = np.asarray(scaler["keep_cols"]).astype(int)
    mu_model, sd_model = _scaler_statistics(scaler)

    X_list: List[np.ndarray] = []
    L_list: List[int] = []
    y_list: List[int] = []

    m = meta.copy()
    if "label" not in m.columns:
        m["label"] = (m["decision_type"].astype(str).str.lower() == "incorrect").astype(
            int
        )
    shared_lengths = (
        m.groupby("query_id")["n_frames"]
        .min()
        .clip(lower=1, upper=tgt_len)
        .astype(int)
        .to_dict()
    )

    for _, row in m.iterrows():
        fp = row["file_path"]
        L = shared_lengths[row["query_id"]]

        X = coerce_numeric(np.load(fp, allow_pickle=False)).astype(
            np.float32, copy=False
        )
        X = X[:, keep_cols]

        if X.shape[0] > L:
            X = X[-L:, :]

        X = pad_or_trim(X, tgt_len)
        X = (X - mu_model) / sd_model
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_list.append(X.astype(np.float32, copy=False))
        L_list.append(L)
        y_list.append(int(row["label"]))

    X_all = np.stack(X_list, axis=0)
    L_all = np.asarray(L_list, dtype=np.int32)
    y_all = np.asarray(y_list, dtype=np.int32)
    return X_all, L_all, y_all


@torch.no_grad()
def score_single_clips(
    model: nn.Module,
    X: np.ndarray,
    L: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """One score is calculated for each clip."""
    model.eval()
    N = X.shape[0]
    out = np.empty((N,), dtype=np.float32)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        tX = torch.from_numpy(X[s:e]).to(device)
        tL = torch.from_numpy(L[s:e]).to(device)
        out[s:e] = model(tX, tL).detach().cpu().numpy().astype(np.float32)
    return out


def summarise_feature_last_k_seconds(
    X_std: np.ndarray,
    L: np.ndarray,
    feat_idx: int,
    *,
    mu: float,
    sd: float,
    fps: int,
    k_seconds: int,
) -> np.ndarray:
    """A final-window feature mean is calculated in the original units."""
    N, T, _ = X_std.shape
    win = int(max(1, round(k_seconds * fps)))

    out = np.empty((N,), dtype=np.float32)

    for i in range(N):
        Li = int(L[i])
        s = max(0, Li - win)
        e = Li
        xs = X_std[i, s:e, feat_idx]
        xs = xs * sd + mu

        out[i] = float(xs.mean())
    return out


def fit_isotonic_calibrator(scores: np.ndarray, y: np.ndarray):
    """Isotonic calibration is fitted on validation scores."""
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception as e:
        raise RuntimeError(
            "scikit learn is not available for isotonic calibration."
        ) from e

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(scores.astype(np.float64), y.astype(np.int32))
    return iso


def bin_table(
    values: np.ndarray, p: np.ndarray, *, n_bins: int, min_count: int
) -> pd.DataFrame:
    """Mean predicted risk is calculated within feature quantile bins."""
    v = np.asarray(values, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(v, qs)
    edges = np.unique(edges)

    if edges.size < 3:
        edges = np.array([v.min(), (v.min() + v.max()) / 2, v.max()], dtype=np.float64)

    bin_id = np.digitize(v, edges[1:-1], right=True)

    rows = []
    for b in range(int(bin_id.min()), int(bin_id.max()) + 1):
        idx = np.where(bin_id == b)[0]
        if idx.size == 0:
            continue
        rows.append(
            {
                "bin": int(b),
                "lo": float(v[idx].min()),
                "hi": float(v[idx].max()),
                "n": int(idx.size),
                "risk_mean": float(p[idx].mean()),
                "risk_std": float(p[idx].std(ddof=0)),
            }
        )

    df = pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)
    df["warn_low_n"] = df["n"] < int(min_count)
    return df


def main():
    """The selected explanation and calibration stages are completed."""
    set_seed()
    device = get_device()
    print("Device:", device, flush=True)

    out_dir = Path(RESULTS_DIR)
    schema_path = OUTPUT_DIR / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema.json at {schema_path}")

    meta = load_metadata()
    meta = filter_valid_queries_listwise(meta, top_k=TOP_K_CORRECT)

    meta_train, meta_val, meta_test, q_tr, q_va, q_te = split_dataset(meta)
    print(f"Queries: train={len(q_tr)}, val={len(q_va)}, test={len(q_te)}", flush=True)
    print(
        f"Rows   : train={len(meta_train)}, val={len(meta_val)}, test={len(meta_test)}",
        flush=True,
    )

    scaler = fit_scaler(meta_train)
    print(f"[setup] model feature_dim = {scaler['feature_dim']}", flush=True)

    model = build_model(int(scaler["feature_dim"])).to(device)
    ckpt_path = out_dir / f"listwise_model_{str(MODEL_NAME).lower()}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[model] loaded {ckpt_path}", flush=True)

    feat_names = model_input_feature_names(schema_path, scaler)
    print(f"[features] names_in_model={len(feat_names)}", flush=True)

    data = None
    if PERM_ENABLE or WINDOW_ENABLE:
        from torch.utils.data import DataLoader
        from dataset import QueryListDataset

        test_items = build_query_lists(meta_test, top_k=TOP_K_CORRECT)
        use_mps = device.type == "mps"
        test_loader = DataLoader(
            QueryListDataset(test_items, scaler),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0 if use_mps else 2,
            pin_memory=not use_mps,
        )
        data = collect_listwise_to_cpu(test_loader)

    perm_imp = None
    if PERM_ENABLE:
        perm_imp = permutation_importance_listwise(
            model=model,
            data=data,
            feature_names=feat_names,
            device=device,
            batch_size=PERM_BATCH_SIZE,
            metric=PERM_METRIC,
            n_repeats=PERM_REPEATS,
            seed=PERM_SEED,
        )

        out_perm = out_dir / f"permutation_importance_listwise_{PERM_METRIC}.csv"
        dfp = pd.DataFrame(
            {
                "feature": perm_imp["names"],
                "drop": perm_imp["drops"],
                "units": perm_imp["units"],
            }
        )
        dfp.sort_values("drop", ascending=False).to_csv(out_perm, index=False)
        print(f"Saved: {out_perm}", flush=True)

    if WINDOW_ENABLE:
        windowed = windowed_permutation_importance_listwise(
            model=model,
            data=data,
            feature_names=feat_names,
            device=device,
            batch_size=WINDOW_BATCH_SIZE,
            metric=WINDOW_METRIC,
            window=WINDOW_FRAMES,
            step=WINDOW_STEP_FRAMES,
            seed=WINDOW_SEED,
        )

        drops = windowed["drops"]
        offsets = windowed["offsets"]
        best_w = drops.argmax(axis=1)

        peak_rows = []
        for i in range(drops.shape[0]):
            wi = int(best_w[i])
            peak_rows.append(
                {
                    "feature": str(windowed["names"][i]),
                    "peak_drop": float(drops[i, wi]),
                    "units": windowed["units"],
                    "offset_frames_before_event": int(offsets[wi]),
                    "window_frames": int(windowed["window"]),
                }
            )

        df_peak = (
            pd.DataFrame(peak_rows)
            .sort_values("peak_drop", ascending=False)
            .reset_index(drop=True)
        )
        out_peak = out_dir / f"windowed_permutation_peaks_{WINDOW_METRIC}.csv"
        df_peak.to_csv(out_peak, index=False)
        print(f"\nSaved: {out_peak}", flush=True)

        surf_rows = []
        for i in range(drops.shape[0]):
            for wi, off in enumerate(offsets):
                surf_rows.append(
                    {
                        "feature": str(windowed["names"][i]),
                        "offset_frames_before_event": int(off),
                        "drop": float(drops[i, wi]),
                        "units": windowed["units"],
                    }
                )
        out_surface = out_dir / f"windowed_permutation_surface_{WINDOW_METRIC}.csv"
        pd.DataFrame(surf_rows).to_csv(out_surface, index=False)
        print(f"Saved: {out_surface}", flush=True)

    if RISK_ENABLE:
        print(
            "\n[risk] building single clip arrays for calibration (VAL) and profiling (TEST)...",
            flush=True,
        )

        X_cal, L_cal, y_cal = build_single_clip_arrays(
            meta_val, scaler, tgt_len=TGT_LEN
        )
        X_test, L_test, y_test = build_single_clip_arrays(
            meta_test, scaler, tgt_len=TGT_LEN
        )

        print(f"[risk] val clips={len(y_cal)} | test clips={len(y_test)}", flush=True)
        print(
            f"[risk] y_val mean (incorrect rate)={y_cal.mean():.3f} | y_test mean={y_test.mean():.3f}",
            flush=True,
        )

        s_cal = score_single_clips(model, X_cal, L_cal, device=device, batch_size=256)
        s_test = score_single_clips(
            model, X_test, L_test, device=device, batch_size=256
        )

        cal = fit_isotonic_calibrator(s_cal, y_cal)

        p_test = np.asarray(cal.predict(s_test), dtype=np.float32)
        print(f"[risk] calibrated test mean risk={p_test.mean():.3f}", flush=True)

        cols = ["query_id", "decision_type", "candidate_rank", "file_path", "n_frames"]
        df_pred = meta_test[cols].copy()
        df_pred["score"] = s_test
        df_pred["p_incorrect"] = p_test
        out_pred = out_dir / "clip_risk_predictions_test.csv"
        df_pred.to_csv(out_pred, index=False)
        print(f"Saved: {out_pred}", flush=True)

        risk_features = [str(n) for n in feat_names.tolist()]

        name2idx = {str(n): i for i, n in enumerate(feat_names)}
        mu_model, sd_model = _scaler_statistics(scaler)

        all_rows = []

        for feat in risk_features:
            j = int(name2idx[feat])
            mu = float(mu_model[j])
            sd = float(sd_model[j])

            vals = summarise_feature_last_k_seconds(
                X_test,
                L_test,
                j,
                mu=mu,
                sd=sd,
                fps=RISK_FPS,
                k_seconds=RISK_LAST_SECONDS,
            )

            dfb = bin_table(
                vals,
                p_test,
                n_bins=RISK_N_BINS,
                min_count=RISK_MIN_COUNT_PER_BIN,
            )

            dfb.insert(0, "feature", feat)
            dfb.insert(1, "summary", f"mean_last_{RISK_LAST_SECONDS}s")

            all_rows.append(dfb)

        if not all_rows:
            print("[risk][WARN] No risk profiles were produced.", flush=True)
        else:
            df_all = pd.concat(all_rows, axis=0, ignore_index=True)

            df_all["feature"] = df_all["feature"].astype(str)
            df_all = df_all.sort_values(["feature", "bin"]).reset_index(drop=True)

            out_all = out_dir / "risk_profiles_all_features.csv"
            df_all.to_csv(out_all, index=False)
            print(f"[risk] Saved combined risk profiles: {out_all}", flush=True)


if __name__ == "__main__":
    main()
