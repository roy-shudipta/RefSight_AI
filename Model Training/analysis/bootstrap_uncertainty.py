"""Query-level bootstrap intervals are calculated for the consensus model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_FEATURE_COLUMNS, OUTPUT_DIR, FPS  # noqa: E402

WINDOW_FRAMES = int(FPS * 5)
SEQ_DIR = Path(OUTPUT_DIR) / "sequences"
CONSENSUS_CSV = (
    PROJECT_ROOT
    / "results"
    / "aggregate"
    / "consensus_risk"
    / "consensus_clip_risk.csv"
)
OUT_DIR = PROJECT_ROOT / "results" / "aggregate" / "bootstrap_uncertainty"

K_MAX_RANK = 7
CI_LOW, CI_HIGH = 2.5, 97.5

FEATURE_LABELS = {
    "referee_speed": "Referee speed",
    "ball_height": "Ball height",
    "ball_speed": "Ball speed",
    "referee_to_ball_distance": "Referee to ball distance",
    "interaction_angle": "Interaction angle",
    "occlusion_count": "Occlusion count",
    "lateral_view_angle": "Lateral view angle",
    "referee_to_lead_assistant_distance": "Referee to lead assistant distance",
    "ball_angle_referee_vs_lead_assistant": "Ball angle (referee vs lead assistant)",
    "complementary_coverage_index": "Complementary coverage index",
    "distance_covered_previous_5s": "Distance covered (previous 5 s)",
    "trajectory_directness_5s": "Trajectory directness (5 s)",
    "fastest_player_speed": "Fastest player speed",
    "closest_player_speed": "Closest player speed",
    "turn_angle": "Turn angle",
    "trajectory_directness_2s": "Trajectory directness (2 s)",
    "home_team_length": "Home team length",
    "home_team_width": "Home team width",
    "home_team_stretch_index": "Home team stretch index",
    "home_team_spread": "Home team spread",
    "home_team_surface_area": "Home team surface area",
    "home_team_centroid_x": "Home team centroid x",
    "home_team_centroid_y": "Home team centroid y",
    "home_team_centroid_to_ball_distance": "Home team centroid to ball distance",
    "away_team_length": "Away team length",
    "away_team_width": "Away team width",
    "away_team_stretch_index": "Away team stretch index",
    "away_team_spread": "Away team spread",
    "away_team_surface_area": "Away team surface area",
    "away_team_centroid_x": "Away team centroid x",
    "away_team_centroid_y": "Away team centroid y",
    "away_team_centroid_to_ball_distance": "Away team centroid to ball distance",
}


def schema_index():
    """Feature names are mapped to stored array columns."""
    with open(Path(OUTPUT_DIR) / "schema.json") as f:
        ordered = json.load(f)["ordered_columns"]
    if tuple(ordered) != MODEL_FEATURE_COLUMNS:
        raise ValueError("The stored feature order does not match config.py.")
    return {n: i for i, n in enumerate(ordered)}


def final_means(arr, col_idx, shared_length):
    """Final five-second feature means are calculated over finite values."""
    X = np.asarray(arr, dtype=float)[-int(shared_length) :, :]
    out = np.full(len(col_idx), np.nan)

    for k, c in enumerate(col_idx):
        col = X[-WINDOW_FRAMES:, c]
        finite = np.isfinite(col)
        if finite.any():
            out[k] = float(np.mean(col[finite]))

    return out


def build_clip_table(features):
    """Consensus risks and final five-second feature means are combined."""
    if not CONSENSUS_CSV.exists():
        raise FileNotFoundError(
            f"{CONSENSUS_CSV} not found. Run "
            "consensus_aggregation/aggregate_consensus_risk.py first."
        )

    cons = pd.read_csv(CONSENSUS_CSV)
    cons["consensus_risk"] = pd.to_numeric(cons["consensus_risk"], errors="coerce")
    cons["candidate_rank"] = pd.to_numeric(cons["candidate_rank"], errors="coerce")
    cons = cons.dropna(subset=["consensus_risk", "query_id", "candidate_rank"])
    sidx = schema_index()
    col_idx = [sidx[c] for c in features]

    if "n_frames" not in cons.columns:
        raise ValueError("The consensus table does not contain n_frames.")
    shared_lengths = cons.groupby("query_id")["n_frames"].min().to_dict()
    feature_rows = []
    for _, row in cons.iterrows():
        local = SEQ_DIR / Path(str(row["file_path"])).name
        if not local.exists():
            raise FileNotFoundError(f"A held-out sequence was not found at {local}")
        means = final_means(
            np.load(local, allow_pickle=False),
            col_idx,
            shared_lengths[row["query_id"]],
        )
        feature_rows.append(means.tolist())

    df = cons.copy()
    df.loc[:, features] = pd.DataFrame(feature_rows, columns=features, index=df.index)

    df["y"] = (df["decision_type"].astype(str).str.lower() == "incorrect").astype(int)
    return df


def ranking_from_nabove(n_above):
    """Top-1, Top-2, and MRR are calculated from candidate ranks."""
    return (
        float(np.mean(n_above == 0)),
        float(np.mean(n_above <= 1)),
        float(np.mean(1.0 / (1.0 + n_above))),
    )


def calibration(p, y, n_bins=10):
    """ECE, Brier score, and AUROC are calculated."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.digitize(p, edges[1:-1], right=True)
    ece, N = 0.0, len(p)

    for k in range(len(edges) - 1):
        m = b == k
        if m.any():
            ece += (m.sum() / N) * abs(p[m].mean() - y[m].mean())

    brier = float(np.mean((p - y) ** 2))
    auroc = np.nan

    try:
        from sklearn.metrics import roc_auc_score

        if y.min() != y.max():
            auroc = float(roc_auc_score(y, p))
    except Exception:
        pass

    return float(ece), brier, auroc


def best_split(values, risk, n_bins=6, min_bin_count=30, min_step_pp=0.5):
    """The largest supported risk step is selected across adjacent bin cuts."""
    ok = np.isfinite(values) & np.isfinite(risk)
    v, r = values[ok], risk[ok]

    if v.size < n_bins * 2:
        return np.nan, "none", np.nan

    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return np.nan, "none", np.nan

    b = np.clip(np.digitize(v, edges[1:-1]), 0, len(edges) - 2)
    nb = len(edges) - 1
    cnt = np.bincount(b, minlength=nb).astype(float)
    rsum = np.bincount(b, weights=r, minlength=nb)
    bin_low = np.array([v[b == index].min() for index in range(nb)])
    bin_high = np.array([v[b == index].max() for index in range(nb)])
    midpoints = 0.5 * (bin_low + bin_high)
    weights = np.where(cnt >= min_bin_count, cnt, 0.0)

    best = None
    for s in range(1, nb):
        lo_c, hi_c = weights[:s].sum(), weights[s:].sum()
        if lo_c <= 0 or hi_c <= 0:
            continue

        lo_r = np.sum(rsum[:s] * (weights[:s] > 0)) / lo_c
        hi_r = np.sum(rsum[s:] * (weights[s:] > 0)) / hi_c
        delta = abs(hi_r - lo_r)

        if best is None or delta > best[0]:
            thr = 0.5 * (midpoints[s - 1] + midpoints[s])
            direction = "high" if hi_r > lo_r else "low"
            best = (delta, float(thr), direction, float(delta * 100.0))

    if best is None or best[3] < min_step_pp:
        return np.nan, "none", np.nan

    return best[1], best[2], best[3]


def ci(arr):
    """A bootstrap median and percentile 95% interval are calculated."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan, np.nan, np.nan

    return (
        float(np.median(arr)),
        float(np.percentile(arr, CI_LOW)),
        float(np.percentile(arr, CI_HIGH)),
    )


def main():
    """Matched query sets are resampled and uncertainty tables are written."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20240620)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    features = [f for f in FEATURE_LABELS if f in schema_index()]
    df = build_clip_table(features)
    print(
        f"[data] clips={len(df)} queries={df['query_id'].nunique()} features={len(features)}"
    )

    rank_df = df[df["candidate_rank"].between(0, K_MAX_RANK)]
    nabove_by_q = {}
    for qid, g in rank_df.groupby("query_id"):
        inc = g[g["candidate_rank"] == 0]
        if len(inc) != 1:
            continue

        r0 = float(inc["consensus_risk"].iloc[0])

        nabove_by_q[qid] = int((g["consensus_risk"].to_numpy() > r0).sum())

    q_rank = np.array(list(nabove_by_q.keys()))
    nabove = np.array([nabove_by_q[q] for q in q_rank])

    queries = df["query_id"].to_numpy()
    uq = pd.unique(queries)
    q_to_rows = {q: np.where(queries == q)[0] for q in uq}
    risk = df["consensus_risk"].to_numpy(float)
    y = df["y"].to_numpy(int)
    featmat = df[features].to_numpy(float)

    p_top1, p_top2, p_mrr = ranking_from_nabove(nabove)
    p_ece, p_brier, p_auroc = calibration(risk, y)
    point_thr = {f: best_split(featmat[:, j], risk) for j, f in enumerate(features)}

    B = args.bootstrap
    perf = np.empty((B, 3))
    cal = np.empty((B, 3))
    thr_vals = {f: [] for f in features}
    dpp_vals = {f: [] for f in features}
    dir_vals = {f: [] for f in features}

    for b in range(B):
        drawn_rank = rng.integers(0, len(q_rank), size=len(q_rank))
        perf[b] = ranking_from_nabove(nabove[drawn_rank])

        drawn = rng.integers(0, len(uq), size=len(uq))
        idx = np.concatenate([q_to_rows[uq[d]] for d in drawn])
        cal[b] = calibration(risk[idx], y[idx])

        for j, f in enumerate(features):
            t, d, dpp = best_split(featmat[idx, j], risk[idx])
            thr_vals[f].append(t)
            dpp_vals[f].append(dpp)
            dir_vals[f].append(d)

        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{B}")

    perf_rows = []
    for j, name in enumerate(["Top-1 accuracy", "Top-2 accuracy", "MRR"]):
        _, lo, hi = ci(perf[:, j])
        perf_rows.append(
            {
                "metric": name,
                "point": round([p_top1, p_top2, p_mrr][j], 4),
                "ci_lower_95": round(lo, 4),
                "ci_upper_95": round(hi, 4),
            }
        )
    pd.DataFrame(perf_rows).to_csv(OUT_DIR / "bootstrap_performance.csv", index=False)

    cal_rows = []
    for j, name in enumerate(["ECE", "Brier", "AUROC"]):
        _, lo, hi = ci(cal[:, j])
        pt = [p_ece, p_brier, p_auroc][j]
        cal_rows.append(
            {
                "metric": name,
                "point": round(pt, 4),
                "ci_lower_95": round(lo, 4),
                "ci_upper_95": round(hi, 4),
            }
        )
    pd.DataFrame(cal_rows).to_csv(OUT_DIR / "bootstrap_calibration.csv", index=False)

    thr_rows = []
    for f in features:
        dirs = [d for d in dir_vals[f] if d in ("low", "high")]
        dom = max(set(dirs), key=dirs.count) if dirs else "none"
        stability = (dirs.count(dom) / len(dirs)) if dirs else np.nan

        t_med, t_lo, t_hi = ci(thr_vals[f])
        d_med, d_lo, d_hi = ci(dpp_vals[f])
        pt_thr, pt_dir, pt_dpp = point_thr[f]

        thr_rows.append(
            {
                "feature": f,
                "feature_label": FEATURE_LABELS.get(f, f),
                "direction": pt_dir,
                "direction_stability": (
                    round(stability, 3) if np.isfinite(stability) else np.nan
                ),
                "threshold_point": round(pt_thr, 4) if np.isfinite(pt_thr) else np.nan,
                "threshold_ci_low": round(t_lo, 4),
                "threshold_ci_high": round(t_hi, 4),
                "delta_risk_pp_point": round(pt_dpp, 3)
                if np.isfinite(pt_dpp)
                else np.nan,
                "delta_risk_pp_ci_low": round(d_lo, 3),
                "delta_risk_pp_ci_high": round(d_hi, 3),
            }
        )
    pd.DataFrame(thr_rows).sort_values("delta_risk_pp_point", ascending=False).to_csv(
        OUT_DIR / "bootstrap_thresholds.csv", index=False
    )

    print("\n=== Performance (95% query bootstrap CI) ===")
    print(pd.DataFrame(perf_rows).to_string(index=False))
    print("\n=== Calibration ===")
    print(pd.DataFrame(cal_rows).to_string(index=False))
    print(f"\nWrote performance / calibration / threshold CIs to {OUT_DIR}")


if __name__ == "__main__":
    main()
