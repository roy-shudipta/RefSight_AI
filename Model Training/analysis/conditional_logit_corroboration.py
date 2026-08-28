"""Matched conditional logistic regression is used for corroboration."""

from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import EXPECTED_TOTAL_QUERIES, OUTPUT_DIR, FPS, TOP_K_CORRECT
from data_prep import load_metadata, filter_valid_queries_listwise, build_query_lists

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "statistical_corroboration")

WINDOW_FRAMES = int(FPS * 5)

# Features outside the DTW matching variables are defined here.
ELIGIBLE_FEATURES = [
    ("trajectory_directness_5s", "Trajectory directness (5 s)", "Referee kinematics"),
    ("trajectory_directness_2s", "Trajectory directness (2 s)", "Referee kinematics"),
    (
        "distance_covered_previous_5s",
        "Distance covered (previous 5 s)",
        "Referee kinematics",
    ),
    ("referee_speed", "Referee speed", "Referee kinematics"),
    ("turn_angle", "Turn angle", "Referee kinematics"),
    ("referee_to_ball_distance", "Referee to ball distance", "Referee ball and view"),
    ("interaction_angle", "Interaction angle", "Referee ball and view"),
    ("occlusion_count", "Occlusion count", "Referee ball and view"),
    ("lateral_view_angle", "Lateral view angle", "Coverage geometry"),
    (
        "ball_angle_referee_vs_lead_assistant",
        "Ball angle (referee vs lead assistant)",
        "Coverage geometry",
    ),
    (
        "referee_to_lead_assistant_distance",
        "Referee to lead assistant distance",
        "Coverage geometry",
    ),
    (
        "complementary_coverage_index",
        "Complementary coverage index",
        "Coverage geometry",
    ),
    ("fastest_player_speed", "Fastest player speed", "Player speed context"),
    ("closest_player_speed", "Closest player speed", "Player speed context"),
    ("ball_speed", "Ball speed", "Ball state"),
    ("ball_height", "Ball height", "Ball state"),
]

# Features used for DTW matching are excluded from the regression.
EXCLUDED_MATCHING_FEATURES = [
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
]


def load_schema_index():
    """Feature names are mapped to stored array columns."""
    with open(os.path.join(OUTPUT_DIR, "schema.json")) as f:
        ordered = json.load(f)["ordered_columns"]
    eligible = {code for code, _, _ in ELIGIBLE_FEATURES}
    expected = eligible | set(EXCLUDED_MATCHING_FEATURES)
    if set(ordered) != expected:
        raise ValueError(
            "The model feature schema does not match the analysis features."
        )
    return {name: i for i, name in enumerate(ordered)}


def final_window_means(arr: np.ndarray, col_idx: list):
    """Final five-second means are calculated for selected features."""
    X = np.asarray(arr, dtype=float)
    out = np.full(len(col_idx), np.nan, dtype=float)
    for k, c in enumerate(col_idx):
        col = X[-WINDOW_FRAMES:, c]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        out[k] = float(np.mean(col[finite]))
    return out


def build_clip_table():
    """One analysis row is prepared for each candidate clip."""
    schema_idx = load_schema_index()
    codes = [c for c, _, _ in ELIGIBLE_FEATURES]
    col_idx = [schema_idx[c] for c in codes]

    meta = load_metadata()
    meta = filter_valid_queries_listwise(meta, top_k=TOP_K_CORRECT)
    query_count = int(meta["query_id"].nunique())
    if query_count != EXPECTED_TOTAL_QUERIES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_QUERIES} complete matched sets, "
            f"but {query_count} were found."
        )

    dec = meta[meta["candidate_rank"] == 0]
    ref_of_query = dict(zip(dec["query_id"], dec["Referee Name"].astype(str)))

    items = build_query_lists(meta, top_k=TOP_K_CORRECT)

    rows = []
    for qid, fps, lens, ranks in items:
        referee = ref_of_query.get(qid, "NA")
        for fp, _, rank in zip(fps, lens, ranks):
            arr = np.load(fp, allow_pickle=False)
            means = final_window_means(arr, col_idx)
            row = {
                "query_id": qid,
                "referee": referee,
                "candidate_rank": int(rank),
                "label": 1 if int(rank) == 0 else 0,
            }
            row.update({code: means[k] for k, code in enumerate(codes)})
            rows.append(row)

    df = pd.DataFrame(rows)
    print(
        f"[data] clips={len(df)}  queries={df['query_id'].nunique()}  "
        f"referees={df['referee'].nunique()}",
        flush=True,
    )
    return df


def clogit1d(x, strat, case_x, n_strata, max_iter=50, tol=1e-9):
    """One exact matched conditional logistic coefficient is estimated."""
    b = 0.0
    for _ in range(max_iter):
        eta = b * x
        stratum_max = np.full(n_strata, -np.inf, dtype=float)
        np.maximum.at(stratum_max, strat, eta)
        w = np.exp(eta - stratum_max[strat])
        S0 = np.bincount(strat, weights=w, minlength=n_strata)
        S1 = np.bincount(strat, weights=w * x, minlength=n_strata)
        S2 = np.bincount(strat, weights=w * x * x, minlength=n_strata)
        E1 = S1 / S0
        E2 = S2 / S0
        grad = float(np.sum(case_x - E1))
        hess = float(-np.sum(E2 - E1 * E1))
        if hess == 0 or not np.isfinite(hess):
            return np.nan, np.nan
        step = grad / hess
        b -= step
        if abs(step) < tol:
            break
    var = -1.0 / hess
    se = float(np.sqrt(var)) if var > 0 else np.nan
    return float(b), se


def feature_design(df: pd.DataFrame, code: str):
    """Valid matched sets are retained and feature values are standardised."""
    sub = df[["query_id", "referee", "label", code]].dropna(subset=[code]).copy()
    g = sub.groupby("query_id")["label"].agg(["sum", "count"])
    good = g[(g["sum"] == 1) & (g["count"] >= 2)].index
    sub = sub[sub["query_id"].isin(good)].reset_index(drop=True)
    mu, sd = sub[code].mean(), sub[code].std(ddof=1)
    sd = sd if sd > 1e-12 else 1.0
    sub["z"] = (sub[code] - mu) / sd
    return sub


def _prep_feature_arrays(sub: pd.DataFrame):
    """Integer strata and referee row indices are prepared."""
    x = sub["z"].to_numpy(dtype=float)
    case_mask = sub["label"].to_numpy().astype(bool)
    strat = pd.factorize(sub["query_id"].to_numpy())[0]
    n_strata = int(strat.max()) + 1
    case_x = np.zeros(n_strata, dtype=float)
    case_x[strat[case_mask]] = x[case_mask]
    referees = sub["referee"].to_numpy()
    ref_rows = {r: np.where(referees == r)[0] for r in pd.unique(referees)}
    return x, strat, case_mask, case_x, n_strata, ref_rows


def primary_linear(df: pd.DataFrame, n_boot: int, rng: np.random.Generator):
    """Separate linear models and referee bootstrap intervals are estimated."""
    results = []
    for code, label, group in ELIGIBLE_FEATURES:
        sub = feature_design(df, code)
        x, strat, case_mask, case_x, n_strata, ref_rows = _prep_feature_arrays(sub)

        beta, _ = clogit1d(x, strat, case_x, n_strata)
        if not np.isfinite(beta):
            results.append(
                {
                    "feature_code": code,
                    "feature_label": label,
                    "group": group,
                    "flag": "fit_failed",
                }
            )
            continue
        ref_keys = list(ref_rows.keys())
        boot_betas = []
        for _ in range(n_boot):
            drawn = rng.integers(0, len(ref_keys), size=len(ref_keys))
            xs, ss, cs = [], [], []
            offset = 0
            for d in drawn:
                rows = ref_rows[ref_keys[d]]
                orig = strat[rows]
                uniq, inv = np.unique(orig, return_inverse=True)
                xs.append(x[rows])
                ss.append(inv + offset)
                cs.append(case_mask[rows])
                offset += uniq.size
            xb = np.concatenate(xs)
            sb = np.concatenate(ss)
            cb = np.concatenate(cs)
            cxb = np.zeros(offset, dtype=float)
            cxb[sb[cb]] = xb[cb]
            bb, _ = clogit1d(xb, sb, cxb, offset)
            if np.isfinite(bb):
                boot_betas.append(bb)

        boot_betas = np.asarray(boot_betas)
        or_point = float(np.exp(beta))
        if boot_betas.size >= max(20, 0.5 * n_boot):
            ci_lo = float(np.exp(np.percentile(boot_betas, 2.5)))
            ci_hi = float(np.exp(np.percentile(boot_betas, 97.5)))
            se_boot = float(np.std(boot_betas, ddof=1))
            p_clustered = (
                float(2 * norm.sf(abs(beta) / se_boot)) if se_boot > 0 else np.nan
            )
        else:
            ci_lo = ci_hi = se_boot = p_clustered = np.nan

        results.append(
            {
                "feature_code": code,
                "feature_label": label,
                "group": group,
                "n_clips": int(len(sub)),
                "n_queries": int(sub["query_id"].nunique()),
                "n_referees": int(sub["referee"].nunique()),
                "OR_per_SD": round(or_point, 4),
                "CI95_low": round(ci_lo, 4) if np.isfinite(ci_lo) else np.nan,
                "CI95_high": round(ci_hi, 4) if np.isfinite(ci_hi) else np.nan,
                "p_referee_clustered": p_clustered,
                "beta": round(float(beta), 5),
                "flag": "",
            }
        )
    return pd.DataFrame(results)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values are returned."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p_values))
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(p_values[finite])]
    ranked = p_values[order] * finite.size / np.arange(1, finite.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def finalise_primary(res: pd.DataFrame):
    """Adjusted results are prepared for reporting."""
    ok = res[res["flag"] == ""].copy()
    q = _benjamini_hochberg(ok["p_referee_clustered"].to_numpy())
    ok["q_FDR_BH"] = np.round(q, 5)

    def direction(row):
        if not np.isfinite(row["q_FDR_BH"]) or row["q_FDR_BH"] >= 0.05:
            return "n.s."
        return "negative" if row["OR_per_SD"] < 1 else "positive"

    def assoc(row):
        d = direction(row)
        if d == "n.s.":
            return "No significant association after FDR"
        side = "Lower" if d == "negative" else "Higher"
        return f"{side} values associated with incorrect decisions"

    ok["direction"] = ok.apply(direction, axis=1)
    ok["association"] = ok.apply(assoc, axis=1)
    ok = ok.sort_values("q_FDR_BH").reset_index(drop=True)
    cols = [
        "feature_label",
        "feature_code",
        "group",
        "n_queries",
        "n_clips",
        "n_referees",
        "OR_per_SD",
        "CI95_low",
        "CI95_high",
        "p_referee_clustered",
        "q_FDR_BH",
        "direction",
        "association",
    ]
    return ok[cols]


def main():
    """Conditional models are fitted and the result table is written."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="referee cluster-bootstrap resamples (default 1000)",
    )
    ap.add_argument("--seed", type=int, default=20240620)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = build_clip_table()

    print(
        f"\n[primary] fitting {len(ELIGIBLE_FEATURES)} linear CLRs "
        f"with {args.bootstrap} referee bootstraps each ...",
        flush=True,
    )
    raw = primary_linear(df, n_boot=args.bootstrap, rng=rng)

    primary = finalise_primary(raw)
    primary.to_csv(os.path.join(OUT_DIR, "clr_corroboration_eligible.csv"), index=False)

    print(
        "\n===== Conditional logistic regression (eligible non-matching features) ====="
    )
    print(
        primary[
            [
                "feature_label",
                "OR_per_SD",
                "CI95_low",
                "CI95_high",
                "q_FDR_BH",
                "association",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
