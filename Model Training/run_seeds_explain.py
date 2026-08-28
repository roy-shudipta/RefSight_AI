#!/usr/bin/env python3
"""Training and per-run analyses are repeated across models and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = ["lstm", "tcn", "transformer"]
DEFAULT_SEEDS = list(range(1, 21))

# Per-run analysis stages are defined in execution order.
ANALYSIS_STAGES = [
    "permutation_importance",
    "calibrated_risk_profiles",
    "operating_range_thresholds",
]
STAGE_ALIASES = {
    "permutation": "permutation_importance",
    "perm": "permutation_importance",
    "importance": "permutation_importance",
    "calibration": "calibrated_risk_profiles",
    "calib": "calibrated_risk_profiles",
    "risk": "calibrated_risk_profiles",
    "thresholds": "operating_range_thresholds",
    "operating_ranges": "operating_range_thresholds",
    "ranges": "operating_range_thresholds",
}


def resolve_stages(values: Iterable[str]) -> list[str]:
    """Requested stage names are resolved in execution order."""
    vals = [str(v).strip().lower() for v in (values or [])]
    if not vals or "all" in vals:
        return list(ANALYSIS_STAGES)
    out: list[str] = []
    for v in vals:
        canon = v if v in ANALYSIS_STAGES else STAGE_ALIASES.get(v)
        if canon is None:
            raise ValueError(
                f"Unknown stage '{v}'. Choices: all, "
                + ", ".join(ANALYSIS_STAGES)
                + " (aliases: "
                + ", ".join(sorted(STAGE_ALIASES))
                + ")."
            )
        if canon not in out:
            out.append(canon)
    return out


@dataclass(frozen=True)
class ScriptSpec:
    name: str
    path: Path
    extra_args: List[str] = field(default_factory=list)


def ts() -> str:
    """A timestamp is returned for a log file name."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    """Command-line arguments are returned."""
    parser = argparse.ArgumentParser(
        description="Train RefSight models across seeds and run per-run explanation/calibration analyses."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model families to run. Valid values are lstm, tcn, and transformer.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds to run.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs",
        help="Folder where per-model and per-seed results will be written.",
    )
    parser.add_argument(
        "--force-explanations",
        action="store_true",
        help="Rerun analysis scripts even when .done markers already exist.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and run only the analysis scripts for existing checkpoints.",
    )
    parser.add_argument(
        "--retrain-existing",
        action="store_true",
        help="Retrain even when a checkpoint already exists.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        help=(
            "Per-run analysis stages to run: 'all' (default), permutation_importance, "
            "calibrated_risk_profiles, operating_range_thresholds. "
            "Aliases: permutation, calibration, thresholds. "
            "Combine with --skip-training --force-explanations to re-run one stage "
            "without retraining. Note: operating_range_thresholds needs the "
            "calibrated_risk_profiles outputs to already exist."
        ),
    )
    return parser.parse_args()


def normalise_models(models: Iterable[str]) -> list[str]:
    """Model names are validated and normalised."""
    valid = set(DEFAULT_MODELS)
    out = []
    for model in models:
        name = str(model).strip().lower()
        if name not in valid:
            raise ValueError(f"Unknown model '{model}'. Valid models: {sorted(valid)}")
        out.append(name)
    return out


def analysis_scripts(results_dir: Path, model_name: str) -> list[ScriptSpec]:
    """Per-run analysis commands are defined."""
    exp_dir = PROJECT_ROOT / "explanation_and_calibration"
    return [
        ScriptSpec("permutation_importance", exp_dir / "permutation_importance.py"),
        ScriptSpec("calibrated_risk_profiles", exp_dir / "calibrated_risk_profiles.py"),
        ScriptSpec(
            "operating_range_thresholds",
            exp_dir / "operating_range_thresholds.py",
            ["--results-dir", str(results_dir), "--model-name", model_name],
        ),
    ]


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    """One command is run and its output is written to a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CWD:\n" + str(cwd) + "\n\n")
        f.write("CMD:\n" + " ".join(cmd) + "\n\n")
        f.write("ENV_OVERRIDES:\n")
        for key in ["OVERRIDE_RESULTS_DIR", "OVERRIDE_MODEL_NAME", "OVERRIDE_SEED"]:
            f.write(f"{key}={env.get(key, '')}\n")
        f.write("\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc = proc.wait()

    if rc != 0:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            tail = "".join(deque(f, maxlen=40)).strip()
        raise RuntimeError(
            f"Command failed with exit code {rc}. Log file: {log_path}\n\n"
            f"Last log lines:\n{tail}"
        )


def append_run_metrics_csv(
    runs_root: Path, seed: int, model_name: str, results_dir: Path
) -> None:
    """One completed training result is added to the metrics table."""
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"[{model_name} seed {seed:03d}] metrics.json was not found.", flush=True)
        return

    with metrics_path.open("r", encoding="utf-8") as f:
        row = json.load(f)

    row["seed_folder"] = str(results_dir)
    row["model_name"] = str(model_name)
    row["seed"] = int(seed)

    out_csv = runs_root / "run_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[{model_name} seed {seed:03d}] metrics appended -> {out_csv}", flush=True)


def main() -> None:
    """All requested training and analysis runs are completed."""
    args = parse_args()
    models = normalise_models(args.models)
    stages = resolve_stages(args.stages)
    runs_root = args.runs_root.expanduser().resolve()
    train_script = PROJECT_ROOT / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(train_script)

    print(
        f"Models: {models} | seeds: {len(args.seeds)} | "
        f"training: {'skipped' if args.skip_training else 'on'} | "
        f"stages: {stages}",
        flush=True,
    )

    for model_name in models:
        model_root = runs_root / model_name
        model_root.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            run_dir = model_root / f"seed_{seed:03d}"
            logs_dir = run_dir / "logs"
            run_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            env = dict(os.environ)
            env["OVERRIDE_SEED"] = str(seed)
            env["OVERRIDE_MODEL_NAME"] = model_name
            env["OVERRIDE_RESULTS_DIR"] = str(run_dir)

            ckpt_path = run_dir / f"listwise_model_{model_name}.pt"

            trained = False
            if args.skip_training:
                print(
                    f"[{model_name} seed {seed:03d}] training skipped by flag.",
                    flush=True,
                )
            elif ckpt_path.exists() and not args.retrain_existing:
                print(
                    f"[{model_name} seed {seed:03d}] checkpoint exists; training skipped.",
                    flush=True,
                )
            else:
                print(f"[{model_name} seed {seed:03d}] training...", flush=True)
                run_cmd(
                    [sys.executable, str(train_script)],
                    cwd=PROJECT_ROOT,
                    env=env,
                    log_path=logs_dir / f"train_{ts()}.log",
                )
                if not ckpt_path.exists():
                    raise FileNotFoundError(
                        f"Training finished but checkpoint was not found: {ckpt_path}"
                    )
                trained = True

            # A metrics row is added only after training is completed.
            if trained:
                append_run_metrics_csv(model_root, seed, model_name, run_dir)

            for spec in analysis_scripts(run_dir, model_name):
                if spec.name not in stages:
                    continue
                if not spec.path.exists():
                    raise FileNotFoundError(spec.path)

                done_flag = run_dir / f".done_{spec.name}"
                if done_flag.exists() and not args.force_explanations:
                    print(
                        f"[{model_name} seed {seed:03d}] {spec.name} already done.",
                        flush=True,
                    )
                    continue

                print(
                    f"[{model_name} seed {seed:03d}] running {spec.name}...", flush=True
                )
                cmd = [sys.executable, str(spec.path), *spec.extra_args]
                run_cmd(
                    cmd,
                    cwd=PROJECT_ROOT,
                    env=env,
                    log_path=logs_dir / f"{spec.name}_{ts()}.log",
                )
                done_flag.write_text(ts(), encoding="utf-8")

            print(f"[{model_name} seed {seed:03d}] done.\n", flush=True)

    print("All requested model/seed runs finished.")


if __name__ == "__main__":
    main()
