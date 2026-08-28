#!/usr/bin/env python3
"""One listwise RefSight model is trained and evaluated here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from config import (
    BATCH_SIZE,
    EPOCHS,
    GRADIENT_CLIP_NORM,
    LISTWISE_TAU,
    LR,
    MODEL_NAME,
    RESULTS_DIR,
    SEED,
    SPLIT_SEED,
    TGT_LEN,
    TOP_K_CORRECT,
    WEIGHT_DECAY,
)
from data_prep import (
    build_query_lists,
    filter_valid_queries_listwise,
    fit_scaler,
    load_metadata,
    split_dataset,
)
from dataset import make_listwise_loaders
from engine import eval_listwise, train_epoch_listwise
from models import build_model
from utils import get_device, set_seed


def _parameter_counts(model: torch.nn.Module) -> Dict[str, int]:
    """Total and trainable parameter counts are returned."""
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
    }


def _run_configuration(model: torch.nn.Module, feature_dim: int) -> Dict[str, Any]:
    """The complete model and training configuration is recorded."""
    return {
        "model_name": MODEL_NAME,
        "model_class": type(model).__name__,
        "feature_dim": feature_dim,
        "model_hyperparameters": model.hparams,
        "parameter_counts": _parameter_counts(model),
        "training_hyperparameters": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "target_sequence_length": TGT_LEN,
            "correct_candidates_per_query": TOP_K_CORRECT,
            "listwise_temperature": LISTWISE_TAU,
            "training_seed": SEED,
            "split_seed": SPLIT_SEED,
        },
    }


def main() -> None:
    """The configured training run is completed."""
    set_seed()
    device = get_device()
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}", flush=True)
    print(f"Model: {MODEL_NAME} | seed: {SEED}", flush=True)

    metadata = filter_valid_queries_listwise(
        load_metadata(),
        top_k=TOP_K_CORRECT,
    )
    (
        train_meta,
        validation_meta,
        test_meta,
        train_ids,
        validation_ids,
        test_ids,
    ) = split_dataset(metadata)
    print(
        f"Queries: train={len(train_ids)}, validation={len(validation_ids)}, "
        f"test={len(test_ids)}",
        flush=True,
    )

    scaler = fit_scaler(train_meta)
    feature_dim = int(scaler["feature_dim"])
    train_items = build_query_lists(train_meta)
    validation_items = build_query_lists(validation_meta)
    test_items = build_query_lists(test_meta)
    train_loader, validation_loader, test_loader = make_listwise_loaders(
        train_items,
        validation_items,
        test_items,
        scaler,
        device,
    )

    model = build_model(feature_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    run_configuration = _run_configuration(model, feature_dim)
    config_path = results_dir / "run_config.json"
    config_path.write_text(json.dumps(run_configuration, indent=2), encoding="utf-8")

    checkpoint_path = results_dir / f"listwise_model_{MODEL_NAME}.pt"
    best_epoch = 0
    best_validation = {
        "loss": float("nan"),
        "top1": -1.0,
        "top2": float("nan"),
        "mrr": float("nan"),
    }

    for epoch in range(1, EPOCHS + 1):
        training_loss = train_epoch_listwise(model, optimizer, device, train_loader)
        (
            validation_loss,
            validation_top1,
            validation_top2,
            validation_mrr,
        ) = eval_listwise(model, device, validation_loader)
        print(
            f"Epoch {epoch:02d} | train loss={training_loss:.6f} | "
            f"validation loss={validation_loss:.6f} "
            f"Top 1={validation_top1:.3f} Top 2={validation_top2:.3f} "
            f"MRR={validation_mrr:.3f}",
            flush=True,
        )

        if validation_top1 > best_validation["top1"]:
            best_epoch = epoch
            best_validation = {
                "loss": validation_loss,
                "top1": validation_top1,
                "top2": validation_top2,
                "mrr": validation_mrr,
            }
            torch.save(model.state_dict(), checkpoint_path)

    if best_epoch == 0:
        raise RuntimeError("No model checkpoint was saved.")

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    test_loss, test_top1, test_top2, test_mrr = eval_listwise(
        model,
        device,
        test_loader,
    )
    print(
        f"Test | loss={test_loss:.6f} Top 1={test_top1:.3f} "
        f"Top 2={test_top2:.3f} MRR={test_mrr:.3f}",
        flush=True,
    )

    metrics = {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "best_epoch": best_epoch,
        "val_loss": best_validation["loss"],
        "val_top1": best_validation["top1"],
        "val_top2": best_validation["top2"],
        "val_mrr": best_validation["mrr"],
        "test_loss": test_loss,
        "test_top1": test_top1,
        "test_top2": test_top2,
        "test_mrr": test_mrr,
        "feature_dim": feature_dim,
        "checkpoint_path": str(checkpoint_path),
        "run_config_path": str(config_path),
        "model_hyperparameters": run_configuration["model_hyperparameters"],
        "parameter_counts": run_configuration["parameter_counts"],
        "training_hyperparameters": run_configuration["training_hyperparameters"],
    }
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
