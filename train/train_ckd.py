"""Train the released Camera Keyframe Detection baseline from YAML."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from accelerate.utils import set_seed

from common.config import load_config, require_mapping
from common.paths import resolve_project_path
from data.ckd_dataset import build_ckd_dataset
from models.checkpoint import load_checkpoint_file, load_model_weights
from models.ckd import build_ckd_model
from train.adan import Adan

from train.train_utils import (
    ExponentialMovingAverage,
    append_metrics,
    create_accelerator,
    load_data_config,
    prepare_run_directory,
    save_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def train(config: dict[str, Any]) -> Path:
    accelerator = create_accelerator()
    experiment = require_mapping(config, "experiment")
    set_seed(int(experiment.get("seed", 42)), device_specific=True)
    training = require_mapping(config, "training")
    data_config = load_data_config(config)
    accelerator.print("Building virtual CKD windows and fitting train-split normalizers...")
    dataset = build_ckd_dataset(data_config, "train")
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 1024)),
        shuffle=True,
        num_workers=int(training.get("num_workers", 8)),
        pin_memory=True,
        drop_last=bool(training.get("drop_last", True)),
    )
    model = build_ckd_model(config)
    optimizer = Adan(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 0.02)),
    )
    ema = ExponentialMovingAverage(model, float(training.get("ema_decay", 0.9999)))
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    ema.model.to(accelerator.device)
    run = prepare_run_directory(config, accelerator)

    start_epoch = 1
    global_step = 0
    resume = training.get("resume_checkpoint")
    if resume:
        checkpoint = load_checkpoint_file(resolve_project_path(str(resume)), accelerator.device)
        load_model_weights(accelerator.unwrap_model(model), checkpoint, use_ema=False, strict=True)
        ema.model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))

    class_weights = torch.tensor(
        [float(training.get("negative_weight", 0.5)), float(training.get("positive_weight", 2.0))],
        device=accelerator.device,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    history = int(require_mapping(config, "model").get("history_len", 60))
    epochs = int(training.get("epochs", 1500))
    ema_interval = int(training.get("ema_interval", 1))
    save_every = int(training.get("save_every", 300))
    max_steps = training.get("max_steps_per_epoch")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = correct = valid_count = positive_count = predicted_positive = 0.0
        for step, batch in enumerate(loader, start=1):
            target = batch["camera_keyframe"]
            padding = batch["padding_mask"]
            logits = model(target, padding, batch["motion"], batch["music"])
            inference_logits = logits[:, history:]
            inference_target = target[:, history:, 0]
            valid = padding[:, history:, 0]
            loss_map = loss_fn(inference_logits.transpose(1, 2), inference_target) * valid
            loss = loss_map.mean()
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            global_step += 1
            if global_step % ema_interval == 0:
                ema.update(accelerator.unwrap_model(model))

            prediction = inference_logits.argmax(dim=-1)
            loss_sum += float(loss.detach())
            correct += float(((prediction == inference_target) * valid).sum().detach())
            valid_count += float(valid.sum().detach())
            positive_count += float((inference_target * valid).sum().detach())
            predicted_positive += float((prediction * valid).sum().detach())
            if max_steps is not None and step >= int(max_steps):
                break

        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss_sum / max(step, 1),
            "accuracy": correct / max(valid_count, 1),
            "positive_fraction": positive_count / max(valid_count, 1),
            "predicted_positive_fraction": predicted_positive / max(valid_count, 1),
        }
        append_metrics(run, metrics, accelerator)
        accelerator.print(metrics)
        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(
                run / "checkpoints" / f"train-{epoch}.pt",
                accelerator,
                model,
                ema,
                optimizer,
                dataset.normalizers,
                epoch,
                global_step,
                {"stage": "ckd", "experiment": dict(experiment)},
            )
    return run


def main() -> None:
    args = build_parser().parse_args()
    run = train(load_config(args.config))
    print(f"Training complete: {run}")


if __name__ == "__main__":
    main()
