"""Train the style-conditioned Camera Synthesis baseline with optional GAN loss."""

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
from data.cs_dataset import build_cs_dataset
from models.checkpoint import load_checkpoint_file, load_model_weights
from models.cs import build_cs_model
from models.discriminator import CameraAdversarialLoss, CameraStyleDiscriminator
from train.adan import Adan

from train.train_utils import (
    ExponentialMovingAverage,
    append_metrics,
    create_accelerator,
    load_data_config,
    prepare_run_directory,
    save_checkpoint,
)
from train.losses import calculate_cs_losses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def _wrong_style_labels(labels: torch.Tensor, number_of_styles: int = 16) -> torch.Tensor:
    offset = torch.randint(1, number_of_styles, labels.shape, device=labels.device)
    return (labels + offset) % number_of_styles


def _gradient_penalty(
    discriminator: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    condition: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand(real.shape[0], 1, 1, device=real.device)
    interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    score, _ = discriminator(interpolated, condition, labels)
    gradient = torch.autograd.grad(score.sum(), interpolated, create_graph=True, retain_graph=True)[0]
    return ((gradient.reshape(real.shape[0], -1).norm(2, dim=1) - 1) ** 2).mean()


def train(config: dict[str, Any]) -> Path:
    accelerator = create_accelerator()
    experiment = require_mapping(config, "experiment")
    set_seed(int(experiment.get("seed", 42)), device_specific=True)
    training = require_mapping(config, "training")
    adversarial = config.get("adversarial", {})
    if not isinstance(adversarial, dict):
        raise TypeError("`adversarial` must be a YAML mapping")
    losses = require_mapping(config, "losses")
    use_gan = bool(training.get("use_gan", True))
    data_section = require_mapping(config, "data")
    data_config = load_data_config(config)
    style_vocabulary = str(data_section.get("style_vocabulary", "legacy_dsc_v1"))
    accelerator.print("Building virtual CS windows and fitting train-split normalizers...")
    dataset = build_cs_dataset(data_config, "train", style_vocabulary=style_vocabulary)
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 128)),
        shuffle=True,
        num_workers=int(training.get("num_workers", 8)),
        pin_memory=True,
        drop_last=bool(training.get("drop_last", True)),
    )

    model = build_cs_model(config)
    optimizer = Adan(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 0.02)),
    )
    ema = ExponentialMovingAverage(model, float(training.get("ema_decay", 0.9999)))
    discriminator: torch.nn.Module | None = None
    discriminator_optimizer: torch.optim.Optimizer | None = None
    if use_gan:
        discriminator = CameraStyleDiscriminator(cam_dim=8, cond_dim=215, num_styles=16, hidden=256)
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=float(adversarial.get("discriminator_learning_rate", 1e-5)),
            betas=(0.5, 0.999),
        )
        model, discriminator, optimizer, discriminator_optimizer, loader = accelerator.prepare(
            model,
            discriminator,
            optimizer,
            discriminator_optimizer,
            loader,
        )
    else:
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
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if use_gan:
            assert discriminator is not None
            assert discriminator_optimizer is not None
            if "discriminator_state_dict" in checkpoint:
                accelerator.unwrap_model(discriminator).load_state_dict(
                    checkpoint["discriminator_state_dict"], strict=True
                )
                if "discriminator_optimizer_state_dict" in checkpoint:
                    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
            else:
                accelerator.print("Resume checkpoint has no discriminator; initialized a new discriminator.")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    adversarial_loss = CameraAdversarialLoss(loss_type="wgan") if use_gan else None
    style_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.015) if use_gan else None
    history = int(require_mapping(config, "model").get("history_len", 60))
    weights = {name: float(value) for name, value in losses.items()}
    epochs = int(training.get("epochs", 1200))
    ema_interval = int(training.get("ema_interval", 1))
    save_every = int(training.get("save_every", 200))
    max_steps = training.get("max_steps_per_epoch")
    gradient_clip = float(training.get("gradient_clip", 1.0))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        if use_gan:
            assert discriminator is not None
            assert discriminator_optimizer is not None
            discriminator.train()
        if use_gan and epoch == int(adversarial.get("learning_rate_change_epoch", 501)):
            assert discriminator_optimizer is not None
            for group in discriminator_optimizer.param_groups:
                group["lr"] = float(adversarial.get("late_discriminator_learning_rate", 5e-6))
        gp_weight = float(
            adversarial.get("late_gradient_penalty", 0.8)
            if epoch > int(adversarial.get("schedule_after_epoch", 300))
            else adversarial.get("gradient_penalty", 0.5)
        )

        for step, batch in enumerate(loader, start=1):
            target = batch["camera"][:, :, :8]
            inference_mask = batch["camera_inference_mask"]
            motion = batch["motion"]
            music = batch["music"]
            style = batch["style"]
            result = model(target, inference_mask, motion, music, style)
            reconstruction_total, reconstruction_parts = calculate_cs_losses(
                result,
                target,
                inference_mask,
                batch["bone_mask"],
                motion,
                dataset.normalizers,
                history,
                weights,
            )
            values = dict(reconstruction_parts)
            generator_loss = reconstruction_total
            if use_gan:
                assert discriminator is not None
                assert discriminator_optimizer is not None
                assert adversarial_loss is not None
                assert style_loss is not None
                style_labels = style[:, 0].argmax(dim=-1).long()
                combined_condition = torch.cat((music, motion), dim=-1)
                wrong_labels = _wrong_style_labels(style_labels)

                real_score, _ = discriminator(target, combined_condition, style_labels)
                fake_score, _ = discriminator(result.detach(), combined_condition, style_labels)
                discriminator_loss = adversarial_loss(real_score, True, True) + adversarial_loss(
                    fake_score, False, True
                )
                if gp_weight > 0:
                    discriminator_loss = discriminator_loss + gp_weight * _gradient_penalty(
                        discriminator, target, result, combined_condition, style_labels
                    )
                discriminator_optimizer.zero_grad()
                accelerator.backward(discriminator_loss)
                accelerator.clip_grad_norm_(discriminator.parameters(), gradient_clip)
                discriminator_optimizer.step()

                fake_score, fake_style = discriminator(result, combined_condition, style_labels)
                (correct_style_score, wrong_style_score), _ = discriminator(
                    result, combined_condition, style_labels, wrong_labels
                )
                generator_adversarial = float(adversarial.get("generator_weight", 0.0005)) * adversarial_loss(
                    fake_score, True, False
                )
                generator_style = float(adversarial.get("style_weight", 0.00002)) * style_loss(
                    fake_style, style_labels
                )
                generator_focus = float(adversarial.get("focus_weight", 0.0002)) * (
                    adversarial_loss(correct_style_score, True, False)
                    + adversarial_loss(wrong_style_score, False, False)
                )
                generator_loss = reconstruction_total + generator_adversarial + generator_style + generator_focus
                values.update(
                    {
                        "discriminator": discriminator_loss,
                        "generator_adversarial": generator_adversarial,
                        "generator_style": generator_style,
                        "generator_focus": generator_focus,
                    }
                )
            optimizer.zero_grad()
            accelerator.backward(generator_loss)
            accelerator.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            global_step += 1
            if global_step % ema_interval == 0:
                ema.update(accelerator.unwrap_model(model))

            values["generator_total"] = generator_loss
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            if max_steps is not None and step >= int(max_steps):
                break

        metrics = {"epoch": epoch, "global_step": global_step}
        metrics.update({name: value / max(step, 1) for name, value in totals.items()})
        append_metrics(run, metrics, accelerator)
        accelerator.print(metrics)
        if epoch % save_every == 0 or epoch == epochs:
            extra: dict[str, Any] = {
                "stage": "cs",
                "experiment": dict(experiment),
                "style_vocabulary": style_vocabulary,
                "training": {"use_gan": use_gan},
            }
            if use_gan:
                assert discriminator is not None
                assert discriminator_optimizer is not None
                extra.update(
                    {
                        "discriminator_state_dict": accelerator.unwrap_model(discriminator).state_dict(),
                        "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
                    }
                )
            save_checkpoint(
                run / "checkpoints" / f"train-{epoch}.pt",
                accelerator,
                model,
                ema,
                optimizer,
                dataset.normalizers,
                epoch,
                global_step,
                extra,
            )
    return run


def main() -> None:
    args = build_parser().parse_args()
    run = train(load_config(args.config))
    print(f"Training complete: {run}")


if __name__ == "__main__":
    main()
