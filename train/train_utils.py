"""Shared, portable training utilities."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs

from common.config import load_config, require_mapping
from common.paths import resolve_project_path
from data.normalization import NormalizerBundle


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, source: torch.nn.Module) -> None:
        source_parameters = dict(source.named_parameters())
        for name, target in self.model.named_parameters():
            target.lerp_(source_parameters[name].detach(), 1.0 - self.decay)
        source_buffers = dict(source.named_buffers())
        for name, target in self.model.named_buffers():
            target.copy_(source_buffers[name])


def create_accelerator() -> Accelerator:
    return Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )


def load_data_config(config: Mapping[str, Any]) -> dict[str, Any]:
    data = require_mapping(config, "data")
    return load_config(resolve_project_path(str(data["config"])))


def prepare_run_directory(config: Mapping[str, Any], accelerator: Accelerator) -> Path:
    experiment = require_mapping(config, "experiment")
    training = require_mapping(config, "training")
    run = resolve_project_path(str(training.get("output_root", "runs"))) / str(experiment["name"])
    resume = training.get("resume_checkpoint")
    if accelerator.is_main_process:
        if run.exists() and resume is None:
            raise FileExistsError(f"Training run already exists: {run}")
        run.mkdir(parents=True, exist_ok=True)
        (run / "checkpoints").mkdir(exist_ok=True)
        with (run / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(config), handle, sort_keys=False)
    accelerator.wait_for_everyone()
    return run


def normalizers_to_state(normalizers: NormalizerBundle) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            "minimum": torch.from_numpy(stats.minimum.copy()),
            "maximum": torch.from_numpy(stats.maximum.copy()),
        }
        for name, stats in normalizers.fields.items()
    }


def save_checkpoint(
    path: Path,
    accelerator: Accelerator,
    model: torch.nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    normalizers: NormalizerBundle,
    epoch: int,
    global_step: int,
    extra: Mapping[str, Any] | None = None,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    checkpoint: dict[str, Any] = {
        "format_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "ema_state_dict": ema.model.state_dict(),
        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "normalizers": normalizers_to_state(normalizers),
    }
    if extra:
        checkpoint.update(extra)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def append_metrics(run: Path, values: Mapping[str, Any], accelerator: Accelerator) -> None:
    if not accelerator.is_main_process:
        return
    with (run / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        json.dump(dict(values), handle, ensure_ascii=True)
        handle.write("\n")
