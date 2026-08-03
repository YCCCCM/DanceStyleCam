"""Load released checkpoints without retaining legacy pickle classes."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterator

import numpy as np
import torch

from data.normalization import MinMaxStats, NormalizerBundle


class LegacyMinMaxScaler:
    """State-only target for `dataset.scaler.MinMaxScaler` pickle records."""


class LegacyNormalizer:
    """State-only target for `dataset.preprocess.Normalizer` pickle records."""


@contextmanager
def _legacy_dataset_modules() -> Iterator[None]:
    names = ("dataset", "dataset.preprocess", "dataset.scaler")
    previous = {name: sys.modules.get(name) for name in names}
    package = ModuleType("dataset")
    package.__path__ = []
    preprocess = ModuleType("dataset.preprocess")
    scaler = ModuleType("dataset.scaler")
    preprocess.Normalizer = LegacyNormalizer
    scaler.MinMaxScaler = LegacyMinMaxScaler
    package.preprocess = preprocess
    package.scaler = scaler
    sys.modules.update({"dataset": package, "dataset.preprocess": preprocess, "dataset.scaler": scaler})
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_checkpoint_file(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    with _legacy_dataset_modules():
        return torch.load(path, map_location=map_location, weights_only=False)


def _strip_distributed_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_model_weights(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    use_ema: bool = True,
    strict: bool = True,
) -> None:
    key = "ema_state_dict" if use_ema and "ema_state_dict" in checkpoint else "model_state_dict"
    if key not in checkpoint:
        raise KeyError(f"Checkpoint has no model weights under `{key}`")
    model.load_state_dict(_strip_distributed_prefix(checkpoint[key]), strict=strict)


def normalizers_from_checkpoint(checkpoint: dict[str, Any]) -> NormalizerBundle:
    if "normalizers" in checkpoint:
        return NormalizerBundle(
            {
                name: MinMaxStats(
                    np.asarray(values["minimum"], dtype=np.float32),
                    np.asarray(values["maximum"], dtype=np.float32),
                )
                for name, values in checkpoint["normalizers"].items()
            }
        )

    legacy_keys = {
        "pose": "normalizer_pose",
        "camera_distance": "normalizer_camera_dis",
        "camera_position": "normalizer_camera_pos",
        "camera_rotation": "normalizer_camera_rot",
        "camera_fov": "normalizer_camera_fov",
        "camera_eye": "normalizer_camera_eye",
    }
    fields: dict[str, MinMaxStats] = {}
    for name, key in legacy_keys.items():
        if key not in checkpoint:
            continue
        scaler = checkpoint[key].scaler
        fields[name] = MinMaxStats(
            minimum=scaler.data_min_.detach().cpu().numpy().astype(np.float32),
            maximum=scaler.data_max_.detach().cpu().numpy().astype(np.float32),
        )
    if "pose" not in fields:
        raise ValueError("Checkpoint does not contain pose normalization statistics")
    return NormalizerBundle(fields)

