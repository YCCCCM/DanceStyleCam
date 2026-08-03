"""Baseline CKD -> style-conditioned CS inference without video rendering."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import random
from typing import Any

import numpy as np
import torch

from common.config import require_mapping
from common.paths import resolve_project_path
from data.camera_geometry import camera_centric_axes
from data.ckd_dataset import CKDDataset, build_ckd_dataset
from data.cs_dataset import CSDataset, build_cs_dataset
from data.normalization import NormalizerBundle
from models.checkpoint import (
    load_checkpoint_file,
    load_model_weights,
    normalizers_from_checkpoint,
)
from models.ckd import build_ckd_model
from models.cs import build_cs_model

from infer.result_io import GenerationRun


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_uses_ema(generation: Mapping[str, Any]) -> bool:
    source = str(generation.get("checkpoint_weights", "model")).lower()
    if source not in {"model", "ema"}:
        raise ValueError("generation.checkpoint_weights must be either `model` or `ema`")
    return source == "ema"


def _tensor(value: Any, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = torch.as_tensor(value, device=device)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result.unsqueeze(0)


def _clip_indices(dataset: CKDDataset | CSDataset, sample_ids: set[str] | None) -> list[int]:
    available = {clip.name: index for index, clip in enumerate(dataset.context.clips)}
    if sample_ids is None:
        return list(range(len(available)))
    missing = sorted(sample_ids - available.keys())
    if missing:
        raise ValueError(f"Unknown sample ids for selected split: {', '.join(missing)}")
    return [index for index, clip in enumerate(dataset.context.clips) if clip.name in sample_ids]


def _window_range(dataset: CKDDataset | CSDataset, clip_index: int) -> range:
    start = 0 if clip_index == 0 else dataset.subsequence_end_index[clip_index - 1]
    return range(start, dataset.subsequence_end_index[clip_index])


@torch.inference_mode()
def infer_keyframes(
    model: torch.nn.Module,
    dataset: CKDDataset,
    device: torch.device,
    sample_ids: set[str] | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    history = dataset.context.history_len
    generated: dict[str, np.ndarray] = {}
    for clip_index in _clip_indices(dataset, sample_ids):
        clip = dataset.context.clips[clip_index]
        condition: torch.Tensor | None = None
        pieces: list[np.ndarray] = []
        for window_index in _window_range(dataset, clip_index):
            sample = dataset[window_index]
            padding = _tensor(sample["padding_mask"], device, torch.float32)
            keyframes = _tensor(sample["camera_keyframe"], device, torch.long)
            motion = _tensor(sample["motion"], device, torch.float32)
            music = _tensor(sample["music"], device, torch.float32)
            valid = int(padding[:, history:].sum().item())
            model_input = torch.zeros_like(keyframes) if condition is None else condition
            logits = model(model_input, padding, motion, music)
            prediction = logits.argmax(dim=-1, keepdim=True)
            condition = torch.zeros_like(keyframes)
            condition[:, :history] = prediction[:, valid : history + valid]
            pieces.append(prediction[0, history : history + valid, 0].cpu().numpy().astype(np.uint8))
        mask = np.concatenate(pieces)[: clip.frames]
        generated[clip.name] = mask
    return generated


@torch.inference_mode()
def infer_cameras(
    model: torch.nn.Module,
    dataset: CSDataset,
    device: torch.device,
    sample_ids: set[str] | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    history = dataset.context.history_len
    generated: dict[str, np.ndarray] = {}
    for clip_index in _clip_indices(dataset, sample_ids):
        clip = dataset.context.clips[clip_index]
        condition: torch.Tensor | None = None
        pieces: list[np.ndarray] = []
        for window_index in _window_range(dataset, clip_index):
            sample = dataset[window_index]
            camera = _tensor(sample["camera"][:, :8], device, torch.float32)
            inference_mask = _tensor(sample["camera_inference_mask"], device, torch.float32)
            motion = _tensor(sample["motion"], device, torch.float32)
            music = _tensor(sample["music"], device, torch.float32)
            style = _tensor(sample["style"], device, torch.float32)
            active = int(inference_mask.sum().item())
            model_input = camera * 0 + camera[:, 0:1] if condition is None else condition
            output = model(model_input, inference_mask, motion, music, style)
            condition = torch.zeros_like(output)
            condition[:, :history] = output[:, active : history + active]
            pieces.append(output[0, history : history + active].cpu().numpy().astype(np.float32))
        normalized = np.concatenate(pieces)[: clip.frames]
        generated[clip.name] = polar8_to_camera20(normalized, dataset.normalizers)
    return generated


def polar8_to_camera20(value: np.ndarray, normalizers: NormalizerBundle) -> np.ndarray:
    distance = normalizers["camera_distance"].denormalize(value[:, 0:1])
    position = normalizers["camera_position"].denormalize(value[:, 1:4])
    rotation = normalizers["camera_rotation"].denormalize(value[:, 4:7])
    fov = normalizers["camera_fov"].denormalize(value[:, 7:8])
    eye, axis_x, axis_y, axis_z = camera_centric_axes(distance, position, rotation)
    return np.concatenate((distance, position, rotation, fov, eye, axis_x, axis_y, axis_z), axis=1).astype(
        np.float32
    )


def run_generation(
    config: Mapping[str, Any],
    data_config: dict[str, Any],
    run_name: str | None = None,
    sample_ids: Iterable[str] | None = None,
) -> GenerationRun:
    experiment = require_mapping(config, "experiment")
    checkpoint_paths = require_mapping(config, "checkpoints")
    generation = require_mapping(config, "generation")
    split = str(require_mapping(config, "data").get("split", "test"))
    selected = None if sample_ids is None else set(sample_ids)
    seed_everything(int(experiment.get("seed", 42)))
    device = resolve_device(str(generation.get("device", "auto")))
    use_ema = checkpoint_uses_ema(generation)

    ckd_checkpoint = load_checkpoint_file(resolve_project_path(str(checkpoint_paths["ckd"])), device)
    ckd_model = build_ckd_model(config).to(device)
    load_model_weights(ckd_model, ckd_checkpoint, use_ema=use_ema, strict=True)
    ckd_normalizers = normalizers_from_checkpoint(ckd_checkpoint)
    ckd_dataset = build_ckd_dataset(data_config, split, normalizers=ckd_normalizers)
    keyframes = infer_keyframes(ckd_model, ckd_dataset, device, selected)
    del ckd_model, ckd_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cs_checkpoint = load_checkpoint_file(resolve_project_path(str(checkpoint_paths["cs"])), device)
    cs_model = build_cs_model(config).to(device)
    load_model_weights(cs_model, cs_checkpoint, use_ema=use_ema, strict=True)
    cs_normalizers = normalizers_from_checkpoint(cs_checkpoint)
    cs_dataset = build_cs_dataset(
        data_config,
        split,
        normalizers=cs_normalizers,
        generated_keyframe_masks=keyframes,
        style_vocabulary=str(generation.get("checkpoint_style_vocabulary", "legacy_dsc_v1")),
    )
    cameras = infer_cameras(cs_model, cs_dataset, device, selected)

    output_root = resolve_project_path(str(generation.get("output_root", "generation")))
    result = GenerationRun.create(output_root, run_name or str(experiment["name"]), config)
    clips = {clip.name: clip for clip in cs_dataset.context.clips}
    for sample_id, camera in cameras.items():
        clip = clips[sample_id]
        result.save_camera(
            sample_id,
            camera,
            {
                "sequence_id": clip.sequence_id,
                "source_start_frame": clip.start,
                "source_end_frame": clip.end,
                "source_items": list(clip.source_items),
                "style": cs_dataset.context.annotations.style_name(clip.sequence_id),
            },
        )
        result.save_keyframes(sample_id, keyframes[sample_id])
    return result
