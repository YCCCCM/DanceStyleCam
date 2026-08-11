"""Generate the configured test split with optional temporal, spatial, and style controls."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config, require_mapping
from common.paths import resolve_project_path
from data.ckd_dataset import CKDDataset, build_ckd_dataset
from data.cs_dataset import CSDataset, build_cs_dataset
from data.normalization import NormalizerBundle
from data.style_labels import STYLE_VOCABULARIES, normalize_style_name
from infer.pipeline import (
    _clip_indices,
    _tensor,
    _window_range,
    checkpoint_uses_ema,
    infer_keyframes,
    polar8_to_camera20,
    resolve_device,
    seed_everything,
)
from infer.result_io import GenerationRun
from models.checkpoint import load_checkpoint_file, load_model_weights, normalizers_from_checkpoint
from models.ckd import build_ckd_model
from models.cs import build_cs_model


@dataclass(frozen=True)
class SpatioTemporalControl:
    """Sparse camera8 constraints indexed by clip-local frame."""

    frames: np.ndarray
    camera8: np.ndarray

    def __post_init__(self) -> None:
        raw_frames = np.asarray(self.frames)
        if raw_frames.ndim != 1:
            raise ValueError(f"Control frames must have shape [K], got {raw_frames.shape}")
        if not np.issubdtype(raw_frames.dtype, np.integer):
            if not np.isfinite(raw_frames).all() or not np.equal(raw_frames, np.floor(raw_frames)).all():
                raise ValueError("Spatio-temporal frame indices must be integers")
        frames = raw_frames.astype(np.int64, copy=False)
        camera = np.asarray(self.camera8, dtype=np.float32)
        if camera.ndim != 2 or camera.shape[1] not in {8, 20}:
            raise ValueError(f"Controlled cameras must have shape [K,8] or [K,20], got {camera.shape}")
        if len(frames) != len(camera):
            raise ValueError("Spatio-temporal frame and camera counts do not match")
        if len(np.unique(frames)) != len(frames):
            raise ValueError("Spatio-temporal control contains duplicate frame indices")
        if not np.isfinite(camera).all():
            raise ValueError("Spatio-temporal camera values must be finite")
        order = np.argsort(frames)
        object.__setattr__(self, "frames", frames[order])
        object.__setattr__(self, "camera8", camera[order, :8])

    def by_frame(self) -> dict[int, np.ndarray]:
        return {int(frame): self.camera8[index] for index, frame in enumerate(self.frames)}


def _read_control_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError(f"Control files must be JSON or NPY: {path}")


def _sample_payload(payload: Any, sample_id: str, sample_count: int) -> Any | None:
    if not isinstance(payload, Mapping):
        if sample_count != 1:
            raise ValueError("A shared non-mapping control file requires exactly one selected sample")
        return payload
    if sample_id in payload:
        return payload[sample_id]
    if sample_count == 1:
        return payload
    return None


def _load_directory_payload(path: Path, sample_id: str) -> Any | None:
    for suffix in (".npy", ".json"):
        candidate = path / f"{sample_id}{suffix}"
        if candidate.is_file():
            return _read_control_file(candidate)
    return None


def _control_payloads(path: str | Path, frames_by_sample: Mapping[str, int]) -> dict[str, Any]:
    source = resolve_project_path(path)
    if not source.exists():
        raise FileNotFoundError(f"Control path does not exist: {source}")
    if source.is_dir():
        return {
            sample_id: payload
            for sample_id in frames_by_sample
            if (payload := _load_directory_payload(source, sample_id)) is not None
        }
    payload = _read_control_file(source)
    selected: dict[str, Any] = {}
    for sample_id in frames_by_sample:
        value = _sample_payload(payload, sample_id, len(frames_by_sample))
        if value is not None:
            selected[sample_id] = value
    return selected


def _binary_mask_or_frames(value: Any, total_frames: int) -> np.ndarray:
    if isinstance(value, Mapping):
        if "mask" in value:
            value = value["mask"]
        elif "frames" in value:
            value = value["frames"]
        elif "keyframes" in value:
            value = value["keyframes"]
        else:
            raise ValueError("Temporal JSON objects require `mask`, `frames`, or `keyframes`")
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"Temporal control must be one-dimensional, got {array.shape}")
    is_binary_mask = len(array) == total_frames and np.isin(array, (0, 1, False, True)).all()
    if is_binary_mask:
        return array.astype(np.uint8, copy=False)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError("Temporal frame indices must be integers")
    indices = array.astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= total_frames):
        raise ValueError(f"Temporal frame indices must be within [0, {total_frames - 1}]")
    mask = np.zeros(total_frames, dtype=np.uint8)
    mask[indices] = 1
    return mask


def load_temporal_controls(
    path: str | Path | None,
    frames_by_sample: Mapping[str, int],
) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    payloads = _control_payloads(path, frames_by_sample)
    if not payloads:
        raise ValueError("Temporal control path has no entry for any selected sample")
    controls: dict[str, np.ndarray] = {}
    for sample_id, payload in payloads.items():
        controls[sample_id] = _binary_mask_or_frames(payload, frames_by_sample[sample_id])
    return controls


def _spatio_temporal_value(value: Any, total_frames: int) -> SpatioTemporalControl:
    if isinstance(value, Mapping):
        if "frames" in value and "camera" in value:
            frames = np.asarray(value["frames"])
            camera = np.asarray(value["camera"])
        else:
            try:
                ordered = sorted(((int(frame), camera) for frame, camera in value.items()), key=lambda item: item[0])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Spatio-temporal JSON must contain `frames`/`camera` or numeric frame keys"
                ) from error
            frames = np.asarray([item[0] for item in ordered])
            camera = np.asarray([item[1] for item in ordered])
    elif isinstance(value, list) and value and isinstance(value[0], Mapping):
        frames = np.asarray([item["frame"] for item in value])
        camera = np.asarray([item["camera"] for item in value])
    else:
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError(f"Spatio-temporal NPY must be two-dimensional, got {array.shape}")
        if array.shape[1] == 9:
            frames, camera = array[:, 0], array[:, 1:]
        elif array.shape[0] == total_frames and array.shape[1] in {8, 20}:
            finite_rows = np.isfinite(array).all(axis=1)
            partial_rows = np.isfinite(array).any(axis=1) & ~finite_rows
            if partial_rows.any():
                raise ValueError("Dense spatio-temporal control rows must be fully finite or fully NaN")
            frames = np.flatnonzero(finite_rows)
            camera = array[finite_rows]
        else:
            raise ValueError(
                "Spatio-temporal NPY must have shape [K,9], [T,8], or [T,20]"
            )
    control = SpatioTemporalControl(frames, camera)
    if np.any(control.frames < 0) or np.any(control.frames >= total_frames):
        raise ValueError(f"Spatio-temporal frame indices must be within [0, {total_frames - 1}]")
    return control


def load_spatio_temporal_controls(
    path: str | Path | None,
    frames_by_sample: Mapping[str, int],
) -> dict[str, SpatioTemporalControl]:
    if path is None:
        return {}
    payloads = _control_payloads(path, frames_by_sample)
    if not payloads:
        raise ValueError("Spatio-temporal control path has no entry for any selected sample")
    controls: dict[str, SpatioTemporalControl] = {}
    for sample_id, payload in payloads.items():
        controls[sample_id] = _spatio_temporal_value(payload, frames_by_sample[sample_id])
    return controls


def validate_style(style: str, vocabulary: str) -> str:
    normalized = normalize_style_name(style)
    if vocabulary not in STYLE_VOCABULARIES:
        raise ValueError(f"Unknown checkpoint style vocabulary: {vocabulary}")
    if normalized not in STYLE_VOCABULARIES[vocabulary]:
        choices = ", ".join(STYLE_VOCABULARIES[vocabulary])
        raise ValueError(f"Unknown style `{style}`. Available styles: {choices}")
    return normalized


def style_features(style: str, vocabulary: str, frames: int) -> np.ndarray:
    normalized = validate_style(style, vocabulary)
    one_hot = np.zeros(len(STYLE_VOCABULARIES[vocabulary]), dtype=np.float32)
    one_hot[STYLE_VOCABULARIES[vocabulary].index(normalized)] = 1.0
    return np.repeat(one_hot[None, :], frames, axis=0)


def normalize_camera8(camera8: np.ndarray, normalizers: NormalizerBundle) -> np.ndarray:
    value = np.asarray(camera8, dtype=np.float32)
    if value.shape[-1] != 8:
        raise ValueError(f"camera8 must end in 8 features, got {value.shape}")
    return np.concatenate(
        (
            normalizers["camera_distance"].normalize(value[..., 0:1]),
            normalizers["camera_position"].normalize(value[..., 1:4]),
            normalizers["camera_rotation"].normalize(value[..., 4:7]),
            normalizers["camera_fov"].normalize(value[..., 7:8]),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def apply_keyframe_overrides(
    generated: dict[str, np.ndarray],
    temporal: Mapping[str, np.ndarray],
    spatio_temporal: Mapping[str, SpatioTemporalControl],
) -> dict[str, np.ndarray]:
    combined = {sample_id: np.asarray(mask, dtype=np.uint8).copy() for sample_id, mask in generated.items()}
    for sample_id, mask in temporal.items():
        if sample_id not in combined:
            continue
        if len(mask) != len(combined[sample_id]):
            raise ValueError(f"Temporal control length mismatch for {sample_id}")
        combined[sample_id] = np.asarray(mask, dtype=np.uint8).copy()
    for sample_id, control in spatio_temporal.items():
        if sample_id not in combined:
            continue
        combined[sample_id][control.frames] = 1
    return combined


@torch.inference_mode()
def infer_cameras_controlled(
    model: torch.nn.Module,
    dataset: CSDataset,
    device: torch.device,
    sample_ids: set[str] | None = None,
    spatio_temporal: Mapping[str, SpatioTemporalControl] | None = None,
    style_override: str | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    history = dataset.context.history_len
    inference = dataset.context.inference_len
    vocabulary = dataset.context.style_vocabulary
    controls = {} if spatio_temporal is None else dict(spatio_temporal)
    normalized_controls = {
        sample_id: {
            frame: normalized
            for frame, normalized in zip(
                control.frames,
                normalize_camera8(control.camera8, dataset.normalizers),
            )
        }
        for sample_id, control in controls.items()
    }
    generated: dict[str, np.ndarray] = {}
    for clip_index in _clip_indices(dataset, sample_ids):
        clip = dataset.context.clips[clip_index]
        clip_controls = normalized_controls.get(clip.name, {})
        condition: torch.Tensor | None = None
        pieces: list[np.ndarray] = []
        for window_index in _window_range(dataset, clip_index):
            sample = dataset[window_index]
            camera = _tensor(sample["camera"][:, :8], device, torch.float32)
            inference_mask = _tensor(sample["camera_inference_mask"], device, torch.float32)
            motion = _tensor(sample["motion"], device, torch.float32)
            music = _tensor(sample["music"], device, torch.float32)
            style_array = sample["style"]
            if style_override is not None:
                style_array = style_features(style_override, vocabulary, len(style_array))
            style = _tensor(style_array, device, torch.float32)
            active = int(inference_mask.sum().item())
            model_input = camera * 0 + camera[:, 0:1] if condition is None else condition
            output = model(model_input, inference_mask, motion, music, style)

            window = dataset.windows[window_index]
            start_control = clip_controls.get(window.keyframe)
            end_frame = window.keyframe + active
            end_control = clip_controls.get(end_frame) if active < inference else None
            if start_control is not None or end_control is not None:
                keyframe_input = output.detach().clone()
                keyframe_input[:, :history] = model_input[:, :history]
                controlled_mask = inference_mask.clone()
                if start_control is not None:
                    keyframe_input[0, history] = torch.as_tensor(start_control, device=device)
                if end_control is not None:
                    end_index = history + active
                    keyframe_input[0, end_index] = torch.as_tensor(end_control, device=device)
                    controlled_mask[0, end_index, 0] = 1.0
                output = model.forward_with_keyframe(
                    keyframe_input,
                    controlled_mask,
                    motion,
                    music,
                    style,
                )

            condition = torch.zeros_like(output)
            condition[:, :history] = output[:, active : history + active]
            pieces.append(output[0, history : history + active].cpu().numpy().astype(np.float32))
        normalized = np.concatenate(pieces)[: clip.frames]
        generated[clip.name] = polar8_to_camera20(normalized, dataset.normalizers)
    return generated


def _selected_clip_frames(dataset: CKDDataset, selected: set[str] | None) -> dict[str, int]:
    return {
        clip.name: clip.frames
        for clip in dataset.context.clips
        if selected is None or clip.name in selected
    }


def run_controlled_test_generation(
    config: Mapping[str, Any],
    data_config: dict[str, Any],
    *,
    run_name: str | None = None,
    sample_ids: Iterable[str] | None = None,
    temporal_control: str | Path | None = None,
    spatio_temporal_control: str | Path | None = None,
    style: str | None = None,
) -> GenerationRun:
    experiment = require_mapping(config, "experiment")
    checkpoint_paths = require_mapping(config, "checkpoints")
    generation = require_mapping(config, "generation")
    split = str(require_mapping(config, "data").get("split", "test"))
    selected = None if sample_ids is None else set(sample_ids)
    seed_everything(int(experiment.get("seed", 42)))
    device = resolve_device(str(generation.get("device", "auto")))
    use_ema = checkpoint_uses_ema(generation)
    vocabulary = str(generation.get("checkpoint_style_vocabulary", "legacy_dsc_v1"))
    if style is not None:
        style = validate_style(style, vocabulary)

    ckd_checkpoint = load_checkpoint_file(resolve_project_path(str(checkpoint_paths["ckd"])), device)
    ckd_model = build_ckd_model(config).to(device)
    load_model_weights(ckd_model, ckd_checkpoint, use_ema=use_ema, strict=True)
    ckd_dataset = build_ckd_dataset(
        data_config,
        split,
        normalizers=normalizers_from_checkpoint(ckd_checkpoint),
    )
    frames_by_sample = _selected_clip_frames(ckd_dataset, selected)
    if selected is not None:
        missing = selected - frames_by_sample.keys()
        if missing:
            raise ValueError(f"Unknown sample ids for selected split: {', '.join(sorted(missing))}")
    temporal = load_temporal_controls(temporal_control, frames_by_sample)
    spatial = load_spatio_temporal_controls(spatio_temporal_control, frames_by_sample)
    generated_masks = infer_keyframes(ckd_model, ckd_dataset, device, selected)
    keyframes = apply_keyframe_overrides(generated_masks, temporal, spatial)
    del ckd_model, ckd_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cs_checkpoint = load_checkpoint_file(resolve_project_path(str(checkpoint_paths["cs"])), device)
    cs_model = build_cs_model(config).to(device)
    load_model_weights(cs_model, cs_checkpoint, use_ema=use_ema, strict=True)
    cs_dataset = build_cs_dataset(
        data_config,
        split,
        normalizers=normalizers_from_checkpoint(cs_checkpoint),
        generated_keyframe_masks=keyframes,
        style_vocabulary=vocabulary,
    )
    cameras = infer_cameras_controlled(
        cs_model,
        cs_dataset,
        device,
        selected,
        spatio_temporal=spatial,
        style_override=style,
    )

    effective_config = deepcopy(dict(config))
    effective_config["runtime_controls"] = {
        "temporal": None if temporal_control is None else str(temporal_control),
        "spatio_temporal": None if spatio_temporal_control is None else str(spatio_temporal_control),
        "style": style,
    }
    output_root = resolve_project_path(str(generation.get("output_root", "generation")))
    result = GenerationRun.create(output_root, run_name or str(experiment["name"]), effective_config)
    clips = {clip.name: clip for clip in cs_dataset.context.clips}
    for sample_id, camera in cameras.items():
        clip = clips[sample_id]
        selected_style = style or cs_dataset.context.annotations.style_name(clip.sequence_id)
        result.save_camera(
            sample_id,
            camera,
            {
                "sequence_id": clip.sequence_id,
                "source_start_frame": clip.start,
                "source_end_frame": clip.end,
                "source_items": list(clip.source_items),
                "style": selected_style,
                "temporal_control": sample_id in temporal,
                "spatio_temporal_control": sample_id in spatial,
            },
        )
        result.save_keyframes(sample_id, keyframes[sample_id])
    return result


def _optional_path(cli_value: Path | None, config_value: Any) -> Path | None:
    value = cli_value if cli_value is not None else config_value
    if value in (None, ""):
        return None
    return resolve_project_path(str(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", default=None, help="Override experiment.name")
    parser.add_argument("--sample-ids", default=None, help="Comma-separated virtual clip ids")
    parser.add_argument("--temporal-control", type=Path, default=None)
    parser.add_argument("--spatio-temporal-control", type=Path, default=None)
    parser.add_argument("--style", default=None, help="One global style override for all selected clips")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    data_path = resolve_project_path(str(require_mapping(config, "data")["config"]))
    data_config = load_config(data_path)
    controls = config.get("controls", {})
    if not isinstance(controls, Mapping):
        raise ValueError("Optional config section `controls` must be a mapping")
    sample_ids = None if args.sample_ids is None else [value.strip() for value in args.sample_ids.split(",")]
    style = args.style if args.style is not None else controls.get("style")
    result = run_controlled_test_generation(
        config,
        data_config,
        run_name=args.run_name,
        sample_ids=sample_ids,
        temporal_control=_optional_path(args.temporal_control, controls.get("temporal")),
        spatio_temporal_control=_optional_path(
            args.spatio_temporal_control,
            controls.get("spatio_temporal"),
        ),
        style=None if style in (None, "") else str(style),
    )
    print(f"Controlled test generation complete: {result.root}")


if __name__ == "__main__":
    main()
