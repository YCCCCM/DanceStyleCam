"""Convert public DCM files directly into memory-mappable DCM-style++ arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.config import load_config, require_mapping
from common.paths import DatasetPaths
from data.audio_features import audio_frame_count, extract_music35_clip
from data.camera_geometry import (
    CameraAlignment,
    align_camera_keyframes,
    detect_bone_mask,
    global_transforms_to_keypoints,
    interpolate_camera,
)
from data.raw_dcm import RawDCM
from data.schema import ARRAY_SPECS, FPS, SCHEMA_VERSION
from data.style_labels import StyleAnnotations


def _array_contract() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "directory": spec.directory,
            "trailing_shape": list(spec.trailing_shape),
            "dtype": spec.dtype,
        }
        for name, spec in ARRAY_SPECS.items()
    }


def _empty_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "fps": FPS,
        "alignment_policy": "legacy_v1_last_camera_keyframe_excluded",
        "music_feature_scope": "full_aligned_sequence",
        "arrays": _array_contract(),
        "sequences": {},
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    temporary.replace(path)


def _save_npy_atomic(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_is_complete(root: Path, sequence_id: str, entry: dict[str, Any]) -> bool:
    if int(entry.get("frames", -1)) <= 0:
        return False
    return all((root / spec.directory / f"{sequence_id}.npy").is_file() for spec in ARRAY_SPECS.values())


def _load_or_create_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        return _empty_manifest()
    manifest = _read_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Cannot resume unsupported DCM-style++ schema: {manifest.get('schema_version')}")
    if manifest.get("arrays") != _array_contract():
        raise ValueError("Cannot resume because the stored array contract has changed")
    return manifest


def convert_sequence(raw: RawDCM, sequence_id: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    files = raw.sequence_files(sequence_id)
    missing = raw.missing_files(sequence_id)
    if missing:
        raise FileNotFoundError("Missing raw DCM files:\n" + "\n".join(str(path) for path in missing))

    camera_data = _read_json(files.camera)
    motion_data = _read_json(files.aligned_motion or files.motion)
    source_audio_frames = audio_frame_count(files.audio, fps=FPS)
    source_motion_frames = int(motion_data["BoneKeyFrameNumber"])
    aligned_camera = _read_json(files.camera_centric) if files.camera_centric is not None else None
    if aligned_camera is not None:
        output_frames = len(aligned_camera["camera_eye"])
        retained = [
            item
            for item in sorted(camera_data["CameraKeyFrameRecord"], key=lambda item: item["FrameTime"])
            if int(item["FrameTime"]) <= output_frames
        ]
        alignment = CameraAlignment(retained, output_frames + 1, output_frames)
    else:
        alignment = align_camera_keyframes(camera_data, source_audio_frames, source_motion_frames)

    if files.camera_centric is not None:
        # Reuse the aligned public camera representation when available.  It
        # is the output of the released DCM preprocessing and avoids tiny
        # float differences from reimplementing GLM on another platform.
        assert aligned_camera is not None
        camera20 = np.concatenate(
            (
                np.asarray(aligned_camera["Distance"], dtype=np.float32)[:, None],
                np.asarray(aligned_camera["Position"], dtype=np.float32),
                np.asarray(aligned_camera["Rotation"], dtype=np.float32),
                np.asarray(aligned_camera["Fov"], dtype=np.float32)[:, None],
                np.asarray(aligned_camera["camera_eye"], dtype=np.float32),
                np.asarray(aligned_camera["camera_x"], dtype=np.float32),
                np.asarray(aligned_camera["camera_y"], dtype=np.float32),
                np.asarray(aligned_camera["camera_z"], dtype=np.float32),
            ),
            axis=1,
        )[: alignment.output_frames]
        keyframe_mask = np.zeros(alignment.output_frames, dtype=np.uint8)
        for item in alignment.keyframes:
            frame = int(item["FrameTime"])
            if frame < alignment.output_frames:
                keyframe_mask[frame] = 1
    else:
        camera20, keyframe_mask = interpolate_camera(alignment)
    motion180 = global_transforms_to_keypoints(
        motion_data["BoneKeyFrameTransformRecord"],
        alignment.output_frames,
    )
    bone_mask60 = detect_bone_mask(camera20, motion180)
    music35 = extract_music35_clip(
        files.audio,
        start_frame=0,
        end_frame=None,
        output_frames=alignment.output_frames,
        aligned_frame_limit=None if files.aligned_audio is not None else alignment.aligned_frame_limit,
    )

    arrays = {
        "motion180": motion180.astype(np.float32, copy=False),
        "camera20": camera20.astype(np.float32, copy=False),
        "music35": music35.astype(np.float32, copy=False),
        "keyframe_mask": keyframe_mask.astype(np.uint8, copy=False),
        "bone_mask60": bone_mask60.astype(np.uint8, copy=False),
    }
    lengths = {name: len(value) for name, value in arrays.items()}
    if set(lengths.values()) != {alignment.output_frames}:
        raise ValueError(f"Sequence {sequence_id} is not frame-aligned: {lengths}")

    metadata = {
        "frames": alignment.output_frames,
        "aligned_frame_limit": alignment.aligned_frame_limit,
        "source_audio_frames": source_audio_frames,
        "source_motion_frames": source_motion_frames,
        "source_camera_keyframes": len(camera_data["CameraKeyFrameRecord"]),
    }
    return arrays, metadata


def prepare_dataset(config: dict[str, Any], sequence_ids: list[str] | None = None) -> dict[str, Any]:
    paths = DatasetPaths.from_config(config)
    processing = require_mapping(config, "processing")
    if int(processing.get("fps", FPS)) != FPS:
        raise ValueError(f"DanceStyleCam preprocessing currently requires {FPS} FPS")
    if processing.get("alignment_policy", "legacy_v1") != "legacy_v1":
        raise ValueError("Only the checkpoint-compatible `legacy_v1` alignment policy is supported")

    annotations = StyleAnnotations.load(paths.style_file)
    requested = sequence_ids if sequence_ids is not None else processing.get("sequence_ids")
    selected = list(annotations.by_sequence) if requested is None else [str(value) for value in requested]
    unknown = sorted(set(selected) - set(annotations.by_sequence), key=int)
    if unknown:
        raise ValueError(f"Sequence IDs have no style annotation: {unknown}")
    selected = sorted(set(selected), key=int)

    output_root = paths.processed_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(output_root)
    resume = bool(processing.get("resume", True))
    overwrite = bool(processing.get("overwrite", False))
    if manifest["sequences"] and not resume and not overwrite:
        raise FileExistsError(f"DCM-style++ already contains data: {output_root}")

    raw = RawDCM(paths.raw_root)
    for position, sequence_id in enumerate(selected, start=1):
        existing = manifest["sequences"].get(sequence_id)
        if resume and not overwrite and existing and _sequence_is_complete(output_root, sequence_id, existing):
            print(f"[{position:03d}/{len(selected):03d}] skip sequence {sequence_id} (complete)")
            continue

        print(f"[{position:03d}/{len(selected):03d}] convert sequence {sequence_id}")
        arrays, metadata = convert_sequence(raw, sequence_id)
        checksums: dict[str, str] = {}
        for feature, value in arrays.items():
            output_path = output_root / ARRAY_SPECS[feature].directory / f"{sequence_id}.npy"
            checksums[feature] = _save_npy_atomic(output_path, value)
        metadata["sha256"] = checksums
        manifest["sequences"][sequence_id] = metadata
        _write_json_atomic(output_root / "manifest.json", manifest)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--sequence-ids",
        default=None,
        help="Optional comma-separated subset for validation or resuming selected sequences",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sequence_ids = None if args.sequence_ids is None else [value.strip() for value in args.sequence_ids.split(",")]
    manifest = prepare_dataset(load_config(args.config), sequence_ids=sequence_ids)
    print(f"DCM-style++ contains {len(manifest['sequences'])} converted sequences")


if __name__ == "__main__":
    main()
