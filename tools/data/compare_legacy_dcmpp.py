"""Compare DCM-style++ arrays and virtual clips with a legacy DCM++ tree."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from common.config import load_config, require_mapping
from common.paths import DatasetPaths
from data.audio_features import extract_music35_clip
from data.raw_dcm import RawDCM
from data.splits import load_segment_ranges
from data.store import SequenceStore


FEATURE_TOLERANCES = {
    "camera20": 1e-6,
    "keyframe_mask": 0.0,
    "bone_mask60": 0.0,
    "motion180": 0.0,
    "music35_clip": 1e-6,
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _clip_range(
    clip_name: str,
    segments: dict[str, list[tuple[int, int]]],
    sequence_frames: int,
) -> tuple[str, int, int, int | None]:
    parts = clip_name.split("_", maxsplit=1)
    sequence_id = parts[0]
    if len(parts) == 1:
        return sequence_id, 0, sequence_frames, None
    segment_range = parts[1]
    if "~" in segment_range:
        first, last = (int(value) for value in segment_range.split("~", maxsplit=1))
    else:
        first = last = int(segment_range)
    start = int(segments[sequence_id][first][0])
    audio_end = int(segments[sequence_id][last][1]) + 1
    end = min(audio_end, sequence_frames)
    return sequence_id, start, end, audio_end


def _legacy_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    value = _read_json(path)
    camera = np.concatenate(
        (
            np.asarray(value["Distance"], dtype=np.float32)[:, None],
            np.asarray(value["Position"], dtype=np.float32),
            np.asarray(value["Rotation"], dtype=np.float32),
            np.asarray(value["Fov"], dtype=np.float32)[:, None],
            np.asarray(value["camera_eye"], dtype=np.float32),
            np.asarray(value["camera_x"], dtype=np.float32),
            np.asarray(value["camera_y"], dtype=np.float32),
            np.asarray(value["camera_z"], dtype=np.float32),
        ),
        axis=1,
    )
    return camera, np.asarray(value["KeyframeMask"], dtype=np.uint8)


def _legacy_music_path(split_root: Path, clip_name: str) -> Path:
    matches = sorted((split_root / "aist_feats_long").glob(f"a{clip_name}_*.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one legacy music feature for {clip_name}, found {matches}")
    return matches[0]


def _comparison(new: np.ndarray, legacy: np.ndarray, tolerance: float) -> dict[str, Any]:
    overlap = min(len(new), len(legacy))
    new_overlap = np.asarray(new[:overlap])
    legacy_overlap = np.asarray(legacy[:overlap])
    difference = np.abs(new_overlap.astype(np.float64) - legacy_overlap.astype(np.float64))
    max_abs = float(difference.max()) if difference.size else 0.0
    return {
        "new_shape": list(new.shape),
        "legacy_shape": list(legacy.shape),
        "overlap_frames": overlap,
        "max_abs": max_abs,
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "values_over_tolerance": int(np.sum(difference > tolerance)),
        "passed": bool(max_abs <= tolerance and len(new) <= len(legacy)),
    }


def compare_legacy(config: dict[str, Any], legacy_root: Path, split_names: list[str]) -> dict[str, Any]:
    paths = DatasetPaths.from_config(config)
    store = SequenceStore(paths.processed_root)
    raw = RawDCM(paths.raw_root)
    segments = load_segment_ranges(paths.segment_file)
    samples: dict[str, Any] = {}
    compared = 0

    for split_name in split_names:
        split_root = legacy_root / split_name
        for camera_path in sorted((split_root / "CameraCentric").glob("c*.json")):
            clip_name = camera_path.stem[1:]
            sequence_id, start, end, audio_end = _clip_range(
                clip_name,
                segments,
                store.sequence_frames(clip_name.split("_", maxsplit=1)[0]),
            )
            legacy_camera, legacy_keyframes = _legacy_camera(camera_path)
            legacy_bone = np.asarray(
                _read_json(split_root / "BoneMask" / f"bm{clip_name}.json")["bone_mask"],
                dtype=np.uint8,
            )
            legacy_motion = np.asarray(
                _read_json(split_root / "Keypoints3D" / f"m{clip_name}_kps3D.json")["Keypoints3D"],
                dtype=np.float32,
            )
            legacy_music = np.load(_legacy_music_path(split_root, clip_name), allow_pickle=False)
            files = raw.sequence_files(sequence_id)
            aligned_frame_limit = (
                None
                if files.aligned_audio is not None
                else int(store.load_manifest()["sequences"][sequence_id]["aligned_frame_limit"])
            )
            clip_music = extract_music35_clip(
                files.audio,
                start,
                audio_end,
                end - start,
                aligned_frame_limit=aligned_frame_limit,
            )
            arrays = {
                "camera20": (store.load(sequence_id, "camera20")[start:end], legacy_camera),
                "keyframe_mask": (store.load(sequence_id, "keyframe_mask")[start:end], legacy_keyframes),
                "bone_mask60": (store.load(sequence_id, "bone_mask60")[start:end], legacy_bone),
                "motion180": (store.load(sequence_id, "motion180")[start:end], legacy_motion),
                "music35_clip": (clip_music, legacy_music),
            }
            samples[f"{split_name}/{clip_name}"] = {
                name: _comparison(new, legacy, FEATURE_TOLERANCES[name])
                for name, (new, legacy) in arrays.items()
            }
            compared += 1
            if compared == 1 or compared % 10 == 0:
                print(f"Compared {compared} legacy clips (latest: {split_name}/{clip_name})", flush=True)

    feature_summary: dict[str, Any] = {}
    for feature, tolerance in FEATURE_TOLERANCES.items():
        values = [sample[feature] for sample in samples.values()]
        feature_summary[feature] = {
            "tolerance": tolerance,
            "samples": len(values),
            "passed": sum(value["passed"] for value in values),
            "failed": sum(not value["passed"] for value in values),
            "max_abs": max((value["max_abs"] for value in values), default=0.0),
            "values_over_tolerance": sum(value["values_over_tolerance"] for value in values),
        }
    return {
        "schema_version": 1,
        "legacy_root": str(legacy_root.resolve()),
        "processed_root": str(paths.processed_root),
        "split_names": split_names,
        "clips": len(samples),
        "passed": all(value["failed"] == 0 for value in feature_summary.values()),
        "features": feature_summary,
        "samples": samples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--splits", default="Train,Test", help="Comma-separated legacy subdirectories")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    require_mapping(config, "paths")
    report = compare_legacy(
        config,
        args.legacy_root,
        [value.strip() for value in args.splits.split(",") if value.strip()],
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "clips": report["clips"], **report["features"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
