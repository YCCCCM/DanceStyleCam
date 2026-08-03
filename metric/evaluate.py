"""Evaluate a generation run against virtual DCM-style++ reference clips."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from common.config import load_config, require_mapping
from common.paths import resolve_project_path
from data.camera_geometry import detect_bone_mask
from data.store import SequenceStore
from infer.result_io import GenerationRun
from metric.features import (
    average_pairwise_distance,
    fid,
    kinetic_features,
    normalize_features,
    shot_features,
    stack_or_empty,
    style_features,
)


def _trajectory_metrics(camera: np.ndarray) -> dict[str, float]:
    camera = np.asarray(camera, dtype=np.float32)
    velocity = np.diff(camera[:, 8:11], axis=0)
    acceleration = np.diff(velocity, axis=0)
    return {
        "camera_eye_velocity_l2": float(np.linalg.norm(velocity, axis=1).mean()) if len(velocity) else 0.0,
        "camera_eye_acceleration_l2": float(np.linalg.norm(acceleration, axis=1).mean()) if len(acceleration) else 0.0,
        "fov_velocity_l1": float(np.abs(np.diff(camera[:, 7])).mean()) if len(camera) > 1 else 0.0,
    }


def _reference_arrays(store: SequenceStore, metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    required = ("sequence_id", "source_start_frame", "source_end_frame")
    if not all(key in metadata for key in required):
        return None
    sequence_id = str(metadata["sequence_id"])
    start = int(metadata["source_start_frame"])
    end = int(metadata["source_end_frame"])
    return (
        np.asarray(store.load(sequence_id, "camera20")[start:end]),
        np.asarray(store.load(sequence_id, "motion180")[start:end]),
        np.asarray(store.load(sequence_id, "bone_mask60")[start:end]),
    )


def _dancer_missing_rate(masks: list[np.ndarray]) -> float | None:
    frame_count = sum(len(mask) for mask in masks)
    if frame_count == 0:
        return None
    missing_count = sum(int(np.sum(np.sum(mask, axis=1) == 0)) for mask in masks)
    return missing_count / frame_count


def _benchmark_metrics(
    generated_kinetic: np.ndarray,
    reference_kinetic: np.ndarray,
    generated_shot: np.ndarray,
    reference_shot: np.ndarray,
    generated_masks: list[np.ndarray],
    reference_masks: list[np.ndarray],
    block_size: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "feature_schema": "dancestylecam_paper_v1",
        "kinetic_feature_count": int(len(generated_kinetic)),
        "reference_kinetic_feature_count": int(len(reference_kinetic)),
        "shot_feature_count": int(len(generated_shot)),
        "reference_shot_feature_count": int(len(reference_shot)),
        "res_dmr": _dancer_missing_rate(generated_masks),
        "test_dmr": _dancer_missing_rate(reference_masks),
    }
    if len(generated_kinetic) and len(reference_kinetic):
        reference_norm, generated_norm = normalize_features(reference_kinetic, generated_kinetic)
        result.update(
            {
                "fid_k_res": fid(generated_norm, reference_norm),
                "div_k_res": average_pairwise_distance(generated_norm, block_size),
                "div_k_test": average_pairwise_distance(reference_norm, block_size),
            }
        )
    else:
        result.update({"fid_k_res": None, "div_k_res": None, "div_k_test": None})
    if len(generated_shot) and len(reference_shot):
        reference_norm, generated_norm = normalize_features(reference_shot, generated_shot)
        result.update(
            {
                "fid_s_res": fid(generated_norm, reference_norm),
                "div_s_res": average_pairwise_distance(generated_norm, block_size),
                "div_s_test": average_pairwise_distance(reference_norm, block_size),
            }
        )
    else:
        result.update({"fid_s_res": None, "div_s_res": None, "div_s_test": None})

    frame_count = 0
    lcd_total = 0.0
    for generated, reference in zip(generated_masks, reference_masks):
        overlap = min(len(generated), len(reference))
        if overlap:
            lcd_total += float(np.sum(np.mean((generated[:overlap] - reference[:overlap]) ** 2, axis=1)))
            frame_count += overlap
    result["lcd"] = lcd_total / max(frame_count, 1)
    return result


def _style_metrics(
    generated: list[tuple[str, np.ndarray]],
    reference: list[tuple[str, np.ndarray]],
) -> dict[str, Any]:
    if not generated or not reference:
        return {"available": False, "reason": "generation metadata has no style/reference samples"}
    reference_features = np.stack([features for _, features in reference]).astype(np.float32)
    generated_features = np.stack([features for _, features in generated]).astype(np.float32)
    min_values = reference_features.min(axis=0)
    ranges = reference_features.max(axis=0) - min_values
    constant = ranges == 0
    ranges = ranges.copy()
    ranges[constant] = 1.0
    reference_norm = 2.0 * (reference_features - min_values) / ranges - 1.0
    generated_norm = 2.0 * (generated_features - min_values) / ranges - 1.0
    reference_norm[:, constant] = 0.0
    generated_norm[:, constant] = 0.0

    prototypes: dict[str, np.ndarray] = {}
    for style in sorted({style for style, _ in reference}):
        prototypes[style] = reference_norm[[index for index, (name, _) in enumerate(reference) if name == style]].mean(axis=0)

    results: list[dict[str, Any]] = []
    for (style, _), feature in zip(generated, generated_norm):
        distances = {name: float(np.linalg.norm(feature - prototype)) for name, prototype in prototypes.items()}
        similarities = {
            name: float(np.dot(feature, prototype) / (np.linalg.norm(feature) * np.linalg.norm(prototype)))
            if np.linalg.norm(feature) > 0 and np.linalg.norm(prototype) > 0
            else 0.0
            for name, prototype in prototypes.items()
        }
        nearest = min(distances, key=distances.get)
        ordered = sorted(distances, key=distances.get)
        results.append(
            {
                "target_style": style,
                "nearest_style": nearest,
                "strict_match": nearest == style,
                "target_distance": distances.get(style),
                "target_similarity": similarities.get(style, 0.0),
                "top_2": style in ordered[:2],
                "top_3": style in ordered[:3],
                "top_5": style in ordered[:5],
            }
        )
    target_similarity = [item["target_similarity"] for item in results]
    target_distances = [item["target_distance"] for item in results if item["target_distance"] is not None]
    confusion: dict[str, int] = {}
    for item in results:
        key = f'{item["target_style"]}-{item["nearest_style"]}'
        confusion[key] = confusion.get(key, 0) + 1
    return {
        "available": True,
        "styles": sorted(prototypes),
        "samples": results,
        "strict_accuracy": float(np.mean([item["strict_match"] for item in results])),
        "top_2_accuracy": float(np.mean([item["top_2"] for item in results])),
        "top_3_accuracy": float(np.mean([item["top_3"] for item in results])),
        "top_5_accuracy": float(np.mean([item["top_5"] for item in results])),
        "avg_target_distance": float(np.mean(target_distances)) if target_distances else None,
        "avg_target_similarity": float(np.mean(target_similarity)),
        "std_target_similarity": float(np.std(target_similarity)),
        "max_target_similarity": float(np.max(target_similarity)),
        "min_target_similarity": float(np.min(target_similarity)),
        "median_target_similarity": float(np.median(target_similarity)),
        "quality_score": float(0.5 * np.mean([item["strict_match"] for item in results]) + 0.5 * (np.mean(target_similarity) + 1.0) / 2.0),
        "confusion_matrix": confusion,
    }


def evaluate_run(config: dict[str, Any], input_root: str | Path) -> dict[str, Any]:
    run = GenerationRun.open(input_root)
    evaluation = require_mapping(config, "evaluation")
    data_config = load_config(resolve_project_path(str(require_mapping(config, "data")["config"])))
    paths = require_mapping(data_config, "paths")
    store = SequenceStore(resolve_project_path(str(paths["processed_root"])))
    manifest = run.load_manifest()
    records: dict[str, dict[str, Any]] = {}
    generated_kinetic: list[np.ndarray] = []
    reference_kinetic: list[np.ndarray] = []
    generated_shot: list[np.ndarray] = []
    reference_shot: list[np.ndarray] = []
    generated_masks: list[np.ndarray] = []
    reference_masks: list[np.ndarray] = []
    generated_style: list[tuple[str, np.ndarray]] = []
    reference_style: list[tuple[str, np.ndarray]] = []

    for sample_id, sample in manifest["samples"].items():
        camera = np.asarray(run.load_camera(sample_id), dtype=np.float32)
        metadata = sample.get("metadata", {})
        record = _trajectory_metrics(camera)
        reference = _reference_arrays(store, metadata)
        if reference is not None:
            target_camera, target_motion, target_mask = reference
            overlap = min(len(target_camera), len(camera), len(target_motion), len(target_mask))
            target_camera = target_camera[:overlap]
            target_motion = target_motion[:overlap]
            target_mask = target_mask[:overlap]
            camera = camera[:overlap]
            record["camera_mse_to_reference"] = float(np.mean((camera - target_camera) ** 2))
            record["reference_frames"] = overlap
            generated_mask = detect_bone_mask(camera, target_motion)
            generated_masks.append(generated_mask)
            reference_masks.append(target_mask)
            generated_kinetic.append(kinetic_features(camera))
            reference_kinetic.append(kinetic_features(target_camera))
            generated_shot.append(shot_features(camera, target_motion, generated_mask))
            reference_shot.append(shot_features(target_camera, target_motion, target_mask))
            style = str(metadata.get("style", "unknown"))
            generated_style.append((style, style_features(camera)))
            reference_style.append((style, style_features(target_camera)))
        records[sample_id] = record

    summary: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": bool(evaluation.get("benchmark", True)),
        "style_consistency": bool(evaluation.get("style_consistency", True)),
        "samples": records,
    }
    if records:
        numeric_keys = sorted(
            {key for record in records.values() for key, value in record.items() if isinstance(value, (int, float))}
        )
        summary["mean"] = {
            key: float(np.mean([value[key] for value in records.values() if key in value]))
            for key in numeric_keys
        }
    if summary["benchmark"]:
        summary["benchmark_metrics"] = _benchmark_metrics(
            stack_or_empty(generated_kinetic, 10),
            stack_or_empty(reference_kinetic, 10),
            stack_or_empty(generated_shot, 2),
            stack_or_empty(reference_shot, 2),
            generated_masks,
            reference_masks,
            int(evaluation.get("diversity_block_size", 2048)),
        )
    if summary["style_consistency"]:
        summary["style_metrics"] = _style_metrics(generated_style, reference_style)
    output = run.derived_dir(str(evaluation.get("output_subdir", "metrics"))) / "summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Existing generation/<run-name> directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate_run(load_config(args.config), args.input)
    print(json.dumps(summary.get("benchmark_metrics", summary.get("mean", {})), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
