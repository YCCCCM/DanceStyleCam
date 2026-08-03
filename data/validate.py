"""Validate raw metadata, DCM-style++ arrays, checksums, and split ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.config import load_config
from common.paths import DatasetPaths
from data.raw_dcm import RawDCM
from data.schema import ARRAY_SPECS, SCHEMA_VERSION
from data.splits import build_clips, load_segment_ranges, load_split
from data.store import SequenceStore
from data.style_labels import StyleAnnotations


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(
    config: dict[str, Any],
    verify_checksums: bool = True,
    sequence_ids: list[str] | None = None,
) -> dict[str, Any]:
    paths = DatasetPaths.from_config(config)
    annotations = StyleAnnotations.load(paths.style_file)
    train_items = load_split(paths.train_split)
    test_items = load_split(paths.test_split)
    overlap = {item.name for item in train_items} & {item.name for item in test_items}
    if overlap:
        raise ValueError(f"Train/test fragment overlap: {sorted(overlap)}")

    store = SequenceStore(paths.processed_root)
    manifest = store.load_manifest()
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unexpected schema version: {manifest.get('schema_version')}")
    if "style" in manifest or "split" in manifest:
        raise ValueError("DCM-style++ manifest must not embed mutable style or split annotations")

    selected = store.sequence_ids() if sequence_ids is None else sorted(set(sequence_ids), key=int)
    raw = RawDCM(paths.raw_root)
    issues: list[str] = []
    total_frames = 0
    for sequence_id in selected:
        if sequence_id not in manifest["sequences"]:
            issues.append(f"sequence {sequence_id}: missing manifest entry")
            continue
        if sequence_id not in annotations.by_sequence:
            issues.append(f"sequence {sequence_id}: missing style annotation")
        for path in raw.missing_files(sequence_id):
            issues.append(f"sequence {sequence_id}: missing raw file {path}")

        entry = manifest["sequences"][sequence_id]
        expected_frames = int(entry["frames"])
        total_frames += expected_frames
        for feature, spec in ARRAY_SPECS.items():
            path = store.array_path(sequence_id, feature)
            if not path.is_file():
                issues.append(f"sequence {sequence_id}: missing {feature} array")
                continue
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != (expected_frames, *spec.trailing_shape):
                issues.append(f"sequence {sequence_id}: {feature} shape is {array.shape}")
            if array.dtype.name != spec.dtype:
                issues.append(f"sequence {sequence_id}: {feature} dtype is {array.dtype}")
            expected_hash = entry.get("sha256", {}).get(feature)
            if verify_checksums and expected_hash and _sha256(path) != expected_hash:
                issues.append(f"sequence {sequence_id}: {feature} checksum mismatch")

    frames_by_sequence = {
        sequence_id: int(entry["frames"])
        for sequence_id, entry in manifest["sequences"].items()
    }
    segments = load_segment_ranges(paths.segment_file)
    referenced = train_items + test_items
    missing_sequences = sorted({item.sequence_id for item in referenced} - set(frames_by_sequence), key=int)
    if not missing_sequences:
        build_clips(train_items, segments, frames_by_sequence, merge_adjacent=True)
        build_clips(test_items, segments, frames_by_sequence, merge_adjacent=False)
    elif sequence_ids is None:
        issues.append(f"split references unconverted sequences: {missing_sequences}")

    report = {
        "schema_version": SCHEMA_VERSION,
        "sequences": len(selected),
        "frames": total_frames,
        "train_fragments": len(train_items),
        "test_fragments": len(test_items),
        "issues": issues,
    }
    if issues:
        raise ValueError("DCM-style++ validation failed:\n" + "\n".join(f"- {issue}" for issue in issues))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--sequence-ids", default=None, help="Optional comma-separated converted subset")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sequence_ids = None if args.sequence_ids is None else [value.strip() for value in args.sequence_ids.split(",")]
    report = validate_dataset(
        load_config(args.config),
        verify_checksums=not args.skip_checksums,
        sequence_ids=sequence_ids,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
