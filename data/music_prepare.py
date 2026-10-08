"""Pre-extract DCM++-compatible music35 NPY files for configured virtual clips."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from common.config import load_config
from common.paths import DatasetPaths
from data.audio_features import extract_music35
from data.dataset_common import clips_for_split, cs_fragment_start_clips
from data.music_store import MusicClipSpec, MusicFeatureStore
from data.raw_dcm import RawDCM
from data.splits import load_segment_ranges, music_frame_range
from data.store import SequenceStore


def music_specs(config: dict, split: str) -> list[tuple[MusicClipSpec, Path]]:
    paths = DatasetPaths.from_config(config)
    store = SequenceStore(paths.processed_root)
    raw = RawDCM(paths.raw_root)
    segments = load_segment_ranges(paths.segment_file)
    specs: list[tuple[MusicClipSpec, Path]] = []
    seen: dict[str, dict] = {}
    clips = clips_for_split(config, split, store)
    if split == "train":
        clips.extend(cs_fragment_start_clips(config, store))
    for clip in clips:
        files = raw.sequence_files(clip.sequence_id)
        audio_start, audio_end = music_frame_range(clip, segments)
        aligned_frame_limit = (
            None
            if files.aligned_audio is not None
            else int(store.load_manifest()["sequences"][clip.sequence_id]["aligned_frame_limit"])
        )
        spec = MusicClipSpec(clip, audio_start, audio_end, aligned_frame_limit)
        existing = seen.get(spec.key)
        if existing is not None and existing != spec.metadata():
            raise ValueError(f"Conflicting music35 definitions for clip `{spec.key}`")
        seen[spec.key] = spec.metadata()
        specs.append((spec, files.audio))
    return specs


def _source_features(sources: list[Path]) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = {}
    for source in sources:
        roots = [source] if source.name == "aist_feats_long" else source.glob("*/aist_feats_long")
        for root in roots:
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*.npy")):
                stem = path.stem
                if not stem.startswith("a") or "_" not in stem:
                    continue
                key = stem[1:].rsplit("_", 1)[0]
                indexed.setdefault(key, []).append(path)
    for key, paths in indexed.items():
        if len(paths) < 2:
            continue
        hashes = {hashlib.sha256(path.read_bytes()).digest() for path in paths}
        if len(hashes) != 1:
            names = ", ".join(str(path) for path in paths)
            raise ValueError(f"Conflicting DCM++ music35 sources for `{key}`: {names}")
    return indexed


def _load_source_feature(paths: list[Path], spec: MusicClipSpec) -> np.ndarray | None:
    for path in paths:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.ndim == 2 and value.shape[1] == 35 and len(value) >= spec.clip.frames:
            return np.asarray(value[: spec.clip.frames], dtype=np.float32)
    return None


def prepare_music_features(
    config: dict,
    overwrite: bool = False,
    sources: list[Path] | None = None,
    extract_missing: bool = False,
) -> dict[str, int]:
    paths = DatasetPaths.from_config(config)
    store = MusicFeatureStore(paths.processed_root)
    manifest = store.load_manifest()
    requested: dict[str, tuple[MusicClipSpec, Path]] = {}
    for split in ("train", "test"):
        for spec, audio_path in music_specs(config, split):
            existing = requested.get(spec.key)
            if existing is not None and existing[0].metadata() != spec.metadata():
                raise ValueError(f"Conflicting train/test music35 definitions for clip `{spec.key}`")
            requested[spec.key] = (spec, audio_path)

    source_index = _source_features([] if sources is None else sources)
    imported = created = skipped = 0
    missing: list[str] = []
    for index, (spec, audio_path) in enumerate(requested.values(), start=1):
        if not overwrite and store.is_current(spec, manifest):
            skipped += 1
            continue
        music = _load_source_feature(source_index.get(spec.key, []), spec)
        if music is not None:
            imported += 1
        elif extract_missing:
            music = extract_music35(
                audio_path,
                spec.audio_start,
                spec.audio_end,
                spec.clip.frames,
                aligned_frame_limit=spec.aligned_frame_limit,
            )
            created += 1
        else:
            missing.append(spec.key)
            continue
        store.save(spec, music, manifest)
        if index == 1 or index % 10 == 0 or index == len(requested):
            print(f"Prepared music35 for {index}/{len(requested)} clips", flush=True)
    if missing:
        sample = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"No pre-extracted music35 source for {len(missing)} clips ({sample}). "
            "Pass --extract-missing to create only the missing fixed NPY files."
        )
    return {"requested": len(requested), "imported": imported, "created": created, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=[],
        help="Existing DCM++ root or aist_feats_long directory to import before extracting",
    )
    parser.add_argument("--extract-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = prepare_music_features(
        load_config(args.config),
        overwrite=args.overwrite,
        sources=args.source,
        extract_missing=args.extract_missing,
    )
    print(
        "music35 ready: "
        f"requested={report['requested']}, imported={report['imported']}, "
        f"created={report['created']}, skipped={report['skipped']}"
    )


if __name__ == "__main__":
    main()
