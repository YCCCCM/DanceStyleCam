"""Persistent DCM++-compatible music35 features for virtual clips."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .splits import ClipRef


MUSIC_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MusicClipSpec:
    clip: ClipRef
    audio_start: int
    audio_end: int | None
    aligned_frame_limit: int | None

    @property
    def key(self) -> str:
        return self.clip.name

    def metadata(self) -> dict[str, Any]:
        return {
            "sequence_id": self.clip.sequence_id,
            "start": self.clip.start,
            "end": self.clip.end,
            "source_items": list(self.clip.source_items),
            "audio_start": self.audio_start,
            "audio_end": self.audio_end,
            "aligned_frame_limit": self.aligned_frame_limit,
            "frames": self.clip.frames,
        }


class MusicFeatureStore:
    """Read/write clip-local music features without creating train/test copies."""

    def __init__(self, processed_root: str | Path) -> None:
        self.root = Path(processed_root) / "music35"
        self.clip_root = self.root / "clips"
        self.manifest_path = self.root / "manifest.json"

    def feature_path(self, spec: MusicClipSpec) -> Path:
        return self.clip_root / f"{spec.key}.npy"

    def _empty_manifest(self) -> dict[str, Any]:
        return {"schema_version": MUSIC_SCHEMA_VERSION, "clips": {}}

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return self._empty_manifest()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != MUSIC_SCHEMA_VERSION:
            raise ValueError(f"Unsupported music35 schema: {manifest.get('schema_version')}")
        if not isinstance(manifest.get("clips"), dict):
            raise ValueError("music35 manifest must contain a clip mapping")
        return manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
        temporary.replace(self.manifest_path)

    def is_current(self, spec: MusicClipSpec, manifest: dict[str, Any] | None = None) -> bool:
        current = self.load_manifest() if manifest is None else manifest
        metadata = current["clips"].get(spec.key)
        expected = {
            "sequence_id": spec.clip.sequence_id,
            "start": spec.clip.start,
            "end": spec.clip.end,
            "source_items": list(spec.clip.source_items),
            "frames": spec.clip.frames,
        }
        return (
            metadata is not None
            and all(metadata.get(key) == value for key, value in expected.items())
            and self.feature_path(spec).is_file()
        )

    def load(self, spec: MusicClipSpec) -> np.ndarray:
        manifest = self.load_manifest()
        if not self.is_current(spec, manifest):
            raise FileNotFoundError(
                f"Missing pre-extracted music35 for `{spec.key}`. Run "
                "`python tools/data/prepare_music_features.py --config <data-config>`."
            )
        value = np.load(self.feature_path(spec), mmap_mode="r", allow_pickle=False)
        if value.shape != (spec.clip.frames, 35) or value.dtype.name != "float32":
            raise ValueError(f"Invalid music35 array for {spec.key}: {value.shape} {value.dtype}")
        return value

    def load_clip(self, clip: ClipRef) -> np.ndarray:
        manifest = self.load_manifest()
        if not self.is_current(MusicClipSpec(clip, 0, None, None), manifest):
            raise FileNotFoundError(
                f"Missing pre-extracted music35 for `{clip.name}`. Run "
                "`python tools/data/prepare_music_features.py --config <data-config>`."
            )
        value = np.load(self.feature_path(MusicClipSpec(clip, 0, None, None)), mmap_mode="r", allow_pickle=False)
        if value.shape != (clip.frames, 35) or value.dtype.name != "float32":
            raise ValueError(f"Invalid music35 array for {clip.name}: {value.shape} {value.dtype}")
        return value

    def save(self, spec: MusicClipSpec, value: np.ndarray, manifest: dict[str, Any]) -> None:
        feature = np.asarray(value, dtype=np.float32)
        if feature.shape != (spec.clip.frames, 35):
            raise ValueError(f"music35 for {spec.key} must have shape {(spec.clip.frames, 35)}, got {feature.shape}")
        path = self.feature_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, feature, allow_pickle=False)
        temporary.replace(path)
        manifest["clips"][spec.key] = spec.metadata()
        self._write_manifest(manifest)
