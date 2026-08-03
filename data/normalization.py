"""Small normalization artifacts computed from the selected training split."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .splits import ClipRef
from .store import SequenceStore


@dataclass(frozen=True)
class MinMaxStats:
    minimum: np.ndarray
    maximum: np.ndarray

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "minimum.npy", self.minimum, allow_pickle=False)
        np.save(root / "maximum.npy", self.maximum, allow_pickle=False)

    @classmethod
    def load(cls, directory: str | Path) -> "MinMaxStats":
        root = Path(directory)
        return cls(
            minimum=np.load(root / "minimum.npy", allow_pickle=False),
            maximum=np.load(root / "maximum.npy", allow_pickle=False),
        )

    def normalize(self, value: np.ndarray) -> np.ndarray:
        scale = self.maximum - self.minimum
        scale = np.where(scale < 10 * np.finfo(np.float32).eps, 1.0, scale)
        output = ((value - self.minimum) / scale) * 2.0 - 1.0
        return np.clip(output, -1.0, 1.0).astype(np.float32, copy=False)

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        scale = self.maximum - self.minimum
        scale = np.where(scale < 10 * np.finfo(np.float32).eps, 1.0, scale)
        clipped = np.clip(value, -1.0, 1.0)
        return (((clipped + 1.0) * 0.5) * scale + self.minimum).astype(np.float32, copy=False)


NORMALIZER_FIELDS = {
    "pose": ("motion180", slice(0, 180)),
    "camera_distance": ("camera20", slice(0, 1)),
    "camera_position": ("camera20", slice(1, 4)),
    "camera_rotation": ("camera20", slice(4, 7)),
    "camera_fov": ("camera20", slice(7, 8)),
    "camera_eye": ("camera20", slice(8, 11)),
}


@dataclass(frozen=True)
class NormalizerBundle:
    fields: dict[str, MinMaxStats]

    def __getitem__(self, name: str) -> MinMaxStats:
        return self.fields[name]

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for name, stats in self.fields.items():
            stats.save(root / name)
        with (root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "fields": sorted(self.fields)}, handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, directory: str | Path) -> "NormalizerBundle":
        root = Path(directory)
        with (root / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest: dict[str, Any] = json.load(handle)
        if manifest.get("schema_version") != 1:
            raise ValueError(f"Unsupported normalizer schema: {manifest.get('schema_version')}")
        return cls({name: MinMaxStats.load(root / name) for name in manifest["fields"]})


def fit_normalizers(store: SequenceStore, clips: list[ClipRef]) -> NormalizerBundle:
    """Fit legacy min/max statistics by scanning only referenced train frames."""

    if not clips:
        raise ValueError("Cannot fit normalizers on an empty clip list")
    minima: dict[str, np.ndarray] = {}
    maxima: dict[str, np.ndarray] = {}
    for name, (_, field_slice) in NORMALIZER_FIELDS.items():
        width = field_slice.stop - field_slice.start
        # Legacy windows always contain prefix padding at the first anchor.
        minima[name] = np.zeros(width, dtype=np.float32)
        maxima[name] = np.zeros(width, dtype=np.float32)

    for clip in clips:
        for name, (feature, field_slice) in NORMALIZER_FIELDS.items():
            value = store.load(clip.sequence_id, feature)[clip.start : clip.end, field_slice]
            minima[name] = np.minimum(minima[name], np.min(value, axis=0))
            maxima[name] = np.maximum(maxima[name], np.max(value, axis=0))
    return NormalizerBundle(
        {name: MinMaxStats(minima[name], maxima[name]) for name in NORMALIZER_FIELDS}
    )

