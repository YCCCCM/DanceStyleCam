"""Memory-mapped access to per-sequence DCM-style++ NPY arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import ARRAY_SPECS, SCHEMA_VERSION


class SequenceStore:
    def __init__(self, root: str | Path, mmap_mode: str | None = "r") -> None:
        self.root = Path(root)
        self.mmap_mode = mmap_mode
        self._arrays: dict[tuple[str, str], np.ndarray] = {}
        self._manifest: dict[str, Any] | None = None

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def load_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Unsupported DCM-style++ schema: {manifest.get('schema_version')}")
            self._manifest = manifest
        return self._manifest

    def sequence_ids(self) -> list[str]:
        return sorted(self.load_manifest()["sequences"], key=int)

    def sequence_frames(self, sequence_id: str | int) -> int:
        return int(self.load_manifest()["sequences"][str(sequence_id)]["frames"])

    def array_path(self, sequence_id: str | int, feature: str) -> Path:
        if feature not in ARRAY_SPECS:
            raise KeyError(f"Unknown feature `{feature}`")
        return self.root / ARRAY_SPECS[feature].directory / f"{sequence_id}.npy"

    def load(self, sequence_id: str | int, feature: str) -> np.ndarray:
        cache_key = (str(sequence_id), feature)
        if cache_key in self._arrays:
            return self._arrays[cache_key]
        path = self.array_path(sequence_id, feature)
        array = np.load(path, mmap_mode=self.mmap_mode, allow_pickle=False)
        expected = ARRAY_SPECS[feature]
        if array.shape[1:] != expected.trailing_shape:
            raise ValueError(f"Unexpected shape for {path}: {array.shape}")
        if array.dtype.name != expected.dtype:
            raise ValueError(f"Unexpected dtype for {path}: {array.dtype}")
        self._arrays[cache_key] = array
        return array
