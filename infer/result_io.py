"""Portable on-disk contract shared by generation, metrics, and visualization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import yaml

from data.schema import ARRAY_SPECS


RESULT_SCHEMA_VERSION = 1
_SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class GenerationRun:
    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def camera_dir(self) -> Path:
        return self.root / "camera"

    @property
    def keyframe_dir(self) -> Path:
        return self.root / "keyframes"

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        run_name: str,
        config: Mapping[str, Any],
    ) -> "GenerationRun":
        if not _SAFE_SAMPLE_ID.fullmatch(run_name):
            raise ValueError(f"Unsafe run name: {run_name}")
        run = cls(Path(output_root) / run_name)
        run.camera_dir.mkdir(parents=True, exist_ok=False)
        run.keyframe_dir.mkdir(parents=True)
        with (run.root / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(config), handle, sort_keys=False)
        run._write_manifest({"schema_version": RESULT_SCHEMA_VERSION, "samples": {}})
        return run

    @classmethod
    def open(cls, root: str | Path) -> "GenerationRun":
        run = cls(Path(root))
        run.load_manifest()
        return run

    def load_manifest(self) -> dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported generation result schema: {manifest.get('schema_version')}")
        return manifest

    def save_camera(
        self,
        sample_id: str,
        camera: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        if not _SAFE_SAMPLE_ID.fullmatch(sample_id):
            raise ValueError(f"Unsafe sample id: {sample_id}")
        value = np.asarray(camera, dtype=np.float32)
        expected_shape = ARRAY_SPECS["camera20"].trailing_shape
        if value.ndim != 2 or value.shape[1:] != expected_shape:
            raise ValueError(f"Camera output must have shape [frames, 20], got {value.shape}")

        output_path = self.camera_dir / f"{sample_id}.npy"
        np.save(output_path, value, allow_pickle=False)

        manifest = self.load_manifest()
        sample = {
            "camera": output_path.relative_to(self.root).as_posix(),
            "frames": int(value.shape[0]),
            "dtype": value.dtype.name,
        }
        if metadata:
            sample["metadata"] = dict(metadata)
        manifest["samples"][sample_id] = sample
        self._write_manifest(manifest)
        return output_path

    def load_camera(self, sample_id: str, mmap_mode: str | None = "r") -> np.ndarray:
        manifest = self.load_manifest()
        relative_path = manifest["samples"][sample_id]["camera"]
        return np.load(self.root / relative_path, mmap_mode=mmap_mode, allow_pickle=False)

    def save_keyframes(self, sample_id: str, mask: np.ndarray) -> Path:
        if not _SAFE_SAMPLE_ID.fullmatch(sample_id):
            raise ValueError(f"Unsafe sample id: {sample_id}")
        value = np.asarray(mask, dtype=np.uint8)
        if value.ndim != 1:
            raise ValueError(f"Keyframe mask must have shape [frames], got {value.shape}")
        output_path = self.keyframe_dir / f"{sample_id}.npy"
        np.save(output_path, value, allow_pickle=False)

        manifest = self.load_manifest()
        sample = manifest["samples"].setdefault(sample_id, {})
        existing_frames = sample.get("frames")
        if existing_frames is not None and existing_frames != len(value):
            raise ValueError(f"Camera/keyframe frame mismatch for {sample_id}")
        sample["keyframes"] = output_path.relative_to(self.root).as_posix()
        sample["frames"] = int(len(value))
        self._write_manifest(manifest)
        return output_path

    def load_keyframes(self, sample_id: str, mmap_mode: str | None = "r") -> np.ndarray:
        manifest = self.load_manifest()
        relative_path = manifest["samples"][sample_id]["keyframes"]
        return np.load(self.root / relative_path, mmap_mode=mmap_mode, allow_pickle=False)

    def derived_dir(self, name: str) -> Path:
        if name not in {"metrics", "vis", "vmd"}:
            raise ValueError(f"Unknown derived output directory: {name}")
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        temporary.replace(self.manifest_path)
