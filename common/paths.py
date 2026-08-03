"""Portable project paths resolved from the repository root."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, require_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


@dataclass(frozen=True)
class DatasetPaths:
    """All external dataset paths used by preparation and loading."""

    raw_root: Path
    processed_root: Path
    style_file: Path
    segment_file: Path
    train_split: Path
    test_split: Path

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
        project_root: Path = PROJECT_ROOT,
    ) -> "DatasetPaths":
        values = require_mapping(config, "paths")
        required = (
            "raw_root",
            "processed_root",
            "style_file",
            "segment_file",
            "train_split",
            "test_split",
        )
        missing = [key for key in required if not isinstance(values.get(key), (str, Path))]
        if missing:
            raise ConfigError(f"Missing dataset path values: {', '.join(missing)}")
        return cls(**{key: resolve_project_path(values[key], project_root) for key in required})
