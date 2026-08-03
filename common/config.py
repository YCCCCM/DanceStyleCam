"""YAML loading shared by all `python -m ... --config` entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing or malformed."""


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML mapping, optionally inheriting another file via `base_config`."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Config file does not exist: {config_path}")

    seen = set() if _seen is None else _seen
    if config_path in seen:
        raise ConfigError(f"Recursive base_config reference: {config_path}")
    seen.add(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must contain a YAML mapping: {config_path}")

    base_config = data.pop("base_config", None)
    if base_config is None:
        return data

    base_path = Path(base_config)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    return _deep_merge(load_config(base_path, seen), data)


def require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a required mapping section with a focused error message."""

    section = config.get(key)
    if not isinstance(section, Mapping):
        raise ConfigError(f"Required config section `{key}` is missing or is not a mapping")
    return section

