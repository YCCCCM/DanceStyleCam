import json
from pathlib import Path

import numpy as np

from data.ckd_dataset import build_ckd_dataset
from data.cs_dataset import build_cs_dataset
from data.schema import ARRAY_SPECS, SCHEMA_VERSION


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _dataset_config(tmp_path: Path) -> dict:
    raw = tmp_path / "DCM_data"
    processed = tmp_path / "DCM-style++"
    frames = 90
    arrays = {
        "motion180": np.linspace(-2.0, 2.0, frames * 180, dtype=np.float32).reshape(frames, 180),
        "camera20": np.linspace(-1.0, 1.0, frames * 20, dtype=np.float32).reshape(frames, 20),
        "music35": np.zeros((frames, 35), dtype=np.float32),
        "keyframe_mask": np.zeros(frames, dtype=np.uint8),
        "bone_mask60": np.ones((frames, 60), dtype=np.uint8),
    }
    arrays["keyframe_mask"][[0, 20, 89]] = 1
    for name, value in arrays.items():
        directory = processed / ARRAY_SPECS[name].directory
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "0.npy", value, allow_pickle=False)
    _write_json(
        processed / "manifest.json",
        {"schema_version": SCHEMA_VERSION, "sequences": {"0": {"frames": frames}}},
    )
    _write_json(raw / "music_style_16cat.json", {"a0.wav": "ShenYun"})
    _write_json(raw / "split/train_pre.json", ["C_0"])
    _write_json(raw / "split/test_pre.json", ["C_0"])
    _write_json(raw / "split/long2short.json", {"0": []})
    return {
        "paths": {
            "raw_root": str(raw),
            "processed_root": str(processed),
            "style_file": str(raw / "music_style_16cat.json"),
            "segment_file": str(raw / "split/long2short.json"),
            "train_split": str(raw / "split/train_pre.json"),
            "test_split": str(raw / "split/test_pre.json"),
        },
        "dataset": {
            "history_len": 60,
            "inference_len": 60,
            "merge_adjacent_train": True,
            "style_vocabulary": "canonical_v1",
            "ckd_train_stride": 15,
            "ckd_test_stride": 60,
        },
    }


def test_ckd_windows_are_created_without_materialized_cache(tmp_path) -> None:
    dataset = build_ckd_dataset(_dataset_config(tmp_path), "train")
    sample = dataset[0]
    assert len(dataset) == 6
    assert sample["motion"].shape == (120, 180)
    assert sample["music"].shape == (120, 35)
    assert sample["padding_mask"][:60].sum() == 0
    assert sample["padding_mask"][60:].sum() == 60
    assert not list(tmp_path.rglob("*.pkl"))


def test_cs_windows_keep_visualization_independent(tmp_path) -> None:
    dataset = build_cs_dataset(_dataset_config(tmp_path), "train")
    sample = dataset[0]
    assert len(dataset) == 4
    assert sum(dataset.inserted_keyframe) == 1
    assert sample["camera"].shape == (120, 11)
    assert sample["bone_mask"].shape == (120, 60)
    assert sample["style"][0].argmax() == 10
    assert sample["pre_padding"] == 60
    assert sample["camera_inference_mask"].sum() == 20

