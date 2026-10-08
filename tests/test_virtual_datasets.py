import json
from pathlib import Path

import numpy as np
import pytest

from data.ckd_dataset import build_ckd_dataset
from data.cs_dataset import build_cs_dataset
from data.music_store import MusicClipSpec, MusicFeatureStore
from data.schema import ARRAY_SPECS, SCHEMA_VERSION
from data.splits import ClipRef
from infer.pipeline import ckd_normalizers_for_inference


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_ckd_benchmark_normalization_matches_dsc_dataset_statistics(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    checkpoint = {
        "normalizers": {
            "pose": {
                "minimum": np.full(180, -100.0, dtype=np.float32),
                "maximum": np.full(180, 100.0, dtype=np.float32),
            }
        }
    }
    dataset = build_ckd_dataset(
        config, "test", normalizers=ckd_normalizers_for_inference(checkpoint, {})
    )
    sample = dataset[0]["motion"]
    np.testing.assert_allclose(sample[60], -np.ones(180), atol=1e-6)

    checkpoint_dataset = build_ckd_dataset(
        config,
        "test",
        normalizers=ckd_normalizers_for_inference(checkpoint, {"ckd_normalization": "checkpoint"}),
    )
    assert np.max(np.abs(checkpoint_dataset[0]["motion"][60])) < 0.021


def _dataset_config(tmp_path: Path) -> dict:
    raw = tmp_path / "DCM_data"
    processed = tmp_path / "DCM-style++"
    frames = 90
    arrays = {
        "motion180": np.linspace(-2.0, 2.0, frames * 180, dtype=np.float32).reshape(frames, 180),
        "camera20": np.linspace(-1.0, 1.0, frames * 20, dtype=np.float32).reshape(frames, 20),
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
    aligned_audio = raw / "amc_aligned_data" / "Audio" / "a0.wav"
    aligned_audio.parent.mkdir(parents=True, exist_ok=True)
    aligned_audio.touch()
    music_store = MusicFeatureStore(processed)
    music_store.save(
        MusicClipSpec(ClipRef("0", "0", 0, frames, ("C_0",)), 0, None, None),
        np.zeros((frames, 35), dtype=np.float32),
        music_store.load_manifest(),
    )
    _write_json(raw / "music_style_16cat.json", {"a0.wav": "ShenYun"})
    _write_json(raw / "split/train.json", ["C_0"])
    _write_json(raw / "split/test.json", ["C_0"])
    _write_json(raw / "split/long2short.json", {"0": []})
    return {
        "paths": {
            "raw_root": str(raw),
            "processed_root": str(processed),
            "style_file": str(raw / "music_style_16cat.json"),
            "segment_file": str(raw / "split/long2short.json"),
            "train_split": str(raw / "split/train.json"),
            "test_split": str(raw / "split/test.json"),
        },
        "dataset": {
            "history_len": 60,
            "inference_len": 60,
            "merge_adjacent_train": True,
            "style_vocabulary": "dsc",
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
    assert sample["style"][0].argmax() == 13
    assert sample["pre_padding"] == 60
    assert sample["camera_inference_mask"].sum() == 20


def _fragment_config(tmp_path: Path) -> dict:
    config = _dataset_config(tmp_path)
    paths = config["paths"]
    _write_json(Path(paths["train_split"]), ["C_0_0", "C_0_1"])
    _write_json(Path(paths["test_split"]), ["C_0_1"])
    _write_json(Path(paths["segment_file"]), {"0": [[0, 29], [30, 59], [60, 89]]})
    music = MusicFeatureStore(paths["processed_root"])
    for name, start, end, items, value in [
        ("0_0~1", 0, 60, ("C_0_0", "C_0_1"), 10.0),
        ("0_0", 0, 30, ("C_0_0",), 20.0),
        ("0_1", 30, 60, ("C_0_1",), 30.0),
        ("0_2", 60, 90, ("C_0_2",), 40.0),
    ]:
        clip = ClipRef(name, "0", start, end, items)
        music.save(MusicClipSpec(clip, start, end, None), np.full((end-start, 35), value, dtype=np.float32), music.load_manifest())
    return config


def test_cs_start_windows_match_standalone_padding_and_preserve_base(tmp_path: Path) -> None:
    config = _fragment_config(tmp_path)
    original = build_cs_dataset(config, "train")
    config["dataset"]["cs_start_window_frames"] = 60
    augmented = build_cs_dataset(config, "train")
    independent = build_cs_dataset(config, "test", normalizers=original.normalizers)
    assert augmented.start_window_count == 5
    assert len(augmented) == len(original) + 5
    assert independent.start_window_count == 0
    fields = ["camera", "camera_inference_mask", "motion", "music", "bone_mask", "style"]
    for index in range(len(original)):
        for field in fields:
            np.testing.assert_array_equal(original[index][field], augmented[index][field])
    for field in original.normalizers.fields:
        np.testing.assert_array_equal(original.normalizers[field].minimum, augmented.normalizers[field].minimum)
        np.testing.assert_array_equal(original.normalizers[field].maximum, augmented.normalizers[field].maximum)
    starts = [index for index, window in enumerate(augmented.windows)
              if window.clip_index in augmented.prefix_frames_by_clip
              and augmented.context.clips[window.clip_index].name == "0_1"]
    assert len(starts) == len(independent)
    for index, reference in zip(starts, range(len(independent))):
        for field in fields:
            np.testing.assert_array_equal(augmented[index][field], independent[reference][field])
    assert augmented[starts[0]]["pre_padding"] == 60
    assert np.count_nonzero(augmented[starts[0]]["music"][:60]) == 0
    assert np.all(augmented[starts[0]]["music"][60:90] == 30.0)


def test_cs_start_windows_use_only_training_fragments_and_prepare_their_music(tmp_path: Path) -> None:
    from data.music_prepare import music_specs
    config = _fragment_config(tmp_path)
    _write_json(Path(config["paths"]["test_split"]), ["C_0_2"])
    config["dataset"]["cs_start_window_frames"] = 20
    dataset = build_cs_dataset(config, "train")
    assert set(clip.name for clip in dataset.context.clips) == {"0_0~1", "0_0", "0_1"}
    assert dataset.start_window_count == 2
    assert all(window.keyframe < 20 for window in dataset.windows if window.clip_index in dataset.prefix_frames_by_clip)
    assert {spec.key for spec, _ in music_specs(config, "train")} == {"0_0~1", "0_0", "0_1"}
    assert {spec.key for spec, _ in music_specs(config, "test")} == {"0_2"}


def test_cs_start_windows_do_not_duplicate_existing_standalone_clips(tmp_path: Path) -> None:
    config = _fragment_config(tmp_path)
    config["dataset"].update(merge_adjacent_train=False, cs_start_window_frames=60)
    dataset = build_cs_dataset(config, "train")
    assert dataset.start_window_count == 0
    assert len(dataset.context.clips) == 2


@pytest.mark.parametrize("sample_id, expected_match", [("0", True), ("unexpected", False)])
def test_benchmark_checks_configured_test_membership(tmp_path: Path, sample_id: str, expected_match: bool) -> None:
    from infer.result_io import GenerationRun
    from metric.evaluate import evaluate_run

    data_config = _dataset_config(tmp_path)
    config_path = tmp_path / "data.yaml"
    _write_json(config_path, data_config)
    run = GenerationRun.create(tmp_path / "generation", "test", {})
    run.save_camera(sample_id, np.zeros((90, 20), dtype=np.float32))
    config = {
        "data": {"config": str(config_path)},
        "evaluation": {"sample_split": "test", "benchmark": False, "style_consistency": False},
    }
    if expected_match:
        assert set(evaluate_run(config, run.root)["samples"]) == {"0"}
    else:
        with pytest.raises(ValueError, match="configured test split"):
            evaluate_run(config, run.root)
