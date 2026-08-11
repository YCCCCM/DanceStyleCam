import json

import numpy as np
import pytest

from infer.generate_test_controlled import (
    SpatioTemporalControl,
    apply_keyframe_overrides,
    load_spatio_temporal_controls,
    load_temporal_controls,
    style_features,
    validate_style,
)


def test_temporal_control_supports_single_json_and_binary_npy(tmp_path) -> None:
    frame_file = tmp_path / "frames.json"
    frame_file.write_text(json.dumps({"frames": [2, 5]}), encoding="utf-8")
    mask_file = tmp_path / "mask.npy"
    np.save(mask_file, np.asarray([1, 0, 0, 1, 0, 0], dtype=np.uint8), allow_pickle=False)

    frames = load_temporal_controls(frame_file, {"sample": 6})["sample"]
    mask = load_temporal_controls(mask_file, {"sample": 6})["sample"]

    assert frames.tolist() == [0, 0, 1, 0, 0, 1]
    assert mask.tolist() == [1, 0, 0, 1, 0, 0]


def test_temporal_control_mapping_can_leave_samples_on_model_default(tmp_path) -> None:
    control_file = tmp_path / "temporal.json"
    control_file.write_text(json.dumps({"clip_a": [1, 4]}), encoding="utf-8")
    controls = load_temporal_controls(control_file, {"clip_a": 6, "clip_b": 6})
    assert set(controls) == {"clip_a"}


def test_spatio_temporal_control_accepts_sparse_rows_and_joins_temporal_mask(tmp_path) -> None:
    camera_file = tmp_path / "camera.npy"
    rows = np.asarray(
        [
            [2, -7, 0, 10, 0, 0.1, 0, 0, 45],
            [5, -9, 1, 11, 0, 0.2, 0.3, 0, 50],
        ],
        dtype=np.float32,
    )
    np.save(camera_file, rows, allow_pickle=False)
    spatial = load_spatio_temporal_controls(camera_file, {"sample": 6})
    generated = {"sample": np.zeros(6, dtype=np.uint8)}
    combined = apply_keyframe_overrides(generated, {}, spatial)

    assert spatial["sample"].frames.tolist() == [2, 5]
    assert spatial["sample"].camera8.shape == (2, 8)
    assert combined["sample"].tolist() == [0, 0, 1, 0, 0, 1]


def test_spatio_temporal_control_sorts_json_frame_keys() -> None:
    control = SpatioTemporalControl(
        np.asarray([5, 2]),
        np.asarray([[5] * 8, [2] * 8], dtype=np.float32),
    )
    assert control.frames.tolist() == [2, 5]
    assert control.camera8[:, 0].tolist() == [2, 5]


def test_spatio_temporal_control_rejects_fractional_frames() -> None:
    with pytest.raises(ValueError, match="integers"):
        SpatioTemporalControl(
            np.asarray([1.5]),
            np.zeros((1, 8), dtype=np.float32),
        )


def test_style_control_uses_complete_checkpoint_one_hot() -> None:
    features = style_features("Choreography", "legacy_dsc_v1", 12)
    assert validate_style("Cherography", "legacy_dsc_v1") == "Choreography"
    assert features.shape == (12, 16)
    assert np.all(features.sum(axis=1) == 1)
    assert np.all(features.argmax(axis=1) == 10)
