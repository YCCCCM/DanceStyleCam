import json
import struct

import numpy as np

import infer.generate_custom as custom


def _fixed(value: str, size: int) -> bytes:
    encoded = value.encode("shift_jis")[:size]
    return encoded + bytes(size - len(encoded))


def _bone_record(name: str, frame: int, x: float) -> bytes:
    return (
        _fixed(name, 15)
        + struct.pack("<Ifffffff", frame, x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        + bytes(64)
    )


def _write_center_vmd(path) -> None:
    records = [_bone_record("\u30bb\u30f3\u30bf\u30fc", 0, 0.0), _bone_record("\u30bb\u30f3\u30bf\u30fc", 2, 2.0)]
    path.write_bytes(
        _fixed("Vocaloid Motion Data 0002", 30)
        + _fixed("test", 20)
        + struct.pack("<I", len(records))
        + b"".join(records)
        + struct.pack("<I", 0)
    )


def test_vmd_to_motion180_interpolates_and_uses_dcm_bone_order(tmp_path, monkeypatch) -> None:
    vmd_path = tmp_path / "dance.vmd"
    _write_center_vmd(vmd_path)
    monkeypatch.setattr(
        custom,
        "read_pmx_bone_positions",
        lambda _path: {bone: np.zeros(3, dtype=np.float32) for bone in custom.FK_BONES},
    )

    motion = custom.motion180_from_vmd(vmd_path, tmp_path / "model.pmx")

    assert motion.shape == (3, 180)
    assert motion.dtype == np.float32
    assert motion[:, 0].tolist() == [0.0, 1.0, 2.0]
    assert np.isfinite(motion).all()


def test_load_motion180_accepts_npy_and_global_transform_json(tmp_path) -> None:
    expected = np.arange(2 * 180, dtype=np.float32).reshape(2, 180)
    npy_path = tmp_path / "dance.npy"
    np.save(npy_path, expected, allow_pickle=False)
    json_path = tmp_path / "dance.json"
    transforms = np.zeros((2, 60, 16), dtype=np.float32)
    transforms[:, :, 12:15] = expected.reshape(2, 60, 3)
    json_path.write_text(
        json.dumps(
            {
                "BoneKeyFrameTransformRecord": [
                    {"FrameTime": index, "Transform": row.reshape(-1).tolist()}
                    for index, row in enumerate(transforms)
                ]
            }
        ),
        encoding="utf-8",
    )

    assert np.array_equal(custom.load_motion180(npy_path), expected)
    assert np.array_equal(custom.load_motion180(json_path), expected)
