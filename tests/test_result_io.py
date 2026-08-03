import numpy as np

from infer.result_io import GenerationRun
from tools.visualization.vmd import write_camera_vmd


def test_generation_result_keeps_camera_and_visualization_separate(tmp_path) -> None:
    run = GenerationRun.create(tmp_path, "smoke", {"experiment": {"name": "smoke"}})
    camera = np.zeros((12, 20), dtype=np.float32)
    run.save_camera("C_4_0", camera, {"style": "ShenYun"})
    run.save_keyframes("C_4_0", np.array([1] + [0] * 10 + [1], dtype=np.uint8))

    loaded = run.load_camera("C_4_0", mmap_mode="r")
    assert loaded.shape == (12, 20)
    assert run.load_keyframes("C_4_0").tolist() == [1] + [0] * 10 + [1]
    assert not (run.root / "vis").exists()
    assert run.derived_dir("vis") == run.root / "vis"


def test_vmd_export_has_camera_records(tmp_path) -> None:
    output = write_camera_vmd(np.zeros((3, 20), dtype=np.float32), tmp_path / "camera.vmd")
    value = output.read_bytes()
    camera_count = int.from_bytes(value[30 + 20 + 4 + 4 : 30 + 20 + 4 + 4 + 4], "little")
    assert value[:25] == b"Vocaloid Motion Data 0002"
    assert camera_count == 3
