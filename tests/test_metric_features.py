from __future__ import annotations

import numpy as np

from metric.evaluate import _dancer_missing_rate
from metric.features import average_pairwise_distance, kinetic_features, shot_features, style_features


def _camera(frames: int) -> np.ndarray:
    camera = np.zeros((frames, 20), dtype=np.float32)
    camera[:, 7] = 60.0
    camera[:, 8:11] = np.array((0.0, 0.0, 0.0), dtype=np.float32)
    camera[:, 11:14] = np.array((1.0, 0.0, 0.0), dtype=np.float32)
    camera[:, 14:17] = np.array((0.0, 1.0, 0.0), dtype=np.float32)
    camera[:, 17:20] = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    return camera


def test_paper_features_are_array_native() -> None:
    camera = _camera(150)
    motion = np.zeros((150, 180), dtype=np.float32)
    motion[:, 2::3] = 5.0
    mask = np.ones((150, 60), dtype=np.uint8)

    kinetic = kinetic_features(camera)
    shot = shot_features(camera, motion, mask)
    style = style_features(camera)

    assert kinetic.shape == (2, 10)
    assert shot.shape == (149, 2)
    assert style.shape == (30,)
    assert np.isfinite(kinetic).all()
    assert np.isfinite(shot).all()
    assert np.isfinite(style).all()


def test_pairwise_distance_uses_unique_pairs() -> None:
    values = np.array([[0.0], [3.0], [4.0]], dtype=np.float32)
    assert average_pairwise_distance(values, block_size=2) == 8.0 / 3.0


def test_dancer_missing_rate_is_weighted_by_frame_like_original_evaluator() -> None:
    short = np.ones((2, 3), dtype=np.uint8)
    long = np.ones((8, 3), dtype=np.uint8)
    short[0] = 0

    assert _dancer_missing_rate([short, long]) == 0.1
    assert _dancer_missing_rate([]) is None
