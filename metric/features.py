"""Paper benchmark features computed directly from DCM-style++ arrays.

The original project materialized JSON-derived feature caches before running
evaluation.  The released project keeps the same definitions but computes
them on demand from ``camera20``, ``motion180`` and ``bone_mask60`` arrays.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _camera_parts(camera20: np.ndarray) -> tuple[np.ndarray, ...]:
    camera = np.asarray(camera20, dtype=np.float32)
    if camera.ndim != 2 or camera.shape[1] != 20:
        raise ValueError(f"Expected camera20 with shape [frames, 20], got {camera.shape}")
    return (
        camera[:, 8:11],
        camera[:, 17:20],
        camera[:, 14:17],
        camera[:, 11:14],
        camera[:, 7],
    )


def _average_velocity(subject: np.ndarray, index: int, sliding_window: int = 2, frame_time: float = 1.0 / 30.0) -> float:
    total = np.zeros(subject.shape[-1] if subject.ndim > 1 else 1, dtype=np.float64)
    count = 0
    for offset in range(-sliding_window, sliding_window + 1):
        if index + offset - 1 < 0 or index + offset >= len(subject):
            continue
        total += subject[index + offset] - subject[index + offset - 1]
        count += 1
    return float(np.linalg.norm(total / max(count, 1) / frame_time))


def _average_acceleration(
    subject: np.ndarray,
    index: int,
    sliding_window: int = 2,
    frame_time: float = 1.0 / 30.0,
) -> float:
    total = np.zeros(subject.shape[-1] if subject.ndim > 1 else 1, dtype=np.float64)
    count = 0
    for offset in range(-sliding_window, sliding_window + 1):
        if index + offset - 1 < 0 or index + offset + 1 >= len(subject):
            continue
        v2 = (subject[index + offset + 1] - subject[index + offset]) / frame_time
        v1 = (subject[index + offset] - subject[index + offset - 1]) / frame_time
        total += (v2 - v1) / frame_time
        count += 1
    return float(np.linalg.norm(total / max(count, 1)))


def kinetic_feature(camera20: np.ndarray, sliding_window: int = 2) -> np.ndarray:
    """Return the legacy ten-dimensional kinetic feature for one 75-frame clip."""

    subjects = _camera_parts(camera20)
    if len(camera20) < 2:
        return np.zeros(10, dtype=np.float32)
    values: list[float] = []
    for subject in subjects:
        velocity_energy = np.mean(
            [_average_velocity(subject, index, sliding_window) ** 2 for index in range(1, len(subject))]
        )
        acceleration_energy = np.mean(
            [_average_acceleration(subject, index, sliding_window) for index in range(1, len(subject))]
        )
        values.extend((float(velocity_energy), float(acceleration_energy)))
    return np.asarray(values, dtype=np.float32)


def kinetic_features(camera20: np.ndarray, window: int = 75) -> np.ndarray:
    """Extract non-overlapping 2.5-second kinetic windows as ``[N, 10]``."""

    camera = np.asarray(camera20, dtype=np.float32)
    windows = [kinetic_feature(camera[start : start + window]) for start in range(0, len(camera) - window + 1, window)]
    return np.stack(windows) if windows else np.empty((0, 10), dtype=np.float32)


def shot_features(camera20: np.ndarray, motion180: np.ndarray, bone_mask60: np.ndarray) -> np.ndarray:
    """Return the legacy per-frame shot features as ``[frames - 1, 2]``.

    The two columns are body area inside the shot divided by the shot area and
    by the projected body area.  The implementation intentionally preserves
    the released script's treatment of joints behind the camera.
    """

    camera = np.asarray(camera20, dtype=np.float32)
    motion = np.asarray(motion180, dtype=np.float32)
    mask = np.asarray(bone_mask60, dtype=np.float32)
    if camera.ndim != 2 or camera.shape[1] != 20:
        raise ValueError(f"Expected camera20 with shape [frames, 20], got {camera.shape}")
    frame_count = min(len(camera), len(motion), len(mask))
    if frame_count < 2:
        return np.empty((0, 2), dtype=np.float32)
    camera = camera[:frame_count]
    pose = motion[:frame_count].reshape(frame_count, 60, 3)
    bone_mask = mask[:frame_count, :60]
    eye = camera[:, 8:11]
    axis_z = camera[:, 17:20]
    axis_y = camera[:, 14:17]
    axis_x = camera[:, 11:14]
    fov = camera[:, 7]

    pose_to_eye = pose - eye[:, None, :]
    pose_to_z = np.sum(pose_to_eye * axis_z[:, None, :], axis=-1)
    pose_to_y = np.sum(pose_to_eye * axis_y[:, None, :], axis=-1)
    pose_to_x = np.sum(pose_to_eye * axis_x[:, None, :], axis=-1)
    pose_to_y_z = pose_to_y / (pose_to_z + 1e-20)
    pose_to_x_z = pose_to_x / (pose_to_z + 1e-20)

    inside_area = np.zeros(frame_count, dtype=np.float64)
    body_area = np.zeros(frame_count, dtype=np.float64)
    for index in range(frame_count):
        visible = bone_mask[index] != 0
        if np.any(visible):
            inside_area[index] = (
                (pose_to_y_z[index, visible].max() - pose_to_y_z[index, visible].min())
                * (pose_to_x_z[index, visible].max() - pose_to_x_z[index, visible].min())
            )

        positive = pose_to_z[index] > 0
        if np.any(positive):
            xs = pose_to_x_z[index, positive]
            ys = pose_to_y_z[index, positive]
            body_area[index] = (ys.max() - ys.min()) * (xs.max() - xs.min())
        else:
            # Matches the old while-loop fallback when all joints are behind
            # the camera: the final joint is used as a degenerate box.
            body_area[index] = 0.0

    tan_fov = np.tan(fov * 0.5 / 180.0 * math.pi)
    inside_divide_shot = inside_area / (4.0 * tan_fov * tan_fov + 1e-20)
    inside_divide_body = inside_area / (body_area + 1e-20)
    return np.stack((inside_divide_shot[:-1], inside_divide_body[:-1]), axis=-1).astype(np.float32)


def style_features(camera20: np.ndarray) -> np.ndarray:
    """Return the 30-D style-consistency feature used by the released script."""

    camera = np.asarray(camera20, dtype=np.float32)
    eye = camera[:, 8:11]
    fov = camera[:, 7]
    velocity = np.diff(eye, axis=0) if len(eye) > 1 else np.zeros((0, 3), dtype=np.float32)
    eye_mean = np.mean(eye, axis=0)
    eye_std = np.std(eye, axis=0)
    fov_mean = np.mean(fov) if len(fov) else 0.0
    fov_std = np.std(fov) if len(fov) else 0.0
    vel_mean = np.mean(velocity, axis=0) if len(velocity) else np.zeros(3, dtype=np.float32)
    vel_std = np.std(velocity, axis=0) if len(velocity) else np.zeros(3, dtype=np.float32)
    base = np.concatenate((eye_mean, eye_std, [fov_mean, fov_std], vel_mean, vel_std)).astype(np.float32)
    eye_range = 2.0 * eye_std
    vel_max = np.abs(vel_mean) + 2.0 * vel_std
    acc_mean = vel_std
    acc_std = vel_std * 0.5
    motion_complexity = np.sum(vel_std)
    horizontal_movement = np.sqrt(vel_mean[0] ** 2 + vel_mean[2] ** 2)
    vertical_dominance = abs(vel_mean[1]) / (horizontal_movement + 1e-10)
    advanced = np.concatenate(
        (
            eye_range,
            vel_max,
            acc_mean,
            acc_std,
            [motion_complexity, vertical_dominance, fov_std, np.sum(eye_std)],
        )
    )
    return np.concatenate((base, advanced)).astype(np.float32)


def normalize_features(reference: np.ndarray, generated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize generated/reference features using reference statistics."""

    mean = np.mean(reference, axis=0)
    std = np.std(reference, axis=0)
    return (reference - mean) / (std + 1e-10), (generated - mean) / (std + 1e-10)


def average_pairwise_distance(features: np.ndarray, block_size: int = 2048) -> float:
    """Exact average pairwise Euclidean distance with bounded temporary memory."""

    values = np.asarray(features, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for start in range(0, len(values), block_size):
        left = values[start : start + block_size]
        for other_start in range(start, len(values), block_size):
            right = values[other_start : other_start + block_size]
            distance = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
            if other_start == start:
                tri = np.triu_indices(len(left), k=1)
                total += float(distance[tri].sum())
                pairs += len(tri[0])
            else:
                total += float(distance.sum())
                pairs += int(distance.size)
    return total / max(pairs, 1)


def fid(generated: np.ndarray, reference: np.ndarray) -> float | None:
    """Fréchet distance matching the original SciPy implementation."""

    from scipy import linalg

    generated = np.asarray(generated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if len(generated) < 2 or len(reference) < 2:
        return None
    mu_gen = generated.mean(axis=0)
    mu_ref = reference.mean(axis=0)
    sigma_gen = np.atleast_2d(np.cov(generated, rowvar=False))
    sigma_ref = np.atleast_2d(np.cov(reference, rowvar=False))
    diff = mu_gen - mu_ref
    eps = 1e-5
    covmean, _ = linalg.sqrtm(sigma_gen.dot(sigma_ref), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_gen.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_gen + offset).dot(sigma_ref + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma_gen) + np.trace(sigma_ref) - 2.0 * np.trace(covmean))


def stack_or_empty(values: Iterable[np.ndarray], width: int) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float32) for value in values if len(value)]
    return np.concatenate(arrays, axis=0) if arrays else np.empty((0, width), dtype=np.float32)
