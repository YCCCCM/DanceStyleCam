"""Legacy-compatible camera interpolation and geometric feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraAlignment:
    keyframes: list[dict[str, Any]]
    aligned_frame_limit: int
    output_frames: int


class VmdBezier:
    def __init__(self, curve: list[int] | tuple[int, ...]) -> None:
        if len(curve) < 4:
            raise ValueError("VMD camera curve must contain at least four values")
        self.cp1 = np.array((curve[0], curve[2]), dtype=np.float64) / 127.0
        self.cp2 = np.array((curve[1], curve[3]), dtype=np.float64) / 127.0

    def _eval_x(self, value: float) -> float:
        inverse = 1.0 - value
        return (
            value**3
            + 3.0 * value**2 * inverse * self.cp2[0]
            + 3.0 * value * inverse**2 * self.cp1[0]
        )

    def eval_y_from_time(self, time: float) -> float:
        start, stop = 0.0, 1.0
        value = 0.5
        while abs(time - self._eval_x(value)) > 1e-5:
            if time < self._eval_x(value):
                stop = value
            else:
                start = value
            value = (start + stop) * 0.5
        inverse = 1.0 - value
        return float(
            value**3
            + 3.0 * value**2 * inverse * self.cp2[1]
            + 3.0 * value * inverse**2 * self.cp1[1]
        )


def align_camera_keyframes(
    camera_data: dict[str, Any],
    audio_frames: int,
    motion_frames: int,
) -> CameraAlignment:
    limit = min(int(audio_frames), int(motion_frames))
    keyframes = sorted(camera_data["CameraKeyFrameRecord"], key=lambda item: item["FrameTime"])
    retained = [item for item in keyframes if int(item["FrameTime"]) < limit]
    if not retained:
        raise ValueError("No camera keyframe remains after audio/motion alignment")

    aligned_frame_limit = int(retained[-1]["FrameTime"]) + 1
    # The released pipeline iterated range(last_keyframe_time), excluding the
    # final camera keyframe. Preserve that behavior for checkpoint parity.
    output_frames = int(retained[-1]["FrameTime"])
    if output_frames <= 0:
        raise ValueError("Aligned camera sequence has no interpolated frames")
    return CameraAlignment(retained, aligned_frame_limit, output_frames)


def _vector3(value: dict[str, float]) -> np.ndarray:
    return np.array((value["x"], value["y"], value["z"]), dtype=np.float64)


def interpolate_camera(alignment: CameraAlignment) -> tuple[np.ndarray, np.ndarray]:
    keyframes = alignment.keyframes
    frame_count = alignment.output_frames
    distance = np.empty((frame_count, 1), dtype=np.float32)
    position = np.empty((frame_count, 3), dtype=np.float32)
    rotation = np.empty((frame_count, 3), dtype=np.float32)
    fov = np.empty((frame_count, 1), dtype=np.float32)
    keyframe_mask = np.zeros(frame_count, dtype=np.uint8)

    for item in keyframes:
        frame = int(item["FrameTime"])
        if frame < frame_count:
            keyframe_mask[frame] = 1

    left_index = 0
    right_index = 1
    for frame in range(frame_count):
        while right_index < len(keyframes) and frame > int(keyframes[right_index]["FrameTime"]):
            left_index = right_index
            right_index += 1

        left = keyframes[left_index]
        if frame <= int(left["FrameTime"]) or right_index >= len(keyframes):
            weight = 0.0
            right = left
        else:
            right = keyframes[right_index]
            right_frame = int(right["FrameTime"])
            if frame == right_frame:
                weight = 1.0
            else:
                linear_time = (frame - int(left["FrameTime"])) / (right_frame - int(left["FrameTime"]))
                weight = VmdBezier(left["Curve"]).eval_y_from_time(linear_time)

        distance[frame, 0] = np.float32((1.0 - weight) * float(left["Distance"]) + weight * float(right["Distance"]))
        position[frame] = ((1.0 - weight) * _vector3(left["Position"]) + weight * _vector3(right["Position"])).astype(
            np.float32
        )
        rotation[frame] = ((1.0 - weight) * _vector3(left["Rotation"]) + weight * _vector3(right["Rotation"])).astype(
            np.float32
        )
        fov[frame, 0] = np.float32(
            (1.0 - weight) * float(left["ViewAngle"]) + weight * float(right["ViewAngle"])
        )

        if weight == 1.0:
            left_index = right_index
            right_index += 1

    camera_axes = camera_centric_axes(distance, position, rotation)
    camera20 = np.concatenate((distance, position, rotation, fov, *camera_axes), axis=1).astype(np.float32)
    return camera20, keyframe_mask


def _rotation_matrices(rotation: np.ndarray) -> np.ndarray:
    x, y, z = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    matrices = np.empty((len(rotation), 3, 3), dtype=np.float64)
    for index in range(len(rotation)):
        rotate_y = np.array(((cy[index], 0.0, sy[index]), (0.0, 1.0, 0.0), (-sy[index], 0.0, cy[index])))
        rotate_negative_z = np.array(
            ((cz[index], sz[index], 0.0), (-sz[index], cz[index], 0.0), (0.0, 0.0, 1.0))
        )
        rotate_x = np.array(((1.0, 0.0, 0.0), (0.0, cx[index], -sx[index]), (0.0, sx[index], cx[index])))
        matrices[index] = rotate_y @ rotate_negative_z @ rotate_x
    return matrices


def camera_centric_axes(
    distance: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrices = _rotation_matrices(rotation.astype(np.float64, copy=False))
    translation = np.einsum(
        "tij,tj->ti",
        matrices,
        np.column_stack((np.zeros(len(distance)), np.zeros(len(distance)), np.abs(distance[:, 0]))),
    )
    eye = translation + position.astype(np.float64) * np.array((1.0, 1.0, -1.0))
    axis_x = matrices[:, :, 0]
    axis_y = matrices[:, :, 1]
    axis_z = -matrices[:, :, 2]
    return tuple(value.astype(np.float32) for value in (eye, axis_x, axis_y, axis_z))


def global_transforms_to_keypoints(records: list[dict[str, Any]], frame_count: int) -> np.ndarray:
    transforms = np.asarray([record["Transform"] for record in records[:frame_count]], dtype=np.float32)
    if transforms.ndim != 2 or transforms.shape[1] % 16 != 0:
        raise ValueError(f"Unexpected global transform shape: {transforms.shape}")
    keypoints = transforms.reshape(len(transforms), -1, 16)[:, :, 12:15]
    if keypoints.shape[1] != 60:
        raise ValueError(f"DanceStyleCam expects 60 joints, found {keypoints.shape[1]}")
    return keypoints.reshape(len(transforms), 180)


def detect_bone_mask(camera20: np.ndarray, motion180: np.ndarray) -> np.ndarray:
    camera_eye = camera20[:, 8:11]
    camera_x = camera20[:, 11:14]
    camera_y = camera20[:, 14:17]
    camera_z = camera20[:, 17:20]
    camera_fov = camera20[:, 7]
    keypoints = motion180.reshape(len(motion180), 60, 3).transpose(1, 0, 2)

    key_to_eye = keypoints - camera_eye
    projected_yz = key_to_eye - camera_x * np.sum(key_to_eye * camera_x, axis=-1, keepdims=True)
    projected_xz = key_to_eye - camera_y * np.sum(key_to_eye * camera_y, axis=-1, keepdims=True)
    cosine_yz = np.sum(projected_yz * camera_z, axis=-1)
    cosine_xz = np.sum(projected_xz * camera_z, axis=-1)
    cosine_fov = np.cos(camera_fov * 0.5 / 180.0 * math.pi)

    visible_y = cosine_yz >= cosine_fov * np.linalg.norm(projected_yz, axis=-1)
    visible_x = cosine_xz >= cosine_fov * np.linalg.norm(projected_xz, axis=-1)
    return np.logical_and(visible_y, visible_x).T.astype(np.uint8)

