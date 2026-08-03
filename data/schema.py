"""Stable numeric and style contracts for DCM-style++."""

from __future__ import annotations

from dataclasses import dataclass


SCHEMA_VERSION = 1
FPS = 30


@dataclass(frozen=True)
class ArraySpec:
    directory: str
    trailing_shape: tuple[int, ...]
    dtype: str


ARRAY_SPECS = {
    "motion180": ArraySpec("motion180", (180,), "float32"),
    "camera20": ArraySpec("camera20", (20,), "float32"),
    "music35": ArraySpec("music35", (35,), "float32"),
    "keyframe_mask": ArraySpec("keyframe_mask", (), "uint8"),
    "bone_mask60": ArraySpec("bone_mask60", (60,), "uint8"),
}

CAMERA20_FIELDS = (
    "distance",
    "position_x",
    "position_y",
    "position_z",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "fov",
    "eye_x",
    "eye_y",
    "eye_z",
    "axis_x_x",
    "axis_x_y",
    "axis_x_z",
    "axis_y_x",
    "axis_y_y",
    "axis_y_z",
    "axis_z_x",
    "axis_z_y",
    "axis_z_z",
)

# This is the only style-to-index order used by new checkpoints.
STYLE_NAMES_CANONICAL_V1 = (
    "Breaking",
    "Popping",
    "Locking",
    "Hiphop",
    "Urban",
    "Jazz",
    "Tai",
    "Uighur",
    "Hmong",
    "HanTang",
    "ShenYun",
    "Kun",
    "DunHuang",
    "Korean",
    "Choreography",
    "Chinese",
)

