"""Generate a camera from user-provided MMD dance motion and music."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, BinaryIO

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config, require_mapping
from common.paths import resolve_project_path
from data.audio_features import audio_frame_count, extract_music35
from data.camera_geometry import global_transforms_to_keypoints
from data.ckd_dataset import CKDWindow
from data.cs_dataset import CSWindow, _keyframe_positions
from data.dataset_common import padded_window
from data.normalization import NormalizerBundle
from data.splits import ClipRef
from infer.generate_test_controlled import (
    apply_keyframe_overrides,
    infer_cameras_controlled,
    load_spatio_temporal_controls,
    load_temporal_controls,
    style_features,
    validate_style,
)
from infer.pipeline import (
    checkpoint_uses_ema,
    infer_keyframes,
    resolve_device,
    seed_everything,
)
from infer.result_io import GenerationRun
from models.checkpoint import load_checkpoint_file, load_model_weights, normalizers_from_checkpoint
from models.ckd import build_ckd_model
from models.cs import build_cs_model


FPS = 30
IDENTITY_QUATERNION = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

# This order is the exact order used by DCM GlobalTransform and motion180.
DCM_BONES = (
    "\u4e0b\u534a\u8eab",
    "\u4e0a\u534a\u8eab",
    "\u4e0a\u534a\u8eab2",
    "\u9996",
    "\u982d",
    "\u4e21\u76ee",
    "\u5de6\u76ee",
    "\u53f3\u76ee",
    "\u5de6\u8db3",
    "\u5de6\u3072\u3056",
    "\u5de6\u8db3\u9996",
    "\u5de6\u3064\u307e\u5148",
    "\u53f3\u8db3",
    "\u53f3\u3072\u3056",
    "\u53f3\u8db3\u9996",
    "\u53f3\u3064\u307e\u5148",
    "\u5de6\u80a9P",
    "\u5de6\u80a9",
    "\u5de6\u8155",
    "\u5de6\u3072\u3058",
    "\u5de6\u624b\u9996",
    "\u5de6\u8155\u6369",
    "\u5de6\u624b\u6369",
    "\u5de6\u89aa\u6307\uff10",
    "\u5de6\u89aa\u6307\uff11",
    "\u5de6\u89aa\u6307\uff12",
    "\u5de6\u4eba\u6307\uff11",
    "\u5de6\u4eba\u6307\uff12",
    "\u5de6\u4eba\u6307\uff13",
    "\u5de6\u4e2d\u6307\uff11",
    "\u5de6\u4e2d\u6307\uff12",
    "\u5de6\u4e2d\u6307\uff13",
    "\u5de6\u85ac\u6307\uff11",
    "\u5de6\u85ac\u6307\uff12",
    "\u5de6\u85ac\u6307\uff13",
    "\u5de6\u5c0f\u6307\uff11",
    "\u5de6\u5c0f\u6307\uff12",
    "\u5de6\u5c0f\u6307\uff13",
    "\u53f3\u80a9P",
    "\u53f3\u80a9",
    "\u53f3\u8155",
    "\u53f3\u3072\u3058",
    "\u53f3\u624b\u9996",
    "\u53f3\u8155\u6369",
    "\u53f3\u624b\u6369",
    "\u53f3\u89aa\u6307\uff10",
    "\u53f3\u89aa\u6307\uff11",
    "\u53f3\u89aa\u6307\uff12",
    "\u53f3\u4eba\u6307\uff11",
    "\u53f3\u4eba\u6307\uff12",
    "\u53f3\u4eba\u6307\uff13",
    "\u53f3\u4e2d\u6307\uff11",
    "\u53f3\u4e2d\u6307\uff12",
    "\u53f3\u4e2d\u6307\uff13",
    "\u53f3\u85ac\u6307\uff11",
    "\u53f3\u85ac\u6307\uff12",
    "\u53f3\u85ac\u6307\uff13",
    "\u53f3\u5c0f\u6307\uff11",
    "\u53f3\u5c0f\u6307\uff12",
    "\u53f3\u5c0f\u6307\uff13",
)

PARENT: dict[str, str | None] = {
    "\u5168\u3066\u306e\u89aa": None,
    "\u30bb\u30f3\u30bf\u30fc": "\u5168\u3066\u306e\u89aa",
    "\u5de6\u8db3\uff29\uff2b": "\u5168\u3066\u306e\u89aa",
    "\u53f3\u8db3\uff29\uff2b": "\u5168\u3066\u306e\u89aa",
    "\u5de6\u3064\u307e\u5148\uff29\uff2b": "\u5de6\u8db3\uff29\uff2b",
    "\u53f3\u3064\u307e\u5148\uff29\uff2b": "\u53f3\u8db3\uff29\uff2b",
    "\u30b0\u30eb\u30fc\u30d6": "\u30bb\u30f3\u30bf\u30fc",
    "\u8170": "\u30b0\u30eb\u30fc\u30d6",
    "\u4e0b\u534a\u8eab": "\u8170",
    "\u4e0a\u534a\u8eab": "\u8170",
    "\u4e0a\u534a\u8eab2": "\u4e0a\u534a\u8eab",
    "\u9996": "\u4e0a\u534a\u8eab2",
    "\u982d": "\u9996",
    "\u4e21\u76ee": "\u982d",
    "\u5de6\u76ee": "\u982d",
    "\u53f3\u76ee": "\u982d",
    "\u5de6\u8db3": "\u4e0b\u534a\u8eab",
    "\u5de6\u3072\u3056": "\u5de6\u8db3",
    "\u5de6\u8db3\u9996": "\u5de6\u3072\u3056",
    "\u5de6\u3064\u307e\u5148": "\u5de6\u8db3\u9996",
    "\u53f3\u8db3": "\u4e0b\u534a\u8eab",
    "\u53f3\u3072\u3056": "\u53f3\u8db3",
    "\u53f3\u8db3\u9996": "\u53f3\u3072\u3056",
    "\u53f3\u3064\u307e\u5148": "\u53f3\u8db3\u9996",
    "\u5de6\u80a9P": "\u4e0a\u534a\u8eab2",
    "\u5de6\u80a9": "\u5de6\u80a9P",
    "\u5de6\u8155": "\u5de6\u80a9",
    "\u5de6\u8155\u6369": "\u5de6\u8155",
    "\u5de6\u3072\u3058": "\u5de6\u8155\u6369",
    "\u5de6\u624b\u6369": "\u5de6\u3072\u3058",
    "\u5de6\u624b\u9996": "\u5de6\u624b\u6369",
    "\u53f3\u80a9P": "\u4e0a\u534a\u8eab2",
    "\u53f3\u80a9": "\u53f3\u80a9P",
    "\u53f3\u8155": "\u53f3\u80a9",
    "\u53f3\u8155\u6369": "\u53f3\u8155",
    "\u53f3\u3072\u3058": "\u53f3\u8155\u6369",
    "\u53f3\u624b\u6369": "\u53f3\u3072\u3058",
    "\u53f3\u624b\u9996": "\u53f3\u624b\u6369",
}

for side in ("\u5de6", "\u53f3"):
    wrist = f"{side}\u624b\u9996"
    PARENT.update(
        {
            f"{side}\u89aa\u6307\uff10": wrist,
            f"{side}\u89aa\u6307\uff11": f"{side}\u89aa\u6307\uff10",
            f"{side}\u89aa\u6307\uff12": f"{side}\u89aa\u6307\uff11",
        }
    )
    for finger in ("\u4eba\u6307", "\u4e2d\u6307", "\u85ac\u6307", "\u5c0f\u6307"):
        PARENT[f"{side}{finger}\uff11"] = wrist
        PARENT[f"{side}{finger}\uff12"] = f"{side}{finger}\uff11"
        PARENT[f"{side}{finger}\uff13"] = f"{side}{finger}\uff12"

FK_BONES = (
    "\u5168\u3066\u306e\u89aa",
    "\u30bb\u30f3\u30bf\u30fc",
    "\u30b0\u30eb\u30fc\u30d6",
    "\u8170",
    "\u4e0b\u534a\u8eab",
    "\u4e0a\u534a\u8eab",
    "\u4e0a\u534a\u8eab2",
    "\u9996",
    "\u982d",
    "\u4e21\u76ee",
    "\u5de6\u76ee",
    "\u53f3\u76ee",
    "\u5de6\u8db3",
    "\u5de6\u3072\u3056",
    "\u5de6\u8db3\u9996",
    "\u5de6\u3064\u307e\u5148",
    "\u53f3\u8db3",
    "\u53f3\u3072\u3056",
    "\u53f3\u8db3\u9996",
    "\u53f3\u3064\u307e\u5148",
    "\u5de6\u80a9P",
    "\u5de6\u80a9",
    "\u5de6\u8155",
    "\u5de6\u8155\u6369",
    "\u5de6\u3072\u3058",
    "\u5de6\u624b\u6369",
    "\u5de6\u624b\u9996",
    *DCM_BONES[23:38],
    "\u53f3\u80a9P",
    "\u53f3\u80a9",
    "\u53f3\u8155",
    "\u53f3\u8155\u6369",
    "\u53f3\u3072\u3058",
    "\u53f3\u624b\u6369",
    "\u53f3\u624b\u9996",
    *DCM_BONES[45:60],
)

IK_BONES = (
    "\u5de6\u8db3\uff29\uff2b",
    "\u53f3\u8db3\uff29\uff2b",
    "\u5de6\u3064\u307e\u5148\uff29\uff2b",
    "\u53f3\u3064\u307e\uff29\uff2b",
)

CORE_PMX_BONES = (
    "\u30bb\u30f3\u30bf\u30fc",
    "\u4e0b\u534a\u8eab",
    "\u4e0a\u534a\u8eab",
    "\u9996",
    "\u982d",
    "\u5de6\u8db3",
    "\u5de6\u3072\u3056",
    "\u5de6\u8db3\u9996",
    "\u53f3\u8db3",
    "\u53f3\u3072\u3056",
    "\u53f3\u8db3\u9996",
    "\u5de6\u80a9",
    "\u5de6\u8155",
    "\u5de6\u3072\u3058",
    "\u5de6\u624b\u9996",
    "\u53f3\u80a9",
    "\u53f3\u8155",
    "\u53f3\u3072\u3058",
    "\u53f3\u624b\u9996",
)

BONE_ALIASES = {
    "\u4e0a\u534a\u8eab\uff12": "\u4e0a\u534a\u8eab2",
    "\u5de6\u80a9\uff30": "\u5de6\u80a9P",
    "\u53f3\u80a9\uff30": "\u53f3\u80a9P",
}


@dataclass(frozen=True)
class VmdMotion:
    model_name: str
    max_frame: int
    bones: dict[str, list[tuple[int, np.ndarray, np.ndarray]]]


def _decode_fixed(value: bytes) -> str:
    decoded = value.split(b"\0", 1)[0].decode("shift_jis", errors="ignore")
    return BONE_ALIASES.get(decoded, decoded)


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    normalized = quaternion / np.maximum(norm, 1e-8)
    return np.where(normalized[..., 3:4] < 0.0, -normalized, normalized).astype(np.float32)


def parse_motion_vmd(path: str | Path) -> VmdMotion:
    source = Path(path)
    data = source.read_bytes()
    if len(data) < 44:
        raise ValueError(f"VMD file is too small: {source}")
    header = data[:30].split(b"\0", 1)[0].decode("ascii", errors="ignore")
    if header.startswith("Vocaloid Motion Data 0002"):
        model_bytes = 20
    elif header.startswith("Vocaloid Motion Data file"):
        model_bytes = 10
    else:
        raise ValueError(f"Unsupported VMD header in {source}: {header!r}")
    model_name = _decode_fixed(data[30 : 30 + model_bytes])
    offset = 30 + model_bytes
    bone_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    bones: defaultdict[str, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    max_frame = 0
    known = set(FK_BONES)
    for _ in range(bone_count):
        if offset + 111 > len(data):
            raise ValueError(f"Unexpected EOF in VMD bone records: {source}")
        name = _decode_fixed(data[offset : offset + 15])
        frame, px, py, pz, qx, qy, qz, qw = struct.unpack_from("<Ifffffff", data, offset + 15)
        max_frame = max(max_frame, int(frame))
        if name in known:
            bones[name].append(
                (
                    int(frame),
                    np.asarray([px, py, pz], dtype=np.float32),
                    _normalize_quaternion(np.asarray([qx, qy, qz, qw], dtype=np.float32)),
                )
            )
        offset += 111
    if not bones:
        raise ValueError(f"VMD contains no recognized DCM/MMD bones: {source}")
    for records in bones.values():
        records.sort(key=lambda item: item[0])
    return VmdMotion(model_name=model_name, max_frame=max_frame, bones=dict(bones))


def _slerp(q0: np.ndarray, q1: np.ndarray, weights: np.ndarray) -> np.ndarray:
    left = _normalize_quaternion(q0)
    right = _normalize_quaternion(q1)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    weights = np.asarray(weights, dtype=np.float64)
    if dot > 0.9995:
        output = left[None, :] + weights[:, None] * (right - left)[None, :]
        return _normalize_quaternion(output)
    theta = math.acos(dot)
    denominator = math.sin(theta)
    scale_left = np.sin((1.0 - weights) * theta) / denominator
    scale_right = np.sin(weights * theta) / denominator
    return _normalize_quaternion(scale_left[:, None] * left + scale_right[:, None] * right)


def dense_bone_motion(
    records: list[tuple[int, np.ndarray, np.ndarray]],
    frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.zeros((frames, 3), dtype=np.float32)
    rotation = np.repeat(IDENTITY_QUATERNION[None, :], frames, axis=0)
    if not records:
        return position, rotation
    frame_indices = np.asarray([np.clip(item[0], 0, frames - 1) for item in records], dtype=np.int64)
    positions = np.stack([item[1] for item in records]).astype(np.float32)
    rotations = np.stack([item[2] for item in records]).astype(np.float32)
    frame_indices, unique = np.unique(frame_indices, return_index=True)
    positions = positions[unique]
    rotations = rotations[unique]
    if len(frame_indices) == 1:
        position[:] = positions[0]
        rotation[:] = rotations[0]
        return position, rotation
    axis = np.arange(frames, dtype=np.float32)
    for dimension in range(3):
        position[:, dimension] = np.interp(axis, frame_indices, positions[:, dimension])
    rotation[: frame_indices[0]] = rotations[0]
    rotation[frame_indices[-1] + 1 :] = rotations[-1]
    for index in range(len(frame_indices) - 1):
        start = int(frame_indices[index])
        end = int(frame_indices[index + 1])
        weights = np.linspace(0.0, 1.0, end - start + 1, dtype=np.float64)
        rotation[start : end + 1] = _slerp(rotations[index], rotations[index + 1], weights)
    return position, rotation


def _read_index(handle: BinaryIO, size: int, signed: bool = True) -> int:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError("Unexpected EOF while reading PMX index")
    return int.from_bytes(data, "little", signed=signed)


def _read_text(handle: BinaryIO, encoding: str) -> str:
    data = handle.read(4)
    if len(data) != 4:
        raise EOFError("Unexpected EOF while reading PMX text length")
    length = struct.unpack("<i", data)[0]
    raw = handle.read(length)
    if len(raw) != length:
        raise EOFError("Unexpected EOF while reading PMX text")
    return raw.decode(encoding, errors="ignore")


def _skip_floats(handle: BinaryIO, count: int) -> None:
    handle.seek(4 * count, 1)


def read_pmx_bone_positions(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    with source.open("rb") as handle:
        if handle.read(4) != b"PMX ":
            raise ValueError(f"Not a PMX file: {source}")
        handle.seek(4, 1)
        header_size = struct.unpack("<B", handle.read(1))[0]
        header = handle.read(header_size)
        if len(header) != header_size or header_size < 8:
            raise ValueError(f"Invalid PMX header: {source}")
        encoding = "utf-16-le" if header[0] == 0 else "utf-8"
        extra_uv_count = header[1]
        vertex_index_size = header[2]
        texture_index_size = header[3]
        bone_index_size = header[5]
        for _ in range(4):
            _read_text(handle, encoding)

        vertex_count = struct.unpack("<i", handle.read(4))[0]
        for _ in range(vertex_count):
            _skip_floats(handle, 3 + 3 + 2 + extra_uv_count * 4)
            weight_type = struct.unpack("<B", handle.read(1))[0]
            if weight_type == 0:
                handle.seek(bone_index_size, 1)
            elif weight_type == 1:
                handle.seek(bone_index_size * 2 + 4, 1)
            elif weight_type in {2, 4}:
                handle.seek(bone_index_size * 4 + 16, 1)
            elif weight_type == 3:
                handle.seek(bone_index_size * 2 + 4 + 36, 1)
            else:
                raise ValueError(f"Unsupported PMX weight type {weight_type}")
            handle.seek(4, 1)

        face_index_count = struct.unpack("<i", handle.read(4))[0]
        handle.seek(face_index_count * vertex_index_size, 1)
        texture_count = struct.unpack("<i", handle.read(4))[0]
        for _ in range(texture_count):
            _read_text(handle, encoding)
        material_count = struct.unpack("<i", handle.read(4))[0]
        for _ in range(material_count):
            _read_text(handle, encoding)
            _read_text(handle, encoding)
            _skip_floats(handle, 4 + 3 + 1 + 3)
            handle.seek(1, 1)
            _skip_floats(handle, 4 + 1)
            handle.seek(texture_index_size * 2 + 1, 1)
            toon_flag = struct.unpack("<B", handle.read(1))[0]
            handle.seek(texture_index_size if toon_flag == 0 else 1, 1)
            _read_text(handle, encoding)
            handle.seek(4, 1)

        bone_count = struct.unpack("<i", handle.read(4))[0]
        bones: dict[str, np.ndarray] = {}
        for _ in range(bone_count):
            japanese_name = _read_text(handle, encoding)
            _read_text(handle, encoding)  # English name
            name = BONE_ALIASES.get(japanese_name, japanese_name)
            position = np.asarray(struct.unpack("<fff", handle.read(12)), dtype=np.float32)
            _read_index(handle, bone_index_size)
            handle.seek(4, 1)
            flags = struct.unpack("<H", handle.read(2))[0]
            if flags & 0x0001:
                handle.seek(bone_index_size, 1)
            else:
                _skip_floats(handle, 3)
            if flags & (0x0100 | 0x0200):
                handle.seek(bone_index_size + 4, 1)
            if flags & 0x0400:
                _skip_floats(handle, 3)
            if flags & 0x0800:
                _skip_floats(handle, 6)
            if flags & 0x2000:
                handle.seek(4, 1)
            if flags & 0x0020:
                handle.seek(bone_index_size + 8, 1)
                link_count = struct.unpack("<i", handle.read(4))[0]
                for _link in range(link_count):
                    handle.seek(bone_index_size, 1)
                    has_limit = struct.unpack("<B", handle.read(1))[0]
                    if has_limit:
                        _skip_floats(handle, 6)
            bones[name] = position
    missing = [bone for bone in CORE_PMX_BONES if bone not in bones]
    if missing:
        raise ValueError(f"PMX is missing required standard MMD bones: {', '.join(missing)}")
    return bones


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    output = np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )
    return _normalize_quaternion(output)


def _quaternion_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = _normalize_quaternion(quaternion)
    vector = np.asarray(vector, dtype=np.float32)
    imaginary = quaternion[..., :3]
    real = quaternion[..., 3:4]
    dot = np.sum(imaginary * vector, axis=-1, keepdims=True)
    norm = np.sum(imaginary * imaginary, axis=-1, keepdims=True)
    return (
        2.0 * dot * imaginary
        + (real * real - norm) * vector
        + 2.0 * real * np.cross(imaginary, vector)
    ).astype(np.float32)


def _descendants(root: str, bones: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for candidate in bones:
        parent = PARENT.get(candidate)
        while parent is not None:
            if parent == root:
                result.append(candidate)
                break
            parent = PARENT.get(parent)
    return result


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = np.linalg.norm(source, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    source_unit = source / np.maximum(source_norm, 1e-8)
    target_unit = target / np.maximum(target_norm, 1e-8)
    cross = np.cross(source_unit, target_unit)
    sine = np.linalg.norm(cross, axis=-1, keepdims=True)
    cosine = np.sum(source_unit * target_unit, axis=-1, keepdims=True)
    axis = cross / np.maximum(sine, 1e-8)
    half_angle = 0.5 * np.arctan2(sine, cosine)
    quaternion = np.concatenate((axis * np.sin(half_angle), np.cos(half_angle)), axis=-1)
    quaternion[sine[:, 0] < 1e-6] = IDENTITY_QUATERNION
    return _normalize_quaternion(quaternion)


def _apply_leg_ik(
    global_position: dict[str, np.ndarray],
    global_rotation: dict[str, np.ndarray],
    target_name: str,
    end_name: str,
    links: tuple[str, ...],
    bones: tuple[str, ...],
) -> None:
    if target_name not in global_position:
        return
    target = global_position[target_name]
    descendants = {link: _descendants(link, bones) + [link] for link in links}
    for _ in range(6):
        for link in links:
            joint = global_position[link]
            delta = _rotation_between(global_position[end_name] - joint, target - joint)
            for descendant in descendants[link]:
                offset = global_position[descendant] - joint
                global_position[descendant] = joint + _quaternion_rotate(delta, offset)
                global_rotation[descendant] = _quaternion_multiply(delta, global_rotation[descendant])


def _rest_position(rest: Mapping[str, np.ndarray], bone: str | None) -> np.ndarray:
    current = bone
    while current is not None:
        if current in rest:
            return np.asarray(rest[current], dtype=np.float32)
        current = PARENT.get(current)
    return np.zeros(3, dtype=np.float32)


def motion180_from_vmd(vmd_path: str | Path, pmx_path: str | Path, frames: int | None = None) -> np.ndarray:
    motion = parse_motion_vmd(vmd_path)
    output_frames = motion.max_frame + 1 if frames is None else min(motion.max_frame + 1, int(frames))
    if output_frames <= 1:
        raise ValueError("VMD dance must contain at least two frames")
    rest = read_pmx_bone_positions(pmx_path)
    global_position: dict[str, np.ndarray] = {}
    global_rotation: dict[str, np.ndarray] = {}
    zeros = np.zeros((output_frames, 3), dtype=np.float32)
    identity = np.repeat(IDENTITY_QUATERNION[None, :], output_frames, axis=0)
    all_bones = (*FK_BONES, *IK_BONES)
    for bone in all_bones:
        local_position, local_rotation = dense_bone_motion(motion.bones.get(bone, []), output_frames)
        parent = PARENT.get(bone)
        parent_position = global_position.get(parent, zeros)
        parent_rotation = global_rotation.get(parent, identity)
        rest_offset = _rest_position(rest, bone) - _rest_position(rest, parent)
        global_position[bone] = parent_position + _quaternion_rotate(
            parent_rotation,
            local_position + rest_offset[None, :],
        )
        global_rotation[bone] = _quaternion_multiply(parent_rotation, local_rotation)
    if motion.bones.get("\u5de6\u8db3\uff29\uff2b"):
        _apply_leg_ik(
            global_position,
            global_rotation,
            "\u5de6\u8db3\uff29\uff2b",
            "\u5de6\u8db3\u9996",
            ("\u5de6\u3072\u3056", "\u5de6\u8db3"),
            all_bones,
        )
    if motion.bones.get("\u53f3\u8db3\uff29\uff2b"):
        _apply_leg_ik(
            global_position,
            global_rotation,
            "\u53f3\u8db3\uff29\uff2b",
            "\u53f3\u8db3\u9996",
            ("\u53f3\u3072\u3056", "\u53f3\u8db3"),
            all_bones,
        )
    return np.stack([global_position[bone] for bone in DCM_BONES], axis=1).reshape(output_frames, 180)


def load_motion180(
    dance_path: str | Path,
    pmx_path: str | Path | None = None,
    frames: int | None = None,
) -> np.ndarray:
    source = Path(dance_path)
    suffix = source.suffix.lower()
    if suffix == ".vmd":
        if pmx_path is None:
            raise ValueError("VMD input requires a matching PMX model because VMD does not store rest bone positions")
        motion = motion180_from_vmd(source, pmx_path, frames=frames)
    elif suffix == ".npy":
        motion = np.load(source, allow_pickle=False)
    elif suffix == ".json":
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("BoneKeyFrameTransformRecord", payload.get("BoneFrameTransformRecord"))
        if not isinstance(records, list):
            raise ValueError("GlobalTransform JSON has no bone transform record list")
        records = sorted(records, key=lambda item: int(item.get("FrameTime", 0)))
        requested = len(records) if frames is None else min(len(records), int(frames))
        motion = global_transforms_to_keypoints(records, requested)
    else:
        raise ValueError("Dance input must be .vmd, dataset-compatible .npy, or GlobalTransform .json")
    value = np.asarray(motion, dtype=np.float32)
    if value.ndim == 3 and value.shape[1:] == (60, 3):
        value = value.reshape(len(value), 180)
    if value.ndim != 2 or value.shape[1] != 180:
        raise ValueError(f"Dance motion must have shape [T,180] or [T,60,3], got {value.shape}")
    if frames is not None:
        value = value[: int(frames)]
    if len(value) <= 1 or not np.isfinite(value).all():
        raise ValueError("Dance motion must contain at least two finite frames")
    return value.astype(np.float32, copy=False)


@dataclass(frozen=True)
class InMemoryContext:
    clips: list[ClipRef]
    normalizers: NormalizerBundle
    history_len: int
    inference_len: int
    style_vocabulary: str
    motion180: np.ndarray
    music35: np.ndarray
    style: str
    audio_path: Path


class CustomCKDDataset:
    def __init__(self, context: InMemoryContext, stride: int) -> None:
        self.context = context
        self.windows = [CKDWindow(0, anchor) for anchor in range(0, context.clips[0].frames, stride)]
        self.subsequence_end_index = [len(self.windows)]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        clip = self.context.clips[0]
        history = self.context.history_len
        inference = self.context.inference_len
        motion, valid = padded_window(self.context.motion180, clip, window.anchor, history, inference)
        music, _ = padded_window(self.context.music35, clip, window.anchor, history, inference)
        keyframes = np.zeros(history + inference, dtype=np.uint8)
        clip_last = clip.frames - 1
        if window.anchor - history <= clip_last < window.anchor + inference:
            keyframes[history + clip_last - window.anchor] = 1
        return {
            "camera_keyframe": keyframes[:, None].astype(np.int64),
            "padding_mask": valid[:, None],
            "motion": self.context.normalizers["pose"].normalize(motion),
            "music": music.astype(np.float32, copy=False),
            "sample_id": clip.name,
        }


class CustomCSDataset:
    def __init__(self, context: InMemoryContext, keyframe_mask: np.ndarray) -> None:
        self.context = context
        positions, inserted = _keyframe_positions(keyframe_mask, context.inference_len)
        self.windows = [
            CSWindow(
                0,
                keyframe,
                positions[index + 1] if index + 1 < len(positions) else None,
                inserted[index],
            )
            for index, keyframe in enumerate(positions)
        ]
        self.subsequence_end_index = [len(self.windows)]
        self._camera20 = np.zeros((context.clips[0].frames, 20), dtype=np.float32)

    @property
    def normalizers(self) -> NormalizerBundle:
        return self.context.normalizers

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        clip = self.context.clips[0]
        history = self.context.history_len
        inference = self.context.inference_len
        motion, _ = padded_window(self.context.motion180, clip, window.keyframe, history, inference)
        music, _ = padded_window(self.context.music35, clip, window.keyframe, history, inference)
        camera, _ = padded_window(self._camera20, clip, window.keyframe, history, inference)
        camera_normalized = np.concatenate(
            (
                self.normalizers["camera_distance"].normalize(camera[:, 0:1]),
                self.normalizers["camera_position"].normalize(camera[:, 1:4]),
                self.normalizers["camera_rotation"].normalize(camera[:, 4:7]),
                self.normalizers["camera_fov"].normalize(camera[:, 7:8]),
                self.normalizers["camera_eye"].normalize(camera[:, 8:11]),
            ),
            axis=1,
        )
        available = min(inference, clip.frames - window.keyframe)
        active = available if window.next_keyframe is None else window.next_keyframe - window.keyframe
        inference_mask = np.zeros(history + inference, dtype=np.float32)
        inference_mask[history : history + active] = 1.0
        return {
            "camera": camera_normalized.astype(np.float32, copy=False),
            "camera_inference_mask": inference_mask[:, None],
            "motion": self.normalizers["pose"].normalize(motion),
            "music": music.astype(np.float32, copy=False),
            "style": style_features(
                self.context.style,
                self.context.style_vocabulary,
                history + inference,
            ),
            "sample_id": clip.name,
        }


def _context(
    sample_id: str,
    dance_path: Path,
    music_path: Path,
    motion180: np.ndarray,
    music35: np.ndarray,
    normalizers: NormalizerBundle,
    history_len: int,
    inference_len: int,
    vocabulary: str,
    style: str,
) -> InMemoryContext:
    clip = ClipRef(sample_id, "custom", 0, len(motion180), (dance_path.name, music_path.name))
    return InMemoryContext(
        clips=[clip],
        normalizers=normalizers,
        history_len=history_len,
        inference_len=inference_len,
        style_vocabulary=vocabulary,
        motion180=motion180,
        music35=music35,
        style=style,
        audio_path=music_path,
    )


def run_custom_generation(
    config: Mapping[str, Any],
    *,
    dance_path: str | Path,
    music_path: str | Path,
    pmx_path: str | Path | None = None,
    temporal_control: str | Path | None = None,
    spatio_temporal_control: str | Path | None = None,
    style: str = "Choreography",
    sample_id: str = "custom",
    run_name: str | None = None,
    max_frames: int | None = None,
) -> GenerationRun:
    experiment = require_mapping(config, "experiment")
    checkpoints = require_mapping(config, "checkpoints")
    generation = require_mapping(config, "generation")
    model_config = config.get("model", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("Optional config section `model` must be a mapping")
    history_len = int(model_config.get("history_len", 60))
    inference_len = int(model_config.get("inference_len", 60))
    vocabulary = str(generation.get("checkpoint_style_vocabulary", "legacy_dsc_v1"))
    style = validate_style(style, vocabulary)
    dance = resolve_project_path(dance_path)
    music = resolve_project_path(music_path)
    pmx = None if pmx_path is None else resolve_project_path(pmx_path)
    if not dance.is_file():
        raise FileNotFoundError(f"Dance input does not exist: {dance}")
    if not music.is_file():
        raise FileNotFoundError(f"Music input does not exist: {music}")
    if pmx is not None and not pmx.is_file():
        raise FileNotFoundError(f"PMX input does not exist: {pmx}")

    requested_frames = audio_frame_count(music, fps=FPS)
    if max_frames is not None:
        if max_frames <= 1:
            raise ValueError("max_frames must be greater than one")
        requested_frames = min(requested_frames, int(max_frames))
    motion180 = load_motion180(dance, pmx, frames=requested_frames)
    frames = min(len(motion180), requested_frames)
    motion180 = motion180[:frames]
    music35 = extract_music35(music, aligned_frame_limit=frames, output_frames=frames)
    frames_by_sample = {sample_id: frames}
    temporal = load_temporal_controls(temporal_control, frames_by_sample)
    spatial = load_spatio_temporal_controls(spatio_temporal_control, frames_by_sample)

    seed_everything(int(experiment.get("seed", 42)))
    device = resolve_device(str(generation.get("device", "auto")))
    use_ema = checkpoint_uses_ema(generation)
    ckd_checkpoint = load_checkpoint_file(resolve_project_path(str(checkpoints["ckd"])), device)
    ckd_model = build_ckd_model(config).to(device)
    load_model_weights(ckd_model, ckd_checkpoint, use_ema=use_ema, strict=True)
    ckd_context = _context(
        sample_id,
        dance,
        music,
        motion180,
        music35,
        normalizers_from_checkpoint(ckd_checkpoint),
        history_len,
        inference_len,
        vocabulary,
        style,
    )
    stride = int(model_config.get("ckd_test_stride", inference_len))
    ckd_dataset = CustomCKDDataset(ckd_context, stride=stride)
    generated_masks = infer_keyframes(ckd_model, ckd_dataset, device)
    keyframes = apply_keyframe_overrides(generated_masks, temporal, spatial)
    del ckd_model, ckd_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cs_checkpoint = load_checkpoint_file(resolve_project_path(str(checkpoints["cs"])), device)
    cs_model = build_cs_model(config).to(device)
    load_model_weights(cs_model, cs_checkpoint, use_ema=use_ema, strict=True)
    cs_context = _context(
        sample_id,
        dance,
        music,
        motion180,
        music35,
        normalizers_from_checkpoint(cs_checkpoint),
        history_len,
        inference_len,
        vocabulary,
        style,
    )
    cs_dataset = CustomCSDataset(cs_context, keyframes[sample_id])
    cameras = infer_cameras_controlled(
        cs_model,
        cs_dataset,  # type: ignore[arg-type]
        device,
        spatio_temporal=spatial,
        style_override=style,
    )

    output_root = resolve_project_path(str(generation.get("output_root", "generation")))
    result = GenerationRun.create(output_root, run_name or str(experiment["name"]), config)
    input_directory = result.root / "inputs"
    input_directory.mkdir()
    np.save(input_directory / "motion180.npy", motion180, allow_pickle=False)
    np.save(input_directory / "music35.npy", music35, allow_pickle=False)
    result.save_camera(
        sample_id,
        cameras[sample_id],
        {
            "dance": str(dance),
            "music": str(music),
            "pmx": None if pmx is None else str(pmx),
            "style": style,
            "temporal_control": sample_id in temporal,
            "spatio_temporal_control": sample_id in spatial,
            "motion180": "inputs/motion180.npy",
            "music35": "inputs/music35.npy",
        },
    )
    result.save_keyframes(sample_id, keyframes[sample_id])
    return result


def _config_or_cli_path(cli_value: Path | None, config_value: Any, name: str, required: bool) -> Path | None:
    value = cli_value if cli_value is not None else config_value
    if value in (None, ""):
        if required:
            raise ValueError(f"Custom inference requires `{name}` in config or --{name.replace('_', '-')}")
        return None
    return resolve_project_path(str(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dance", type=Path, default=None)
    parser.add_argument("--music", type=Path, default=None)
    parser.add_argument("--pmx", type=Path, default=None, help="Required for VMD dance input")
    parser.add_argument("--temporal-control", type=Path, default=None)
    parser.add_argument("--spatio-temporal-control", type=Path, default=None)
    parser.add_argument("--style", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    inputs = require_mapping(config, "input")
    controls = config.get("controls", {})
    if not isinstance(controls, Mapping):
        raise ValueError("Optional config section `controls` must be a mapping")
    dance = _config_or_cli_path(args.dance, inputs.get("dance"), "dance", True)
    music = _config_or_cli_path(args.music, inputs.get("music"), "music", True)
    pmx = _config_or_cli_path(args.pmx, inputs.get("pmx"), "pmx", False)
    temporal = _config_or_cli_path(
        args.temporal_control,
        controls.get("temporal"),
        "temporal_control",
        False,
    )
    spatial = _config_or_cli_path(
        args.spatio_temporal_control,
        controls.get("spatio_temporal"),
        "spatio_temporal_control",
        False,
    )
    style = args.style or controls.get("style") or "Choreography"
    sample_id = args.sample_id or inputs.get("sample_id") or "custom"
    result = run_custom_generation(
        config,
        dance_path=dance,  # type: ignore[arg-type]
        music_path=music,  # type: ignore[arg-type]
        pmx_path=pmx,
        temporal_control=temporal,
        spatio_temporal_control=spatial,
        style=str(style),
        sample_id=str(sample_id),
        run_name=args.run_name,
        max_frames=args.max_frames if args.max_frames is not None else inputs.get("max_frames"),
    )
    print(f"Custom generation complete: {result.root}")


if __name__ == "__main__":
    main()
