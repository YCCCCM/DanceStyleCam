"""Small dependency-free VMD camera writer for generated camera20 arrays."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _fixed(value: str, size: int, encoding: str = "shift_jis") -> bytes:
    encoded = value.encode(encoding, errors="ignore")[:size]
    return encoded + bytes(size - len(encoded))


def write_camera_vmd(camera20: np.ndarray, path: str | Path, stride: int = 1) -> Path:
    value = np.asarray(camera20, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 20:
        raise ValueError(f"camera20 must have shape [frames, 20], got {value.shape}")
    if stride <= 0:
        raise ValueError("VMD stride must be positive")
    frame_indices = list(range(0, len(value), stride))
    if frame_indices[-1] != len(value) - 1:
        frame_indices.append(len(value) - 1)

    chunks = [b"Vocaloid Motion Data 0002".ljust(30, b"\0"), _fixed("DanceStyleCam", 20)]
    chunks.append(struct.pack("<I", 0))  # bone keyframes
    chunks.append(struct.pack("<I", 0))  # morph keyframes
    chunks.append(struct.pack("<I", len(frame_indices)))
    for frame in frame_indices:
        row = value[frame]
        chunks.append(
            struct.pack(
                "<Ifffffff4B20xfB",
                frame,
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
                20,
                20,
                20,
                20,
                float(row[7]),
                0,
            )
        )
    chunks.append(struct.pack("<I", 0))  # light keyframes
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"".join(chunks))
    return output
