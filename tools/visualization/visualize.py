"""Render a lightweight 3D diagnostic video from an existing generation run."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from common.config import load_config, require_mapping
from common.paths import resolve_project_path
from data.store import SequenceStore

from infer.result_io import GenerationRun


def _render_sample(
    run: GenerationRun,
    sample_id: str,
    motion: np.ndarray,
    camera: np.ndarray,
    output: Path,
    fps: int,
    width: int,
    height: int,
    max_frames: int | None,
) -> None:
    try:
        import imageio.v2 as imageio
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Visualization requires the `visualization` optional dependencies") from error

    frames = min(len(motion), len(camera))
    if max_frames is not None:
        frames = min(frames, max_frames)
    figure = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    axis = figure.add_subplot(111, projection="3d")
    writer = imageio.get_writer(output, fps=fps, codec="libx264")
    try:
        points = motion[:frames].reshape(frames, 60, 3)
        camera = camera[:frames]
        for index in range(frames):
            axis.clear()
            pose = points[index]
            cam = camera[index]
            axis.scatter(pose[:, 0], pose[:, 1], pose[:, 2], s=8, c="tab:blue")
            axis.scatter([cam[8]], [cam[9]], [cam[10]], s=40, c="tab:red")
            axis.plot(camera[: index + 1, 8], camera[: index + 1, 9], camera[: index + 1, 10], c="tab:red", lw=1)
            extent = np.ptp(pose, axis=0).max()
            center = pose.mean(axis=0)
            radius = max(float(extent) * 0.7, 1.0)
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.set_title(f"DanceStyleCam {sample_id}  frame {index}")
            axis.set_xlabel("X")
            axis.set_ylabel("Y")
            axis.set_zlabel("Z")
            figure.tight_layout()
            figure.canvas.draw()
            image = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(image)
    finally:
        writer.close()
        plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Existing generation/<run-name> directory")
    parser.add_argument("--sample-ids", default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per sample for quick inspection")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run = GenerationRun.open(args.input)
    values = require_mapping(config, "visualization")
    data_config = load_config(resolve_project_path(str(require_mapping(config, "data")["config"])))
    store = SequenceStore(resolve_project_path(str(require_mapping(data_config, "paths")["processed_root"])))
    selected = None if args.sample_ids is None else {value.strip() for value in args.sample_ids.split(",")}
    output = run.derived_dir(str(values.get("output_subdir", "vis")))
    manifest = run.load_manifest()
    for sample_id, sample in manifest["samples"].items():
        if selected is not None and sample_id not in selected:
            continue
        metadata: dict[str, Any] = sample.get("metadata", {})
        sequence_id = str(metadata["sequence_id"])
        start = int(metadata["source_start_frame"])
        end = int(metadata["source_end_frame"])
        motion = store.load(sequence_id, "motion180")[start:end]
        _render_sample(
            run,
            sample_id,
            motion,
            run.load_camera(sample_id),
            output / f"{sample_id}.mp4",
            int(values.get("fps", 30)),
            int(values.get("width", 1280)),
            int(values.get("height", 720)),
            args.max_frames,
        )
    print(f"Visualization complete: {output}")


if __name__ == "__main__":
    main()
