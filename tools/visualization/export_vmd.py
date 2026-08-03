"""Export camera20 NPY files from a generation run to VMD camera files."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infer.result_io import GenerationRun
from tools.visualization.vmd import write_camera_vmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing generation/<run-name> directory")
    parser.add_argument("--stride", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run = GenerationRun.open(args.input)
    output = run.derived_dir("vmd")
    manifest = run.load_manifest()
    for sample_id in manifest["samples"]:
        write_camera_vmd(run.load_camera(sample_id), output / f"{sample_id}.vmd", args.stride)
    print(f"VMD export complete: {output}")


if __name__ == "__main__":
    main()
