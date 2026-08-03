"""Generate camera NPY files without rendering videos."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config
from common.config import require_mapping
from common.paths import resolve_project_path

from infer.pipeline import run_generation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", default=None, help="Override experiment.name")
    parser.add_argument(
        "--sample-ids",
        default=None,
        help="Comma-separated virtual clip ids for a focused run, for example 66_0,66_1",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    data_path = resolve_project_path(str(require_mapping(config, "data")["config"]))
    data_config = load_config(data_path)
    sample_ids = None if args.sample_ids is None else [value.strip() for value in args.sample_ids.split(",")]
    result = run_generation(config, data_config, run_name=args.run_name, sample_ids=sample_ids)
    print(f"Generation complete: {result.root}")


if __name__ == "__main__":
    main()
