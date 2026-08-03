# Training Runs

Training logs, resolved configurations, normalizer statistics, and periodic
checkpoints are written below this directory. Runtime files are ignored by Git.

Each run contains a resolved `config.yaml`, `metrics.jsonl`, and portable
checkpoints under `checkpoints/`. Checkpoints contain tensors and numeric
metadata only; they do not pickle Dataset or custom Python normalizer objects.
