# DanceStyleCam-Official Migration Scope

This file records the architecture and compatibility decisions for the open-source migration.
It is intentionally kept at the repository root so later changes do not silently drift from the
agreed project structure.

## Project layout

The project follows the top-level module style used by AnchorDance. There is no project package
directory such as `dancestylecam/`.

```text
DanceStyleCam-Official/
├── DCM_data/                 # original public dataset, read-only
├── DCM-style++/              # generated per-sequence NPY dataset
├── common/                   # config loading and portable project paths
├── configs/                  # data, train, infer, visualization and eval YAML
├── data/                     # dataset/preprocessing source code only
├── models/                   # CKD, CS, transformer backbone and optional discriminator
├── train/                    # train_ckd.py and train_cs.py entrypoints
├── infer/                    # CKD -> CS inference entrypoints
├── metric/                   # evaluation implementations
├── tools/                    # executable data, visualization and evaluation tools
├── generation/               # generated runtime results, one directory per run
├── runs/                     # training logs and checkpoints
├── checkpoints/              # downloaded/released model files
└── tests/                    # software tests only
```

All commands use a YAML config and a Python entrypoint. Shell scripts are not part of the required
workflow.

## Dataset contract

- `DCM_data/` remains the unmodified public DCM layout from DanceCamAnimator-Official.
- `DCM_data/music_style_16cat.json` is the only style annotation source.
- The default split files are `DCM_data/split/train_pre.json` and `DCM_data/split/test_pre.json`.
- `DCM-style++/` stores one NPY file per complete source sequence for motion, camera, music,
  keyframe mask and bone mask, plus a small manifest and checksums.
- The stored `music35` NPY is the full-sequence representation. The default
  `dataset.music_feature_mode: legacy_clip` computes clip-local music features from `DCM_data`
  audio in memory, matching the legacy cut-WAV-then-extract behavior without writing split caches.
- Split selection and train/test virtual clips are resolved at runtime from JSON. Changing a split
  does not rebuild or duplicate the processed NPY arrays.
- The processed manifest must not embed mutable split or style assignments.
- No giant DCM++ NPZ, pickle dataset cache, copied audio tree or physical Train/Test data copy is
  introduced.

## Model and checkpoint scope

- Public baseline `DSC_CKD.pt`: CKD stage, released 1500-epoch model.
- Public baseline `DSC_CS.pt`: CS stage, released 1200-epoch GAN-trained generator.
- Released inference uses `model_state_dict`, matching the original test scripts' `EMA_tag=False`.
  `configs/infer/default.yaml` therefore defaults to `checkpoint_weights: model`; EMA inference must
  be selected explicitly with `checkpoint_weights: ema`.
- The CS generator architecture is identical for GAN and no-GAN training.
- Long-sequence, LSC and AWTS branches are explicitly out of this migration.
- The historical public CS style-channel order is represented by an explicit compatibility
  vocabulary; it is never inferred from filesystem ordering.

## GAN and no-GAN training

`train/train_cs.py` is the only CS training entrypoint. The YAML controls the optional
discriminator:

```yaml
training:
  use_gan: true       # false selects reconstruction-only no-GAN training
```

GAN mode creates and trains the discriminator and stores its state in the checkpoint. no-GAN mode
does not allocate a discriminator and stores only generator/EMA/optimizer/normalizer state.

When resuming with `use_gan: true`, a missing discriminator state is allowed: a fresh discriminator
and discriminator optimizer are initialized while the generator is loaded from the checkpoint.
When `use_gan: false`, any discriminator state in a checkpoint is ignored.

Inference never loads or requires discriminator state. GAN and no-GAN generator checkpoints use the
same `infer/generate.py` pipeline.

## Runtime result contract

`generation/<run-name>/` is the single runtime result root:

```text
manifest.json
config.yaml
camera/*.npy
keyframes/*.npy
vis/       # created only by visualization tools
metrics/   # created only by metric tools
vmd/       # created only by VMD export tools
```

`tests/` contains test code, not inference outputs. A separate ambiguous top-level `outputs/`
directory is not needed: training artifacts belong under `runs/`, and generated artifacts belong
under `generation/`.

## Required entrypoints

```bash
python tools/data/prepare_dcm_style_pp.py --config configs/data/dcm_style_pp.yaml
python tools/data/validate_dcm_style_pp.py --config configs/data/dcm_style_pp.yaml
python train/train_ckd.py --config configs/train/ckd.yaml
python train/train_cs.py --config configs/train/cs.yaml
python train/train_cs.py --config configs/train/cs_nogan.yaml
python infer/generate.py --config configs/infer/default.yaml
python tools/visualization/visualize.py --config configs/visualization/default.yaml --input generation/<run>
python tools/eval/run_benchmark.py --config configs/eval/benchmark.yaml --input generation/<run>
```
