# Controllable inference

DanceStyleCam exposes two additional inference entrypoints. The original
`infer/generate.py` behavior is unchanged.

## Test-split inference

`infer/generate_test_controlled.py` reads the configured test split. With all
control values set to `null`, CKD predicts keyframe times, CS predicts camera
parameters, and each test clip uses its annotated dance style.

```bash
python infer/generate_test_controlled.py --config configs/infer/controlled_test.yaml
```

The command line can override any control without editing the YAML:

```bash
python infer/generate_test_controlled.py \
  --config configs/infer/controlled_test.yaml \
  --sample-ids 66_0 \
  --temporal-control controls/temporal.json \
  --spatio-temporal-control controls/camera.json \
  --style Jazz
```

For multiple test clips, a JSON file maps clip ids to controls. A directory is
also accepted; it may contain `<clip-id>.json` or `<clip-id>.npy` files. Clips
without an entry retain the model-generated result and dataset style.

## Custom dance and music

`infer/generate_custom.py` extracts the 35D music feature with librosa at run
time and converts the supplied dance to the DCM `[frames, 180]` joint-position
contract.

```bash
python infer/generate_custom.py \
  --config configs/infer/custom.yaml \
  --dance input/dance.vmd \
  --pmx input/model.pmx \
  --music input/music.wav
```

VMD stores local bone animation but does not store the model's rest skeleton.
A matching PMX is therefore required for VMD input. The converter interpolates
the sparse VMD records at 30 FPS, performs PMX forward kinematics, applies a
lightweight two-leg CCD solve when `左/右足ＩＫ` tracks are present, and writes
the fixed DCM 60-bone order. The selected PMX determines character proportions.

The `--dance` input can alternatively be one of these already-converted forms:

- `.npy`: `[T,180]` or `[T,60,3]` float joint positions.
- `.json`: DCM `BoneKeyFrameTransformRecord` or `BoneFrameTransformRecord`.

These two forms do not require `--pmx`. Dance and music are trimmed to their
shared duration. The extracted `motion180.npy` and `music35.npy` are retained
under `generation/<run-name>/inputs/` for inspection. When no style is given,
every frame uses `Choreography`. When no keyframe control is given, CKD and CS
generate the temporal and spatial camera results normally.

## Temporal control

Temporal control sets camera keyframe times. It accepts a binary `[T]` NPY
mask, a one-dimensional list of frame indices, or either JSON form below:

```json
{"frames": [0, 30, 75, 119]}
```

```json
{
  "66_0": {"frames": [0, 30, 75, 119]},
  "66_1": {"mask": [1, 0, 0, 0]}
}
```

The second example abbreviates the mask; a real mask must contain one value
per clip frame. First and last frames are always made safe keyframe boundaries,
and gaps longer than the model inference window receive internal boundaries.

## Spatio-temporal control

Spatio-temporal control fixes camera values at selected frame indices. Each
camera row is the first eight DCM camera values in this order:

`distance, position xyz, rotation xyz, field-of-view`.

Rotation is in radians and field-of-view is in degrees. JSON supports either
parallel arrays or numeric frame keys:

```json
{
  "frames": [0, 60],
  "camera": [
    [-7.0, 0.0, 10.0, 0.0, 0.1, 0.0, 0.0, 45.0],
    [-9.0, 1.0, 11.0, 0.0, 0.2, 0.3, 0.0, 50.0]
  ]
}
```

An NPY file can use `[K,9]` rows with the frame index in column zero, or a
dense `[T,8]`/`[T,20]` array with unconstrained rows filled entirely with NaN.
Spatially controlled frames are automatically added to the temporal keyframe
mask. Camera parameters at all other keyframes remain model-generated.

## Style control

`--style` accepts one of the 16 checkpoint vocabulary names:

`Breaking`, `Popping`, `Locking`, `Hiphop`, `Urban`, `Jazz`, `Tai`, `Uighur`,
`Hmong`, `Korean`, `Choreography`, `Chinese`, `HanTang`, `ShenYun`, `Kun`, or
`DunHuang`.
