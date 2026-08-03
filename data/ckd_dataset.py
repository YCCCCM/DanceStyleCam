"""Virtual sliding-window dataset for Camera Keyframe Detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from common.config import require_mapping

from .dataset_common import DatasetContext, build_context, padded_window
from .normalization import NormalizerBundle


@dataclass(frozen=True)
class CKDWindow:
    clip_index: int
    anchor: int


class CKDDataset:
    def __init__(self, context: DatasetContext, stride: int) -> None:
        if stride <= 0:
            raise ValueError("CKD stride must be positive")
        self.context = context
        self.stride = stride
        self.windows: list[CKDWindow] = []
        self.subsequence_end_index: list[int] = []
        for clip_index, clip in enumerate(context.clips):
            self.windows.extend(CKDWindow(clip_index, anchor) for anchor in range(0, clip.frames, stride))
            self.subsequence_end_index.append(len(self.windows))

    @property
    def normalizers(self) -> NormalizerBundle:
        return self.context.normalizers

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        clip = self.context.clips[window.clip_index]
        history = self.context.history_len
        inference = self.context.inference_len
        motion, valid = padded_window(
            self.context.store.load(clip.sequence_id, "motion180"), clip, window.anchor, history, inference
        )
        music = self.context.music_window(clip, window.anchor)
        keyframes, _ = padded_window(
            self.context.store.load(clip.sequence_id, "keyframe_mask"), clip, window.anchor, history, inference
        )
        clip_last = clip.frames - 1
        if window.anchor - history <= clip_last < window.anchor + inference:
            keyframes[history + clip_last - window.anchor] = 1

        audio_path = self.context.raw.sequence_files(clip.sequence_id).audio
        return {
            "camera_keyframe": keyframes[:, None].astype(np.int64),
            "padding_mask": valid[:, None],
            "motion": self.normalizers["pose"].normalize(motion),
            "music": music.astype(np.float32, copy=False),
            "sample_id": clip.name,
            "sequence_id": clip.sequence_id,
            "audio_path": str(audio_path),
            "audio_start_frame": clip.start,
            "anchor_frame": clip.start + window.anchor,
        }


def build_ckd_dataset(
    config: dict[str, Any],
    split: str,
    normalizers: NormalizerBundle | None = None,
) -> CKDDataset:
    context = build_context(config, split, normalizers=normalizers)
    dataset_config = require_mapping(config, "dataset")
    stride_key = "ckd_train_stride" if split == "train" else "ckd_test_stride"
    return CKDDataset(context, stride=int(dataset_config.get(stride_key, 15 if split == "train" else 60)))
