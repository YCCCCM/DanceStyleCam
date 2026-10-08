"""Virtual keyframe-centered dataset for style-conditioned Camera Synthesis."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .dataset_common import DatasetContext, build_context, cs_fragment_start_clips, padded_window
from .music_store import MusicFeatureStore
from .normalization import NormalizerBundle


@dataclass(frozen=True)
class CSWindow:
    clip_index: int
    keyframe: int
    next_keyframe: int | None
    inserted: bool


def _keyframe_positions(mask: np.ndarray, inference_len: int) -> tuple[list[int], list[bool]]:
    value = np.asarray(mask, dtype=np.uint8).copy()
    value[0] = 1
    value[-1] = 1
    original = np.flatnonzero(value).tolist()
    positions = [original[0]]
    inserted = [False]
    for target in original[1:]:
        current = positions[-1]
        while target - current > inference_len:
            current += inference_len
            positions.append(current)
            inserted.append(True)
        positions.append(target)
        inserted.append(False)
    return positions, inserted


class CSDataset:
    def __init__(
        self,
        context: DatasetContext,
        generated_keyframe_masks: dict[str, np.ndarray] | None = None,
        prefix_frames_by_clip: dict[int, int] | None = None,
    ) -> None:
        self.context = context
        self.prefix_frames_by_clip = {} if prefix_frames_by_clip is None else prefix_frames_by_clip
        self.windows: list[CSWindow] = []
        self.subsequence_end_index: list[int] = []
        for clip_index, clip in enumerate(context.clips):
            if generated_keyframe_masks is None:
                mask = context.store.load(clip.sequence_id, "keyframe_mask")[clip.start : clip.end]
            else:
                mask = generated_keyframe_masks.get(clip.name)
                if mask is None:
                    mask = context.store.load(clip.sequence_id, "keyframe_mask")[clip.start : clip.end]
                if len(mask) != clip.frames:
                    raise ValueError(f"Generated keyframe mask length mismatch for {clip.name}")
            positions, inserted = _keyframe_positions(mask, context.inference_len)
            for keyframe_index, keyframe in enumerate(positions):
                if keyframe >= self.prefix_frames_by_clip.get(clip_index, clip.frames):
                    continue
                next_keyframe = positions[keyframe_index + 1] if keyframe_index + 1 < len(positions) else None
                self.windows.append(CSWindow(clip_index, keyframe, next_keyframe, inserted[keyframe_index]))
            self.subsequence_end_index.append(len(self.windows))
        self.start_window_count = sum(window.clip_index in self.prefix_frames_by_clip for window in self.windows)

    @property
    def normalizers(self) -> NormalizerBundle:
        return self.context.normalizers

    @property
    def inserted_keyframe(self) -> list[int]:
        return [int(window.inserted) for window in self.windows]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        clip = self.context.clips[window.clip_index]
        history = self.context.history_len
        inference = self.context.inference_len
        store = self.context.store

        motion, _ = padded_window(store.load(clip.sequence_id, "motion180"), clip, window.keyframe, history, inference)
        music = self.context.music_window(clip, window.keyframe)
        camera, _ = padded_window(store.load(clip.sequence_id, "camera20"), clip, window.keyframe, history, inference)
        bone_mask, _ = padded_window(
            store.load(clip.sequence_id, "bone_mask60"), clip, window.keyframe, history, inference
        )

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
        inference_mask = np.zeros(history + inference, dtype=np.float32)
        available = min(inference, clip.frames - window.keyframe)
        active = available if window.next_keyframe is None else window.next_keyframe - window.keyframe
        inference_mask[history : history + active] = 1.0

        pre_padding = max(0, history - window.keyframe)
        suffix_padding = max(0, inference - available)
        style = self.context.annotations.one_hot(clip.sequence_id, self.context.style_vocabulary)
        style_features = np.repeat(style[None, :], history + inference, axis=0)
        return {
            "camera": camera_normalized.astype(np.float32, copy=False),
            "camera_inference_mask": inference_mask[:, None],
            "bone_mask": bone_mask.astype(np.float32),
            "motion": self.normalizers["pose"].normalize(motion),
            "music": music.astype(np.float32, copy=False),
            "style": style_features,
            "sample_id": clip.name,
            "sequence_id": clip.sequence_id,
            "pre_padding": pre_padding,
            "suffix_padding": suffix_padding,
            "start_frame": max(0, window.keyframe - history),
            "end_frame": min(clip.frames - 1, window.keyframe + inference - 1),
            "anchor_frame": clip.start + window.keyframe,
            "inserted_keyframe": int(window.inserted),
        }


def build_cs_dataset(
    config: dict[str, Any],
    split: str,
    normalizers: NormalizerBundle | None = None,
    generated_keyframe_masks: dict[str, np.ndarray] | None = None,
    style_vocabulary: str | None = None,
) -> CSDataset:
    context = build_context(
        config,
        split,
        normalizers=normalizers,
        style_vocabulary=style_vocabulary,
    )
    prefix_frames_by_clip = {}
    if split == "train":
        start_clips = cs_fragment_start_clips(config, context.store)
        if start_clips:
            count = len(context.clips)
            prefix_frames = int(config["dataset"]["cs_start_window_frames"])
            prefix_frames_by_clip = {count + index: prefix_frames for index in range(len(start_clips))}
            music_store = MusicFeatureStore(context.store.root)
            context = replace(
                context,
                clips=[*context.clips, *start_clips],
                music_by_clip={
                    **context.music_by_clip,
                    **{clip.name: music_store.load_clip(clip) for clip in start_clips},
                },
            )
    return CSDataset(context, generated_keyframe_masks, prefix_frames_by_clip)
