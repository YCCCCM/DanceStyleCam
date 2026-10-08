"""Shared virtual clip and zero-padded window helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from common.config import require_mapping
from common.paths import DatasetPaths

from .music_store import MusicFeatureStore
from .normalization import NormalizerBundle, fit_normalizers
from .splits import ClipRef, build_clips, load_segment_ranges, load_split
from .store import SequenceStore
from .style_labels import StyleAnnotations


@dataclass(frozen=True)
class DatasetContext:
    store: SequenceStore
    annotations: StyleAnnotations
    clips: list[ClipRef]
    normalizers: NormalizerBundle
    history_len: int
    inference_len: int
    style_vocabulary: str
    music_by_clip: dict[str, np.ndarray]

    def music_window(self, clip: ClipRef, anchor: int) -> np.ndarray:
        value = self.music_by_clip[clip.name]
        local_clip = ClipRef(clip.name, clip.sequence_id, 0, clip.frames, clip.source_items)
        return padded_window(value, local_clip, anchor, self.history_len, self.inference_len)[0]


def clips_for_split(config: dict[str, Any], split: str, store: SequenceStore) -> list[ClipRef]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    paths = DatasetPaths.from_config(config)
    dataset_config = require_mapping(config, "dataset")
    items = load_split(paths.train_split if split == "train" else paths.test_split)
    segments = load_segment_ranges(paths.segment_file)
    frames = {sequence_id: store.sequence_frames(sequence_id) for sequence_id in store.sequence_ids()}
    merge = split == "train" and bool(dataset_config.get("merge_adjacent_train", True))
    return build_clips(items, segments, frames, merge_adjacent=merge)


def cs_fragment_start_clips(config: dict[str, Any], store: SequenceStore) -> list[ClipRef]:
    dataset_config = require_mapping(config, "dataset")
    frames = int(dataset_config.get("cs_start_window_frames", 0))
    if frames < 0 or frames > int(dataset_config.get("history_len", 60)):
        raise ValueError("cs_start_window_frames must be between zero and history_len")
    if frames == 0:
        return []
    fragment_config = {
        **config,
        "dataset": {**dataset_config, "merge_adjacent_train": False},
    }
    existing = set(clips_for_split(config, "train", store))
    return [clip for clip in clips_for_split(fragment_config, "train", store) if clip not in existing]


def build_context(
    config: dict[str, Any],
    split: str,
    normalizers: NormalizerBundle | None = None,
    style_vocabulary: str | None = None,
) -> DatasetContext:
    paths = DatasetPaths.from_config(config)
    dataset_config = require_mapping(config, "dataset")
    store = SequenceStore(paths.processed_root)
    clips = clips_for_split(config, split, store)
    if normalizers is None:
        train_clips = clips if split == "train" else clips_for_split(config, "train", store)
        normalizers = fit_normalizers(store, train_clips)
    music_by_clip: dict[str, np.ndarray] = {}
    music_store = MusicFeatureStore(paths.processed_root)
    for clip in clips:
        music_by_clip[clip.name] = music_store.load_clip(clip)

    return DatasetContext(
        store=store,
        annotations=StyleAnnotations.load(paths.style_file),
        clips=clips,
        normalizers=normalizers,
        history_len=int(dataset_config.get("history_len", 60)),
        inference_len=int(dataset_config.get("inference_len", 60)),
        style_vocabulary=(
            style_vocabulary
            if style_vocabulary is not None
            else str(dataset_config.get("style_vocabulary", "dsc"))
        ),
        music_by_clip=music_by_clip,
    )


def padded_window(
    value: np.ndarray,
    clip: ClipRef,
    anchor: int,
    history_len: int,
    inference_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Copy `[anchor-history, anchor+inference)` within one virtual clip."""

    window_frames = history_len + inference_len
    output = np.zeros((window_frames, *value.shape[1:]), dtype=value.dtype)
    valid = np.zeros(window_frames, dtype=np.float32)
    local_start = max(0, anchor - history_len)
    local_end = min(clip.frames, anchor + inference_len)
    destination_start = history_len - (anchor - local_start)
    destination_end = destination_start + (local_end - local_start)
    output[destination_start:destination_end] = value[clip.start + local_start : clip.start + local_end]
    valid[destination_start:destination_end] = 1.0
    return output, valid
