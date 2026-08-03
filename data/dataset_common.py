"""Shared virtual clip and zero-padded window helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from common.config import require_mapping
from common.paths import DatasetPaths

from .audio_features import extract_music35_clip
from .normalization import NormalizerBundle, fit_normalizers
from .raw_dcm import RawDCM
from .splits import ClipRef, build_clips, load_segment_ranges, load_split, music_frame_range
from .store import SequenceStore
from .style_labels import StyleAnnotations


@dataclass(frozen=True)
class DatasetContext:
    store: SequenceStore
    raw: RawDCM
    annotations: StyleAnnotations
    clips: list[ClipRef]
    normalizers: NormalizerBundle
    history_len: int
    inference_len: int
    style_vocabulary: str
    music_by_clip: dict[str, np.ndarray] | None = None

    def music_window(self, clip: ClipRef, anchor: int) -> np.ndarray:
        if self.music_by_clip is None:
            value = self.store.load(clip.sequence_id, "music35")
            return padded_window(value, clip, anchor, self.history_len, self.inference_len)[0]
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
    raw = RawDCM(paths.raw_root)
    music_by_clip: dict[str, np.ndarray] | None = None
    music_feature_mode = str(dataset_config.get("music_feature_mode", "sequence_npy"))
    if music_feature_mode == "legacy_clip":
        music_by_clip = {}
        segments = load_segment_ranges(paths.segment_file)
        for position, clip in enumerate(clips, start=1):
            files = raw.sequence_files(clip.sequence_id)
            audio_path = files.audio
            audio_start, audio_end = music_frame_range(clip, segments)
            aligned_frame_limit = (
                None
                if files.aligned_audio is not None
                else int(store.load_manifest()["sequences"][clip.sequence_id]["aligned_frame_limit"])
            )
            music_by_clip[clip.name] = extract_music35_clip(
                audio_path,
                audio_start,
                audio_end,
                clip.frames,
                aligned_frame_limit=aligned_frame_limit,
            )
            if position == 1 or position % 10 == 0 or position == len(clips):
                print(f"Extracted legacy music features for {position}/{len(clips)} clips", flush=True)
    elif music_feature_mode != "sequence_npy":
        raise ValueError(f"Unsupported music feature mode: {music_feature_mode}")

    return DatasetContext(
        store=store,
        raw=raw,
        annotations=StyleAnnotations.load(paths.style_file),
        clips=clips,
        normalizers=normalizers,
        history_len=int(dataset_config.get("history_len", 60)),
        inference_len=int(dataset_config.get("inference_len", 60)),
        style_vocabulary=(
            style_vocabulary
            if style_vocabulary is not None
            else str(dataset_config.get("style_vocabulary", "canonical_v1"))
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
