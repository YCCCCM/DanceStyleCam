"""Resolve lightweight split JSON entries into full-song frame ranges."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class SplitItem:
    name: str
    legacy_category: str
    sequence_id: str
    segment_index: int | None

    @classmethod
    def parse(cls, value: str) -> "SplitItem":
        parts = value.split("_")
        if len(parts) not in (2, 3) or not parts[0] or not parts[1].isdigit():
            raise ValueError(f"Invalid DCM split item: {value}")
        segment_index = None if len(parts) == 2 else int(parts[2])
        return cls(value, parts[0], parts[1], segment_index)


@dataclass(frozen=True)
class FrameRange:
    item: SplitItem
    start: int
    end: int | None


@dataclass(frozen=True)
class ClipRef:
    """A virtual clip backed by a frame range in one full-sequence NPY."""

    name: str
    sequence_id: str
    start: int
    end: int
    source_items: tuple[str, ...]

    @property
    def frames(self) -> int:
        return self.end - self.start


def load_split(path: str | Path) -> list[SplitItem]:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Split must be a JSON string list: {path}")
    return [SplitItem.parse(value) for value in values]


def load_segment_ranges(path: str | Path) -> dict[str, list[tuple[int, int]]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    return {str(key): [tuple(pair) for pair in ranges] for key, ranges in values.items()}


def resolve_frame_range(item: SplitItem, segments: dict[str, list[tuple[int, int]]]) -> FrameRange:
    if item.segment_index is None:
        return FrameRange(item=item, start=0, end=None)
    sequence_segments = segments.get(item.sequence_id, [])
    try:
        start, end = sequence_segments[item.segment_index]
    except IndexError as error:
        raise ValueError(f"Segment index out of range for {item.name}") from error
    return FrameRange(item=item, start=int(start), end=int(end))


def _item_clip(
    item: SplitItem,
    segments: dict[str, list[tuple[int, int]]],
    frames_by_sequence: dict[str, int],
) -> ClipRef:
    frame_range = resolve_frame_range(item, segments)
    sequence_frames = frames_by_sequence[item.sequence_id]
    start = min(frame_range.start, sequence_frames)
    end = sequence_frames if frame_range.end is None else min(frame_range.end + 1, sequence_frames)
    if end <= start:
        raise ValueError(f"Split item has an empty frame range after alignment: {item.name}")
    suffix = "" if item.segment_index is None else f"_{item.segment_index}"
    return ClipRef(f"{item.sequence_id}{suffix}", item.sequence_id, start, end, (item.name,))


def build_clips(
    items: list[SplitItem],
    segments: dict[str, list[tuple[int, int]]],
    frames_by_sequence: dict[str, int],
    merge_adjacent: bool = False,
) -> list[ClipRef]:
    """Build physical-DCM++-compatible clips without copying array data."""

    if not merge_adjacent:
        return [_item_clip(item, segments, frames_by_sequence) for item in items]

    ordered = sorted(
        items,
        key=lambda item: (
            int(item.sequence_id),
            -1 if item.segment_index is None else item.segment_index,
        ),
    )
    groups: list[list[SplitItem]] = []
    for item in ordered:
        if not groups:
            groups.append([item])
            continue
        previous = groups[-1][-1]
        consecutive = (
            item.sequence_id == previous.sequence_id
            and item.segment_index is not None
            and previous.segment_index is not None
            and item.segment_index == previous.segment_index + 1
        )
        if consecutive:
            groups[-1].append(item)
        else:
            groups.append([item])

    clips: list[ClipRef] = []
    for group in groups:
        first = _item_clip(group[0], segments, frames_by_sequence)
        last = _item_clip(group[-1], segments, frames_by_sequence)
        first_segment = group[0].segment_index
        last_segment = group[-1].segment_index
        if first_segment is None:
            name = group[0].sequence_id
        elif first_segment == last_segment:
            name = f"{group[0].sequence_id}_{first_segment}"
        else:
            name = f"{group[0].sequence_id}_{first_segment}~{last_segment}"
        clips.append(
            ClipRef(
                name=name,
                sequence_id=group[0].sequence_id,
                start=first.start,
                end=last.end,
                source_items=tuple(item.name for item in group),
            )
        )
    return clips


def music_frame_range(
    clip: ClipRef,
    segments: dict[str, list[tuple[int, int]]],
) -> tuple[int, int | None]:
    """Return the legacy audio slice before camera-length truncation."""

    items = [SplitItem.parse(value) for value in clip.source_items]
    if len(items) == 1 and items[0].segment_index is None:
        return 0, None
    first = items[0].segment_index
    last = items[-1].segment_index
    if first is None or last is None:
        raise ValueError(f"Cannot mix split and unsplit items in one clip: {clip.source_items}")
    return int(segments[clip.sequence_id][first][0]), int(segments[clip.sequence_id][last][1]) + 1
