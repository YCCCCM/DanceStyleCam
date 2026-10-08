import json

from common.paths import PROJECT_ROOT
from data.schema import STYLE_NAMES_DSC
from data.splits import build_clips, load_segment_ranges, load_split, resolve_frame_range


def test_default_splits_are_the_original_non_overlapping_protocol() -> None:
    train = load_split(PROJECT_ROOT / "DCM_data/split/train.json")
    validation = load_split(PROJECT_ROOT / "DCM_data/split/validation.json")
    test = load_split(PROJECT_ROOT / "DCM_data/split/test.json")
    assert (len(train), len(validation), len(test)) == (362, 49, 43)
    train_names = {item.name for item in train}
    validation_names = {item.name for item in validation}
    test_names = {item.name for item in test}
    assert not (train_names & validation_names)
    assert not (train_names & test_names)
    assert not (validation_names & test_names)


def test_all_segment_ids_resolve() -> None:
    segments = load_segment_ranges(PROJECT_ROOT / "DCM_data/split/long2short.json")
    items = load_split(PROJECT_ROOT / "DCM_data/split/train.json")
    items += load_split(PROJECT_ROOT / "DCM_data/split/validation.json")
    items += load_split(PROJECT_ROOT / "DCM_data/split/test.json")
    assert all(resolve_frame_range(item, segments).start >= 0 for item in items)


def test_training_fragments_are_merged_virtually() -> None:
    segments = load_segment_ranges(PROJECT_ROOT / "DCM_data/split/long2short.json")
    train = load_split(PROJECT_ROOT / "DCM_data/split/train.json")
    frames = {str(sequence_id): 100_000 for sequence_id in range(108)}
    clips = build_clips(train, segments, frames, merge_adjacent=True)
    assert len(clips) == 145
    assert any(clip.name == "4_0~3" for clip in clips)



def test_annotations_use_the_dsc_style_names() -> None:
    with (PROJECT_ROOT / "DCM_data/music_style_16cat.json").open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    assert len(annotations) == 108
    assert set(annotations.values()).issubset(STYLE_NAMES_DSC)
