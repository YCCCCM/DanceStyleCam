from data.schema import ARRAY_SPECS, CAMERA20_FIELDS, STYLE_NAMES_DSC


def test_array_contract() -> None:
    assert set(ARRAY_SPECS) == {
        "motion180",
        "camera20",
        "keyframe_mask",
        "bone_mask60",
    }
    assert len(CAMERA20_FIELDS) == 20


def test_style_vocabulary_has_stable_size() -> None:
    assert len(STYLE_NAMES_DSC) == 16
    assert len(set(STYLE_NAMES_DSC)) == 16
