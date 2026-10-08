import numpy as np
import pytest

from data.music_store import MusicClipSpec, MusicFeatureStore
from data.splits import ClipRef


def _spec() -> MusicClipSpec:
    return MusicClipSpec(ClipRef("7_0", "7", 10, 15, ("C_7_0",)), 10, 15, 99)


def test_music_store_loads_fixed_clip_feature_without_audio(tmp_path) -> None:
    store = MusicFeatureStore(tmp_path)
    spec = _spec()
    store.save(spec, np.zeros((5, 35), dtype=np.float32), store.load_manifest())

    loaded = store.load_clip(spec.clip)

    assert loaded.shape == (5, 35)
    assert loaded.dtype == np.float32


def test_music_store_rejects_changed_clip_range(tmp_path) -> None:
    store = MusicFeatureStore(tmp_path)
    spec = _spec()
    store.save(spec, np.zeros((5, 35), dtype=np.float32), store.load_manifest())
    changed = ClipRef("7_0", "7", 11, 16, ("C_7_0",))

    with pytest.raises(FileNotFoundError, match="prepare_music_features"):
        store.load_clip(changed)
