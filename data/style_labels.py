"""Central style annotation and checkpoint vocabulary handling."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .schema import STYLE_NAMES_CANONICAL_V1


STYLE_ALIASES = {
    "jazz": "Jazz",
    "Dai": "Tai",
    "Wei": "Uighur",
    "Miao": "Hmong",
    "Cherography": "Choreography",
    "Choerography": "Choreography",
}

# Existing DanceStyleCam training code used this accidental order when it
# converted the canonical filename label back into a one-hot vector.
STYLE_NAMES_LEGACY_DSC_V1 = (
    "Breaking",
    "Popping",
    "Locking",
    "Hiphop",
    "Urban",
    "Jazz",
    "Tai",
    "Uighur",
    "Hmong",
    "Korean",
    "Choreography",
    "Chinese",
    "HanTang",
    "ShenYun",
    "Kun",
    "DunHuang",
)

STYLE_VOCABULARIES = {
    "canonical_v1": STYLE_NAMES_CANONICAL_V1,
    "legacy_dsc_v1": STYLE_NAMES_LEGACY_DSC_V1,
}


def normalize_style_name(value: str) -> str:
    return STYLE_ALIASES.get(value, value)


@dataclass(frozen=True)
class StyleAnnotations:
    by_sequence: dict[str, str]

    @classmethod
    def load(cls, path: str | Path) -> "StyleAnnotations":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        annotations: dict[str, str] = {}
        for audio_name, style in raw.items():
            if not audio_name.startswith("a") or not audio_name.endswith(".wav"):
                raise ValueError(f"Invalid style annotation key: {audio_name}")
            sequence_id = audio_name[1:-4]
            normalized = normalize_style_name(style)
            if normalized not in STYLE_NAMES_CANONICAL_V1:
                raise ValueError(f"Unknown style annotation `{style}` for {audio_name}")
            annotations[sequence_id] = normalized
        return cls(annotations)

    def style_name(self, sequence_id: str | int) -> str:
        return self.by_sequence[str(sequence_id)]

    def one_hot(self, sequence_id: str | int, vocabulary: str = "canonical_v1") -> np.ndarray:
        names = STYLE_VOCABULARIES[vocabulary]
        value = np.zeros(len(names), dtype=np.float32)
        value[names.index(self.style_name(sequence_id))] = 1.0
        return value

